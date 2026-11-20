"""ViT/BERT VQA classifier. No training loop or implicit input normalization.

Inputs from PlantVQADataset are already normalized: use image_format='normalized'
(the default) so normalization is NOT applied twice. Raw RGB batches of shape
(B,3,224,224) can instead declare image_format='uint8' or 'unit_float'. These are
converted/normalized from the actual timm pretrained config before ViT sees them.
Resize/crop stays in the data pipeline; this model never silently resizes images.
"""
from contextlib import nullcontext
import os

# This project uses the PyTorch backend only. Set before any Transformers import.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import torch
from torch import nn

from .config import ABLATIONS, IMAGE_MODEL, TEXT_MODEL


class ViTBERT(nn.Module):
    """Return logits (B,num_classes); use BERT's final CLS hidden state.

    Both encoders output 768 features, each active branch projects to 512.
    Ablations do not construct the disabled encoder OR its projection. The absent
    branch is zero-filled at fusion to keep the specified 1024->512->C head.
    In frozen mode only the two encoders are frozen; projections/head learn.
    eval() changes dropout behavior, but does not globally disable autograd;
    callers use torch.no_grad() or inference_mode() for inference.
    """

    def __init__(self, num_classes: int, mode: str = "full", *, pretrained: bool = True):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if mode not in ABLATIONS:
            raise ValueError(f"Unsupported mode {mode!r}; expected {ABLATIONS}")
        self.mode = mode
        self.num_classes = num_classes
        self.image_encoder = None
        self.text_encoder = None
        self.image_projection = None
        self.text_projection = None
        self.image_data_config = None
        if mode != "question_only":
            import timm
            from timm.data import resolve_data_config
            self.image_encoder = timm.create_model(IMAGE_MODEL, pretrained=pretrained, num_classes=0)
            if self.image_encoder.num_features != 768:
                raise ValueError("ViT must produce 768-dimensional features")
            self.image_data_config = resolve_data_config(model=self.image_encoder)
            if tuple(self.image_data_config["input_size"]) != (3, 224, 224):
                raise ValueError("ViT pretrained input config must be (3,224,224)")
            self.register_buffer("image_mean", torch.tensor(self.image_data_config["mean"]).reshape(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor(self.image_data_config["std"]).reshape(1, 3, 1, 1))
            self.image_projection = nn.Linear(768, 512)
        if mode != "image_only":
            from transformers import BertConfig, BertModel
            if pretrained:
                self.text_encoder = BertModel.from_pretrained(
                    TEXT_MODEL, add_pooling_layer=False, attn_implementation="eager")
            else:
                self.text_encoder = BertModel(
                    BertConfig(attn_implementation="eager"), add_pooling_layer=False)
            if self.text_encoder.config.hidden_size != 768:
                raise ValueError("BERT must produce 768-dimensional hidden states")
            self.text_projection = nn.Linear(768, 512)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        if mode == "frozen":
            for encoder in (self.image_encoder, self.text_encoder):
                encoder.requires_grad_(False)
                encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "frozen":
            # Parent train() recursively enables dropout, so undo it for encoders.
            self.image_encoder.eval()
            self.text_encoder.eval()
        return self

    def prepare_images(self, pixel_values, *, image_format="normalized"):
        """Explicit formats prevent accidental uint8 input and double normalization."""
        if self.image_encoder is None:
            raise ValueError("This mode has no image encoder")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("pixel_values must be a tensor")
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, 224, 224) or pixel_values.shape[0] < 1:
            raise ValueError("pixel_values must have shape (B,3,224,224), B > 0")
        if image_format == "normalized":
            if not pixel_values.is_floating_point():
                raise ValueError("Normalized images must be floating point; declare image_format='uint8' for raw uint8")
            result = pixel_values
        elif image_format == "uint8":
            if pixel_values.dtype != torch.uint8:
                raise ValueError("image_format='uint8' requires torch.uint8")
            result = pixel_values.float() / 255.0
        elif image_format == "unit_float":
            if not pixel_values.is_floating_point():
                raise ValueError("unit_float images must be floating point in [0,1]")
            if not ((pixel_values >= 0) & (pixel_values <= 1)).all():
                raise ValueError("unit_float images must be in [0,1]")
            result = pixel_values.float()
        else:
            raise ValueError("image_format must be normalized, uint8 or unit_float")
        if not torch.isfinite(result).all():
            raise ValueError("Images contain nonfinite values")
        if image_format != "normalized":
            result = (result - self.image_mean.to(result)) / self.image_std.to(result)
        return result

    @staticmethod
    def _validate_text(input_ids, attention_mask, token_type_ids):
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have shape (B,L), B and L > 0")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        for name, value in (("attention_mask", attention_mask), ("token_type_ids", token_type_ids)):
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.shape != input_ids.shape:
                    raise ValueError(f"{name} must have the same shape as input_ids")
                if value.device != input_ids.device:
                    raise ValueError(f"{name} and input_ids must be on the same device")
        if attention_mask.dtype not in (torch.bool, torch.long):
            raise ValueError("attention_mask must have dtype bool or torch.long")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only zero/one")
        if token_type_ids is not None and token_type_ids.dtype != torch.long:
            raise ValueError("token_type_ids must have dtype torch.long")
        return attention_mask

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                token_type_ids=None, *, image_format="normalized"):
        # Ignore an unused modality completely, even if supplied by a common loader.
        image_input = None
        if self.image_encoder is not None:
            if pixel_values is None:
                raise ValueError(f"pixel_values is required in {self.mode} mode")
            image_input = self.prepare_images(pixel_values, image_format=image_format)
        if self.text_encoder is not None:
            if input_ids is None:
                raise ValueError(f"input_ids is required in {self.mode} mode")
            attention_mask = self._validate_text(input_ids, attention_mask, token_type_ids)
            if image_input is not None:
                if image_input.shape[0] != input_ids.shape[0]:
                    raise ValueError("Image/text batch sizes must match")
                if image_input.device != input_ids.device:
                    raise ValueError("Image and text inputs must be on the same device")
        image_features = text_features = None
        # no_grad only around frozen encoders: the projections/head still need gradients.
        with torch.no_grad() if self.mode == "frozen" else nullcontext():
            if self.image_encoder is not None:
                image_features = self.image_encoder(image_input)
                if image_features.shape != (image_input.shape[0], 768):
                    raise ValueError("ViT output must have shape (B,768)")
            if self.text_encoder is not None:
                output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids, return_dict=True)
                if output.last_hidden_state.shape != (*input_ids.shape, 768):
                    raise ValueError("BERT last_hidden_state must have shape (B,L,768)")
                text_features = output.last_hidden_state[:, 0, :]
        image_projected = self.image_projection(image_features) if image_features is not None else None
        text_projected = self.text_projection(text_features) if text_features is not None else None
        if image_projected is None:
            image_projected = torch.zeros_like(text_projected)
        if text_projected is None:
            text_projected = torch.zeros_like(image_projected)
        return self.fusion(torch.cat((image_projected, text_projected), dim=-1))
