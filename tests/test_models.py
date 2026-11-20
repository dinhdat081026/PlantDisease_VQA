"""Offline architecture/autograd tests; no optimizer, training run or weight download."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.config import IMAGE_MODEL, SEED, TEXT_MODEL
from src.models import ViTBERT


class FakeViT(nn.Module):
    num_features = 768
    pretrained_cfg = {"input_size": (3, 224, 224), "mean": (0.5, 0.4, 0.3),
                      "std": (0.2, 0.3, 0.4), "interpolation": "bicubic", "crop_pct": .9}

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)
        self.dropout = nn.Dropout(.5)
        self.calls = []
        self.features = []

    def forward(self, pixels):
        self.calls.append({"grad_enabled": torch.is_grad_enabled(), "training": self.training,
                           "pixels": pixels.detach().clone()})
        features = self.dropout(self.projection(pixels.mean(dim=(-2, -1))))
        self.features.append(features.detach().clone())
        return features


class FakeBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.embedding = nn.Embedding(32, 768)
        self.dropout = nn.Dropout(.5)
        self.calls = []
        self.features = []

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        self.calls.append({"grad_enabled": torch.is_grad_enabled(), "training": self.training,
                           "attention_mask": attention_mask.detach().clone(), "return_dict": return_dict})
        hidden = self.dropout(self.embedding(input_ids))
        self.features.append(hidden.detach().clone())
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def factories(monkeypatch):
    import timm
    import transformers
    torch.manual_seed(SEED)
    calls = {"image": [], "text": []}

    def make_image(name, **kwargs):
        calls["image"].append((name, kwargs))
        return FakeViT()

    def make_text(name, **kwargs):
        calls["text"].append((name, kwargs))
        return FakeBERT()

    monkeypatch.setattr(timm, "create_model", make_image)
    monkeypatch.setattr(transformers.BertModel, "from_pretrained", make_text)
    return calls


@pytest.fixture
def inputs():
    return dict(pixel_values=torch.randn(2, 3, 224, 224),
                input_ids=torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
                token_type_ids=torch.zeros(2, 4, dtype=torch.long))


@pytest.mark.parametrize("mode", ["full", "frozen", "question_only", "image_only"])
def test_architecture_logits_shape_and_only_needed_backbones(factories, inputs, mode):
    model = ViTBERT(7, mode).eval()
    logits = model(**inputs)
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()
    assert model.fusion[0].in_features == 1024
    assert model.fusion[0].out_features == 512
    assert isinstance(model.fusion[1], nn.ReLU)
    assert isinstance(model.fusion[2], nn.Dropout) and model.fusion[2].p == .5
    assert model.fusion[3].in_features == 512 and model.fusion[3].out_features == 7
    for branch, enabled in (("image", mode != "question_only"), ("text", mode != "image_only")):
        encoder = getattr(model, branch + "_encoder")
        projection = getattr(model, branch + "_projection")
        assert (encoder is not None) == enabled
        assert (projection is not None) == enabled
        assert len(factories[branch]) == int(enabled)
        if enabled:
            assert (projection.in_features, projection.out_features) == (768, 512)
    if factories["image"]:
        assert factories["image"][0] == (IMAGE_MODEL, {"pretrained": True, "num_classes": 0})
    if factories["text"]:
        assert factories["text"][0] == (TEXT_MODEL, {"add_pooling_layer": False, "attn_implementation": "eager"})


@pytest.mark.parametrize("mode", ["full", "frozen", "question_only", "image_only"])
def test_backward_active_parameters_and_frozen_boundaries(factories, inputs, mode):
    model = ViTBERT(3, mode).train()
    inputs["pixel_values"].requires_grad_()
    model(**inputs).square().mean().backward()
    for name, parameter in model.named_parameters():
        is_encoder = name.startswith(("image_encoder.", "text_encoder."))
        if mode == "frozen" and is_encoder:
            assert not parameter.requires_grad
            assert parameter.grad is None
        else:
            assert parameter.requires_grad
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for branch in ("image", "text"):
        encoder = getattr(model, branch + "_encoder")
        if encoder is not None:
            assert encoder.calls[-1]["grad_enabled"] == (mode != "frozen")
            if mode != "frozen":
                assert sum(p.grad.abs().sum().item() for p in encoder.parameters()) > 0
    if mode in ("full", "image_only"):
        assert inputs["pixel_values"].grad is not None
    else:
        assert inputs["pixel_values"].grad is None


def test_frozen_dropout_stays_off_across_train_eval_and_head_keeps_grad(factories, inputs):
    model = ViTBERT(4, "frozen")
    assert model.training
    for _ in range(2):
        model.eval()
        assert not any(module.training for module in model.modules())
        assert model.train() is model
        assert model.fusion[2].training
        assert model.image_projection.training and model.text_projection.training
        assert not any(module.training for module in model.image_encoder.modules())
        assert not any(module.training for module in model.text_encoder.modules())
    first = model(**inputs)
    model(**inputs)
    assert torch.equal(model.image_encoder.features[-2], model.image_encoder.features[-1])
    assert torch.equal(model.text_encoder.features[-2], model.text_encoder.features[-1])
    assert first.requires_grad  # no_grad did not encompass projection/head


def test_eval_is_deterministic_and_respects_callers_no_grad(factories, inputs):
    model = ViTBERT(3).eval()
    assert torch.equal(model(**inputs), model(**inputs))
    # Standard PyTorch eval semantics: callers may still request gradients.
    assert model(**inputs).requires_grad
    with torch.no_grad():
        output = model(**inputs)
    assert not output.requires_grad
    assert not model.image_encoder.calls[-1]["grad_enabled"]
    assert not model.text_encoder.calls[-1]["grad_enabled"]


@pytest.mark.parametrize("mode,missing", [("full", "pixel_values"), ("full", "input_ids"),
    ("frozen", "pixel_values"), ("frozen", "input_ids"),
    ("question_only", "input_ids"), ("image_only", "pixel_values")])
def test_required_modality_missing_is_clear_error(factories, inputs, mode, missing):
    model = ViTBERT(2, mode)
    inputs.pop(missing)
    with pytest.raises(ValueError, match=missing):
        model(**inputs)


def test_unused_modalities_are_ignored_and_fusion_has_zero_branch(factories, inputs):
    text_model = ViTBERT(2, "question_only").eval()
    captured = []
    hook = text_model.fusion[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    assert text_model(input_ids=inputs["input_ids"]).shape == (2, 2)
    # Invalid images are irrelevant to the question-only ablation.
    assert text_model(pixel_values=object(), input_ids=inputs["input_ids"], image_format="invalid").shape == (2, 2)
    assert torch.equal(captured[-1][:, :512], torch.zeros(2, 512))
    hook.remove()
    image_model = ViTBERT(2, "image_only").eval()
    captured.clear()
    hook = image_model.fusion[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    assert image_model(pixel_values=inputs["pixel_values"], input_ids=object(), attention_mask=object()).shape == (2, 2)
    assert torch.equal(captured[-1][:, 512:], torch.zeros(2, 512))
    hook.remove()


def test_uint8_is_converted_and_normalized_using_pretrained_config(factories):
    model = ViTBERT(2, "image_only").eval()
    raw = torch.full((2, 3, 224, 224), 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="floating point"):
        model(pixel_values=raw)
    model(pixel_values=raw, image_format="uint8")
    observed = model.image_encoder.calls[-1]["pixels"]
    expected = (raw.float() / 255 - model.image_mean) / model.image_std
    assert observed.dtype == torch.float32
    assert torch.allclose(observed, expected)
    # Already normalized pipeline output must be passed through unchanged.
    model(pixel_values=expected)
    assert torch.equal(model.image_encoder.calls[-1]["pixels"], expected)
    model(pixel_values=raw.float() / 255, image_format="unit_float")
    assert torch.allclose(model.image_encoder.calls[-1]["pixels"], expected)
    assert (raw == 128).all()  # preprocessing never mutates the source


@pytest.mark.parametrize("kind", ["shape", "nan", "unit_range", "wrong_raw_dtype", "unknown_format"])
def test_bad_images_rejected_before_vit(factories, kind):
    model = ViTBERT(2, "image_only")
    pixels = torch.zeros(2, 3, 224, 224)
    image_format = "normalized"
    if kind == "shape":
        pixels = torch.zeros(2, 224, 224, 3)
    elif kind == "nan":
        pixels[0, 0, 0, 0] = float("nan")
    elif kind == "unit_range":
        pixels.fill_(255)
        image_format = "unit_float"
    elif kind == "wrong_raw_dtype":
        image_format = "uint8"
    else:
        image_format = "guess"
    with pytest.raises(ValueError):
        model(pixel_values=pixels, image_format=image_format)
    assert not model.image_encoder.calls


def test_text_masks_cls_and_batch_validation(factories, inputs):
    model = ViTBERT(2).eval()
    with pytest.raises(ValueError, match="batch sizes"):
        model(pixel_values=inputs["pixel_values"][:1], input_ids=inputs["input_ids"])
    with pytest.raises(ValueError, match="attention_mask"):
        model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"], attention_mask=torch.ones(2, 3))
    with pytest.raises(ValueError, match="torch.long"):
        model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"].float())
    captured = []
    hook = model.text_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
    assert torch.equal(model.text_encoder.calls[-1]["attention_mask"], torch.ones_like(inputs["input_ids"]))
    assert torch.equal(captured[-1], model.text_encoder.features[-1][:, 0, :])
    hook.remove()


@pytest.mark.parametrize("num_classes,mode", [(0, "full"), (-1, "full"), (True, "full"), (2.5, "full"), (2, "unknown")])
def test_invalid_configuration_fails_before_loading(factories, num_classes, mode):
    with pytest.raises(ValueError):
        ViTBERT(num_classes, mode)
    assert factories == {"image": [], "text": []}
