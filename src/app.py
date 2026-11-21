"""Run from repo root: streamlit run src/app.py."""
from pathlib import Path
import io
import importlib.util
import math
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
from PIL import Image
from src.config import IMAGE_MODEL, TEXT_MODEL
from src.evaluate import load_checkpoint, restore_tokenizer
from src.models import ViTBERT

SAMPLES = {
    "Detailed Verification": "Is this leaf diseased?",
    "General Health Assessment": "Is this leaf healthy?",
    "Plant Species Identification": "What plant species is this?",
    "Visual Attribute Grounding": "Are there spots on the leaf?",
}


def check_runtime_dependencies():
    """Show install commands for the interpreter actually running Streamlit."""
    required = {"transformers": "transformers==4.49.0",
                "timm": "timm==1.0.15", "matplotlib": "matplotlib==3.10.1"}
    missing = [package for module, package in required.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return True
    st.error("Môi trường Python chạy Streamlit thiếu dependency: " + ", ".join(missing))
    st.write("Dừng app, chạy các lệnh sau trong PowerShell, rồi mở lại app:")
    # PowerShell quoted executable invocation also handles paths with spaces.
    executable = sys.executable.replace("'", "''")
    app_path = str(ROOT / "src/app.py").replace("'", "''")
    st.code(f"& '{executable}' -m pip install {' '.join(missing)}\n"
            f"& '{executable}' -m streamlit run '{app_path}'", language="powershell")
    st.caption(f"Python đang dùng: {sys.executable}")
    return False


def decode_upload(content):
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Maximum upload size is 10 MiB")
    with Image.open(io.BytesIO(content)) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPEG and PNG are supported")
        if image.width * image.height > 20_000_000:
            raise ValueError("Maximum image size is 20 megapixels")
        image.load()
        return image.convert("RGB")


def attention_rollout(attentions):
    """Average heads, add residual, normalize rows, compose layers in order."""
    if not attentions:
        raise ValueError("No attention matrices captured")
    tokens = attentions[0].shape[-1]
    side = math.isqrt(tokens - 1)
    if side * side != tokens - 1:
        raise ValueError("Expected CLS plus square patch grid")
    joint = torch.eye(tokens, device=attentions[0].device)
    for attention in attentions:
        if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2:] != (tokens, tokens):
            raise ValueError("Expected attention shape (1, heads, tokens, tokens)")
        if not torch.isfinite(attention).all() or (attention < 0).any():
            raise ValueError("Invalid attention weights")
        matrix = attention.float().mean(1)[0] + torch.eye(tokens, device=joint.device)
        matrix = matrix / matrix.sum(-1, keepdim=True)
        joint = matrix @ joint
    mask = joint[0, 1:].reshape(side, side)
    return mask / mask.max().clamp_min(1e-12)


@st.cache_resource
def load_bundle(path_string, mtime_ns, size, device):
    path = Path(path_string).resolve()
    if not path.is_relative_to((ROOT / "outputs/models").resolve()):
        raise ValueError("Checkpoint must be inside outputs/models")
    checkpoint = load_checkpoint(path)
    metadata = checkpoint["preprocessing"]
    if metadata["image_model"] != IMAGE_MODEL or metadata["text_model"] != TEXT_MODEL or metadata["image_format"] != "normalized":
        raise ValueError("Incompatible preprocessing metadata")
    from timm.data import create_transform
    transform = create_transform(**metadata["data_config"], is_training=False)
    with tempfile.TemporaryDirectory() as directory:
        tokenizer = restore_tokenizer(checkpoint["tokenizer"], directory)
    model = ViTBERT(len(checkpoint["answer_to_idx"]), checkpoint["mode"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, transform, checkpoint, threading.RLock()


def predict(bundle, image, question, device):
    model, tokenizer, transform, checkpoint, lock = bundle
    pixels = transform(image).unsqueeze(0).to(device)
    tokens = tokenizer(question, truncation=True, padding="max_length",
                       max_length=checkpoint["preprocessing"]["max_length"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in tokens.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
    captured, handles, original = [], [], []
    # Cached model is shared: serialize temporary attention hooks across sessions.
    with lock, torch.inference_mode():
        model.eval()
        try:
            if model.image_encoder is not None:
                for block in model.image_encoder.blocks:
                    attention = block.attn
                    original.append((attention, attention.fused_attn))
                    attention.fused_attn = False
                    handles.append(attention.attn_drop.register_forward_hook(
                        lambda module, args, output: captured.append(output.detach())))
            logits = model(pixel_values=pixels, **inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite predictions")
            values, indices = logits.float().softmax(-1)[0].topk(min(3, logits.shape[-1]))
            vocab = {v: k for k, v in checkpoint["answer_to_idx"].items()}
            results = [{"Answer": vocab[i], "Probability": p} for i, p in zip(indices.tolist(), values.tolist())]
            mask = attention_rollout(captured).cpu() if captured else None
        finally:
            for handle in handles:
                handle.remove()
            for attention, fused in original:
                attention.fused_attn = fused
    cfg = checkpoint["preprocessing"]["data_config"]
    view = pixels[0].cpu() * torch.tensor(cfg["std"])[:, None, None] + torch.tensor(cfg["mean"])[:, None, None]
    return results, view.clamp(0, 1).permute(1, 2, 0).numpy(), mask


def main():
    st.title("Plant Disease VQA")
    st.warning("Chỉ phục vụ học tập, không thay thế chuyên gia nông nghiệp.")
    if not check_runtime_dependencies():
        return
    candidates = sorted(p for p in (ROOT / "outputs/models").glob("*.pt") if not p.stem.endswith("_last"))
    if not candidates:
        st.error("Thiếu checkpoint. Sao chép best checkpoint đã train vào outputs/models/.")
        return
    selected = st.selectbox("Checkpoint", candidates, format_func=lambda p: p.name)
    upload = st.file_uploader("Ảnh lá cây (tối đa 10 MiB, 20 megapixel)", type=["jpg", "jpeg", "png"])
    kind = st.selectbox("Question type", list(SAMPLES))
    sample = st.selectbox("Câu hỏi mẫu", [SAMPLES[kind], "Tự nhập"])
    question = st.text_input("Question (English)", value="" if sample == "Tự nhập" else sample, max_chars=500)
    st.caption("Question type chỉ chọn mẫu; mô hình nhận ảnh và nội dung câu hỏi.")
    if st.button("Predict", disabled=upload is None or not question.strip()):
        try:
            image = decode_upload(upload.getvalue())
            device = "cuda" if torch.cuda.is_available() else "cpu"
            stat = selected.stat()
            with st.spinner("Predicting…"):
                bundle = load_bundle(str(selected), stat.st_mtime_ns, stat.st_size, device)
                answers, view, mask = predict(bundle, image, question.strip(), device)
            st.dataframe(answers, hide_index=True)
            st.caption("Softmax probabilities trên vocabulary train; chưa được hiệu chuẩn.")
            if mask is None:
                st.info("Question-only checkpoint không có image attention.")
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.imshow(view)
                ax.imshow(mask.numpy(), cmap="jet", alpha=.45, interpolation="bilinear",
                          extent=(-.5, view.shape[1]-.5, view.shape[0]-.5, -.5))
                ax.set_title("ViT attention rollout — model input crop")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Attention rollout trực quan hóa attention trên crop đầu vào, không phải giải thích nhân quả và không đặc hiệu cho câu hỏi hay đáp án.")
        except Exception as error:
            st.error(f"Không thể suy luận: {error}")


if __name__ == "__main__":
    main()
