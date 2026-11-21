"""Offline rollout checks, independent of trained weights and Streamlit UI."""
import io
import pytest
import torch
from PIL import Image
from src.app import attention_rollout, decode_upload


def test_rollout_known_uniform_attention():
    attention = torch.full((1, 2, 5, 5), .2)
    result = attention_rollout([attention, attention])
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.ones(2, 2))


def test_rollout_composes_layers_in_forward_order():
    first = torch.eye(5).reshape(1, 1, 5, 5)
    second = first.clone()
    first[0, 0, 1] = torch.tensor([0., 0., 1., 0., 0.])
    second[0, 0, 0] = torch.tensor([0., 1., 0., 0., 0.])
    expected = ((second[0, 0] + torch.eye(5)) / 2) @ ((first[0, 0] + torch.eye(5)) / 2)
    expected = expected[0, 1:].reshape(2, 2)
    assert torch.allclose(attention_rollout([first, second]), expected / expected.max())


def test_rollout_timm_attention_capture():
    from timm.models.vision_transformer import VisionTransformer
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=2).eval()
    captured, handles = [], []
    for block in model.blocks:
        block.attn.fused_attn = False
        handles.append(block.attn.attn_drop.register_forward_hook(lambda m, a, out: captured.append(out)))
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 32, 32))
            result = attention_rollout(captured)
        assert len(captured) == 2
        assert result.shape == (2, 2) and torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()
    finally:
        for handle in handles:
            handle.remove()


def test_rollout_rejects_bad_input():
    with pytest.raises(ValueError):
        attention_rollout([])
    with pytest.raises(ValueError):
        attention_rollout([torch.full((1, 1, 5, 5), float('nan'))])


def test_upload_validation():
    stream = io.BytesIO()
    Image.new('RGB', (8, 8)).save(stream, format='PNG')
    assert decode_upload(stream.getvalue()).size == (8, 8)
    with pytest.raises(ValueError, match='10 MiB'):
        decode_upload(b'x' * (10 * 1024 * 1024 + 1))
