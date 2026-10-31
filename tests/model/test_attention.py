import pytest
import torch

attention = pytest.importorskip("aurora.model.attention")  # skipped until implemented


@pytest.fixture
def attn():
    # TODO: adjust to the agreed config in aurora/model/config.py
    from aurora.model.config import ModelConfig
    cfg = ModelConfig(n_embd=64, n_head=4, block_size=32)
    return attention.CausalSelfAttention(cfg).eval()


def test_output_shape(attn):
    x = torch.randn(2, 16, 64)
    assert attn(x).shape == (2, 16, 64)


def test_future_tokens_do_not_affect_past(attn):
    x = torch.randn(1, 12, 64)
    x2 = x.clone()
    x2[:, 8:] = torch.randn(1, 4, 64)  # change only positions 8..11
    with torch.no_grad():
        y, y2 = attn(x), attn(x2)
    assert torch.allclose(y[:, :8], y2[:, :8], atol=1e-6)   # past unchanged
    assert not torch.allclose(y[:, 8:], y2[:, 8:])          # sanity: future did change
