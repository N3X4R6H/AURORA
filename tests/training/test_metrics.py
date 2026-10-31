import math

import pytest
import torch

from aurora.training import metrics
from aurora.training.metrics import (
    IGNORE_INDEX, LossAverager, ThroughputMeter, compute_loss, perplexity,
)


def test_uniform_logits_give_log_vocab():
    V = 50
    logits = torch.zeros(2, 8, V)
    targets = torch.randint(0, V, (2, 8))
    assert compute_loss(logits, targets).item() == pytest.approx(math.log(V), rel=1e-5)


def test_ignored_positions_do_not_affect_loss():
    logits = torch.randn(1, 4, 10)
    t = torch.tensor([[1, 2, 3, 4]])
    t_masked = torch.tensor([[1, 2, IGNORE_INDEX, IGNORE_INDEX]])
    expected = compute_loss(logits[:, :2], t[:, :2])
    assert compute_loss(logits, t_masked).item() == pytest.approx(expected.item(), rel=1e-5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(2, 4, 10), torch.zeros(2, 5, dtype=torch.long))


def test_perplexity():
    assert perplexity(math.log(50)) == pytest.approx(50)
    assert perplexity(10_000) == float("inf")


def test_loss_averager_is_token_weighted():
    avg = LossAverager()
    a_logits, a_t = torch.randn(1, 2, 10), torch.randint(0, 10, (1, 2))
    b_logits, b_t = torch.randn(1, 8, 10), torch.randint(0, 10, (1, 8))
    avg.update(a_logits, a_t)
    avg.update(b_logits, b_t)
    all_loss = (compute_loss(a_logits, a_t, reduction="sum")
                + compute_loss(b_logits, b_t, reduction="sum")) / 10
    assert avg.mean == pytest.approx(all_loss.item(), rel=1e-5)


def test_throughput_excludes_warmup(monkeypatch):
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0])  # init, then 4 updates
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(clock))
    m = ThroughputMeter(device="cpu", warmup_steps=2)
    for _ in range(4):
        m.update(100)
    # warmup resets the clock at t=2; two counted steps end at t=4 -> 200 tokens / 2 s
    assert m.tokens_per_sec == pytest.approx(100.0)


def test_peak_memory_none_on_cpu():
    assert metrics.peak_memory_mb("cpu") is None


@pytest.mark.gpu
def test_peak_memory_reports_on_cuda():
    metrics.reset_peak_memory("cuda")
    _ = torch.zeros(1024, 1024, device="cuda")
    out = metrics.peak_memory_mb("cuda")
    assert out["allocated_mb"] >= 4.0 and out["reserved_mb"] >= out["allocated_mb"]
