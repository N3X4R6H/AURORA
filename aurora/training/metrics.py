"""Loss, throughput and memory metrics for AURORA training/evaluation."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # targets with this value are excluded from the loss


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 ignore_index: int = IGNORE_INDEX,
                 reduction: str = "mean") -> torch.Tensor:
    """Next-token cross-entropy.

    logits:  [batch, seq, vocab]
    targets: [batch, seq]  (already aligned/shifted by the dataset)
    Returns a scalar. Computed in float32 for numerical stability.
    """
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
        )
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def perplexity(mean_loss: float) -> float:
    """exp(mean per-token loss); inf on overflow."""
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


class LossAverager:
    """Token-weighted mean loss, for validation over many batches."""

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        self.ignore_index = ignore_index
        self.total_loss = 0.0
        self.total_tokens = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        n = int((targets != self.ignore_index).sum())
        if n == 0:
            return
        self.total_loss += compute_loss(logits, targets, self.ignore_index, "sum").item()
        self.total_tokens += n

    @property
    def mean(self) -> float:
        return self.total_loss / self.total_tokens if self.total_tokens else float("nan")


class ThroughputMeter:
    """Wall-clock tokens/sec, excluding warmup steps."""

    def __init__(self, device: torch.device | str | None = None, warmup_steps: int = 2):
        self.device = torch.device(device) if device is not None else None
        self.warmup_steps = warmup_steps
        self.tokens = 0
        self._seen = 0
        self._last = None
        self._sync()
        self._t0 = time.perf_counter()

    def _sync(self) -> None:
        # CUDA kernels are async; without this the clock stops too early.
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self, num_tokens: int) -> None:
        """Call once per optimizer step, after the step completes."""
        self._sync()
        now = time.perf_counter()
        self._seen += 1
        if self._seen <= self.warmup_steps:
            self._t0 = now  # restart the clock after each warmup step
            return
        self.tokens += num_tokens
        self._last = now

    @property
    def tokens_per_sec(self) -> float:
        if self._last is None:
            return 0.0
        elapsed = self._last - self._t0
        return self.tokens / elapsed if elapsed > 0 else 0.0


def reset_peak_memory(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device | str) -> dict[str, float] | None:
    """Peak CUDA memory since the last reset. None on CPU (use OS tools)."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    mb = 1024 ** 2
    return {
        "allocated_mb": torch.cuda.max_memory_allocated(device) / mb,
        "reserved_mb": torch.cuda.max_memory_reserved(device) / mb,
    }
