"""Paired bootstrap for retrieval configuration comparisons."""
from __future__ import annotations
import random
from collections.abc import Sequence


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (observed mean delta, 95% CI low, 95% CI high).

    Paired, because both configurations answer the SAME questions — an
    unpaired test throws away the pairing and widens the interval for nothing.
    Resamples questions with replacement; the interval is the 2.5/97.5
    percentiles of the resampled mean differences.
    """
    if len(baseline) != len(candidate):
        raise ValueError("configurations must cover the same questions")
    n = len(baseline)
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    observed = sum(deltas) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations)]
    return observed, lo, hi


def verdict(observed: float, lo: float, hi: float) -> str:
    """One line a reader can act on."""
    if lo > 0:
        return f"+{observed:.3f} [95% CI {lo:+.3f}, {hi:+.3f}] — improvement"
    if hi < 0:
        return f"{observed:.3f} [95% CI {lo:+.3f}, {hi:+.3f}] — regression"
    return f"{observed:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] — not distinguishable"