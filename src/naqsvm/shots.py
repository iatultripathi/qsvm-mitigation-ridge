"""Finite-shot measurement model and shot-allocation rules.

Measurement model (exact): each kernel entry e is a Bernoulli experiment with
success probability p_e = Pr[0...0]; with S_e shots the count n_e ~ Binomial(S_e, p_e)
and k_hat_e = n_e / S_e is unbiased with Var = p_e (1 - p_e) / S_e.

Allocation rules
----------------
uniform  : S_e = B / M for all M entries (largest-remainder rounding; sum == B exactly).
neyman   : two-phase. Phase 1 spends a pilot of s0 shots on every entry. Phase 2 spends
           the rest R = B - M s0 to minimise  sum_e w_e Var(k_hat_e)  =  sum_e w_e v_e / S_e
           subject to sum_e S_e = B. Lagrange / Cauchy-Schwarz gives S_e ∝ sqrt(w_e v_e)
           (Neyman 1934 optimal allocation). v_e is estimated from the pilot with the
           Laplace-smoothed proportion p~ = (n+1)/(s0+2) so zero counts still get shots.
           w_e = 1 minimises the expected squared Frobenius error of the kernel estimate.
Note: pooling pilot and phase-2 counts after a data-dependent allocation introduces an
O(1/S) bias; the kernel-error metrics reported in the experiments include it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer vector proportional to weights that sums exactly to total."""
    weights = np.asarray(weights, dtype=float)
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=np.int64)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones_like(weights)
    raw = total * weights / weights.sum()
    base = np.floor(raw).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1
    return base


@dataclass
class Measurement:
    counts: np.ndarray   # number of 0...0 outcomes per entry
    shots: np.ndarray    # shots spent per entry

    @property
    def estimate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.shots > 0, self.counts / np.maximum(self.shots, 1), np.nan)

    @property
    def variance(self) -> np.ndarray:
        """Plug-in variance of k_hat with Laplace smoothing (never exactly zero)."""
        pt = (self.counts + 1.0) / (self.shots + 2.0)
        return pt * (1 - pt) / np.maximum(self.shots, 1)

    @property
    def total_shots(self) -> int:
        return int(self.shots.sum())


def measure_uniform(p: np.ndarray, budget: int, rng: np.random.Generator) -> Measurement:
    S = largest_remainder(np.ones(len(p)), budget)
    return Measurement(rng.binomial(S, np.clip(p, 0, 1)), S)


def measure_two_phase(p: np.ndarray, budget: int, rng: np.random.Generator,
                      pilot_fraction: float = 0.25,
                      weight_fn: Optional[Callable[[Measurement], np.ndarray]] = None) -> Measurement:
    """Pilot + Neyman allocation. weight_fn(pilot_measurement) -> importance weights w_e."""
    M = len(p)
    if M == 0:
        return Measurement(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    s0 = int(max(1, np.floor(pilot_fraction * budget / M)))
    if s0 * M > budget:                          # budget too small for a pilot: uniform
        return measure_uniform(p, budget, rng)
    S1 = np.full(M, s0, dtype=np.int64)
    n1 = rng.binomial(S1, np.clip(p, 0, 1))
    pilot = Measurement(n1, S1)
    pt = (n1 + 1.0) / (s0 + 2.0)
    v = pt * (1 - pt)
    w = np.ones(M) if weight_fn is None else np.asarray(weight_fn(pilot), dtype=float)
    S2 = largest_remainder(np.sqrt(np.maximum(w * v, 1e-12)), budget - s0 * M)
    n2 = rng.binomial(S2, np.clip(p, 0, 1))
    return Measurement(n1 + n2, S1 + S2)


def neyman_efficiency(p: np.ndarray) -> float:
    """Oracle variance ratio Var_neyman / Var_uniform at equal budget (<= 1).

    sum v/S with S ∝ sqrt(v) gives (sum sqrt v)^2 / B; uniform gives M sum v / B.
    """
    v = np.clip(p, 0, 1) * (1 - np.clip(p, 0, 1))
    if v.sum() <= 0:
        return 1.0
    return float(np.sum(np.sqrt(v)) ** 2 / (len(v) * v.sum()))
