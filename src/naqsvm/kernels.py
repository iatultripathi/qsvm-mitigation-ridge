"""Kernel post-processing and kernel-quality metrics.

ESTABLISHED operations (not claimed as contributions):
  * eigenvalue clipping  = Frobenius-nearest PSD matrix (Higham 1988); this is what
    Qiskit's FidelityQuantumKernel(enforce_psd=True) does to training kernels;
  * eigenvalue flipping and diagonal shifting (indefinite-kernel literature,
    e.g. Chen et al., JMLR 2009);
  * depolarizing-noise mitigation from the measured diagonal ("average" and
    "split channel" strategies, Hubregtsen et al., PRA 2022). Implemented here from
    the noise model stated in that paper, not copied from any library.

Model behind the mitigation (d = 2**n):
    global depolarizing with survival prob. Lam :  k_obs = Lam * k + (1 - Lam) / d
    since k(x, x) = 1:                             Lam  = (k_obs(x,x) - 1/d) / (1 - 1/d)
    inverse:                                       k    = (k_obs - 1/d) / Lam + 1/d
"""
from __future__ import annotations

import numpy as np


def symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def min_eig(K: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(symmetrize(K))[0])


def psd_clip(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(symmetrize(K))
    return symmetrize((V * np.clip(w, 0.0, None)) @ V.T)


def psd_flip(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(symmetrize(K))
    return symmetrize((V * np.abs(w)) @ V.T)


def psd_shift(K: np.ndarray) -> np.ndarray:
    lam = min_eig(K)
    return K + max(0.0, -lam) * np.eye(len(K))


def is_psd(K: np.ndarray, tol: float = 1e-9) -> bool:
    return min_eig(K) >= -tol * max(1.0, np.abs(K).max())


def survival_from_diag(diag_obs: np.ndarray, d: int, floor: float = 0.05) -> np.ndarray:
    lam = (np.asarray(diag_obs, dtype=float) - 1.0 / d) / (1.0 - 1.0 / d)
    return np.clip(lam, floor, 1.0)


def mitigate(K_obs: np.ndarray, lam, d: int) -> np.ndarray:
    """Invert global depolarizing noise. lam: scalar or array broadcastable to K_obs."""
    return (K_obs - 1.0 / d) / lam + 1.0 / d


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def offdiag(K: np.ndarray) -> np.ndarray:
    return K[~np.eye(len(K), dtype=bool)]


def rel_fro_error_offdiag(K: np.ndarray, K_ref: np.ndarray) -> float:
    a, b = offdiag(K), offdiag(K_ref)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def centered_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Centered kernel alignment (Cortes, Mohri & Rostamizadeh, JMLR 2012).

    Invariant to positive rescaling and to adding a constant matrix, so it measures
    *shape* agreement with the ideal kernel, not calibration.
    """
    n = len(K1)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ K1 @ H, H @ K2 @ H
    den = np.linalg.norm(A) * np.linalg.norm(B)
    return float(np.sum(A * B) / den) if den > 0 else float("nan")
