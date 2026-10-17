"""Revision experiments (v3): answers to the four reviewer-style weaknesses of v2.

W1  C1 "fails" at 64 shots.
    Corollary 1 (finite-shot, model-free identity). Let K_raw be ANY measured kernel with diag fixed to 1,
    let Lam_hat be ANY scalar in (0, 1] (e.g. the diagonal-based estimate) and let
        K_mit = (K_raw - 1/d)/Lam_hat + 1/d   (off-diagonal),  diag(K_mit) = 1      (global "average" mitigation)
    applied identically to validation/test rows. Then, algebraically,
        K_raw = Lam_hat * (K_mit + r_hat I) + ((1 - Lam_hat)/d) 11^T,   r_hat = (1-Lam_hat)(1-1/d)/Lam_hat,
    and every test row satisfies k_raw = Lam_hat * k_mit + (1-Lam_hat)/d. By the SVM scaling/offset lemma,
        SVM(K_raw, C)  ==  SVM(K_mit + r_hat I, Lam_hat C)        (identical decision functions)
    for every shot budget and every noise channel. No noise model is needed: global-average mitigation
    IS the removal of the ridge r_hat plus a rescaling of C. Theorem 1 (ideal kernel + ridge) is the special
    case "global depolarizing + infinite shots", where K_mit = K_ideal. The v2 test compared the finite-shot
    raw SVM with the *noise-free* ideal kernel + ridge, so it mixed the ridge with shot noise; the correct
    finite-shot comparison is against K_mit + r_hat I measured from the same record.
    Consequence: global mitigation does not change the signal-to-noise ratio of the kernel at all
    (signal and shot noise are both divided by Lam_hat); it changes only the ridge and the scale of C.

W2  "Mitigation can hurt" was weakly supported.
    Prediction (from Cor. 1): the accuracy effect of mitigation equals the effect of removing a ridge r_hat
    from the SVM. A ridge on the kernel acts like an extra (squared-hinge) soft-margin term, which overlaps
    with the box constraint C. Hence (P2a) at FIXED C, mitigation changes accuracy, and the change grows with
    r_hat (stronger noise); (P2b) with C TUNED on validation, the effect shrinks towards zero (equivalence).
    Tested at fixed C in {0.1, 1, 10, 100} and with tuned C, on the same measurement record (exactly paired).

W3  Hidden-ridge accuracy test used fixed C = 1.
    Re-run with C tuned on the validation split for the ideal kernel with and without the hidden ridge,
    plus a ridge sweep (multiples of r) with tuned C.

W4  Small problems / simulation only / classical baseline stronger.
    Scale-up runs: N_train = 200 (4 qubits) and 6 qubits; heterogeneous device-like noise models
    (per-qubit depolarizing + T1/T2 relaxation + asymmetric readout + coherent ZZ over-rotation);
    a quantum-structured dataset (Havlicek et al. 2019 "ad hoc" construction) where the ideal quantum kernel
    is expected to beat the RBF-SVM. Hardware runs remain out of scope (no hardware access in this study).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

from . import kernels as kx
from .config import NoiseSpec
from .simulators import (NumpyDensityMatrixSimulator, apply_readout_all_zero, exact_all_zero_probability,
                         feature_statevectors, op_angles)

# ----------------------------------------------------------------------------------------------
# Device-like heterogeneous noise model
# ----------------------------------------------------------------------------------------------


@dataclass
class DeviceNoise(NoiseSpec):
    """Heterogeneous per-qubit noise ('device-like'; illustrative, not a specific backend).

    After every H: 1-qubit depolarizing p1_q[q] then thermal relaxation (T1_q, T2_q, t_1q).
    After every CX: 2-qubit depolarizing p2 (per pair) then thermal relaxation on both qubits (t_2q).
    Phase gates P are virtual (noise-free) except that the ZZ phase angles carry a COHERENT relative
    over-rotation `zz_overrot` (not removable by depolarizing mitigation).
    Readout: asymmetric per qubit, P(read 1 | 0) = ro10_q, P(read 0 | 1) = ro01_q.
    """
    p1_q: List[float] = field(default_factory=list)
    p2_pair: List[float] = field(default_factory=list)
    t1_q: List[float] = field(default_factory=list)
    t2_q: List[float] = field(default_factory=list)
    ro10_q: List[float] = field(default_factory=list)
    ro01_q: List[float] = field(default_factory=list)
    zz_overrot: float = 0.0


DEVICE_PROFILES = {
    # medians (heterogeneity: log-normal, sigma = 0.3, fixed seed per profile & qubit count)
    "device A (low-error class)": dict(p1=2.5e-4, p2=4e-3, t1=250e-6, t2=150e-6, t_1q=32e-9, t_2q=84e-9,
                                       ro10=0.008, ro01=0.020, zz_overrot=0.005),
    "device B (higher-error class)": dict(p1=4e-4, p2=1.2e-2, t1=200e-6, t2=120e-6, t_1q=36e-9, t_2q=500e-9,
                                          ro10=0.015, ro01=0.035, zz_overrot=0.01),
}


def make_device_noise(name: str, n: int, seed: int = 7) -> DeviceNoise:
    m = DEVICE_PROFILES[name]
    rng = np.random.default_rng([seed, n, sum(map(ord, name))])
    ln = lambda med, k: (med * np.exp(rng.normal(0, 0.3, k))).tolist()
    t1 = ln(m["t1"], n)
    t2 = [min(a, 2 * b) for a, b in zip(ln(m["t2"], n), t1)]          # physical: T2 <= 2 T1
    return DeviceNoise(name=name, kind="device", readout=0.0, t_1q=m["t_1q"], t_2q=m["t_2q"],
                       p1_q=ln(m["p1"], n), p2_pair=ln(m["p2"], max(n - 1, 1) + n * (n - 1) // 2),
                       t1_q=t1, t2_q=t2, ro10_q=np.clip(ln(m["ro10"], n), 0, 0.2).tolist(),
                       ro01_q=np.clip(ln(m["ro01"], n), 0, 0.2).tolist(), zz_overrot=m["zz_overrot"])


class DeviceDMSimulator(NumpyDensityMatrixSimulator):
    def __init__(self, n, reps, entanglement, noise: DeviceNoise):
        super().__init__(n, reps, entanglement, NoiseSpec("tmp", kind="none"))
        self.dev = noise
        self.pair_index = {}

    def _pair_p(self, a, b):
        key = (min(a, b), max(a, b))
        if key not in self.pair_index:
            self.pair_index[key] = len(self.pair_index)
        return self.dev.p2_pair[self.pair_index[key] % len(self.dev.p2_pair)]

    # --- fast in-place channels (index slicing on the (B, 2..2, 2..2) tensor; no moveaxis copies) ---
    def _ix(self, assign):
        idx = [slice(None)] * (1 + 2 * self.n)
        for ax, v in assign.items():
            idx[ax] = v
        return tuple(idx)

    def _own(self, rho):
        return rho if rho.flags.writeable and rho.base is None else rho.copy()

    def _thermal_q(self, rho, q, t):
        g = 1.0 - np.exp(-t / self.dev.t1_q[q])
        f = np.exp(-t / self.dev.t2_q[q])
        rho = self._own(rho)
        r, c = 1 + q, 1 + self.n + q
        i00, i11 = self._ix({r: 0, c: 0}), self._ix({r: 1, c: 1})
        rho[i00] += g * rho[i11]
        rho[i11] *= (1 - g)
        rho[self._ix({r: 0, c: 1})] *= f
        rho[self._ix({r: 1, c: 0})] *= f
        return rho

    def _h(self, rho, q):
        rho = self._own(rho)
        s = 1 / np.sqrt(2)
        for ax in (1 + q, 1 + self.n + q):
            i0, i1 = self._ix({ax: 0}), self._ix({ax: 1})
            a, b = rho[i0].copy(), rho[i1].copy()
            rho[i0] = s * (a + b)
            rho[i1] = s * (a - b)
        return rho

    def _p(self, rho, q, th):
        rho = self._own(rho)
        ph = np.exp(1j * th).reshape((-1,) + (1,) * (2 * self.n - 1))
        rho[self._ix({1 + q: 1})] *= ph
        rho[self._ix({1 + self.n + q: 1})] *= ph.conj()
        return rho

    def _cx(self, rho, c, t):
        rho = self._own(rho)
        for off in (0, self.n):
            i10, i11 = self._ix({1 + off + c: 1, 1 + off + t: 0}), self._ix({1 + off + c: 1, 1 + off + t: 1})
            tmp = rho[i10].copy()
            rho[i10] = rho[i11]
            rho[i11] = tmp
        return rho

    def _depol1(self, rho, q, p):
        if p <= 0:
            return rho
        rho = self._own(rho)
        r, c = 1 + q, 1 + self.n + q
        i00, i11 = self._ix({r: 0, c: 0}), self._ix({r: 1, c: 1})
        tr = rho[i00] + rho[i11]
        rho *= (1 - p)
        rho[i00] += 0.5 * p * tr
        rho[i11] += 0.5 * p * tr
        return rho

    def _depol2(self, rho, a, b, p):
        if p <= 0:
            return rho
        rho = self._own(rho)
        ra, rb, ca, cb = 1 + a, 1 + b, 1 + self.n + a, 1 + self.n + b
        ids = [self._ix({ra: i, rb: j, ca: i, cb: j}) for i in (0, 1) for j in (0, 1)]
        tr = sum(rho[k] for k in ids)
        rho *= (1 - p)
        for k in ids:
            rho[k] += 0.25 * p * tr
        return rho

    def basis_probabilities(self, X, Y):
        B, n = X.shape
        D = 2 ** n
        rho = np.zeros((B,) + (2,) * (2 * n), dtype=complex)
        rho[(slice(None),) + (0,) * (2 * n)] = 1.0
        dv = self.dev
        for (g, qs, spec) in self.ops:
            if g == "h":
                rho = self._h(rho, qs[0])
                rho = self._depol1(rho, qs[0], dv.p1_q[qs[0]])
                rho = self._thermal_q(rho, qs[0], dv.t_1q)
            elif g == "p":
                th = op_angles(spec, X, Y)
                if spec[0] == "pair":
                    th = th * (1.0 + dv.zz_overrot)
                rho = self._p(rho, qs[0], th)
            elif g == "cx":
                rho = self._cx(rho, qs[0], qs[1])
                rho = self._depol2(rho, qs[0], qs[1], self._pair_p(*qs))
                rho = self._thermal_q(rho, qs[0], dv.t_2q)
                rho = self._thermal_q(rho, qs[1], dv.t_2q)
        diag = np.real(np.diagonal(rho.reshape(B, D, D), axis1=1, axis2=2))
        return np.clip(diag, 0.0, 1.0)


def readout_all_zero_asym(P, n, ro10, ro01):
    s = np.arange(2 ** n)
    w = np.ones(2 ** n)
    for q in range(n):
        bit = (s >> (n - 1 - q)) & 1
        w *= np.where(bit == 0, 1 - ro10[q], ro01[q])
    return P @ w


def exact_p(X, Y, reps, ent, noise, chunk=1500):
    """Exact Pr[observe 0...0] for each pair; dispatches the device model, else the package simulator."""
    t0 = time.perf_counter()
    if getattr(noise, "kind", None) == "device":
        n = X.shape[1]
        sim = DeviceDMSimulator(n, reps, ent, noise)
        P = np.vstack([sim.basis_probabilities(X[s:s + chunk], Y[s:s + chunk]) for s in range(0, len(X), chunk)])
        return np.clip(readout_all_zero_asym(P, n, noise.ro10_q, noise.ro01_q), 0, 1), "numpy-device", \
            time.perf_counter() - t0
    return exact_all_zero_probability(X, Y, reps, ent, noise, backend="numpy", chunk=chunk)


def compute_problem_rev(A_tr, A_va, A_te, noise, reps, ent, chunk=1500):
    N = len(A_tr)
    iu = np.triu_indices(N)
    X = np.vstack([A_tr[iu[0]], np.repeat(A_va, N, axis=0), A_va, np.repeat(A_te, N, axis=0), A_te])
    Y = np.vstack([A_tr[iu[1]], np.tile(A_tr, (len(A_va), 1)), A_va, np.tile(A_tr, (len(A_te), 1)), A_te])
    p, used, secs = exact_p(X, Y, reps, ent, noise, chunk)
    o = 0
    P_tr = np.zeros((N, N)); P_tr[iu] = p[o:o + len(iu[0])]; o += len(iu[0])
    P_tr = P_tr + np.triu(P_tr, 1).T
    nv, nt = len(A_va), len(A_te)
    P_va = p[o:o + nv * N].reshape(nv, N); o += nv * N
    P_va_diag = p[o:o + nv]; o += nv
    P_te = p[o:o + nt * N].reshape(nt, N); o += nt * N
    P_te_diag = p[o:o + nt]
    return dict(P_tr=P_tr, P_va=P_va, P_va_diag=P_va_diag, P_te=P_te, P_te_diag=P_te_diag), used, secs


# ----------------------------------------------------------------------------------------------
# Quantum-structured dataset (Havlicek et al., Nature 2019, "ad hoc" construction)
# ----------------------------------------------------------------------------------------------
def load_adhoc(n_qubits: int, n_samples: int = 600, gap: float = 0.1, reps: int = 2,
               entanglement: str = "linear", seed: int = 4242):
    """x ~ U[0, 2pi)^n; label = sign <psi(x)| V^dag Z...Z V |psi(x)> with a Haar-random V (fixed seed),
    kept only if |value| >= gap; balanced classes. psi(x) is the SAME ZZ feature map used by the kernel,
    so the ideal quantum kernel contains the labelling function by construction."""
    rng = np.random.default_rng(seed + 1000 * n_qubits)
    D = 2 ** n_qubits
    G = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(G)
    V = Q * (np.diag(R) / np.abs(np.diag(R)))
    parity = np.array([(-1) ** bin(s).count("1") for s in range(D)], dtype=float)
    O = V.conj().T @ (parity[:, None] * V)
    Xs, ys = [], []
    per = n_samples // 2
    counts = {0: 0, 1: 0}
    while min(counts.values()) < per:
        X = rng.uniform(0, 2 * np.pi, size=(4000, n_qubits))
        psi = feature_statevectors(X, reps, entanglement)
        f = np.real(np.einsum("bi,ij,bj->b", psi.conj(), O, psi))
        keep = np.abs(f) >= gap
        for x, v in zip(X[keep], f[keep]):
            c = int(v > 0)
            if counts[c] < per:
                Xs.append(x); ys.append(c); counts[c] += 1
    return np.array(Xs), np.array(ys)


# ----------------------------------------------------------------------------------------------
# Mechanism block: identity (W1), mitigation effect at fixed vs tuned C (W2), tuned-C ridge test (W3)
# ----------------------------------------------------------------------------------------------
C_GRID_FINE = list(np.round(np.logspace(-2, 3, 11), 6))      # contains 0.1, 1, 10, 100
FIXED_CS = [0.1, 1.0, 10.0, 100.0]
RIDGE_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _fit(K, y, C, exact=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if exact:
            return SVC(kernel="precomputed", C=C, tol=1e-10, max_iter=10 ** 7).fit(K, y)
        return SVC(kernel="precomputed", C=C, max_iter=200000).fit(K, y)


def _cmp(fA, fX, pA, pX):
    fA, fX = np.ravel(fA), np.ravel(fX)
    return dict(rel_diff=float(np.linalg.norm(fA - fX) / max(np.linalg.norm(fA), 1e-15)),
                agree=float(np.mean(pA == pX)))


def _sym(N, iu, v, diag=1.0):
    K = np.zeros((N, N)); K[iu] = v; K = K + K.T; np.fill_diagonal(K, diag)
    return K


def mechanism_block(prob, S: int, rng: np.random.Generator, lambda_floor: float = 0.05,
                    C0: float = 1.0, C_grid=None, fixed_Cs=None, ridge_mults=None, do_accuracy=True):
    """One (split, noise, shot budget) cell. S = 0 means exact probabilities (infinite shots).
    Every variant uses the SAME measurement record (exactly paired)."""
    C_grid = C_grid or C_GRID_FINE
    fixed_Cs = fixed_Cs or FIXED_CS
    ridge_mults = ridge_mults or RIDGE_MULTS
    N, d = len(prob.P_tr), prob.d
    iu = np.triu_indices(N, 1)
    est = (lambda p: p.copy()) if S == 0 else (lambda p: rng.binomial(S, np.clip(p, 0, 1)) / S)
    off, dtr = est(prob.P_tr[iu]), est(np.diag(prob.P_tr))
    Kva_raw, Kte_raw = est(prob.P_va), est(prob.P_te)
    K_raw = _sym(N, iu, off)
    lam_hat = float(np.clip((dtr.mean() - 1 / d) / (1 - 1 / d), lambda_floor, 1.0))
    lam_or = float((np.diag(prob.P_tr).mean() - 1 / d) / (1 - 1 / d))
    rr = lambda lam: (1 - lam) * (1 - 1 / d) / lam
    r_hat, r_or = rr(lam_hat), rr(lam_or)
    mit = lambda K, lam: (K - 1 / d) / lam + 1 / d
    K_mit = mit(K_raw, lam_hat); np.fill_diagonal(K_mit, 1.0)
    Kva_mit, Kte_mit = mit(Kva_raw, lam_hat), mit(Kte_raw, lam_hat)
    K_mito = mit(K_raw, lam_or); np.fill_diagonal(K_mito, 1.0)
    I = np.eye(N)
    y, yv, yt = prob.y_tr, prob.y_va, prob.y_te

    # ---- W1: identity checks at C0 ----
    A = _fit(K_raw, y, C0, True)
    M = _fit(K_mit + r_hat * I, y, lam_hat * C0, True)
    Mo = _fit(K_mito + r_or * I, y, lam_or * C0, True)
    B = _fit(prob.K_tr + r_or * I, y, lam_or * C0, True)
    Z = _fit(prob.K_tr, y, lam_or * C0, True)
    fA, pA = A.decision_function(Kte_raw), A.predict(Kte_raw)
    Kte_mito = mit(Kte_raw, lam_or)
    ident = dict(lam_hat=lam_hat, lam_oracle=lam_or, r_hat=r_hat, r_oracle=r_or,
                 min_eig_raw=kx.min_eig(K_raw), min_eig_mit=kx.min_eig(K_mit),
                 snr_raw=_snr(K_raw, prob.K_tr, lam_or, d, scale=lam_or),
                 snr_mit=_snr(K_mit, prob.K_tr, lam_or, d, scale=1.0))
    for tag, mdl, Kt in (("M", M, Kte_mit), ("Mo", Mo, Kte_mito), ("B", B, prob.K_te), ("Z", Z, prob.K_te)):
        c = _cmp(fA, mdl.decision_function(Kt), pA, mdl.predict(Kt))
        ident[f"{tag}_rel_diff"], ident[f"{tag}_agree"] = c["rel_diff"], c["agree"]
    # shot-noise-only distance: same ridge, same C, kernels differ only by the measured noise E/Lam
    c = _cmp(Mo.decision_function(Kte_mito), B.decision_function(prob.K_te),
             Mo.predict(Kte_mito), B.predict(prob.K_te))
    ident["Mo_vs_B_rel_diff"], ident["Mo_vs_B_agree"] = c["rel_diff"], c["agree"]
    if not do_accuracy:
        return ident, []

    # ---- W2/W3: accuracy at fixed and tuned C ----
    variants = {
        "raw": (K_raw, Kva_raw, Kte_raw),
        "mitigated": (K_mit, Kva_mit, Kte_mit),
        "raw+clip": (kx.psd_clip(K_raw), Kva_raw, Kte_raw),
        "mitigated+clip": (kx.psd_clip(K_mit), Kva_mit, Kte_mit),
        "ideal (no ridge)": (prob.K_tr, prob.K_va, prob.K_te),
    }
    for m in ridge_mults:
        variants[f"mitigated + {m}r_hat I"] = (K_mit + m * r_hat * I, Kva_mit, Kte_mit)
        variants[f"ideal + {m}r I"] = (prob.K_tr + m * r_or * I, prob.K_va, prob.K_te)
    acc_rows = []
    for name, (Kt, Kv, Ke) in variants.items():
        best, row = None, dict(variant=name)
        for C in C_grid:
            try:
                mdl = _fit(Kt, y, C)
                sv = balanced_accuracy_score(yv, mdl.predict(Kv))
                st = balanced_accuracy_score(yt, mdl.predict(Ke))
            except Exception:
                continue
            if any(abs(C - fc) < 1e-9 for fc in fixed_Cs):
                row[f"bacc_C{fc_label(C)}"] = st
            if best is None or sv > best[0] + 1e-12:
                best = (sv, C, st)
        if best is not None:
            row.update(bacc_tuned=best[2], C_tuned=best[1], val_tuned=best[0])
        acc_rows.append(row)
    return ident, acc_rows


def fc_label(C):
    return f"{C:g}"


def _snr(K_est, K_ideal, lam, d, scale):
    """Signal-to-noise of the off-diagonal kernel estimate after removing the deterministic affine part:
    SNR = std(signal) / rms(residual), with signal = scale*(K_ideal) part. Equal for raw and mitigated
    kernels under global depolarizing noise (Corollary 1); reported to show it empirically."""
    iu = np.triu_indices(len(K_est), 1)
    a = K_est[iu]
    b = K_ideal[iu]
    # best affine fit a ~ s*b + t (removes calibration), residual = noise + model mismatch
    Aq = np.vstack([b, np.ones_like(b)]).T
    coef, *_ = np.linalg.lstsq(Aq, a, rcond=None)
    res = a - Aq @ coef
    return float(abs(coef[0]) * b.std() / max(res.std(), 1e-15))
