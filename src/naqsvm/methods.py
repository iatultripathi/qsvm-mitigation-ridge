"""Classifiers compared in the study.

Families
--------
reference   : ideal (noise-free, infinite-shot) QSVM; classical RBF-SVM (4-D PCA input and
              full standardized input). References, not competitors for robustness claims.
established : standard QSVM variants that post-process ONE shared uniform-shot measurement
              record (raw, PSD clip, flip, shift, depolarizing mitigation 'average' and
              'split channel' + clip).
prior-adaptive : adaptive train-kernel shot allocation from prior work, re-implemented from the
              papers: AQKA Bernoulli-only and AQKA-SVM (Xu et al. 2026), and a margin-aware Neyman
              rule approximating Miroszewski (2026) from its published equations.
proposed    : NA-QSVM (noise-aware, shot-adaptive) - see `run_na_qsvm`.
ablation    : NA-QSVM with one component removed or changed.

Budget accounting (per shot level S = shots per entry):
    train budget  B_tr = S * N(N+1)/2     (all unique train entries incl. diagonal)
    val budget    B_va = S * N_val (N+1)  (cross entries + one diagonal per val point)
    test budget   B_te = S * N_test (N+1)
Established methods spend these uniformly (S per entry). Methods that do not use the
diagonal circuits (raw/clip/flip/shift) are charged only for what they use, so their
reported totals are ~2/(N+1) lower. NA-QSVM spends exactly B_tr + B_va + B_te
unless the 'shot-saving' variant is selected.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC

from . import kernels as kx
from .shots import Measurement, largest_remainder, measure_two_phase, measure_uniform

ESTABLISHED = ["QSVM-raw", "QSVM-clip", "QSVM-flip", "QSVM-shift",
               "QSVM-mitig(avg)+clip", "QSVM-mitig(split)+clip"]
PROPOSED = "NA-QSVM (proposed)"

# name -> options. The PROPOSED configuration was fixed before running the full grid.
NA_VARIANTS: Dict[str, dict] = {
    PROPOSED: dict(mitigate=True, alloc="neyman", ridge="tuned", sv_test=True, test_budget="equal"),
    "Abl: uniform shots": dict(mitigate=True, alloc="uniform", ridge="tuned", sv_test=True, test_budget="equal"),
    "Abl: no noise-aware ridge": dict(mitigate=True, alloc="neyman", ridge="none", sv_test=True, test_budget="equal"),
    "Abl: no mitigation": dict(mitigate=False, alloc="neyman", ridge="tuned", sv_test=True, test_budget="equal"),
    "Abl: no SV-restricted test": dict(mitigate=True, alloc="neyman", ridge="tuned", sv_test=False, test_budget="equal"),
    "Abl: ridge c=1 (untuned)": dict(mitigate=True, alloc="neyman", ridge="fixed1", sv_test=True, test_budget="equal"),
    "Abl: SV-weighted train alloc": dict(mitigate=True, alloc="neyman_sv", ridge="tuned", sv_test=True, test_budget="equal"),
    "Variant: shot-saving (SV test, S/entry)": dict(mitigate=True, alloc="neyman", ridge="tuned", sv_test=True, test_budget="per_entry"),
}


@dataclass
class QProblem:
    """Exact (infinite-shot) probabilities for one dataset/seed/noise level."""
    P_tr: np.ndarray        # (N, N) symmetric; diagonal = Pr[0..0] of the x_i,x_i circuit
    P_va: np.ndarray        # (Nv, N)
    P_va_diag: np.ndarray   # (Nv,)
    P_te: np.ndarray        # (Nt, N)
    P_te_diag: np.ndarray   # (Nt,)
    K_tr: np.ndarray        # ideal kernels
    K_va: np.ndarray
    K_te: np.ndarray
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray
    d: int


# ---------------------------------------------------------------------------
# SVM helpers
# ---------------------------------------------------------------------------
def _scores(y_true, y_pred) -> Dict[str, float]:
    return dict(accuracy=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average="macro"),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred))


def _fit(K, y, C):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return SVC(kernel="precomputed", C=C, max_iter=200000).fit(K, y)


def select_and_fit(train_kernels: List[Tuple[dict, np.ndarray]], K_va_fn, y_tr, y_va,
                   C_grid, metric="balanced_accuracy"):
    """Grid search on the validation split only.

    train_kernels: list of (params, K_train) candidates (e.g. different ridge levels).
    K_va_fn(params) -> validation kernel. Ties keep the first candidate in grid order
    (C ascending; for NA-QSVM ridge multiplier c ascending, so c > 0 must strictly win).
    """
    best = None
    n_fail = 0
    for params, Ktr in train_kernels:
        Kva = K_va_fn(params)
        for C in C_grid:
            try:
                m = _fit(Ktr, y_tr, C)
                s = _scores(y_va, m.predict(Kva))[metric]
            except Exception:
                n_fail += 1
                continue
            if best is None or s > best[0] + 1e-12:
                best = (s, dict(params, C=C), m, Ktr)
    if best is None:
        raise RuntimeError("all SVM fits failed")
    return best[2], best[1], best[0], best[3], n_fail


def _kernel_metrics(K_used, K_ideal) -> dict:
    return dict(kernel_rel_fro_err=kx.rel_fro_error_offdiag(K_used, K_ideal),
                kernel_alignment_gap=1.0 - kx.centered_alignment(K_used, K_ideal))


def _fill_sym(N, iu, values, diag=1.0):
    K = np.zeros((N, N))
    K[iu] = values
    K = K + K.T
    np.fill_diagonal(K, diag)
    return K


# ---------------------------------------------------------------------------
# Reference methods
# ---------------------------------------------------------------------------
def run_ideal_qsvm(prob: QProblem, cfg) -> dict:
    t0 = time.perf_counter()
    m, params, vs, Ktr, nf = select_and_fit([({}, prob.K_tr)], lambda p: prob.K_va,
                                            prob.y_tr, prob.y_va, cfg.C_grid, cfg.selection_metric)
    sc = _scores(prob.y_te, m.predict(prob.K_te))
    return dict(method="Ideal QSVM (exact, noise-free)", family="reference", **sc,
                val_score=vs, chosen=str(params), total_shots=np.nan, train_shots=np.nan,
                val_shots=np.nan, test_shots=np.nan, runtime_s=time.perf_counter() - t0,
                n_sv=len(m.support_), min_eig_before=kx.min_eig(prob.K_tr),
                min_eig_after=kx.min_eig(Ktr), **_kernel_metrics(Ktr, prob.K_tr))


def run_rbf(Ztr, Zva, Zte, ytr, yva, yte, cfg, name) -> dict:
    t0 = time.perf_counter()
    best = None
    for g in cfg.rbf_gamma_grid:
        for C in cfg.C_grid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    m = SVC(kernel="rbf", C=C, gamma=g, max_iter=200000).fit(Ztr, ytr)
                s = _scores(yva, m.predict(Zva))[cfg.selection_metric]
            except Exception:
                continue
            if best is None or s > best[0] + 1e-12:
                best = (s, dict(C=C, gamma=g), m)
    sc = _scores(yte, best[2].predict(Zte))
    return dict(method=name, family="reference", **sc, val_score=best[0], chosen=str(best[1]),
                total_shots=0, train_shots=0, val_shots=0, test_shots=0,
                runtime_s=time.perf_counter() - t0, n_sv=len(best[2].support_))


# ---------------------------------------------------------------------------
# Established QSVM variants on one shared uniform measurement record
# ---------------------------------------------------------------------------
def uniform_record(prob: QProblem, S: int, rng: np.random.Generator) -> dict:
    """Shared train/validation record (and, for the v1 protocol, the full test kernel)."""
    N = len(prob.P_tr)
    iu_off = np.triu_indices(N, 1)
    rec = dict(
        tr_off=measure_uniform(prob.P_tr[iu_off], S * len(iu_off[0]), rng),
        tr_diag=measure_uniform(np.diag(prob.P_tr), S * N, rng),
        va=measure_uniform(prob.P_va.ravel(), S * prob.P_va.size, rng),
        va_diag=measure_uniform(prob.P_va_diag, S * len(prob.P_va_diag), rng),
        te=measure_uniform(prob.P_te.ravel(), S * prob.P_te.size, rng),
        te_diag=measure_uniform(prob.P_te_diag, S * len(prob.P_te_diag), rng),
    )
    rec["iu_off"] = iu_off
    return rec


def measure_test_sv(prob: QProblem, model, S: int, rng, with_diag: bool):
    """Fair test protocol (v2): every method spends the test budget S*Nt*(N+1) on the
    support-vector columns of its own final model (standard SVM property). Methods that
    need per-point diagonals (split-channel mitigation) spend S*Nt of it on those.
    Returns (K_test_raw with non-SV columns = 0, test_diag_estimates or None, shots)."""
    N = len(prob.P_tr)
    Nt = prob.P_te.shape[0]
    cols = np.sort(model.support_)
    B = S * Nt * (N + 1)
    diag_est, shots = None, 0
    if with_diag:
        md = measure_uniform(prob.P_te_diag, S * Nt, rng)
        diag_est, shots, B = md.estimate, md.total_shots, B - S * Nt
    m = measure_uniform(prob.P_te[:, cols].ravel(), B, rng)
    K = np.zeros((Nt, N))
    K[:, cols] = m.estimate.reshape(Nt, len(cols))
    return K, diag_est, shots + m.total_shots, cols


def run_established(prob: QProblem, rec: dict, cfg, sampling_time: float,
                    S: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> List[dict]:
    """Established post-processing of the shared record. If cfg.sv_test_all (v2 protocol),
    each method measures its own SV-restricted test kernel with the full test budget."""
    sv_all = bool(getattr(cfg, "sv_test_all", False)) and S is not None and rng is not None
    N, d = len(prob.P_tr), prob.d
    Nv, Nt = prob.P_va.shape[0], prob.P_te.shape[0]
    K_raw = _fill_sym(N, rec["iu_off"], rec["tr_off"].estimate, 1.0)
    Kva_raw = rec["va"].estimate.reshape(Nv, N)
    diag_tr = rec["tr_diag"].estimate
    lam_avg = float(kx.survival_from_diag(diag_tr.mean(), d, cfg.lambda_floor))
    lam_tr = kx.survival_from_diag(diag_tr, d, cfg.lambda_floor)
    lam_va = kx.survival_from_diag(rec["va_diag"].estimate, d, cfg.lambda_floor)

    ident = lambda K, lam_rows: K
    avg_te = lambda K, lam_rows: kx.mitigate(K, lam_avg, d)
    split_te = lambda K, lam_rows: kx.mitigate(K, np.sqrt(np.outer(lam_rows, lam_tr)), d)

    def build(name):
        # returns K_before, K_train, K_val, test_map, uses_diag, lambda_hat
        if name == "QSVM-raw":
            return K_raw, K_raw, Kva_raw, ident, False, np.nan
        if name == "QSVM-clip":
            return K_raw, kx.psd_clip(K_raw), Kva_raw, ident, False, np.nan
        if name == "QSVM-flip":
            return K_raw, kx.psd_flip(K_raw), Kva_raw, ident, False, np.nan
        if name == "QSVM-shift":
            return K_raw, kx.psd_shift(K_raw), Kva_raw, ident, False, np.nan
        if name == "QSVM-mitig(avg)+clip":
            Km = kx.mitigate(K_raw, lam_avg, d); np.fill_diagonal(Km, 1.0)
            return Km, kx.psd_clip(Km), kx.mitigate(Kva_raw, lam_avg, d), avg_te, True, lam_avg
        if name == "QSVM-mitig(split)+clip":
            Km = kx.mitigate(K_raw, np.sqrt(np.outer(lam_tr, lam_tr)), d); np.fill_diagonal(Km, 1.0)
            return (Km, kx.psd_clip(Km), kx.mitigate(Kva_raw, np.sqrt(np.outer(lam_va, lam_tr)), d),
                    split_te, True, float(lam_tr.mean()))
        raise ValueError(name)

    rows = []
    for name in ESTABLISHED:
        try:
            t0 = time.perf_counter()
            K_before, Ktr, Kva, te_map, uses_diag, lam = build(name)
            m, params, vs, Ktr_used, nf = select_and_fit([({}, Ktr)], lambda p: Kva, prob.y_tr,
                                                         prob.y_va, cfg.C_grid, cfg.selection_metric)
            tr_shots = rec["tr_off"].total_shots + (rec["tr_diag"].total_shots if uses_diag else 0)
            va_shots = rec["va"].total_shots + (rec["va_diag"].total_shots if uses_diag else 0)
            if sv_all:
                split = name == "QSVM-mitig(split)+clip"
                Kte_raw, te_diag, te_shots, _ = measure_test_sv(prob, m, S, rng, with_diag=split)
                lam_te = kx.survival_from_diag(te_diag, d, cfg.lambda_floor) if split else None
                Kte = te_map(Kte_raw, lam_te)
                Kte[:, np.setdiff1d(np.arange(N), m.support_)] = 0.0
            else:
                Kte_raw = rec["te"].estimate.reshape(Nt, N)
                lam_te = kx.survival_from_diag(rec["te_diag"].estimate, d, cfg.lambda_floor)
                Kte = te_map(Kte_raw, lam_te)
                te_shots = rec["te"].total_shots + (rec["te_diag"].total_shots if name == "QSVM-mitig(split)+clip" else 0)
            sc = _scores(prob.y_te, m.predict(Kte))
            shots = dict(train_shots=tr_shots, val_shots=va_shots, test_shots=te_shots)
            rt = time.perf_counter() - t0 + sampling_time
            rows.append(dict(method=name, family="established", **sc, val_score=vs, chosen=str(params),
                             total_shots=sum(shots.values()), **shots, runtime_s=rt,
                             n_sv=len(m.support_), lambda_hat=lam,
                             min_eig_before=kx.min_eig(K_before), min_eig_after=kx.min_eig(Ktr_used),
                             psd_after=kx.is_psd(Ktr_used), n_failed_fits=nf,
                             **_kernel_metrics(Ktr_used, prob.K_tr), status="ok"))
        except Exception as exc:
            rows.append(dict(method=name, family="established", status=f"failed: {exc!r}"))
    return rows


# ---------------------------------------------------------------------------
# Prior adaptive-allocation methods (re-implemented from the papers)
# ---------------------------------------------------------------------------
ALLOCATION_BASELINES = ["AQKA-Bernoulli+clip", "AQKA-SVM+clip", "MarginNeyman+clip (Miroszewski-style approx.)"]


def margin_info(K: np.ndarray, y: np.ndarray, C: float = 1.0, V: Optional[np.ndarray] = None):
    """Per-point |eta_i| (= |alpha_i y_i|) and margin-violation probability P_i.

    Multiclass problems are handled one-vs-one (as libsvm does); per point the maximum over
    the binary sub-problems is returned. With V (entry variances of K), P_i is the Gaussian
    approximation  P_i = Phi((1 - y_i f_i) / sd_i),  sd_i^2 = sum_j eta_j^2 V_ij
    (Miroszewski 2026, Eqs. 10-11)."""
    from scipy.stats import norm
    N = len(y)
    eta = np.zeros(N)
    P = np.zeros(N)
    classes = np.unique(y)
    for a_i in range(len(classes)):
        for b_i in range(a_i + 1, len(classes)):
            idx = np.where((y == classes[a_i]) | (y == classes[b_i]))[0]
            ypm = np.where(y[idx] == classes[b_i], 1.0, -1.0)
            try:
                m = _fit(K[np.ix_(idx, idx)], ypm, C)
            except Exception:
                eta[idx] = np.maximum(eta[idx], 1.0)
                continue
            e = np.zeros(len(idx))
            e[m.support_] = m.dual_coef_[0]
            eta[idx] = np.maximum(eta[idx], np.abs(e))
            if V is not None:
                f = m.decision_function(K[np.ix_(idx, idx)])
                sd = np.sqrt(np.maximum((e ** 2)[None, :] @ V[np.ix_(idx, idx)].T, 1e-12)).ravel()
                P[idx] = np.maximum(P[idx], norm.cdf((1.0 - ypm * f) / sd))
    return eta, P


def measure_aqka_svm(p, iu, N, y, budget, rng, C=1.0, warm=0.2, rounds=4, explore=0.2, floor=0.05):
    """AQKA SVM acquisition (Xu et al. 2026, Prop. 4 with the paper's defaults: warm-up 0.2,
    T=4 rounds, exploration 0.2, sensitivity floor 5% of max). Target allocation
    s_ij ∝ |eta_i eta_j| sqrt(K_ij(1-K_ij)); eta from an SVM refit on the PSD-projected
    running estimate each round; deterministic largest-remainder fill."""
    M = len(p)
    s0 = int(max(1, np.floor(warm * budget / M)))
    if s0 * M > budget:
        return measure_uniform(p, budget, rng)
    s = np.full(M, s0, dtype=np.int64)
    c = rng.binomial(s, np.clip(p, 0, 1))
    remaining = budget - int(s.sum())
    per_round = remaining // rounds
    for t in range(rounds):
        Bt = per_round if t < rounds - 1 else budget - int(s.sum())
        if Bt <= 0:
            break
        K = kx.psd_clip(_fill_sym(N, iu, c / s, 1.0))
        eta, _ = margin_info(K, y, C)
        g = eta[iu[0]] * eta[iu[1]]
        g = np.maximum(g, floor * g.max()) if g.max() > 0 else np.ones(M)
        pt = (c + 1.0) / (s + 2.0)
        a = g * np.sqrt(pt * (1 - pt))
        need = np.maximum(0.0, budget * a / a.sum() - s)
        n_exploit = int((1 - explore) * Bt)
        add = largest_remainder(need, n_exploit) if need.sum() > 0 else largest_remainder(np.ones(M), n_exploit)
        add = add + largest_remainder(np.ones(M), Bt - int(add.sum()))
        c = c + rng.binomial(add, np.clip(p, 0, 1))
        s = s + add
    return Measurement(c, s)


def measure_margin_neyman(p, iu, N, y, budget, rng, C=1.0, pilot=0.25):
    """Margin-aware Neyman allocation built ONLY from the published ingredients of
    Miroszewski (2026): margin sensitivity |alpha_i alpha_j| (Eq. 7), margin-violation
    probability P_i (Eq. 11, Gaussian approximation via Eq. 10) and the Bernoulli variance.
    The way they are combined here (u_i = |eta_i| + P_i * mean|eta_SV|, S_ij ∝ u_i u_j sqrt(v_ij))
    is OUR approximation, not the official ShotWise algorithm."""
    def weight_fn(pil: Measurement):
        K = kx.psd_clip(_fill_sym(N, iu, pil.estimate, 1.0))
        V = _fill_sym(N, iu, pil.variance, 0.0)
        eta, P = margin_info(K, y, C, V)
        ref = eta[eta > 0].mean() if np.any(eta > 0) else 1.0
        u = eta + P * ref
        return (u[iu[0]] * u[iu[1]]) ** 2 + 1e-12
    return measure_two_phase(p, budget, rng, pilot, weight_fn)


def run_allocation_baseline(prob: QProblem, S: int, rng: np.random.Generator, cfg, name: str) -> dict:
    """Adaptive train-kernel acquisition from prior work, raw kernel (diag = 1) + PSD clip,
    uniform validation kernel, SV-restricted test kernel. Same total budget as every method."""
    t0 = time.perf_counter()
    N = len(prob.P_tr)
    Nv, Nt = prob.P_va.shape[0], prob.P_te.shape[0]
    iu = np.triu_indices(N, 1)
    B_tr = S * (len(iu[0]) + N)                 # no diagonal circuits needed: whole budget off-diagonal
    p = prob.P_tr[iu]
    if name == "AQKA-Bernoulli+clip":
        m_tr = measure_two_phase(p, B_tr, rng, 0.2)
    elif name == "AQKA-SVM+clip":
        m_tr = measure_aqka_svm(p, iu, N, prob.y_tr, B_tr, rng)
    elif name.startswith("MarginNeyman"):
        m_tr = measure_margin_neyman(p, iu, N, prob.y_tr, B_tr, rng, pilot=cfg.pilot_fraction)
    else:
        raise ValueError(name)
    K_before = _fill_sym(N, iu, m_tr.estimate, 1.0)
    Ktr = kx.psd_clip(K_before)
    m_va = measure_uniform(prob.P_va.ravel(), S * Nv * (N + 1), rng)
    Kva = m_va.estimate.reshape(Nv, N)
    m, params, vs, Ktr_used, nf = select_and_fit([({}, Ktr)], lambda q: Kva, prob.y_tr, prob.y_va,
                                                 cfg.C_grid, cfg.selection_metric)
    Kte, _, te_shots, _ = measure_test_sv(prob, m, S, rng, with_diag=False)
    sc = _scores(prob.y_te, m.predict(Kte))
    shots = dict(train_shots=m_tr.total_shots, val_shots=m_va.total_shots, test_shots=te_shots)
    return dict(method=name, family="prior-adaptive", **sc, val_score=vs, chosen=str(params),
                total_shots=sum(shots.values()), **shots, runtime_s=time.perf_counter() - t0,
                n_sv=len(m.support_), lambda_hat=np.nan, min_eig_before=kx.min_eig(K_before),
                min_eig_after=kx.min_eig(Ktr_used), psd_after=kx.is_psd(Ktr_used), n_failed_fits=nf,
                **_kernel_metrics(Ktr_used, prob.K_tr), status="ok")


# ---------------------------------------------------------------------------
# PROPOSED: NA-QSVM
# ---------------------------------------------------------------------------
def noise_spectral_scale(V: np.ndarray) -> float:
    """rho_hat = 2 * max_i sqrt(sum_j V_ij).

    For a symmetric random matrix E with independent centred entries of variance V_ij,
    E||E||_2 <~ 2 sigma + O(sigma_* sqrt(log N)),  sigma = max_i sqrt(sum_j V_ij)
    (Bandeira & van Handel, Ann. Probab. 2016). 2*sigma is used as the noise scale.
    """
    return float(2.0 * np.sqrt(np.max(V.sum(axis=1))))


def run_na_qsvm(prob: QProblem, S: int, rng: np.random.Generator, cfg, opts: dict,
                name: str, family: str, return_kernels: bool = False) -> dict:
    """Noise-aware, shot-adaptive QSVM.

    1. Calibration: S shots on each of the N diagonal circuits -> pooled survival
       estimate Lam_hat (global-depolarizing model). Uses train data only.
    2. Adaptive measurement of off-diagonal train entries: pilot + Neyman allocation
       of the remaining train budget (minimises the expected Frobenius error).
    3. Mitigation: K_c = (K_hat - 1/d)/Lam_hat + 1/d, diag = 1.
    4. Uncertainty propagation: Var(K_c,ij) ~= p~(1-p~)/(S_ij Lam_hat^2)
       (independent shot noise; the shared error of Lam_hat is a global rescaling and is
       absorbed by C - see notebook Sec. 3).
    5. Noise-aware ridge: K_r = K_c + c * rho_hat * I with rho_hat = noise_spectral_scale(Var).
       If c*rho_hat >= ||E||_2 then K_r >= K_ideal + (c rho_hat - ||E||) I is PSD whenever
       the ideal kernel is PSD. c is chosen on the validation split. Any residual negative
       eigenvalue is removed by clipping, so the final training kernel is always PSD (verified).
    6. Validation kernel measured adaptively; (C, c) selected on validation only.
    7. Test: only support-vector columns enter the SVM decision function, so the test
       budget is spent on those columns only (standard SVM property).
    """
    t0 = time.perf_counter()
    N, d = len(prob.P_tr), prob.d
    Nv, Nt = prob.P_va.shape[0], prob.P_te.shape[0]
    iu = np.triu_indices(N, 1)

    # 1) diagonal calibration
    m_diag = measure_uniform(np.diag(prob.P_tr), S * N, rng)
    lam = float(kx.survival_from_diag(m_diag.estimate.mean(), d, cfg.lambda_floor)) if opts["mitigate"] else 1.0
    corr = (lambda K: kx.mitigate(K, lam, d)) if opts["mitigate"] else (lambda K: K)

    # 2) adaptive off-diagonal train measurement
    B_off = S * (len(iu[0]) + N) - m_diag.total_shots
    p_off = prob.P_tr[iu]
    if opts["alloc"] == "uniform":
        m_off = measure_uniform(p_off, B_off, rng)
    elif opts["alloc"] == "neyman":
        m_off = measure_two_phase(p_off, B_off, rng, cfg.pilot_fraction)
    elif opts["alloc"] == "neyman_sv":
        def weight_fn(pilot: Measurement):
            Kp = corr(_fill_sym(N, iu, pilot.estimate, 1.0))
            np.fill_diagonal(Kp, 1.0)
            a = np.full(N, 0.2)
            try:
                a[_fit(kx.psd_clip(Kp), prob.y_tr, 1.0).support_] = 1.0
            except Exception:
                a[:] = 1.0
            return a[iu[0]] * a[iu[1]]
        m_off = measure_two_phase(p_off, B_off, rng, cfg.pilot_fraction, weight_fn)
    else:
        raise ValueError(opts["alloc"])

    # 3) mitigation
    Kc = corr(_fill_sym(N, iu, m_off.estimate, 1.0))
    np.fill_diagonal(Kc, 1.0)
    min_eig_before = kx.min_eig(Kc)

    # 4) uncertainty propagation
    V = _fill_sym(N, iu, m_off.variance / lam ** 2, 0.0)
    rho = noise_spectral_scale(V)

    # 5) candidate ridge levels
    cs = {"tuned": list(cfg.ridge_grid), "none": [0.0], "fixed1": [1.0]}[opts["ridge"]]
    cands, clipped = [], {}
    for c in sorted(cs):                            # c = 0 first: ridge kept only if it strictly helps on val
        Kr = Kc + c * rho * np.eye(N)
        clipped[c] = not kx.is_psd(Kr)
        if clipped[c]:
            Kr = kx.psd_clip(Kr)
        cands.append((dict(c=c), Kr))

    # 6) validation kernel (adaptive) and selection
    B_va = S * Nv * (N + 1)
    m_va = (measure_uniform(prob.P_va.ravel(), B_va, rng) if opts["alloc"] == "uniform"
            else measure_two_phase(prob.P_va.ravel(), B_va, rng, cfg.pilot_fraction))
    Kva = corr(m_va.estimate.reshape(Nv, N))
    model, params, vs, Ktr_used, nf = select_and_fit(cands, lambda p: Kva, prob.y_tr, prob.y_va,
                                                     sorted(cfg.C_grid), cfg.selection_metric)

    # 7) test kernel
    sv = np.sort(model.support_)
    cols = sv if opts["sv_test"] else np.arange(N)
    B_te = S * Nt * (N + 1) if opts["test_budget"] == "equal" else S * Nt * len(cols)
    p_te = prob.P_te[:, cols].ravel()
    m_te = (measure_uniform(p_te, B_te, rng) if opts["alloc"] == "uniform"
            else measure_two_phase(p_te, B_te, rng, cfg.pilot_fraction))
    Kte = np.zeros((Nt, N))
    Kte[:, cols] = corr(m_te.estimate.reshape(Nt, len(cols)))
    sc = _scores(prob.y_te, model.predict(Kte))

    shots = dict(train_shots=m_diag.total_shots + m_off.total_shots,
                 val_shots=m_va.total_shots, test_shots=m_te.total_shots)
    extra = dict(_K_mitigated=Kc, _K_train_used=Ktr_used) if return_kernels else {}
    return dict(method=name, family=family, **sc, val_score=vs, chosen=str(params),
                total_shots=sum(shots.values()), **shots, runtime_s=time.perf_counter() - t0,
                n_sv=len(sv), lambda_hat=lam, **extra, ridge_scale=rho, ridge_used=params["c"] * rho,
                selected_kernel_needed_clip=bool(clipped[params["c"]]),
                ridge_alone_psd_at_c1=(not clipped[1.0]) if 1.0 in clipped else np.nan,
                min_eig_before=min_eig_before,
                min_eig_after=kx.min_eig(Ktr_used), psd_after=kx.is_psd(Ktr_used),
                n_failed_fits=nf, **_kernel_metrics(Ktr_used, prob.K_tr), status="ok")
