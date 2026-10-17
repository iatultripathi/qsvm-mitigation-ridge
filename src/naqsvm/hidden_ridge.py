"""Direct test of the hidden-ridge identity (contribution C1).

Claim. If the measured all-zero probabilities follow a global depolarizing model
p_ij = Lam*k_ij + (1-Lam)/d and the diagonal is fixed to 1 (standard practice), then

    K_raw = Lam * (K_ideal + r I) + ((1-Lam)/d) 11^T,    r = (1-Lam)(1-1/d)/Lam,

so an SVM trained on K_raw with box constraint C is IDENTICAL (same decision function on any
test point, whose kernel row is Lam*k + (1-Lam)/d) to an SVM on the ideal kernel with an explicit
ridge r and box constraint Lam*C. (Scaling: SVM(sK, C) == SVM(K, sC); offset: removed by the
equality constraint sum_i alpha_i y_i = 0.)

Test. For each split we compare the decision values of
    A : SVM(K_raw, C)                    (what practitioners train)
    B : SVM(K_ideal + r I, Lam*C)        (hidden-ridge prediction)
    Z : SVM(K_ideal, Lam*C)              (control: same rescaling, NO ridge)
under (i) an analytic global depolarizing channel (identity must hold to solver precision),
(ii) the local gate-noise and thermal settings (identity holds only approximately), and
(iii) finite shots. If the hidden-ridge account is right, B should track A much more closely than Z.
Lam is the exact mean diagonal survival (an oracle quantity: this is an explanation, not a method).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from . import kernels as kx
from .data import load_dataset, make_split, prepare
from .experiment import _rng, angle_scale_for, compute_problem
from .simulators import ideal_kernel


def _svc(K, y, C):
    return SVC(kernel="precomputed", C=C, tol=1e-10, max_iter=10 ** 7,
               decision_function_shape="ovo").fit(K, y)


def _compare(fA, fX, pA, pX):
    fA, fX = np.ravel(fA), np.ravel(fX)
    corr = float(np.corrcoef(fA, fX)[0, 1]) if np.std(fA) > 0 and np.std(fX) > 0 else np.nan
    return dict(pred_agreement=float(np.mean(pA == pX)), decision_corr=corr,
                decision_rel_diff=float(np.linalg.norm(fA - fX) / max(np.linalg.norm(fA), 1e-15)))


def check_one(P_tr, P_te, K_tr, K_te, y_tr, y_te, lam, d, C, S, rng):
    N = len(K_tr)
    iu = np.triu_indices(N, 1)
    if S > 0:
        P_tr_hat = np.zeros((N, N))
        P_tr_hat[iu] = rng.binomial(S, np.clip(P_tr[iu], 0, 1)) / S
        P_tr_hat = P_tr_hat + P_tr_hat.T
        P_te_hat = rng.binomial(S, np.clip(P_te, 0, 1)) / S
    else:
        P_tr_hat, P_te_hat = P_tr.copy(), P_te.copy()
    K_raw = P_tr_hat.copy()
    np.fill_diagonal(K_raw, 1.0)
    r = (1 - lam) * (1 - 1 / d) / lam
    A = _svc(K_raw, y_tr, C)
    B = _svc(K_tr + r * np.eye(N), y_tr, lam * C)
    Z = _svc(K_tr, y_tr, lam * C)
    fA, fB, fZ = A.decision_function(P_te_hat), B.decision_function(K_te), Z.decision_function(K_te)
    pA, pB, pZ = A.predict(P_te_hat), B.predict(K_te), Z.predict(K_te)
    out = dict(lam=lam, hidden_ridge=r, min_eig_raw=kx.min_eig(K_raw),
               acc_raw=float(np.mean(pA == y_te)), acc_ideal_plus_ridge=float(np.mean(pB == y_te)),
               acc_ideal_no_ridge=float(np.mean(pZ == y_te)))
    out.update({f"B_{k}": v for k, v in _compare(fA, fB, pA, pB).items()})
    out.update({f"Z_{k}": v for k, v in _compare(fA, fZ, pA, pZ).items()})
    return out


def run_hidden_ridge(cfg, out_dir: str, backend: str, verbose: bool = True) -> pd.DataFrame:
    rows = []
    d = 2 ** cfg.n_qubits
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    for di, ds in enumerate(cfg.datasets):
        X, y = load_dataset(ds)
        for seed in cfg.seeds:
            sp = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, seed)
            pr = prepare(X, y, sp, cfg.n_qubits, angle_scale_for(cfg, ds), seed)
            K_tr = ideal_kernel(pr.A_train, pr.A_train, cfg.reps, cfg.entanglement)
            K_te = ideal_kernel(pr.A_test, pr.A_train, cfg.reps, cfg.entanglement)
            settings = [(f"global Λ={l}", "global", l, K_tr * l + (1 - l) / d, K_te * l + (1 - l) / d)
                        for l in cfg.hidden_ridge_global_lams]
            for ni, nz in enumerate(cfg.noise_levels):
                if nz.kind == "none":
                    continue
                Pd, _, _, _ = compute_problem(pr, nz, cfg, backend,
                                              os.path.join(cache_dir, f"P_{ds}_s{seed}_n{ni}.npz"))
                lam = float(kx.survival_from_diag(np.diag(Pd["P_tr"]).mean(), d, 0.0))
                settings.append((nz.name, nz.kind, lam, Pd["P_tr"], Pd["P_te"]))
            for (name, kind, lam, P_tr, P_te) in settings:
                for S in cfg.hidden_ridge_shots:
                    try:
                        r = check_one(P_tr, P_te, K_tr, K_te, pr.y_train, pr.y_test, lam, d,
                                      cfg.hidden_ridge_C, S, _rng(seed, di, name, S, "hidden-ridge"))
                        rows.append(dict(dataset=ds, seed=seed, noise=name, noise_kind=kind,
                                         shots_per_entry=("exact" if S == 0 else S), status="ok", **r))
                    except Exception as exc:
                        rows.append(dict(dataset=ds, seed=seed, noise=name, noise_kind=kind,
                                         shots_per_entry=("exact" if S == 0 else S), status=f"failed: {exc!r}"))
        if verbose:
            print(f"hidden-ridge check: {ds} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hidden_ridge_check.csv"), index=False)
    return df
