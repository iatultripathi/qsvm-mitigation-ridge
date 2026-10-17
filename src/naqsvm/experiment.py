"""Experiment grid runner: datasets x seeds x noise levels x shot budgets x methods.

* exact noisy probabilities are computed once per (dataset, seed, noise) and cached (.npz),
  so interrupted runs resume and shot sweeps do not re-simulate circuits;
* every block is wrapped in try/except: failures are logged to failures.csv and the run
  continues;
* raw results are appended to raw_results.csv after every block.
"""
from __future__ import annotations

import os
import time
import traceback
import zlib

import numpy as np
import pandas as pd

from . import kernels as kx
from .data import load_dataset, make_split, prepare
from .methods import (ALLOCATION_BASELINES, NA_VARIANTS, PROPOSED, QProblem, run_allocation_baseline,
                      run_established, run_ideal_qsvm, run_na_qsvm, run_rbf, uniform_record)
from .shots import neyman_efficiency
from .simulators import exact_all_zero_probability, ideal_kernel


def _rng(*keys) -> np.random.Generator:
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(ints))


def angle_scale_for(cfg, dataset: str) -> float:
    return float(cfg.angle_scale_by_dataset.get(dataset, cfg.angle_scale))


def calibrate_bandwidth(cfg, verbose: bool = True) -> pd.DataFrame:
    """Choose the feature-map bandwidth per dataset BEFORE the main experiment.

    Criterion (label-free, pre-declared): maximise the standard deviation of the
    off-diagonal entries of the ideal TRAINING kernel, averaged over cfg.pilot_seeds
    (disjoint from the evaluation seeds). Small spread means the kernel is concentrated,
    either towards 1/d (large bandwidth; Thanasilp et al. 2024) or towards a near-constant
    matrix (very small bandwidth); in both cases finite-shot noise swamps the signal.
    Labels and test splits are not used. The ideal-QSVM validation score is recorded for
    transparency only. Every quantum method uses the same bandwidth.
    """
    from .methods import run_ideal_qsvm
    rows = []
    d = 2 ** cfg.n_qubits
    for ds in cfg.datasets:
        X, y = load_dataset(ds)
        for a in cfg.bandwidth_grid:
            for seed in cfg.pilot_seeds:
                sp = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, seed)
                pr = prepare(X, y, sp, cfg.n_qubits, a, seed)
                Kt = ideal_kernel(pr.A_train, pr.A_train, cfg.reps, cfg.entanglement)
                Kv = ideal_kernel(pr.A_val, pr.A_train, cfg.reps, cfg.entanglement)
                prob = QProblem(None, None, None, None, None, Kt, Kv, Kv, pr.y_train, pr.y_val,
                                pr.y_val, d)
                r = run_ideal_qsvm(prob, cfg)
                iu = np.triu_indices(len(Kt), 1)
                rows.append(dict(dataset=ds, angle_scale=a, pilot_seed=seed, val_score=r["val_score"],
                                 ideal_offdiag_mean=float(Kt[iu].mean()), ideal_offdiag_std=float(Kt[iu].std())))
    df = pd.DataFrame(rows)
    agg = df.groupby(["dataset", "angle_scale"], as_index=False)[["val_score", "ideal_offdiag_mean",
                                                                 "ideal_offdiag_std"]].mean()
    chosen = {}
    for ds, g in agg.groupby("dataset"):
        chosen[ds] = float(g.loc[g["ideal_offdiag_std"].idxmax(), "angle_scale"])
    cfg.angle_scale_by_dataset = chosen
    agg["chosen"] = agg.apply(lambda r: chosen[r["dataset"]] == r["angle_scale"], axis=1)
    if verbose:
        print("chosen bandwidth per dataset:", chosen)
    return agg


def compute_problem(prep, noise, cfg, backend, cache_path=None):
    """Exact noisy probabilities for train (upper triangle incl. diag), val and test."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        return {k: z[k] for k in z.files}, str(z["backend"]), float(z["sim_seconds"]), True
    A_tr, A_va, A_te = prep.A_train, prep.A_val, prep.A_test
    N = len(A_tr)
    iu = np.triu_indices(N)                          # includes diagonal
    X = np.vstack([A_tr[iu[0]],
                   np.repeat(A_va, N, axis=0), A_va,
                   np.repeat(A_te, N, axis=0), A_te])
    Y = np.vstack([A_tr[iu[1]],
                   np.tile(A_tr, (len(A_va), 1)), A_va,
                   np.tile(A_tr, (len(A_te), 1)), A_te])
    p, used, secs = exact_all_zero_probability(X, Y, cfg.reps, cfg.entanglement, noise,
                                               backend=backend, chunk=cfg.sim_chunk)
    o = 0
    P_tr = np.zeros((N, N))
    P_tr[iu] = p[o:o + len(iu[0])]; o += len(iu[0])
    P_tr = P_tr + np.triu(P_tr, 1).T
    nv, nt = len(A_va), len(A_te)
    P_va = p[o:o + nv * N].reshape(nv, N); o += nv * N
    P_va_diag = p[o:o + nv]; o += nv
    P_te = p[o:o + nt * N].reshape(nt, N); o += nt * N
    P_te_diag = p[o:o + nt]; o += nt
    out = dict(P_tr=P_tr, P_va=P_va, P_va_diag=P_va_diag, P_te=P_te, P_te_diag=P_te_diag)
    if cache_path:
        np.savez_compressed(cache_path, backend=used, sim_seconds=secs, n_circuits=len(X), **out)
    return out, used, secs, False


def run_grid(cfg, out_dir: str, backend: str, verbose: bool = True,
             methods_na=None, log_every: int = 1, methods_prior=None) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    raw_csv = os.path.join(out_dir, "raw_results.csv")
    sim_csv = os.path.join(out_dir, "simulation_log.csv")
    fail_csv = os.path.join(out_dir, "failures.csv")
    split_csv = os.path.join(out_dir, "split_indices.csv")
    for f in (raw_csv, sim_csv, fail_csv, split_csv):
        if os.path.exists(f):
            os.remove(f)
    methods_na = methods_na or list(NA_VARIANTS)
    d = 2 ** cfg.n_qubits
    t_start = time.perf_counter()
    n_blocks = len(cfg.datasets) * len(cfg.seeds)
    b = 0

    store = {raw_csv: [], sim_csv: [], fail_csv: [], split_csv: []}

    def append(path, rows):
        # rows are kept in memory and the whole CSV is rewritten (robust to differing columns)
        if rows:
            store[path].extend(rows)
            pd.DataFrame(store[path]).to_csv(path, index=False)

    for di, ds in enumerate(cfg.datasets):
        X, y = load_dataset(ds)
        for seed in cfg.seeds:
            b += 1
            try:
                split = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, seed)
                append(split_csv, [dict(dataset=ds, seed=seed, part=part, index=int(i))
                                   for part, ids in (("train", split.idx_train), ("val", split.idx_val),
                                                     ("test", split.idx_test)) for i in ids])
                prep = prepare(X, y, split, cfg.n_qubits, angle_scale_for(cfg, ds), seed)
                K_tr = ideal_kernel(prep.A_train, prep.A_train, cfg.reps, cfg.entanglement)
                K_va = ideal_kernel(prep.A_val, prep.A_train, cfg.reps, cfg.entanglement)
                K_te = ideal_kernel(prep.A_test, prep.A_train, cfg.reps, cfg.entanglement)
                base = dict(dataset=ds, seed=seed, n_train=len(prep.y_train), n_val=len(prep.y_val),
                            n_test=len(prep.y_test), n_classes=len(np.unique(y)), angle_scale=angle_scale_for(cfg, ds),
                            pca_explained_var=prep.explained_variance)
                # references (noise/shot independent) - stored once per (dataset, seed)
                ref_rows = []
                dummy = QProblem(None, None, None, None, None, K_tr, K_va, K_te,
                                 prep.y_train, prep.y_val, prep.y_test, d)
                for fn in (lambda: run_ideal_qsvm(dummy, cfg),
                           lambda: run_rbf(prep.Z_train, prep.Z_val, prep.Z_test, prep.y_train,
                                           prep.y_val, prep.y_test, cfg, "RBF-SVM (4-D PCA input)"),
                           lambda: run_rbf(prep.F_train, prep.F_val, prep.F_test, prep.y_train,
                                           prep.y_val, prep.y_test, cfg, "RBF-SVM (all features)")):
                    try:
                        ref_rows.append(dict(base, noise="n/a", shots_per_entry=0, status="ok", **fn()))
                    except Exception as exc:
                        append(fail_csv, [dict(base, stage="reference", error=repr(exc),
                                               trace=traceback.format_exc())])
                append(raw_csv, ref_rows)
            except Exception as exc:
                append(fail_csv, [dict(dataset=ds, seed=seed, stage="split/prepare", error=repr(exc),
                                       trace=traceback.format_exc())])
                continue

            for ni, noise in enumerate(cfg.noise_levels):
                try:
                    cache = os.path.join(cache_dir, f"P_{ds}_s{seed}_n{ni}.npz")
                    Pd, used, secs, cached = compute_problem(prep, noise, cfg, backend, cache)
                    prob = QProblem(Pd["P_tr"], Pd["P_va"], Pd["P_va_diag"], Pd["P_te"], Pd["P_te_diag"],
                                    K_tr, K_va, K_te, prep.y_train, prep.y_val, prep.y_test, d)
                    lam_true = float(kx.survival_from_diag(np.diag(prob.P_tr).mean(), d, 0.0))
                    iu = np.triu_indices(len(K_tr), 1)
                    append(sim_csv, [dict(dataset=ds, seed=seed, noise=noise.name, backend=used,
                                          sim_seconds=secs, from_cache=cached,
                                          n_circuits=int(len(iu[0]) + len(K_tr) + prob.P_va.size
                                                         + len(prob.P_va_diag) + prob.P_te.size
                                                         + len(prob.P_te_diag)),
                                          lambda_true=lam_true,
                                          neyman_oracle_var_ratio=neyman_efficiency(prob.P_tr[iu]),
                                          exact_noisy_rel_err=kx.rel_fro_error_offdiag(prob.P_tr, K_tr),
                                          ideal_offdiag_mean=float(K_tr[iu].mean()),
                                          ideal_offdiag_std=float(K_tr[iu].std()))])
                except Exception as exc:
                    append(fail_csv, [dict(base, noise=noise.name, stage="simulation", error=repr(exc),
                                           trace=traceback.format_exc())])
                    continue
                for si, S in enumerate(cfg.shots_per_entry):
                    rows = []
                    ctx = dict(base, noise=noise.name, noise_index=ni, shots_per_entry=S,
                               sim_backend=used, lambda_true=lam_true)
                    try:
                        t0 = time.perf_counter()
                        rec = uniform_record(prob, S, _rng(seed, di, ni, si, "uniform"))
                        t_samp = time.perf_counter() - t0
                        for r in run_established(prob, rec, cfg, t_samp, S, _rng(seed, di, ni, si, "test-est")):
                            rows.append(dict(ctx, **r))
                    except Exception as exc:
                        append(fail_csv, [dict(ctx, stage="established", error=repr(exc),
                                               trace=traceback.format_exc())])
                    for name in (methods_prior if methods_prior is not None else ALLOCATION_BASELINES):
                        try:
                            r = run_allocation_baseline(prob, S, _rng(seed, di, ni, si, name), cfg, name)
                            rows.append(dict(ctx, **r))
                        except Exception as exc:
                            rows.append(dict(ctx, method=name, family="prior-adaptive", status=f"failed: {exc!r}"))
                            append(fail_csv, [dict(ctx, method=name, stage="prior-adaptive", error=repr(exc),
                                                   trace=traceback.format_exc())])
                    for name in methods_na:
                        fam = "proposed" if name == PROPOSED else "ablation"
                        try:
                            r = run_na_qsvm(prob, S, _rng(seed, di, ni, si, "na"), cfg,
                                            NA_VARIANTS[name], name, fam)
                            rows.append(dict(ctx, **r))
                        except Exception as exc:
                            rows.append(dict(ctx, method=name, family=fam, status=f"failed: {exc!r}"))
                            append(fail_csv, [dict(ctx, method=name, stage="na-qsvm", error=repr(exc),
                                                   trace=traceback.format_exc())])
                    append(raw_csv, rows)
            if verbose and (b % log_every == 0 or b == n_blocks):
                print(f"[{b:3d}/{n_blocks}] {ds:14s} seed={seed:<3d} elapsed {time.perf_counter() - t_start:7.1f}s",
                      flush=True)
    return pd.read_csv(raw_csv)
