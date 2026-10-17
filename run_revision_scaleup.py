"""Stage R2 (scale-up). Usage: python run_revision_scaleup.py A|B [--quick]

A : 4 qubits, N_train = 200 (val 60, test 100); datasets breast_cancer, synthetic_large (1000 samples),
    digits_3v8, adhoc (quantum-structured, Havlicek et al. 2019); noise medium, high, device A, device B.
B : 6 qubits, N_train = 80 (val 30, test 60); datasets breast_cancer, synthetic, adhoc;
    noise device A, device B.
Both: seeds 3000-3004 (fresh), shots/entry 64, 256, 1024. Methods: established QSVM variants, the
Miroszewski-style margin-Neyman allocator, NA-QSVM, ideal QSVM and RBF references, and the mechanism
block (identity + fixed/tuned-C mitigation effect). Bandwidth chosen per dataset by the v2 label-free rule
(max std of ideal off-diagonal training-kernel entries on pilot seeds 100-102); adhoc uses raw angles.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn import datasets as skd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from naqsvm.config import ExperimentConfig, NoiseSpec            # noqa: E402
from naqsvm.data import Prepared, load_dataset, make_split, prepare  # noqa: E402
from naqsvm.experiment import _rng                                # noqa: E402
from naqsvm.methods import (NA_VARIANTS, PROPOSED, QProblem, run_allocation_baseline,  # noqa: E402
                            run_established, run_ideal_qsvm, run_na_qsvm, run_rbf, uniform_record)
from naqsvm.revision import compute_problem_rev, load_adhoc, make_device_noise, mechanism_block  # noqa: E402
from naqsvm.simulators import ideal_kernel                        # noqa: E402

STAGE = sys.argv[1]
QUICK = "--quick" in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "revision")
os.makedirs(OUT, exist_ok=True)

cfg = ExperimentConfig()
if STAGE == "A":
    cfg.n_qubits, cfg.n_train, cfg.n_val, cfg.n_test = 4, 200, 60, 100
    cfg.datasets = ["breast_cancer", "synthetic_large", "digits_3v8", "adhoc"]
    noises = [nz for nz in cfg.noise_levels if nz.name in ("medium", "high")]
    noises += [make_device_noise(n, 4) for n in ("device A (low-error class)", "device B (higher-error class)")]
else:
    cfg.n_qubits, cfg.n_train, cfg.n_val, cfg.n_test = 6, 80, 30, 60
    cfg.datasets = ["breast_cancer", "synthetic", "adhoc"]
    noises = [make_device_noise(n, 6) for n in ("device A (low-error class)", "device B (higher-error class)")]
NSEEDS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--nseeds=")), 5))
cfg.seeds = list(range(3000, 3000 + NSEEDS))
cfg.shots_per_entry = [64, 256, 1024]
if QUICK:
    cfg.datasets, cfg.seeds, cfg.shots_per_entry = cfg.datasets[-1:], [3000], [64]
    noises = noises[-1:]
    cfg.n_train, cfg.n_val, cfg.n_test = 30, 15, 20
d = 2 ** cfg.n_qubits
TAG = f"scaleup_{STAGE}" + ("_quick" if QUICK else "")


def load(ds):
    if ds == "adhoc":
        return load_adhoc(cfg.n_qubits, 600 if STAGE == "A" else 400, 0.1, cfg.reps, cfg.entanglement)
    if ds == "synthetic_large":
        return skd.make_classification(n_samples=1000, n_features=8, n_informative=4, n_redundant=2,
                                       n_classes=2, class_sep=1.0, flip_y=0.03, random_state=12345)
    return load_dataset(ds)


def prep(ds, X, y, sp, a, seed):
    if ds == "adhoc":       # angles are the raw features (the labelling function lives in this space)
        g = lambda idx: X[idx]
        A = [g(sp.idx_train), g(sp.idx_val), g(sp.idx_test)]
        return Prepared(*A, *A, *A, y[sp.idx_train], y[sp.idx_val], y[sp.idx_test], 1.0)
    return prepare(X, y, sp, cfg.n_qubits, a, seed)


def bandwidth(ds, X, y):
    if ds == "adhoc":
        return np.nan
    best = None
    for a in cfg.bandwidth_grid:
        s = []
        for ps in cfg.pilot_seeds:
            sp = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, ps)
            pr = prepare(X, y, sp, cfg.n_qubits, a, ps)
            K = ideal_kernel(pr.A_train, pr.A_train, cfg.reps, cfg.entanglement)
            s.append(K[np.triu_indices(len(K), 1)].std())
        if best is None or np.mean(s) > best[0]:
            best = (np.mean(s), a)
    return best[1]


rows, mech_id, mech_acc, simlog = [], [], [], []
done = set()
_paths = {k: os.path.join(OUT, f"{TAG}_{k}.csv") for k in ("methods", "identity", "accuracy", "simlog")}
if all(os.path.exists(v) for v in _paths.values()):          # resume: keep finished (dataset, seed) blocks
    _old = {k: pd.read_csv(v) for k, v in _paths.items()}
    done = set(map(tuple, _old["simlog"][["dataset", "seed"]].drop_duplicates().values.tolist()))
    _keep = lambda df: df[[(a, b) in done for a, b in zip(df["dataset"], df["seed"])]].to_dict("records")
    rows, mech_id, mech_acc, simlog = (_keep(_old[k]) for k in ("methods", "identity", "accuracy", "simlog"))
    print("resuming; done blocks:", sorted(done), flush=True)
t0 = time.time()
for di, ds in enumerate(cfg.datasets):
    X, y = load(ds)
    a = bandwidth(ds, X, y)
    print(f"{ds}: n={len(y)} bandwidth={a}", flush=True)
    for seed in cfg.seeds:
        if (ds, seed) in done:
            continue
        sp = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, seed)
        pr = prep(ds, X, y, sp, a, seed)
        K_tr = ideal_kernel(pr.A_train, pr.A_train, cfg.reps, cfg.entanglement)
        K_va = ideal_kernel(pr.A_val, pr.A_train, cfg.reps, cfg.entanglement)
        K_te = ideal_kernel(pr.A_test, pr.A_train, cfg.reps, cfg.entanglement)
        base = dict(stage=STAGE, n_qubits=cfg.n_qubits, dataset=ds, seed=seed, n_train=len(pr.y_train),
                    n_test=len(pr.y_test), angle_scale=a)
        dummy = QProblem(None, None, None, None, None, K_tr, K_va, K_te, pr.y_train, pr.y_val, pr.y_test, d)
        rows.append(dict(base, noise="n/a", shots_per_entry=0, status="ok", **run_ideal_qsvm(dummy, cfg)))
        rows.append(dict(base, noise="n/a", shots_per_entry=0, status="ok",
                         **run_rbf(pr.Z_train, pr.Z_val, pr.Z_test, pr.y_train, pr.y_val, pr.y_test, cfg,
                                   "RBF-SVM (same input as quantum)")))
        if ds != "adhoc":
            rows.append(dict(base, noise="n/a", shots_per_entry=0, status="ok",
                             **run_rbf(pr.F_train, pr.F_val, pr.F_test, pr.y_train, pr.y_val, pr.y_test, cfg,
                                       "RBF-SVM (all features)")))
        for ni, nz in enumerate(noises):
            Pd, used, secs = compute_problem_rev(pr.A_train, pr.A_val, pr.A_test, nz, cfg.reps, cfg.entanglement)
            prob = QProblem(Pd["P_tr"], Pd["P_va"], Pd["P_va_diag"], Pd["P_te"], Pd["P_te_diag"],
                            K_tr, K_va, K_te, pr.y_train, pr.y_val, pr.y_test, d)
            lam_true = float((np.diag(prob.P_tr).mean() - 1 / d) / (1 - 1 / d))
            simlog.append(dict(base, noise=nz.name, backend=used, sim_seconds=secs, lambda_true=lam_true))
            for si, S in enumerate(cfg.shots_per_entry):
                ctx = dict(base, noise=nz.name, shots_per_entry=S, lambda_true=lam_true)
                try:
                    rec = uniform_record(prob, S, _rng(seed, di, ni, si, "uniform"))
                    for r in run_established(prob, rec, cfg, 0.0, S, _rng(seed, di, ni, si, "test-est")):
                        rows.append(dict(ctx, **r))
                    nm = "MarginNeyman+clip (Miroszewski-style approx.)"
                    rows.append(dict(ctx, **run_allocation_baseline(prob, S, _rng(seed, di, ni, si, nm), cfg, nm)))
                    rows.append(dict(ctx, **run_na_qsvm(prob, S, _rng(seed, di, ni, si, "na"), cfg,
                                                        NA_VARIANTS[PROPOSED], PROPOSED, "proposed")))
                except Exception as exc:
                    rows.append(dict(ctx, method="(block)", status=f"failed: {exc!r}"))
                try:
                    ident, accs = mechanism_block(prob, S, _rng(seed, di, nz.name, S, "mechanism"))
                    mech_id.append(dict(ctx, status="ok", **ident))
                    mech_acc += [dict(ctx, **x) for x in accs]
                except Exception as exc:
                    mech_id.append(dict(ctx, status=f"failed: {exc!r}"))
        for name, data in (("methods", rows), ("identity", mech_id), ("accuracy", mech_acc), ("simlog", simlog)):
            pd.DataFrame(data).to_csv(os.path.join(OUT, f"{TAG}_{name}.csv"), index=False)
        print(f"[{TAG}] {ds} seed {seed} done, {time.time() - t0:.0f}s", flush=True)
print(f"{TAG} finished in {time.time() - t0:.0f}s")
