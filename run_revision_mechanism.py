"""Stage R1: mechanism study on the ORIGINAL v2 design (4 datasets x 10 seeds 2000-2009, N=60, 4 qubits).
Noise: global Lam in {0.8, 0.5, 0.25} (analytic), low / medium / high local depolarizing, thermal mismatch,
plus device-like A and B. Shots per entry: exact, 16, 64, 256, 1024.
Writes results/revision/mechanism_identity.csv and mechanism_accuracy.csv.
Usage: python run_revision_mechanism.py [--quick]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from naqsvm.config import ExperimentConfig, NoiseSpec          # noqa: E402
from naqsvm.data import load_dataset, make_split, prepare       # noqa: E402
from naqsvm.experiment import _rng                              # noqa: E402
from naqsvm.methods import QProblem                             # noqa: E402
from naqsvm.revision import compute_problem_rev, make_device_noise, mechanism_block  # noqa: E402
from naqsvm.simulators import ideal_kernel                      # noqa: E402

QUICK = "--quick" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "revision")
os.makedirs(OUT, exist_ok=True)

cfg = ExperimentConfig()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiment_config.json")) as f:
    v2 = json.load(f)
cfg.angle_scale_by_dataset = v2["angle_scale_by_dataset"]       # the v2 label-free bandwidth choice
d = 2 ** cfg.n_qubits
noises = [NoiseSpec(f"global Λ={l}", kind="global", p1=l) for l in (0.8, 0.5, 0.25)] + \
         [nz for nz in cfg.noise_levels if nz.kind != "none"] + \
         [make_device_noise(n, cfg.n_qubits) for n in ("device A (low-error class)", "device B (higher-error class)")]
shots = [0, 16, 64, 256, 1024]
datasets, seeds = cfg.datasets, cfg.seeds
if QUICK:
    datasets, seeds, shots = ["iris"], [2000], [0, 64]
    noises = noises[:1] + noises[4:5] + noises[-1:]

id_rows, acc_rows = [], []
t0 = time.time()
for di, ds in enumerate(datasets):
    X, y = load_dataset(ds)
    for seed in seeds:
        sp = make_split(y, cfg.n_train, cfg.n_val, cfg.n_test, seed)
        pr = prepare(X, y, sp, cfg.n_qubits, cfg.angle_scale_by_dataset[ds], seed)
        K_tr = ideal_kernel(pr.A_train, pr.A_train, cfg.reps, cfg.entanglement)
        K_va = ideal_kernel(pr.A_val, pr.A_train, cfg.reps, cfg.entanglement)
        K_te = ideal_kernel(pr.A_test, pr.A_train, cfg.reps, cfg.entanglement)
        for nz in noises:
            Pd, used, secs = compute_problem_rev(pr.A_train, pr.A_val, pr.A_test, nz, cfg.reps, cfg.entanglement)
            prob = QProblem(Pd["P_tr"], Pd["P_va"], Pd["P_va_diag"], Pd["P_te"], Pd["P_te_diag"],
                            K_tr, K_va, K_te, pr.y_train, pr.y_val, pr.y_test, d)
            for S in shots:
                ctx = dict(dataset=ds, seed=seed, noise=nz.name, noise_kind=nz.kind,
                           shots_per_entry=("exact" if S == 0 else S), sim_backend=used)
                try:
                    ident, accs = mechanism_block(prob, S, _rng(seed, di, nz.name, S, "mechanism"))
                    id_rows.append(dict(ctx, status="ok", **ident))
                    acc_rows += [dict(ctx, **a) for a in accs]
                except Exception as exc:
                    id_rows.append(dict(ctx, status=f"failed: {exc!r}"))
        pd.DataFrame(id_rows).to_csv(os.path.join(OUT, "mechanism_identity.csv"), index=False)
        pd.DataFrame(acc_rows).to_csv(os.path.join(OUT, "mechanism_accuracy.csv"), index=False)
        print(f"{ds} seed {seed} done, {time.time() - t0:.0f}s", flush=True)
print("R1 finished", time.time() - t0)
