"""Analyse the hardware counts written by run_ibm_hardware.py.

python analyze_hardware.py            (real QPU results in hw_qpu/)
python analyze_hardware.py --dry-run  (fake-backend results in hw_dryrun/)

For each split and for the full shot budget and a 64-shot sub-sample (exact hypergeometric thinning of the same
counts), it reports:
  * Lambda_hat (diagonal survival) and the hidden ridge r_hat of the device kernel;
  * Corollary 1 on hardware data: SVM(K_raw, C) vs SVM(K_mit + r_hat I, Lambda_hat C) (relative decision difference);
  * balanced accuracy of raw vs mitigated kernels at fixed C in {0.1, 1, 10, 100} and with C tuned on validation;
  * references: ideal (noise-free) QSVM and RBF-SVM on the same split.
Writes hw_*/hardware_results.csv and hw_*/HARDWARE_RESULTS.md.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from naqsvm.config import ExperimentConfig                     # noqa: E402
from naqsvm.data import load_dataset, make_split, prepare       # noqa: E402
from naqsvm.methods import QProblem, run_ideal_qsvm, run_rbf    # noqa: E402
from naqsvm.revision import mechanism_block                     # noqa: E402
from naqsvm.simulators import ideal_kernel                      # noqa: E402

TAG = "dryrun" if "--dry-run" in sys.argv else ("simtest" if "--simtest" in sys.argv else "qpu")
OUT = os.path.join(HERE, f"hw_{TAG}")
runs = sorted(r for r in os.listdir(OUT) if os.path.exists(os.path.join(OUT, r, "counts.csv")))
if not runs:
    sys.exit(f"no counts.csv found under {OUT}/seed*/ - run run_ibm_hardware.py first")
metas = [json.load(open(os.path.join(OUT, r, "metadata.json"))) for r in runs]
meta = dict(metas[0], splits=sum(m["splits"] for m in metas),
            backend=", ".join(sorted(set(m["backend"] for m in metas))))
for m in metas[1:]:
    assert m["dataset"] == meta["dataset"] and m["n_train"] == meta["n_train"], "all runs must use the same design"
cnt = pd.concat([pd.read_csv(os.path.join(OUT, r, "counts.csv")) for r in runs], ignore_index=True)
NQ, REPS, ENT = meta["n_qubits"], meta["reps"], meta["entanglement"]
d = 2 ** NQ
cfg = ExperimentConfig()
rng = np.random.default_rng(123)


def load(ds):
    if ds == "adhoc":
        from naqsvm.revision import load_adhoc
        return load_adhoc(NQ, 600, 0.1, REPS, ENT)
    return load_dataset(ds)


def robust_tuned(Ktr, Kva, Kte, ytr, yva, yte, grid):
    """Small-sample C selection for the hardware split (documented deviation from the simulation protocol, whose
    10-30-point validation sets produce many ties): score(C) = mean of stratified 5-fold CV balanced accuracy on the
    TRAINING kernel and the validation balanced accuracy; ties -> the median C of the tied set (not the smallest,
    which collapses to a one-class predictor). Test data are used only once, for the chosen C."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score as bas
    from naqsvm.revision import _fit
    scores = []
    for C in grid:
        cv = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(ytr)), ytr):
            try:
                m = _fit(Ktr[np.ix_(a, a)], ytr[a], C)
                cv.append(bas(ytr[b], m.predict(Ktr[np.ix_(b, a)])))
            except Exception:
                cv.append(np.nan)
        try:
            v = bas(yva, _fit(Ktr, ytr, C).predict(Kva))
        except Exception:
            v = np.nan
        scores.append(0.5 * np.nanmean(cv) + 0.5 * v)
    scores = np.array(scores)
    tied = [c for c, sc in zip(grid, scores) if sc >= np.nanmax(scores) - 1e-12]
    C = tied[len(tied) // 2]
    return bas(yte, _fit(Ktr, ytr, C).predict(Kte)), C


def thin(z, n, m):
    """Exact sub-sample of m of the n recorded shots (without replacement)."""
    return rng.hypergeometric(z, n - z, m) / m


X, y = load(meta["dataset"])
rows, acc_all, lines = [], [], []
for seed, g in cnt.groupby("split_seed"):
    sp = make_split(y, meta["n_train"], meta["n_val"], meta["n_test"], int(seed))
    if meta["dataset"] == "adhoc":
        A_tr, A_va, A_te = X[sp.idx_train], X[sp.idx_val], X[sp.idx_test]
        Z = (A_tr, A_va, A_te)
    else:
        pr = prepare(X, y, sp, NQ, meta["angle_scale"], int(seed))
        A_tr, A_va, A_te = pr.A_train, pr.A_val, pr.A_test
        Z = (pr.Z_train, pr.Z_val, pr.Z_test)
    y_tr, y_va, y_te = y[sp.idx_train], y[sp.idx_val], y[sp.idx_test]
    K_tr = ideal_kernel(A_tr, A_tr, REPS, ENT)
    K_va = ideal_kernel(A_va, A_tr, REPS, ENT)
    K_te = ideal_kernel(A_te, A_tr, REPS, ENT)
    N, Nv, Nt = len(A_tr), len(A_va), len(A_te)
    ref = QProblem(None, None, None, None, None, K_tr, K_va, K_te, y_tr, y_va, y_te, d)
    ideal = run_ideal_qsvm(ref, cfg)["balanced_accuracy"]
    rbf = run_rbf(*Z, y_tr, y_va, y_te, cfg, "rbf")["balanced_accuracy"]
    for m in (None, 64):
        def est(block, shape):
            b = g[g.block == block]
            v = b.zeros.values / b.shots.values if m is None else \
                np.array([thin(z, n, m) for z, n in zip(b.zeros.values, b.shots.values)])
            M = np.zeros(shape)
            M[b.i.values, b.j.values] = v
            return M, v
        P_tr, _ = est("tr_off", (N, N))
        P_tr = P_tr + P_tr.T
        _, dtr = est("tr_diag", (N, N))
        P_tr[np.arange(N), np.arange(N)] = dtr
        P_va, _ = est("va", (Nv, N))
        _, dva = est("va_diag", (Nv, Nv))
        P_te, _ = est("te", (Nt, N))
        _, dte = est("te_diag", (Nt, Nt))
        prob = QProblem(P_tr, P_va, dva, P_te, dte, K_tr, K_va, K_te, y_tr, y_va, y_te, d)
        ident, accs = mechanism_block(prob, 0, rng)      # S=0: use the measured estimates as they are
        shots_lbl = int(g.shots.iloc[0]) if m is None else m
        iu = np.triu_indices(N, 1)
        rows.append(dict(split_seed=seed, shots=shots_lbl, lambda_hat=ident["lam_hat"], r_hat=ident["r_hat"],
                         corollary1_rel_diff=ident["M_rel_diff"], corollary1_agreement=ident["M_agree"],
                         min_eig_raw=ident["min_eig_raw"],
                         kernel_rel_err_raw=float(np.linalg.norm(P_tr[iu] - K_tr[iu]) / np.linalg.norm(K_tr[iu])),
                         ideal_qsvm=ideal, rbf=rbf))
        from naqsvm.revision import C_GRID_FINE
        lam = ident["lam_hat"]
        Kraw = P_tr.copy(); np.fill_diagonal(Kraw, 1.0)
        mit = lambda K: (K - 1 / d) / lam + 1 / d
        Kmit = mit(Kraw); np.fill_diagonal(Kmit, 1.0)
        from naqsvm import kernels as kx
        robust = {"raw": (Kraw, P_va, P_te), "mitigated": (Kmit, mit(P_va), mit(P_te)),
                  "raw+clip": (kx.psd_clip(Kraw), P_va, P_te), "mitigated+clip": (kx.psd_clip(Kmit), mit(P_va), mit(P_te)),
                  "ideal (no ridge)": (K_tr, K_va, K_te)}
        for a in accs:
            if a["variant"] in robust:
                a["bacc_tuned_robust"], a["C_tuned_robust"] = robust_tuned(*robust[a["variant"]], y_tr, y_va, y_te, C_GRID_FINE)
            acc_all.append(dict(split_seed=seed, shots=shots_lbl, **a))

res = pd.DataFrame(rows)
acc = pd.DataFrame(acc_all)
res.to_csv(os.path.join(OUT, "hardware_results.csv"), index=False)
acc.to_csv(os.path.join(OUT, "hardware_accuracy.csv"), index=False)
cols = ["bacc_C0.1", "bacc_C1", "bacc_C10", "bacc_C100", "bacc_tuned", "bacc_tuned_robust"]
tab = acc[acc.variant.isin(["raw", "mitigated", "raw+clip", "mitigated+clip", "ideal (no ridge)"])].groupby(["shots", "variant"])[cols].mean()
L = [f"# Hardware results ({'FAKE backend dry run' if meta['dry_run'] else 'real QPU'}: {meta['backend']})\n",
     f"Dataset {meta['dataset']}, {meta['splits']} split(s), N_train={meta['n_train']}, N_val={meta['n_val']}, "
     f"N_test={meta['n_test']}, {meta['shots']} shots/circuit, physical qubits {meta.get('layout')}, "
     f"mean two-qubit gates per circuit {meta.get('mean_2q_gates', float('nan')):.1f}.\n",
     "## Device kernel and Corollary 1\n", res.to_markdown(index=False, floatfmt=".4g"), "",
     "## Balanced accuracy: raw vs mitigated, fixed and tuned C (mean over splits)\n", tab.round(3).to_markdown(), "",
     "`bacc_tuned` uses the simulation protocol (validation only, ties -> smallest C); with a 10-point validation set it "
     "often collapses to a one-class predictor (0.5). `bacc_tuned_robust` uses 5-fold CV on the training kernel + "
     "validation, ties -> median C; report this one for the hardware split.\n",
     "Expected from the simulations: Corollary 1 relative difference ~1e-7 unless the kernel is strongly indefinite; "
     "mitigated minus raw positive at small fixed C, negative at large fixed C, near zero with tuned C. With one split "
     "this is a sanity check, not a statistical test."]
open(os.path.join(OUT, "HARDWARE_RESULTS.md"), "w").write("\n".join(L) + "\n")

# LaTeX table for the paper (one row per split and shot budget)
tr = acc.pivot_table(index=["split_seed", "shots"], columns="variant", values="bacc_tuned_robust")
T = res.drop(columns=["ideal_qsvm"]).set_index(["split_seed", "shots"]).join(
    tr[["raw", "mitigated", "ideal (no ridge)"]].rename(columns={"ideal (no ridge)": "ideal_qsvm"})).reset_index()
T = T[["split_seed", "shots", "lambda_hat", "r_hat", "corollary1_rel_diff", "raw", "mitigated", "ideal_qsvm", "rbf"]]
body = T.to_latex(index=False, escape=True, formatters={"lambda_hat": "{:.3f}".format, "r_hat": "{:.3f}".format,
                  "corollary1_rel_diff": "{:.1e}".format, "raw": "{:.3f}".format, "mitigated": "{:.3f}".format,
                  "ideal_qsvm": "{:.3f}".format, "rbf": "{:.3f}".format},
                  header=["Split", "Shots", "$\\hat\\Lambda$", "$\\hat r$", "Cor.~1 diff.", "Raw", "Mitigated",
                          "Ideal QSVM", "RBF"])
body = body.replace("\\textbackslash ", "\\").replace("\\$", "$").replace("\\{", "{").replace("\\}", "}")
cap = (f"Hardware check on {meta['backend'].replace('_', chr(92) + '_')} ({meta['splits']} splits of {meta['dataset'].replace('_', ' ')}, N$_{{train}}$={meta['n_train']}, "
       f"physical qubits {meta.get('layout')}): estimated survival $\\hat\\Lambda$, hidden ridge $\\hat r$, Corollary~1 "
       "relative decision difference, and balanced accuracy of the raw and mitigated device kernels (C tuned by "
       "5-fold CV + validation; the noise-free ideal kernel is tuned the same way), with the RBF-SVM reference.")
open(os.path.join(OUT, "tableR5_hardware.tex"), "w").write(
    "\\begin{table}[!t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:hardware}\n\\resizebox{\\columnwidth}{!}{%\n"
    + body + "}\n\\end{table}\n")
print("\n".join(L))
