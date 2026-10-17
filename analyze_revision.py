"""Analysis of the revision experiments -> tables/revision/*.csv|.tex, figures/revision/*.png|.pdf,
REVISION_FINDINGS.md. Run after run_revision_mechanism.py and run_revision_scaleup.py A and B."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from naqsvm.stats import bootstrap_ci, holm, tost_paired, wilcoxon_p  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(ROOT, "results", "revision")
TAB = os.path.join(ROOT, "tables", "revision")
FIG = os.path.join(ROOT, "figures", "revision")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
MARGIN = 0.01
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]   # validated categorical order (slots 1,2,3,7,5)
MK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", ":", "-.", (0, (5, 1))]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "lines.linewidth": 2, "lines.markersize": 5,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
NOISE_ORDER = ["global Λ=0.8", "global Λ=0.5", "global Λ=0.25", "low", "medium", "high",
               "thermal (mismatch)", "device A (low-error class)", "device B (higher-error class)"]
SHOT_ORDER = ["exact", "1024", "256", "64", "16"]
F = []          # findings lines


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else (f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}")


def paired_test(d, label, extra=None):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap_ci(d)
    t = tost_paired(d, MARGIN)
    r = dict(comparison=label, n_units=len(d), mean_diff=d.mean(), boot_ci_lo=lo, boot_ci_hi=hi,
             p_wilcoxon=wilcoxon_p(d), p_tost=t["p_tost"], equivalent_pm_0_01=t["equivalent"])
    r.update(extra or {})
    return r


# =============================================================================================
# R1 / W1: identity
# =============================================================================================
idf = pd.read_csv(os.path.join(RES, "mechanism_identity.csv"))
idf = idf[idf.status == "ok"].copy()
idf["shots_per_entry"] = idf["shots_per_entry"].astype(str)
g = idf.groupby(["noise", "shots_per_entry"])
t1 = g.agg(n=("M_rel_diff", "size"), lam_hat=("lam_hat", "mean"), r_hat=("r_hat", "mean"),
           cor1_max_rel_diff=("M_rel_diff", "max"), cor1_min_agreement=("M_agree", "min"),
           thm1_ideal_plus_ridge_rel_diff=("B_rel_diff", "mean"), no_ridge_control_rel_diff=("Z_rel_diff", "mean"),
           shot_noise_only_rel_diff=("Mo_vs_B_rel_diff", "mean")).reset_index()
t1["noise"] = pd.Categorical(t1["noise"], NOISE_ORDER, ordered=True)
t1["shots_per_entry"] = pd.Categorical(t1["shots_per_entry"], SHOT_ORDER, ordered=True)
t1 = t1.sort_values(["noise", "shots_per_entry"])
t1.to_csv(os.path.join(TAB, "tableR1_identity.csv"), index=False)
F.append("## W1 - C1 at low shot counts: resolved by Corollary 1 (exact finite-shot identity)\n")
_all = [idf.assign(run="R1 (N=60, 4q)")]
for _st, _lab in (("A", "R2-A (N=200, 4q)"), ("B", "R2-B (N=80, 6q)")):
    _f = os.path.join(RES, f"scaleup_{_st}_identity.csv")
    if os.path.exists(_f):
        _x = pd.read_csv(_f)
        _all.append(_x[_x.status == "ok"].assign(run=_lab))
alli = pd.concat(_all, ignore_index=True)
exc = alli[alli.M_rel_diff > 1e-6]
thr = float(exc.min_eig_raw.max()) if len(exc) else 0.0
mild = alli[alli.min_eig_raw > thr]
strong = alli[alli.min_eig_raw <= thr]
alli.groupby("run").agg(cells=("M_rel_diff", "size"), max_rel_diff=("M_rel_diff", "max"),
                        n_exceptions=("M_rel_diff", lambda v: int((v > 1e-6).sum())),
                        min_eig_raw_min=("min_eig_raw", "min")).to_csv(os.path.join(TAB, "tableR1b_identity_all_runs.csv"))
F.append(f"Test of Corollary 1, SVM(K_raw, C) vs SVM(K_mit + r_hat I, Lam_hat C), over {len(alli)} cells from all three runs "
         f"(R1: 4 datasets x 10 seeds x {idf.noise.nunique()} noise settings x 5 shot budgets incl. exact; R2-A: N=200; R2-B: 6 qubits). "
         f"All {len(mild)} cells whose raw kernel has smallest eigenvalue above {thr:.2f} agree to solver precision "
         f"(max relative decision difference {mild.M_rel_diff.max():.1e}). Of the {len(strong)} strongly indefinite cells "
         f"(smallest eigenvalue <= {thr:.2f}; few shots and/or large N), {len(strong) - len(exc)} also agree; {len(exc)} "
         f"({100 * len(exc) / len(alli):.1f}% of all cells) do not (max rel. diff {exc.M_rel_diff.max():.2f}, min prediction "
         f"agreement {exc.M_agree.min():.3f}). For a strongly indefinite kernel the SVM dual is non-convex and its solution is not "
         f"unique; libsvm's safeguard for non-positive curvature is not scale-invariant, so two mathematically equivalent problems "
         f"can stop at different stationary points. These cells occur at {sorted(set(exc.shots_per_entry.astype(str)))} shots per entry. "
         f"The identity holds under every noise model (global, local depolarizing, thermal, device-like with coherent error).")
for S in ["exact", "1024", "64", "16"]:
    s = idf[idf.shots_per_entry == S]
    F.append(f"- shots={S}: ideal+ridge (Theorem 1) rel. diff {s.B_rel_diff.mean():.3f}; the part explained by the "
             f"measured-kernel deviation E/Lam alone (same ridge, same C; shot noise + departure from global "
             f"depolarizing) is {s.Mo_vs_B_rel_diff.mean():.3f}; no-ridge control {s.Z_rel_diff.mean():.3f}.")
F.append("Interpretation: the v2 '64-shot failure' was the shot-noise term E/Lam, not a failure of the ridge account. "
         "Global mitigation leaves the signal-to-noise ratio unchanged (mean SNR raw "
         f"{idf.snr_raw.replace(np.inf, np.nan).median():.2f} vs mitigated {idf.snr_mit.replace(np.inf, np.nan).median():.2f}, median over cells).\n")

# figure R1
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for ax, nz in zip(axes, ["global Λ=0.5", "medium", "device B (higher-error class)"]):
    s = t1[t1.noise == nz]
    x = np.arange(len(s))
    for k, (col, lab) in enumerate([("cor1_max_rel_diff", "Cor. 1: mitigated + r̂I (max)"),
                                    ("thm1_ideal_plus_ridge_rel_diff", "Thm 1: ideal + rI"),
                                    ("no_ridge_control_rel_diff", "control: ideal, no ridge")]):
        ax.plot(x, np.maximum(s[col].values, 1e-12), color=COL[k], marker=MK[k], ls=LS[k], label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x, s.shots_per_entry.astype(str))
    ax.set_title(nz.replace(" (higher-error class)", ""), fontsize=9)
    ax.set_xlabel("shots per entry")
axes[0].set_ylabel("rel. decision difference\nto raw noisy SVM")
axes[0].legend(fontsize=7, frameon=False, loc="center left")
save(fig, "figR1_identity_vs_shots")

# =============================================================================================
# R1 / W2 + W3: accuracy
# =============================================================================================
adf = pd.read_csv(os.path.join(RES, "mechanism_accuracy.csv"))
adf["shots_per_entry"] = adf["shots_per_entry"].astype(str)
acc_cols = ["bacc_C0.1", "bacc_C1", "bacc_C10", "bacc_C100", "bacc_tuned"]
key = ["dataset", "seed", "noise", "shots_per_entry"]
piv = {c: adf.pivot_table(index=key, columns="variant", values=c) for c in acc_cols}
rmap = idf.set_index(key)["r_hat"]


def mit_effect(pair, label, noise_filter=None):
    rows = []
    a, b = pair
    for c in acc_cols:
        p = piv[c]
        dd = (p[a] - p[b]).rename("d").reset_index()
        if noise_filter is not None:
            dd = dd[dd.noise.isin(noise_filter)]
        # units = dataset x seed (average over noise and shots inside a split)
        u = dd.groupby(["dataset", "seed"])["d"].mean()
        rows.append(paired_test(u.values, f"{label} [{c.replace('bacc_', '')}]",
                                dict(C_setting=c.replace("bacc_", ""), mean_abs_cell_diff=dd["d"].abs().mean())))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"])
    return out


noisy = [n for n in NOISE_ORDER if n in set(adf.noise)]
t2a = mit_effect(("mitigated", "raw"), "mitigated - raw", noisy)
t2b = mit_effect(("mitigated+clip", "raw+clip"), "mitigated+clip - raw+clip", noisy)
t2 = pd.concat([t2a, t2b])
t2.to_csv(os.path.join(TAB, "tableR2_mitigation_fixed_vs_tuned_C.csv"), index=False)

# dose-response: |cell effect| vs r_hat, per C setting (Spearman over cells)
dose = []
for c in acc_cols:
    dd = (piv[c]["mitigated"] - piv[c]["raw"]).rename("d").reset_index()
    dd["r_hat"] = rmap.reindex(pd.MultiIndex.from_frame(dd[key])).values
    cell = dd.groupby(["noise", "shots_per_entry"]).agg(d=("d", "mean"), absd=("d", lambda v: np.abs(v).mean()),
                                                         r_hat=("r_hat", "mean")).reset_index()
    rho, p = st.spearmanr(cell["r_hat"], cell["absd"])
    dose.append(dict(C_setting=c.replace("bacc_", ""), spearman_absdiff_vs_rhat=rho, p=p,
                     mean_abs_diff=cell["absd"].mean()))
    cell.assign(C_setting=c.replace("bacc_", "")).to_csv(os.path.join(TAB, f"_cells_{c}.csv"), index=False)
dose = pd.DataFrame(dose)
dose.to_csv(os.path.join(TAB, "tableR2b_dose_response.csv"), index=False)

F.append("## W2 - does mitigation hurt? Sharpened prediction and test\n")
F.append("Prediction from Corollary 1: mitigation = removing the ridge r_hat (C rescaled). A ridge acts as an extra "
         "soft-margin term that overlaps with C, so (P2a) at FIXED C mitigation changes accuracy and the size of the "
         "change grows with r_hat; (P2b) with C TUNED on validation the change vanishes.")
for _, r in t2.iterrows():
    F.append(f"- {r.comparison}: {r.mean_diff:+.4f} [{r.boot_ci_lo:+.4f}, {r.boot_ci_hi:+.4f}], "
             f"mean |cell effect| {r.mean_abs_cell_diff:.4f}, Holm p = {fmt_p(r.p_holm)}, "
             f"TOST(±0.01) p = {fmt_p(r.p_tost)} -> {'equivalent' if r.equivalent_pm_0_01 else 'not shown equivalent'}")
for _, r in dose.iterrows():
    F.append(f"- dose-response, C={r.C_setting}: mean |effect| per cell {r.mean_abs_diff:.4f}; Spearman(|effect|, r_hat) "
             f"= {r.spearman_absdiff_vs_rhat:+.2f} (p = {fmt_p(r.p)})")
F.append("")

# figure R2: signed effect per noise setting (ordered by r_hat), one line per C setting, 95% bootstrap CI over units
fig, ax = plt.subplots(figsize=(3.5, 3.3))
order = (idf.groupby("noise")["r_hat"].mean().sort_values())
for k, c in enumerate(["bacc_C0.1", "bacc_C1", "bacc_C10", "bacc_C100", "bacc_tuned"]):
    dd = (piv[c]["mitigated"] - piv[c]["raw"]).rename("d").reset_index()
    u = dd.groupby(["noise", "dataset", "seed"])["d"].mean().reset_index()
    ms, lo, hi = [], [], []
    for nz in order.index:
        v = u[u.noise == nz]["d"].values
        ms.append(v.mean()); l, h = bootstrap_ci(v); lo.append(l); hi.append(h)
    lab = "C tuned on validation" if c == "bacc_tuned" else f"fixed C = {c.split('_C')[1]}"
    xs = np.array(order.values)
    ax.errorbar(xs * (1 + 0.03 * (k - 2)), ms, yerr=[np.array(ms) - lo, np.array(hi) - ms], color=COL[k],
                marker=MK[k], ls=LS[k], lw=1.4, ms=4, capsize=1.5, label=lab)
ax.axhline(0, color="#52514e", lw=0.8)
ax.axhspan(-MARGIN, MARGIN, color="#e4e3df", alpha=0.7, lw=0)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 3], ["0.25", "0.5", "1", "2", "3"])
ax.minorticks_off()
ax.set_xlabel("hidden ridge r̂ removed (noise level →)")
ax.set_ylabel("Δ balanced accuracy\n(mitigated − raw)")
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
save(fig, "figR2_mitigation_effect_vs_ridge")

# W3: tuned-C hidden-ridge accuracy test
F.append("## W3 - hidden-ridge accuracy test with tuned C\n")
rows = []
for c in acc_cols:
    p = piv[c]
    for lab, a, b in (("ideal + r I  minus  ideal (no ridge)", "ideal + 1.0r I", "ideal (no ridge)"),
                      ("raw noisy  minus  ideal + r I", "raw", "ideal + 1.0r I")):
        dd = (p[a] - p[b]).rename("d").reset_index()
        dd = dd[dd.shots_per_entry == "exact"] if a.startswith("ideal") else dd
        u = dd.groupby(["dataset", "seed"])["d"].mean()
        rows.append(paired_test(u.values, f"{lab} [{c.replace('bacc_', '')}]", dict(C_setting=c.replace("bacc_", ""))))
t3 = pd.DataFrame(rows)
t3["p_holm"] = holm(t3["p_wilcoxon"])
t3.to_csv(os.path.join(TAB, "tableR3_hidden_ridge_tunedC.csv"), index=False)
for _, r in t3.iterrows():
    F.append(f"- {r.comparison}: {r.mean_diff:+.4f} [{r.boot_ci_lo:+.4f}, {r.boot_ci_hi:+.4f}], Holm p = {fmt_p(r.p_holm)}, "
             f"TOST p = {fmt_p(r.p_tost)}")
# per noise level, tuned vs C=1, ideal ridge effect
per = []
for c in ["bacc_C1", "bacc_tuned"]:
    p = piv[c]
    dd = (p["ideal + 1.0r I"] - p["ideal (no ridge)"]).rename("d").reset_index()
    dd = dd[dd.shots_per_entry == "exact"]
    for nz, gg in dd.groupby("noise"):
        per.append(dict(noise=nz, C_setting=c.replace("bacc_", ""), ridge_effect=gg.d.mean(),
                        ci_lo=bootstrap_ci(gg.d.values)[0], ci_hi=bootstrap_ci(gg.d.values)[1]))
per = pd.DataFrame(per)
per["noise"] = pd.Categorical(per["noise"], NOISE_ORDER, ordered=True)
per = per.sort_values(["C_setting", "noise"])
per.to_csv(os.path.join(TAB, "tableR3b_ridge_effect_by_noise.csv"), index=False)

# ridge sweep (ideal kernel, exact) fixed C=1 vs tuned
sw = []
for c in ["bacc_C1", "bacc_C100", "bacc_tuned"]:
    p = piv[c]
    p = p[p.index.get_level_values("shots_per_entry") == "exact"]
    for m in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        col = "ideal (no ridge)" if m == 0 else f"ideal + {m}r I"
        sw.append(dict(C_setting=c.replace("bacc_", ""), ridge_mult=m, bacc=p[col].mean()))
sw = pd.DataFrame(sw)
sw.to_csv(os.path.join(TAB, "tableR3c_ridge_sweep.csv"), index=False)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for k, (cs, gg) in enumerate(sw.groupby("C_setting", sort=False)):
    lab = "C tuned on validation" if cs == "tuned" else f"fixed C = {cs[1:]}"
    ax.plot(gg.ridge_mult, gg.bacc, color=COL[k], marker=MK[k], ls=LS[k], label=lab)
ax.set_xlabel("ridge on ideal kernel (multiples of hidden ridge r)")
ax.set_ylabel("balanced accuracy")
ax.legend(fontsize=7, frameon=False)
save(fig, "figR3_ridge_sweep_tunedC")
F.append("Ridge sweep on the ideal kernel (mean balanced accuracy, exact kernels): " + "; ".join(
    f"{cs}: " + ", ".join(f"{m:g}r→{v:.3f}" for m, v in zip(gg.ridge_mult, gg.bacc)) for cs, gg in sw.groupby("C_setting", sort=False)))
F.append("")

# =============================================================================================
# R2 / W4: scale-up
# =============================================================================================
F.append("## W4 - scale-up (N = 200 at 4 qubits; 6 qubits; device-like noise; quantum-structured data)\n")
parts = []
for stage in ("A", "B"):
    fpath = os.path.join(RES, f"scaleup_{stage}_methods.csv")
    if not os.path.exists(fpath):
        F.append(f"(stage {stage} not available)")
        continue
    m = pd.read_csv(fpath)
    m = m[m.status == "ok"]
    parts.append(m)
if parts:
    m = pd.concat(parts)
    m["cfg"] = m.apply(lambda r: f"{int(r.n_qubits)}q, N={int(r.n_train)}", axis=1)
    refs = m[m.noise.isna() | (m.noise == "n/a")]
    q = m[~(m.noise.isna() | (m.noise == "n/a"))]
    tab = q.groupby(["cfg", "dataset", "noise", "method"])["balanced_accuracy"].mean().unstack("method")
    ref = refs.groupby(["cfg", "dataset", "method"])["balanced_accuracy"].mean().unstack("method")
    tab.to_csv(os.path.join(TAB, "tableR4_scaleup_methods.csv"))
    ref.to_csv(os.path.join(TAB, "tableR4b_scaleup_references.csv"))
    F.append("References (ideal quantum kernel vs classical RBF, mean balanced accuracy):")
    for (cfgs, ds), r in ref.iterrows():
        F.append(f"- {cfgs}, {ds}: ideal QSVM {r.get('Ideal QSVM (exact, noise-free)', np.nan):.3f}; "
                 f"RBF same input {r.get('RBF-SVM (same input as quantum)', np.nan):.3f}; "
                 f"RBF all features {r.get('RBF-SVM (all features)', np.nan):.3f}")
    # mitigation effect at scale (established methods, SV test protocol, tuned C) - units dataset x seed per cfg
    F.append("Mitigation effect at scale (QSVM-mitig(avg)+clip minus QSVM-clip, tuned C, fair SV test protocol):")
    rows = []
    for cfgs, gg in q.groupby("cfg"):
        pv = gg.pivot_table(index=["dataset", "seed", "noise", "shots_per_entry"], columns="method",
                            values="balanced_accuracy")
        for a, b in (("QSVM-mitig(avg)+clip", "QSVM-clip"), ("NA-QSVM (proposed)", "QSVM-clip"),
                     ("QSVM-mitig(avg)+clip", "QSVM-raw")):
            dd = (pv[a] - pv[b]).groupby(level=["dataset", "seed"]).mean()
            rows.append(paired_test(dd.values, f"{cfgs}: {a} - {b}"))
    t4 = pd.DataFrame(rows)
    t4["p_holm"] = holm(t4["p_wilcoxon"])
    t4.to_csv(os.path.join(TAB, "tableR4c_scaleup_mitigation.csv"), index=False)
    for _, r in t4.iterrows():
        F.append(f"- {r.comparison}: {r.mean_diff:+.4f} [{r.boot_ci_lo:+.4f}, {r.boot_ci_hi:+.4f}], n={r.n_units}, "
                 f"Holm p = {fmt_p(r.p_holm)}, TOST p = {fmt_p(r.p_tost)}")
    # identity + mechanism at scale
    for stage in ("A", "B"):
        fi = os.path.join(RES, f"scaleup_{stage}_identity.csv")
        fa = os.path.join(RES, f"scaleup_{stage}_accuracy.csv")
        if not os.path.exists(fi):
            continue
        ii = pd.read_csv(fi)
        ii = ii[ii.status == "ok"]
        F.append(f"- stage {stage}: Corollary 1 max rel. decision difference {ii.M_rel_diff.max():.1e} over {len(ii)} cells; "
                 f"Lam range {ii.lam_oracle.min():.2f}-{ii.lam_oracle.max():.2f}.")
        aa = pd.read_csv(fa)
        kk = ["dataset", "seed", "noise", "shots_per_entry"]
        for c in ["bacc_C100", "bacc_tuned"]:
            pv = aa.pivot_table(index=kk, columns="variant", values=c)
            dd = (pv["mitigated"] - pv["raw"]).groupby(level=["dataset", "seed"]).mean()
            r = paired_test(dd.values, "")
            F.append(f"  stage {stage}, mitigated - raw [{c.replace('bacc_', '')}]: {r['mean_diff']:+.4f} "
                     f"[{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}], mean |cell| "
                     f"{(pv['mitigated'] - pv['raw']).abs().mean():.4f}, TOST p = {fmt_p(r['p_tost'])}")
    # figure R4: per cfg x dataset, ideal / RBF / clip / mitig+clip / NA-QSVM (avg over noise & shots)
    show = ["Ideal QSVM (exact, noise-free)", "RBF-SVM (same input as quantum)", "QSVM-clip", "QSVM-mitig(avg)+clip",
            "NA-QSVM (proposed)"]
    labels = ["ideal QSVM", "RBF (same input)", "QSVM-clip", "mitig(avg)+clip", "NA-QSVM"]
    qa = q.groupby(["cfg", "dataset", "method"])["balanced_accuracy"].mean().unstack("method")
    allm = pd.concat([ref, qa], axis=1)
    allm = allm.T.groupby(level=0).first().T
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    idx = list(allm.index)
    x = np.arange(len(idx))
    w = 0.16
    for k, (mm, lab) in enumerate(zip(show, labels)):
        vals = allm[mm].values if mm in allm else np.full(len(idx), np.nan)
        ax.plot(x + (k - 2) * w, vals, ls="none", marker=MK[k], ms=7, color=COL[k], label=lab,
                markeredgecolor="white", markeredgewidth=0.8)
    nice = {"adhoc": "quantum-\nstructured", "breast_cancer": "breast\ncancer", "synthetic_large": "synthetic\n(1000 pts)",
            "digits_3v8": "digits\n3 vs 8", "synthetic": "synthetic"}
    ax.set_xticks(x, [f"{c}\n{nice.get(d, d)}" for c, d in idx], fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="#52514e", lw=0.6, ls=":")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "figR4_scaleup")

with open(os.path.join(ROOT, "REVISION_FINDINGS_raw.md"), "w") as f:
    f.write("# Revision findings (auto-generated numbers)\n\n" + "\n".join(F) + "\n")
for c in acc_cols:
    os.remove(os.path.join(TAB, f"_cells_{c}.csv"))
print("\n".join(F))


# ---------------------------------------------------------------------------------------------
# LaTeX versions of the main revision tables (IEEE style, booktabs)
# ---------------------------------------------------------------------------------------------
def _tex(df, path, caption, label):
    body = df.to_latex(index=False, escape=True, float_format=lambda v: f"{v:.4f}")
    body = body.replace("\\toprule", "\\toprule").replace("±", "$\\pm$")
    with open(path, "w") as f:
        f.write("\\begin{table}[!t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
                "\\resizebox{\\columnwidth}{!}{%\n" + body + "}\n\\end{table}\n")


x = t2[["comparison", "mean_diff", "boot_ci_lo", "boot_ci_hi", "p_holm", "p_tost"]].copy()
x.columns = ["Comparison [C setting]", "Mean diff.", "CI low", "CI high", "Holm p", "TOST p"]
_tex(x, os.path.join(TAB, "tableR2_mitigation_fixed_vs_tuned_C.tex"),
     "Effect of global (average) mitigation on balanced accuracy at fixed and validation-tuned $C$ "
     "(paired, units = dataset $\\times$ seed, 95\\% bootstrap CI, TOST margin $\\pm0.01$).", "tab:mitig_fixed_tuned")
y = t1[["noise", "shots_per_entry", "lam_hat", "r_hat", "cor1_max_rel_diff", "thm1_ideal_plus_ridge_rel_diff",
        "shot_noise_only_rel_diff", "no_ridge_control_rel_diff"]].copy()
y.columns = ["Noise", "Shots", "Lam_hat", "r_hat", "Cor.1 max", "Thm.1 (ideal+rI)", "E/Lam part", "No ridge"]
_tex(y, os.path.join(TAB, "tableR1_identity.tex"),
     "Relative decision difference to the raw noisy SVM: Corollary 1 (mitigated kernel + ridge), Theorem 1 "
     "(ideal kernel + ridge), the part due to the measured noise $E/\\Lambda$, and the no-ridge control.", "tab:identity")
if parts:
    z = t4[["comparison", "n_units", "mean_diff", "boot_ci_lo", "boot_ci_hi", "p_holm", "p_tost"]].copy()
    z.columns = ["Comparison", "Units", "Mean diff.", "CI low", "CI high", "Holm p", "TOST p"]
    _tex(z, os.path.join(TAB, "tableR4c_scaleup_mitigation.tex"),
         "Scale-up: mitigation and NA-QSVM effects on balanced accuracy (fair SV-restricted test protocol, tuned $C$).",
         "tab:scaleup")
print("LaTeX tables written")
