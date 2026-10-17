"""Figures (matplotlib). Colours follow a fixed categorical order; every series also has
its own marker so identity is never carried by colour alone."""
from __future__ import annotations

import os

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "QSVM-raw": ("#2a78d6", "o"),
    "QSVM-clip": ("#eb6834", "s"),
    "AQKA-Bernoulli+clip": ("#1baf7a", "h"),
    "QSVM-mitig(avg)+clip": ("#eda100", "D"),
    "AQKA-SVM+clip": ("#e87ba4", "v"),
    "MarginNeyman+clip (Miroszewski-style approx.)": ("#008300", "P"),
    "NA-QSVM (proposed)": ("#6250d6", "*"),
    "Variant: shot-saving (SV test, S/entry)": ("#e34948", "X"),
    "QSVM-shift": ("#8a8984", "^"),
    "QSVM-flip": ("#8a8984", "<"),
    "QSVM-mitig(split)+clip": ("#8a8984", ">"),
}
REF_STYLE = {"Ideal QSVM (exact, noise-free)": ("#0b0b0b", "--"),
             "RBF-SVM (4-D PCA input)": ("#52514e", ":")}
MAIN_METHODS = ["QSVM-raw", "QSVM-clip", "QSVM-mitig(avg)+clip", "AQKA-Bernoulli+clip",
                "AQKA-SVM+clip", "MarginNeyman+clip (Miroszewski-style approx.)", "NA-QSVM (proposed)"]


def _style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8, "lines.linewidth": 1.6})


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        if path.endswith(".png"):
            fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")


def metric_vs_shots(summary: pd.DataFrame, refs: pd.DataFrame, dataset: str, metric: str,
                    noises, methods=MAIN_METHODS, path=None, ylabel=None):
    """summary: output of stats.summarize by [dataset, noise, shots_per_entry, method]."""
    _style()
    fig, axes = plt.subplots(1, len(noises), figsize=(3.0 * len(noises), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, nz in zip(axes, noises):
        g = summary[(summary.dataset == dataset) & (summary.noise == nz)]
        for m in methods:
            h = g[g.method == m].sort_values("shots_per_entry")
            if h.empty:
                continue
            c, mk = STYLE.get(m, ("#888888", "o"))
            lw = 2.4 if "proposed" in m else 1.4
            ax.plot(h.shots_per_entry, h[f"{metric}_mean"], color=c, marker=mk, ms=5, lw=lw, label=m)
            lo_b, hi_b = h[f"{metric}_ci_lo"], h[f"{metric}_ci_hi"]
            if metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                lo_b, hi_b = lo_b.clip(0, 1), hi_b.clip(0, 1)
            ax.fill_between(h.shots_per_entry, lo_b, hi_b, color=c, alpha=0.10, lw=0)
        if refs is not None:
            for rm, (c, ls) in REF_STYLE.items():
                r = refs[(refs.dataset == dataset) & (refs.method == rm)]
                if len(r):
                    ax.axhline(r[metric].mean(), color=c, ls=ls, lw=1.1, label=rm)
        ax.set_xscale("log", base=2)
        ax.set_title(f"noise: {nz}")
        ax.set_xlabel("shots per kernel entry (S)")
    axes[0].set_ylabel(ylabel or metric.replace("_", " "))
    if metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        lo = np.nanmin(summary[(summary.dataset == dataset) & summary.method.isin(methods)][f"{metric}_mean"])
        axes[0].set_ylim(max(0.0, lo - 0.15), 1.0)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{dataset}: {ylabel or metric} (mean and 95% CI over seeds)")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(hs, ls, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path)
    return fig


def kernel_heatmaps(mats, titles, path=None, diff_to=None):
    _style()
    k = len(mats)
    rows = 2 if diff_to is not None else 1
    fig, axes = plt.subplots(rows, k, figsize=(2.6 * k, 2.6 * rows), squeeze=False)
    for j, (M, t) in enumerate(zip(mats, titles)):
        im = axes[0, j].imshow(M, cmap="viridis", vmin=0, vmax=1)
        axes[0, j].set_title(t, fontsize=8)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[0, j].grid(False)
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, label="kernel value")
    if diff_to is not None:
        off = ~np.eye(len(diff_to), dtype=bool)
        lim = max(np.abs((M - diff_to)[off]).max() for M in mats) or 1.0
        for j, (M, t) in enumerate(zip(mats, titles)):
            D = (M - diff_to).astype(float)
            np.fill_diagonal(D, np.nan)             # diagonal (incl. any ridge) not shown
            im2 = axes[1, j].imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[1, j].set_title("minus ideal (off-diagonal)", fontsize=8)
            axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
            axes[1, j].grid(False)
        fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, label="error")
    _save(fig, path)
    return fig


def forest(paired: pd.DataFrame, title: str, path=None, xlabel="mean difference (proposed - other)"):
    _style()
    p = paired.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(p) + 1.2))
    y = np.arange(len(p))
    ax.errorbar(p.mean_diff, y, xerr=[p.mean_diff - p.boot_ci_lo, p.boot_ci_hi - p.mean_diff],
                fmt="o", color="#6250d6", ecolor="#6250d6", capsize=3, ms=5)
    ax.axvline(0, color="#52514e", lw=1)
    lab = [f"{c.split(' vs ')[-1]}  (p_Holm={ph:.3g})" for c, ph in zip(p.comparison, p.p_holm)]
    ax.set_yticks(y, lab)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:+.3f}"))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def lines_by_noise(summary: pd.DataFrame, metric: str, methods, noises, path=None, ylabel=None,
                   logy=False):
    """Averaged over datasets (summary grouped by noise, shots, method)."""
    _style()
    fig, axes = plt.subplots(1, len(noises), figsize=(3.0 * len(noises), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, nz in zip(axes, noises):
        g = summary[summary.noise == nz]
        for m in methods:
            h = g[g.method == m].sort_values("shots_per_entry")
            if h.empty:
                continue
            c, mk = STYLE.get(m, ("#888888", "o"))
            ax.plot(h.shots_per_entry, h[f"{metric}_mean"], color=c, marker=mk, ms=5,
                    lw=2.4 if "proposed" in m else 1.4, label=m)
            ax.fill_between(h.shots_per_entry, h[f"{metric}_ci_lo"], h[f"{metric}_ci_hi"], color=c, alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"noise: {nz}")
        ax.set_xlabel("shots per kernel entry (S)")
    axes[0].set_ylabel(ylabel or metric)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.legend(hs, ls, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path)
    return fig


def scatter_lambda(df: pd.DataFrame, path=None):
    _style()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for i, (nz, g) in enumerate(df.groupby("noise")):
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
        ax.scatter(g.lambda_true, g.lambda_hat, s=10, alpha=0.6, color=cols[i % 5], label=nz)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1)
    ax.set_xlabel("exact mean diagonal survival  Λ")
    ax.set_ylabel("estimated Λ̂ (finite shots)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def acc_vs_total_shots(df: pd.DataFrame, methods, metric="balanced_accuracy", path=None):
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in methods:
        g = df[df.method == m].groupby("shots_per_entry")[[metric, "total_shots"]].mean().sort_values("total_shots")
        c, mk = STYLE.get(m, ("#888888", "o"))
        ax.plot(g.total_shots, g[metric], marker=mk, color=c, label=m, lw=2.4 if "proposed" in m else 1.4)
    ax.set_xscale("log")
    ax.set_xlabel("total measurement shots (train + val + test)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path)
    return fig


def hidden_ridge_plot(hr: pd.DataFrame, path=None):
    """Relative decision-function difference to the raw noisy SVM: hidden-ridge model (B)
    vs no-ridge control (Z), per noise setting and shot level (mean and 95% CI)."""
    _style()
    from scipy import stats as st
    shots = list(dict.fromkeys(hr.shots_per_entry.astype(str)))
    noises = list(dict.fromkeys(hr.noise))
    fig, axes = plt.subplots(1, len(shots), figsize=(3.6 * len(shots), 3.4), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, S in zip(axes, shots):
        g = hr[hr.shots_per_entry.astype(str) == S]
        y = np.arange(len(noises))
        for off, col, lab, key in [(-0.15, "#6250d6", "ideal kernel + hidden ridge (B)", "B_decision_rel_diff"),
                                   (0.15, "#8a8984", "ideal kernel, no ridge (control Z)", "Z_decision_rel_diff")]:
            m, h = [], []
            for nz in noises:
                x = g[g.noise == nz][key].dropna().values
                m.append(x.mean() if len(x) else np.nan)
                h.append(st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
            ax.errorbar(m, y + off, xerr=h, fmt="o", color=col, ms=5, capsize=2, label=lab)
        ax.set_yticks(y, noises)
        ax.set_title(f"shots per entry: {S}")
        ax.set_xlabel("rel. difference of decision values\nto raw noisy SVM (lower = same classifier)")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    hs, ls = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(hs, ls, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path)
    return fig
