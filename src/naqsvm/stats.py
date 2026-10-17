"""Confidence intervals and paired statistical tests.

Statistical units. Rows sharing a (dataset, seed) share the data split, so rows from
different noise levels / shot budgets of the same split are NOT independent. The
primary tests therefore average each method over the grid cells inside a split and use
(dataset, seed) as the independent unit (conservative). Secondary per-noise-level tests
average over shot budgets only. Multiple comparisons are Holm-corrected.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats as st


def t_ci(x, conf: float = 0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = st.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def summarize(df: pd.DataFrame, by: List[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n"] = len(g)
        for m in metrics:
            mean, lo, hi = t_ci(g[m])
            r[f"{m}_mean"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = mean, lo, hi
            r[f"{m}_std"] = float(np.nanstd(g[m], ddof=1)) if len(g) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    pv = p[ok][idx]
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv)
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def bootstrap_ci(x, n_boot: int = 10000, conf: float = 0.95, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return tuple(np.quantile(means, [0.5 - conf / 2, 0.5 + conf / 2]))


def rank_biserial(d) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    r = st.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_p(d) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    try:
        return float(st.wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def paired_comparison(df: pd.DataFrame, target: str, others: List[str], metric: str,
                      unit: List[str], average_over: List[str] | None = None) -> pd.DataFrame:
    """target vs each method in `others`; differences = target - other."""
    sub = df[df["method"].isin([target] + others) & df[metric].notna()]
    piv = sub.pivot_table(index=unit + (average_over or []), columns="method", values=metric)
    if average_over:
        piv = piv.groupby(level=list(range(len(unit)))).mean()
    rows = []
    for o in others:
        if o not in piv or target not in piv:
            continue
        pair = piv[[target, o]].dropna()
        d = (pair[target] - pair[o]).values
        lo, hi = bootstrap_ci(d)
        rows.append(dict(comparison=f"{target} vs {o}", metric=metric, n_units=len(d),
                         mean_diff=float(np.mean(d)) if len(d) else np.nan,
                         median_diff=float(np.median(d)) if len(d) else np.nan,
                         boot_ci_lo=lo, boot_ci_hi=hi,
                         wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                         losses=int((d < -1e-12).sum()),
                         rank_biserial=rank_biserial(d), p_wilcoxon=wilcoxon_p(d)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"])
        out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def friedman(df: pd.DataFrame, methods: List[str], metric: str, block: List[str]) -> dict:
    piv = df[df["method"].isin(methods)].pivot_table(index=block, columns="method", values=metric)
    piv = piv[methods].dropna()
    if len(piv) < 2:
        return dict(n_blocks=len(piv), statistic=np.nan, p_value=np.nan, mean_ranks={})
    s, p = st.friedmanchisquare(*[piv[m].values for m in methods])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    return dict(n_blocks=len(piv), statistic=float(s), p_value=float(p), mean_ranks=ranks.to_dict())


def tost_paired(d, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided paired t-tests (TOST) for equivalence of a mean paired difference
    within (-margin, +margin). Equivalence is concluded when p_tost < alpha."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(n=n, mean=np.nan, p_lower=np.nan, p_upper=np.nan, p_tost=np.nan, equivalent=False)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        p_lo = 0.0 if m > -margin else 1.0
        p_hi = 0.0 if m < margin else 1.0
    else:
        p_lo = 1 - st.t.cdf((m + margin) / se, n - 1)     # H0: mean <= -margin
        p_hi = st.t.cdf((m - margin) / se, n - 1)         # H0: mean >= +margin
    p = max(p_lo, p_hi)
    ci = st.t.interval(1 - 2 * alpha, n - 1, loc=m, scale=se) if se > 0 else (m, m)
    return dict(n=n, mean=m, ci90_lo=ci[0], ci90_hi=ci[1], p_lower=p_lo, p_upper=p_hi, p_tost=p,
                equivalent=bool(p < alpha))
