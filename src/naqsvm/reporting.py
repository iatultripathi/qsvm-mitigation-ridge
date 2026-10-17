"""Environment capture, table writers, README generation and ZIP packaging."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
import zipfile

import numpy as np
import pandas as pd

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "qiskit", "qiskit-aer"]


def environment_info(extra: dict | None = None) -> dict:
    from importlib import metadata
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except Exception:
            vers[p] = None
    info = dict(timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                python=sys.version.replace("\n", " "), platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), packages=vers,
                in_colab="google.colab" in sys.modules)
    if extra:
        info.update(extra)
    return info


def write_requirements(path: str, info: dict) -> None:
    lines = ["# Versions found in the environment that produced these results",
             "# (qiskit / qiskit-aer: None means the NumPy reference simulator was used)"]
    for p, v in info["packages"].items():
        lines.append(f"{p}=={v}" if v else f"# {p} (not installed)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def df_to_latex(df: pd.DataFrame, caption: str, label: str, nd: int = 3) -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    cols = list(df.columns)
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
           r"\small", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
           " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(_fmt(r[c], nd)) for c in cols) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def make_zip(src_dir: str, zip_path: str, extra_files=()) -> list:
    written = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        base = os.path.basename(os.path.normpath(src_dir))
        for root, _, files in os.walk(src_dir):
            if "__pycache__" in root or os.sep + "cache" in root:
                continue
            for fn in sorted(files):
                full = os.path.join(root, fn)
                arc = os.path.join(base, os.path.relpath(full, src_dir))
                z.write(full, arc)
                written.append(arc)
        for f in extra_files:
            if f and os.path.exists(f):
                arc = os.path.join(base, os.path.basename(f))
                z.write(f, arc)
                written.append(arc)
    return written


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
