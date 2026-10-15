"""LaTeX tables and figures written to the results directory (\\input by the paper)."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import accuracy_summary, ate_table

NAMES = {
    "direct": "Direct (no CoT)", "cot": "CoT", "sc": "Self-consistency", "cad": "CAD", "adacad": "AdaCAD",
    "cad_chain": "CAD-chain", "hgcd": "\\textbf{HGCD}", "hgcd_log": "HGCD (log-space)",
    "hgcd_naive": "HGCD (naive mask)", "hgcd_uniform": "HGCD (no graph)",
    "hgcd_placebo": "HGCD (placebo)", "hgcd_ada": "HGCD (adaptive $\\alpha$)",
    "l1+cot": "L1 + CoT", "l1+hgcd": "\\textbf{L1 + HGCD}", "free": "Free generation",
}
ORDER = ["direct", "cot", "sc", "cad", "adacad", "cad_chain", "hgcd_naive", "hgcd_uniform",
         "hgcd_log", "hgcd_ada", "hgcd_placebo", "hgcd", "l1+cot", "l1+hgcd", "free"]


def _fmt(m, lo, hi, pct=True):
    if m is None or np.isnan(m):
        return "--"
    s = 100 if pct else 1
    return f"{m * s:.1f}\\,{{\\scriptsize[{lo * s:.1f},{hi * s:.1f}]}}"


def _write(path: str, s: str):
    with open(path, "w") as f:
        f.write(s)


def main_table(df: pd.DataFrame, out: str, id_max_k: int = 4, col: str = "correct",
               by_split: bool = True) -> pd.DataFrame:
    d = df[df.cond == "orig"].copy()
    d["split"] = np.where(d.k <= id_max_k, "ID", "OOD")
    s2 = accuracy_summary(d, ["method"], col).assign(split="All")
    s = pd.concat([accuracy_summary(d, ["method", "split"], col), s2]) if by_split else s2
    splits = [x for x in ["ID", "OOD", "All"] if x in set(s.split)]
    lines = ["\\begin{tabular}{l" + "c" * len(splits) + "}", "\\toprule",
             "Method & " + " & ".join(f"{x} ($n$={int(s[s.split == x].n.max())})" for x in splits)
             + " \\\\", "\\midrule"]
    for m in [m for m in ORDER if m in set(s.method)]:
        cells = []
        for x in splits:
            r = s[(s.method == m) & (s.split == x)]
            cells.append(_fmt(*r[["mean", "lo", "hi"]].values[0]) if len(r) else "--")
        lines.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")
    return s


def ate_latex(df: pd.DataFrame, out: str, col: str = "correct") -> pd.DataFrame:
    a = ate_table(df, col=col)
    conds = sorted(set(a.cond))
    lines = ["\\begin{tabular}{l" + "c" * len(conds) + "}", "\\toprule",
             "Method & " + " & ".join(c.replace("_", "\\_") for c in conds) + " \\\\",
             "\\midrule"]
    for m in [m for m in ORDER if m in set(a.method)]:
        cells = []
        for c in conds:
            r = a[(a.method == m) & (a.cond == c)]
            cells.append(_fmt(*r[["ate", "lo", "hi"]].values[0]) if len(r) else "--")
        lines.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")
    return a


def estimator_latex(aucs: pd.DataFrame, pos: pd.DataFrame, out_auc: str, out_pos: str):
    en = {"d_total": "$\\Delta^{\\text{total}}$ (graph-aware mask)",
          "d_naive": "$\\Delta^{\\text{naive}}$ (mask hop only)",
          "d_placebo": "$\\Delta^{\\text{placebo}}$ (content-preserving)",
          "d_maskmark": "$\\Delta^{\\text{marker}}$ (mask token, content kept)",
          "pns_rollout": "PNS (rollout)", "drg_relevant": "DRG path indicator"}
    ds = sorted(set(aucs.dataset))
    lines = ["\\begin{tabular}{l" + "cc" * len(ds) + "}", "\\toprule",
             "Estimator & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in ds) + " \\\\",
             " & " + " & ".join(["all", "PS$=$1"] * len(ds)) + " \\\\", "\\midrule"]
    for e in ["d_naive", "d_total", "pns_rollout", "drg_relevant", "d_placebo", "d_maskmark"]:
        if e not in set(aucs.estimator):
            continue
        cells = []
        for d in ds:
            r = aucs[(aucs.dataset == d) & (aucs.estimator == e)]
            cells += ([f"{r.auc_all.values[0]:.3f}", f"{r.auc_ps1.values[0]:.3f}"]
                      if len(r) else ["--", "--"])
        lines.append(f"{en[e]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out_auc, "\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Dataset & Hop type & $\\Delta^{\\text{total}}$ & $\\Delta^{\\text{naive}}$ & "
             "$\\Delta^{\\text{placebo}}$ & $\\Delta^{\\text{marker}}$ \\\\", "\\midrule"]
    for _, r in pos.iterrows():
        lines.append(f"{r.dataset} & {r.position} & {r.d_total:.3f} & {r.d_naive:.3f} & "
                     f"{r.d_placebo:.3f} & {r.d_maskmark:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out_pos, "\n".join(lines) + "\n")


def figures(df: pd.DataFrame, sweep: pd.DataFrame, outdir: str, prefix: str = "clutrr"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.cond == "orig"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m in ["cot", "cad_chain", "hgcd_naive", "hgcd"]:
        g = d[d.method == m].groupby("k").correct.mean()
        if len(g):
            ax.plot(g.index, g.values * 100, marker="o", label=NAMES[m].replace("\\textbf{", "").replace("}", ""))
    ax.set_xlabel("number of hops $k$")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_acc_vs_k.pdf"))
    plt.close(fig)
    if sweep is not None and len(sweep):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for m, g in sweep.groupby("method"):
            ax.plot(g.alpha, g.acc * 100, marker=".", label=NAMES.get(m, m).replace("\\textbf{", "").replace("}", ""))
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_alpha_sweep.pdf"))
        plt.close(fig)
