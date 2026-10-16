"""LaTeX tables and figures written to the results directory (\\input by the paper)."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import accuracy_summary, ate_table

NAMES = {
    "direct": "Direct (no CoT)", "cot": "CoT", "sc": "Self-consistency", "cad": "CAD", "adacad": "AdaCAD",
    "cad_chain": "CAD-chain", "calib": "Calibration (content-free)", "poe": "Direct $\\times$ CoT (PoE)", "hgcd": "\\textbf{HGCD}", "hgcd_log": "HGCD (log-space)",
    "hgcd_naive": "HGCD (naive mask)", "hgcd_uniform": "HGCD (no graph)",
    "hgcd_placebo": "HGCD (placebo)", "hgcd_ada": "HGCD (adaptive $\\alpha$)",
    "l1+cot": "L1 + CoT", "l1+hgcd": "\\textbf{L1 + HGCD}", "free": "Free generation",
}
ORDER = ["direct", "cot", "sc", "cad", "adacad", "cad_chain", "calib", "poe", "hgcd_naive", "hgcd_uniform",
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


EST_NAMES = {
    "d_naive": "Mask hop only (direct)",
    "d_total": "Mask hop + DRG descendants (total)",
    "d_cf_direct": "Replace result, keep downstream (direct)",
    "d_cf_total": "Replace result, propagate (total)",
    "pns_rollout": "Corrupt + resample, resampled baseline (ours)",
    "pns_goldbase": "Corrupt + resample, gold-chain baseline (PNS as in Yu et al.)",
    "d_total_rawlogp": "Mask hop + descendants, raw log-prob (no candidate set)",
    "drg_relevant": "DRG path indicator (structure only)",
    "d_placebo": "Placebo rewrite (control)",
    "d_maskmark": "Mask token, content kept (control)",
    "d_maskmark_tree": "Mask tokens on hop + descendants, content kept (control)",
}
EST_SHORT = {"d_total": "mask-tot", "d_naive": "mask-dir", "d_cf_total": "cf-tot",
             "d_cf_direct": "cf-dir", "pns_rollout": "rollout", "pns_goldbase": "rollout-gold", "d_total_rawlogp": "tot-rawlp",
             "d_maskmark_tree": "marker-tree", "d_placebo": "placebo",
             "d_maskmark": "marker"}


def estimator_latex(aucs: pd.DataFrame, pos: pd.DataFrame, out_auc: str, out_pos: str):
    ds = sorted(set(aucs.dataset))
    lines = ["\\begin{tabular}{l" + "cc" * len(ds) + "}", "\\toprule",
             "Estimator & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d.replace('_', chr(92) + '_')}}}"
                                         for d in ds) + " \\\\",
             " & " + " & ".join(["all", "PS$=$1"] * len(ds)) + " \\\\", "\\midrule"]
    for e in EST_NAMES:
        if e not in set(aucs.estimator):
            continue
        cells = []
        for d in ds:
            r = aucs[(aucs.dataset == d) & (aucs.estimator == e)]
            ok = len(r) and not np.isnan(r.auc_all.values[0])
            cells += ([f"{r.auc_all.values[0]:.3f}", f"{r.auc_ps1.values[0]:.3f}"] if ok else ["--", "--"])
        lines.append(f"{EST_NAMES[e]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out_auc, "\n".join(lines) + "\n")
    cols = [c for c in EST_SHORT if c in pos.columns and not pos[c].isna().all()]
    lines = ["\\begin{tabular}{ll" + "c" * len(cols) + "}", "\\toprule",
             "Dataset & Hop type & " + " & ".join(EST_SHORT[c] for c in cols) + " \\\\", "\\midrule"]
    for _, r in pos.iterrows():
        lines.append(f"{r.dataset.replace('_', chr(92) + '_')} & {r.position} & "
                     + " & ".join(f"{r[c]:.3f}" for c in cols) + " \\\\")
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


def readout_latex(df: pd.DataFrame, out: str, methods=("cot", "sc", "cad_chain", "hgcd_naive",
                                                      "hgcd")) -> pd.DataFrame:
    """Accuracy split by whether the greedy chain reaches every gold-path entity (hop_fid = 1).

    HGCD re-weights answers toward those the chain causally supports, so it can repair *readout*
    errors (correct chain, wrong answer) but not wrong chains.
    """
    d = df[(df.cond == "orig") & df.method.isin(methods)].copy()
    d["chain"] = np.where(d.hop_fid >= 1.0, "gold-path chain", "other chain")
    s = accuracy_summary(d, ["method", "chain"])
    groups = ["gold-path chain", "other chain"]
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Method & " + " & ".join(f"{g} ($n$={int(s[s.chain == g].n.max()) if (s.chain == g).any() else 0})"
                                      for g in groups) + " \\\\", "\\midrule"]
    for m in [m for m in ORDER if m in set(s.method)]:
        cells = []
        for g in groups:
            r = s[(s.method == m) & (s.chain == g)]
            cells.append(_fmt(*r[["mean", "lo", "hi"]].values[0]) if len(r) else "--")
        lines.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")
    return s


def mediation_latex(med: pd.DataFrame, out: str):
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Intervention & Method & TE & NDE (chain fixed) & NIE (via chain) \\\\", "\\midrule"]
    for cond in sorted(set(med.cond)):
        for m in [m for m in ORDER if m in set(med[med.cond == cond].method)]:
            r = med[(med.cond == cond) & (med.method == m)].iloc[0]
            lines.append(f"{cond.replace('_', chr(92) + '_')} & {NAMES.get(m, m)} & "
                         f"{_fmt(r.te, r.te_lo, r.te_hi)} & {_fmt(r.nde, r.nde_lo, r.nde_hi)} & "
                         f"{_fmt(r.nie, r.nie_lo, r.nie_hi)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, "\n".join(lines) + "\n")


def reliance_table(run_dirs: Dict[str, str], out_tex: str, out_pdf: str = None) -> pd.DataFrame:
    """Aggregate estimator runs across models: reliance (mean total effect of gold hops) and
    AUC of each estimator, per dataset and model."""
    from .metrics import estimator_table
    rows = []
    for label, d in run_dirs.items():
        p = os.path.join(d, "estimators.csv")
        if not os.path.exists(p):
            continue
        est = pd.read_csv(p)
        aucs, pos = estimator_table(est.to_dict("records"))
        for ds, g in est.groupby("dataset"):
            gold = g[g.label == 1]
            row = dict(model=label, dataset=ds, reliance_final=gold[gold.is_final == 1].d_total.mean(),
                       reliance_nonfinal=gold[gold.is_final == 0].d_total.mean(),
                       ps1=g.drop_duplicates("uid").full_correct.mean())
            for _, a in aucs[aucs.dataset == ds].iterrows():
                row["auc_" + a.estimator] = a.auc_all
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [c for c in ["auc_d_naive", "auc_d_total", "auc_d_cf_direct", "auc_d_cf_total",
                        "auc_pns_rollout", "auc_pns_goldbase", "auc_d_placebo"] if c in df]
    head = ["Model", "Data", "PS", "Rel.(final)", "Rel.(non-final)"] + \
           [EST_SHORT.get(c[4:], c[4:]) for c in cols]
    lines = ["\\begin{tabular}{ll" + "c" * (3 + len(cols)) + "}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = [r.model, r.dataset, f"{r.ps1:.2f}", f"{r.reliance_final:.3f}",
                 f"{r.reliance_nonfinal:.3f}"] + \
                ["--" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out_tex, "\n".join(lines) + "\n")
    if out_pdf:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for ds, g in df.groupby("dataset"):
            ax.plot(g.model, g.reliance_final, marker="o", label=f"{ds} final hop")
            ax.plot(g.model, g.reliance_nonfinal, marker="s", ls="--", label=f"{ds} non-final")
        ax.set_ylabel("mean total effect on p(gold)")
        ax.legend(fontsize=7, frameon=False)
        plt.xticks(rotation=30, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)
    return df


def form_content_latex(fc: pd.DataFrame, out: str):
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Dataset & Form (masked chain $-$ no chain) & Content (real $-$ masked chain) & $n$ \\\\",
             "\\midrule"]
    for _, r in fc.iterrows():
        lines.append(f"{r.dataset.replace('_', chr(92) + '_')} & {_fmt(r.form, r.form_lo, r.form_hi)} & "
                     f"{_fmt(r.content, r.content_lo, r.content_hi)} & {int(r.n)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")
