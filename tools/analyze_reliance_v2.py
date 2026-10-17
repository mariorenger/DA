"""Aggregate the reliance runs (one estimators.csv + summary.json per model) into the scale table,
an AUC table and the wrong-copy table, with question-clustered bootstrap CIs.

usage: python tools/analyze_reliance_v2.py results/reliance_v2
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ORDER = ["qwen-0.5b", "qwen-1.5b", "smollm2-1.7b", "qwen-3b", "qwen-7b"]
SHORT = {"qwen-0.5b": "Q-0.5B", "qwen-1.5b": "Q-1.5B", "smollm2-1.7b": "Smol-1.7B",
         "qwen-3b": "Q-3B", "qwen-7b": "Q-7B"}


def position(df):
    return np.where(df.label == 0, "redundant", np.where(df.is_final == 1, "gold-final", "gold-nonfinal"))


def cluster_boot(df, col, n_boot=2000, seed=0):
    """Mean of `col` with a 95% CI that resamples questions (uid), not hops."""
    d = df[["uid", col]].dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    sums = d.groupby("uid")[col].agg(["sum", "count"])
    s, c = sums["sum"].values, sums["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def load(root):
    out = {}
    for m in ORDER:
        p = os.path.join(root, m)
        if not os.path.isdir(p):
            continue
        df = pd.read_csv(os.path.join(p, "estimators.csv"))
        df["position"] = position(df)
        out[m] = (df, json.load(open(os.path.join(p, "summary.json"))))
    return out


ROWS = [
    ("CLUTRR, non-final hop: total", "clutrr", "gold-nonfinal", "d_total"),
    ("CLUTRR, non-final hop: placebo", "clutrr", "gold-nonfinal", "d_placebo"),
    ("CLUTRR, final hop: total", "clutrr", "gold-final", "d_total"),
    ("CLUTRR, final hop: placebo", "clutrr", "gold-final", "d_placebo"),
    ("CLUTRR+copy, non-final: total", "clutrr_copy", "gold-nonfinal", "d_total"),
    ("CLUTRR+copy, non-final: direct", "clutrr_copy", "gold-nonfinal", "d_naive"),
    ("CLUTRR+copy, non-final: marker-tree", "clutrr_copy", "gold-nonfinal", "d_maskmark_tree"),
    ("MuSiQue, non-final: total", "musique", "gold-nonfinal", "d_total"),
    ("MuSiQue, non-final: direct", "musique", "gold-nonfinal", "d_naive"),
    ("MuSiQue, non-final: marker-tree", "musique", "gold-nonfinal", "d_maskmark_tree"),
    ("MuSiQue, redundant: total", "musique", "redundant", "d_total"),
    ("Rollout, gold baseline: necessary", "musique", "gold-nonfinal", "pns_goldbase"),
    ("Rollout, gold baseline: redundant", "musique", "redundant", "pns_goldbase"),
    ("Rollout, resampled baseline: necessary", "musique", "gold-nonfinal", "pns_rollout"),
    ("Rollout, resampled baseline: redundant", "musique", "redundant", "pns_rollout"),
]

AUC_EST = [("d_total", "total"), ("d_naive", "direct (naive)"), ("d_cf_total", "cf, propagated"),
           ("d_total_rawlogp", "total, raw log-prob"), ("pns_goldbase", "rollout, gold base"),
           ("pns_rollout", "rollout, resampled"), ("d_maskmark_tree", "marker-tree (control)"),
           ("d_placebo", "placebo (control)")]


def main(root):
    data = load(root)
    models = list(data)
    recs = []
    for m, (df, _) in data.items():
        for label, ds, pos, col in ROWS:
            g = df[(df.dataset == ds) & (df.position == pos)]
            mu, lo, hi = cluster_boot(g, col) if col in g else (np.nan,) * 3
            recs.append(dict(model=m, row=label, dataset=ds, position=pos, est=col,
                             mean=mu, lo=lo, hi=hi, n_q=g.uid.nunique()))
    pos = pd.DataFrame(recs)
    pos.to_csv(os.path.join(root, "positions_ci.csv"), index=False)

    def cell(m, label, ci=False):
        r = pos[(pos.model == m) & (pos.row == label)]
        if r.empty or r["mean"].isna().all():
            return "--"
        r = r.iloc[0]
        s = f"{100 * r['mean']:.1f}"
        return s + f"\\,{{\\scriptsize[{100 * r.lo:.1f},{100 * r.hi:.1f}]}}" if ci else s

    hdr = "Quantity ($\\Delta p(\\text{gold})$, points) & " + " & ".join(SHORT[m] for m in models) + " \\\\"
    lines = ["\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule", hdr, "\\midrule"]
    for i, (label, *_rest) in enumerate(ROWS):
        if i in (4, 7, 11):
            lines.append("\\midrule")
        lines.append(f"{label} & " + " & ".join(cell(m, label) for m in models) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(root, "tab_scale.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # AUC (all hops; necessary vs redundant)
    arecs = []
    for m, (_, s) in data.items():
        for a in s.get("estimator_auc", []):
            arecs.append(dict(model=m, **a))
    auc = pd.DataFrame(arecs)
    auc.to_csv(os.path.join(root, "auc.csv"), index=False)
    lines = ["\\begin{tabular}{ll" + "c" * len(models) + "}", "\\toprule",
             "Dataset & Estimator & " + " & ".join(SHORT[m] for m in models) + " \\\\", "\\midrule"]
    for j, (ds, dname) in enumerate([("musique", "MuSiQue"), ("clutrr_copy", "CLUTRR+copy"), ("clutrr", "CLUTRR")]):
        if j:
            lines.append("\\midrule")
        for est, ename in AUC_EST:
            cells = []
            for m in models:
                r = auc[(auc.model == m) & (auc.dataset == ds) & (auc.estimator == est)]
                cells.append("--" if r.empty or pd.isna(r.auc_all.iloc[0]) else f"{r.auc_all.iloc[0]:.3f}")
            if all(c == "--" for c in cells):
                continue
            lines.append(f"{dname} & {ename} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(root, "tab_auc_scale.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # wrong-answer copy: follow rate overall and when the plain chain is solved (PS = 1)
    crecs = []
    for m, (df, s) in data.items():
        cw = df[(df.dataset == "clutrr_copywrong")].drop_duplicates("uid")
        plain = df[df.dataset == "clutrr"].drop_duplicates("uid").set_index("uid").full_correct
        cw = cw.assign(solv=cw.uid.map(plain))
        rates = {}
        for key, sub in [("all", cw), ("solv", cw[cw.solv == 1]), ("unsolv", cw[cw.solv == 0])]:
            v = sub.follows_wrong.dropna().values
            if len(v):
                rng = np.random.default_rng(0)
                b = rng.choice(v, size=(2000, len(v))).mean(1)
                rates[key] = (v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(v))
            else:
                rates[key] = (np.nan, np.nan, np.nan, 0)
        sw = s.get("copy_wrong", {})
        crecs.append(dict(model=m, **{f"{k}_{n}": rates[k][i] for k in rates
                                      for i, n in enumerate(["rate", "lo", "hi", "n"])},
                          summary_follow=sw.get("follow_rate"),
                          summary_follow_solv=sw.get("follow_rate_when_solvable"),
                          ps_plain=plain.mean()))
    cw = pd.DataFrame(crecs)
    cw.to_csv(os.path.join(root, "copy_wrong.csv"), index=False)
    print(cw.to_string())
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Model & PS of plain chain & follows wrong copy (all) & follows wrong copy $\\mid$ PS$=1$ \\\\",
             "\\midrule"]
    for _, r in cw.iterrows():
        f = lambda k: f"{100 * r[k + '_rate']:.1f}\\,{{\\scriptsize[{100 * r[k + '_lo']:.1f},{100 * r[k + '_hi']:.1f}]}} ($n$={int(r[k + '_n'])})"
        lines.append(f"{SHORT[r.model]} & {100 * r.ps_plain:.1f} & {f('all')} & {f('solv')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(root, "tab_copy_wrong.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
