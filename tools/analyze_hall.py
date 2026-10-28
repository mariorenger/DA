"""Round 13 analysis (REVIEW.md 'Round 13'): pre-registered endpoints H-K, H-L, H-M, H-N. Paired bootstrap over questions.

    python tools/analyze_hall.py  ->  results/hall/analysis.csv
Each run folder results/hall/<run>/{a,b,...}/decisions.csv holds arms evaluated on the same questions.
For Qwen3-4B the stage-1 model of the DPO arm (`sft_before_dpo`, the s_abs_cf recipe) stands in for s_abs_cf if that arm was not run.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hall")


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load_run(run_dirs):
    ds = [pd.read_csv(f) for d in run_dirs for f in sorted(glob.glob(os.path.join(R, d, "*", "decisions.csv")))]   # later folders win
    return pd.concat(ds, ignore_index=True) if ds else None


def vec(d, arm, cond, col):
    g = d[(d.arm == arm) & (d.cond == cond)].drop_duplicates("uid", keep="last").set_index("uid")
    return g[col]


def contrast(rows, run, endpoint, d, a, b, cond, col, label, scale=100.0):
    x, y = vec(d, a, cond, col), vec(d, b, cond, col)
    ids = x.index.intersection(y.index)
    if len(ids) == 0:
        return
    m, lo, hi = boot((x[ids] - y[ids]).values)
    rows.append(dict(run=run, endpoint=endpoint, contrast=f"{label}: {a} - {b} [{cond}]", diff=scale * m, lo=scale * lo,
                     hi=scale * hi, n=len(ids), a=scale * x[ids].mean(), b=scale * y[ids].mean()))


def utility(rows, run, endpoint, d, a, b):
    def u(arm):
        g = d[(d.arm == arm) & d.cond.isin(["ans", "unans"])].drop_duplicates(["cond", "uid"], keep="last")
        return g.set_index(["cond", "uid"]).reward
    x, y = u(a), u(b)
    ids = x.index.intersection(y.index)
    if len(ids) == 0:
        return
    m, lo, hi = boot((x[ids] - y[ids]).values)
    rows.append(dict(run=run, endpoint=endpoint, contrast=f"utility: {a} - {b}", diff=m, lo=lo, hi=hi, n=len(ids),
                     a=x[ids].mean(), b=y[ids].mean()))


runs = {"q25_3b_s0": ["q25_3b_s0"], "q25_3b_s1": ["q25_3b_s1"], "q3_4b_s0": ["q3_4b_run1", "q3_4b_fix"]}
rows = []
for run, dirs in runs.items():
    d = load_run(dirs)
    if d is None:
        continue
    if "s_abs_cf" not in set(d.arm) and "sft_before_dpo" in set(d.arm):
        d = pd.concat([d, d[d.arm == "sft_before_dpo"].assign(arm="s_abs_cf")], ignore_index=True)
    arms = set(d.arm)
    if {"s_abs", "s_plain"} <= arms:
        contrast(rows, run, "H-K", d, "s_abs", "s_plain", "unans", "abstain", "abstain on unanswerable (pts)")
        contrast(rows, run, "H-K", d, "s_abs", "s_plain", "ans", "correct", "EM on answerable (pts)")
    if {"s_abs_cf", "s_abs"} <= arms:
        contrast(rows, run, "H-L", d, "s_abs_cf", "s_abs", "conflict", "correct", "follow context (pts)")
        contrast(rows, run, "H-L", d, "s_abs_cf", "s_abs", "ans", "correct", "EM on answerable (pts)")
        contrast(rows, run, "H-L", d, "s_abs_cf", "s_abs", "unans", "abstain", "abstain on unanswerable (pts)")
    for post, stage in (("s_dpo", "sft_before_dpo"), ("s_rl", "sft_before_rl")):
        if {post, stage} <= arms:
            utility(rows, run, "H-M", d, post, stage)
            contrast(rows, run, "H-M", d, post, stage, "unans", "abstain", "abstain on unanswerable (pts)")
            contrast(rows, run, "H-M", d, post, stage, "ans", "correct", "EM on answerable (pts)")
    if "s_plain" in arms and "s_abs" in arms:
        utility(rows, run, "descr", d, "s_abs", "s_plain")

df = pd.DataFrame(rows)
df.round(2).to_csv(os.path.join(R, "analysis.csv"), index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70)
print(df.round(2).to_string(index=False))
for run in runs:
    for f in glob.glob(os.path.join(R, run, "*", "summary.json")):
        for arm, v in json.load(open(f)).items():
            if "probe" in v:
                print(run, "probe", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v["probe"].items()})
