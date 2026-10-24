"""Round 12 analysis (REVIEW.md 'Round 12'): Qwen3 contrasts, paired bootstrap over questions.

    python tools/analyze_r12.py  ->  results/cfrl/r12_analysis.csv
Runs that are not yet collected are skipped.
"""
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cfrl")
SETS = ("c4", "c6", "c5")


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def load(*p):
    f = os.path.join(R, *p, "decisions.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def wide(d, arm, st):
    g = d[(d.arm == arm) & (d.set == st) & (d.cond == "orig")]
    return g.set_index("uid").correct


rows = []


def contrast(tag, model, da, a, db, b, note=""):
    if da is None or db is None:
        return
    for st in SETS:
        x, y = wide(da, a, st), wide(db, b, st)
        if len(x) == 0 or len(y) == 0:
            continue
        ids = x.index.intersection(y.index)
        m, lo, hi = boot((x[ids] - y[ids]).values)
        rows.append(dict(endpoint=tag, model=model, contrast=f"{a} - {b} [{st}]", diff=m, lo=lo, hi=hi, n=len(ids),
                         acc_a=100 * x[ids].mean(), acc_b=100 * y[ids].mean(), note=note))


# 1.7B (r12_small)
d = load("r12_small", "qwen3-1.7b")
contrast("H-H", "qwen3-1.7b", d, "s_plain", d, "s_ans")
contrast("H-G", "qwen3-1.7b", d, "s_plain", d, "p_fs")
contrast("H-J(expl)", "qwen3-1.7b", d, "s_grpo", d, "sft_before_grpo")
# 4B distillation (r12_star) vs the 4B few-shot run
ds = load("r12_star", "qwen3-4b")
contrast("H-I", "qwen3-4b", ds, "s_star", ds, "s_plain", "non-inferiority needs lo > -6")
d4 = load("r12_4b", "qwen3-4b")
if d4 is not None:
    contrast("H-G", "qwen3-4b", d4, "s_plain", d4, "p_fs")
    contrast("H-H", "qwen3-4b", d4, "s_plain", d4, "s_ans")
    contrast("H-J", "qwen3-4b", d4, "s_grpo", d4, "sft_before_grpo")
    contrast("H-G(think)", "qwen3-4b", d4, "s_plain", d4, "p_think", "subset uids only")
    if ds is not None:
        contrast("H-I", "qwen3-4b", ds, "s_star", d4, "p_fs", "needs diff >= +2, lo > 0")
d8 = load("r12_8b", "qwen3-8b")
if d8 is not None:
    contrast("H-H", "qwen3-8b", d8, "s_plain", d8, "s_ans")
    contrast("H-G(think)", "qwen3-8b", d8, "s_plain", d8, "p_think", "subset uids only")

df = pd.DataFrame(rows)
df.round(2).to_csv(os.path.join(R, "r12_analysis.csv"), index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 40)
print(df.round(1).to_string(index=False))
