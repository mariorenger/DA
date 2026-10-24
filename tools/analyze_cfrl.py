"""Round 11 analysis: pre-registered endpoints H-A .. H-D (REVIEW.md, 'Round 11').

    python tools/analyze_cfrl.py  ->  results/cfrl/analysis.csv (+ printed tables)

Accuracy contrasts are paired bootstrap over questions. The wrong-copy follow rates are stored only as
aggregates (rate, n), so their difference uses an unpaired normal approximation (approximate CI).
"""
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CF = os.path.join(R, "cfrl")
SETS = ("c4", "c6", "c5")


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def load(path):
    return pd.read_csv(os.path.join(path, "decisions.csv"))


def wide(d, arm, st, cond="orig"):
    g = d[(d.arm == arm) & (d.set == st) & (d.cond == cond)]
    return g.set_index("uid").correct


def extras(path):
    e = pd.read_csv(os.path.join(path, "extras.csv"))
    return {(r.arm, r.metric): (r.value, r.n) for r in e.itertuples()}


def prop_diff(a, b):
    (p1, n1), (p2, n2) = a, b
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return 100 * (p1 - p2), 100 * (p1 - p2 - 1.96 * se), 100 * (p1 - p2 + 1.96 * se)


def flips(d, arm, cond):
    o = d[(d.arm == arm) & (d.set == "c6") & (d.cond == "orig")].set_index("uid").pred
    p = d[(d.arm == arm) & (d.set == "c6") & (d.cond == cond)].set_index("uid").pred
    return 100 * float((p != o.reindex(p.index)).mean())


rows = []
add = lambda **k: rows.append(k)

# ---- H-B / H-C: stage-1 SFT vs DPO / GRPO from the same run
for run, stage, arm in (("dpo_s0", "sft_before_dpo", "s_dpo"), ("dpo_s1", "sft_before_dpo", "s_dpo"),
                        ("grpo", "sft_before_grpo", "s_grpo")):
    for model in ("qwen-1.5b", "qwen-3b"):
        p = os.path.join(CF, run, model)
        d = load(p)
        for st in SETS:
            a, b = wide(d, arm, st), wide(d, stage, st)
            m, lo, hi = boot((a - b.reindex(a.index)).values)
            add(run=run, model=model, what=f"acc {arm} - {stage} [{st}]", value=m, lo=lo, hi=hi, n=len(a),
                ref=100 * b.mean(), new=100 * a.mean())
        ex = extras(p)
        for met in ("wrong_copy_follow_given_ps1", "wrong_copy_follow_all", "do_total_effect_nonfinal_hops",
                    "ps_plain_chain"):
            a, b = ex[(arm, met)], ex[(stage, met)]
            if met.startswith("wrong"):
                m, lo, hi = prop_diff(a, b)
            else:
                m, lo, hi = 100 * (a[0] - b[0]), np.nan, np.nan
            add(run=run, model=model, what=f"{met} {arm} - {stage}", value=m, lo=lo, hi=hi, n=a[1],
                ref=100 * b[0], new=100 * a[0])
        for cond in ("rename", "order", "distract"):
            add(run=run, model=model, what=f"flip {cond}: {arm} / {stage}", value=flips(d, arm, cond),
                lo=np.nan, hi=np.nan, n=158, ref=flips(d, stage, cond), new=flips(d, arm, cond))

# ---- H-A: does the chain matter?  s_plain (round 10, same seed) - s_ans (round 11)
for seed, run in ((0, "dpo_s0"), (1, "dpo_s1")):
    for model in ("qwen-1.5b", "qwen-3b"):
        pl = load(os.path.join(R, "cftrain", f"seed{seed}", model))
        an = load(os.path.join(CF, run, model))
        for st in SETS:
            a, b = wide(pl, "s_plain", st), wide(an, "s_ans", st)
            m, lo, hi = boot((a - b.reindex(a.index)).values)
            add(run=f"seed{seed}", model=model, what=f"acc s_plain - s_ans [{st}]", value=m, lo=lo, hi=hi, n=len(a),
                ref=100 * b.mean(), new=100 * a.mean())
d7 = load(os.path.join(CF, "b7", "qwen-7b"))
for st in SETS:
    a, b = wide(d7, "s_plain", st), wide(d7, "s_ans", st)
    m, lo, hi = boot((a - b.reindex(a.index)).values)
    add(run="7b", model="qwen-7b", what=f"acc s_plain - s_ans [{st}]", value=m, lo=lo, hi=hi, n=len(a),
        ref=100 * b.mean(), new=100 * a.mean())
ex7 = extras(os.path.join(CF, "b7", "qwen-7b"))
for met in ("wrong_copy_follow_given_ps1", "wrong_copy_follow_all", "do_total_effect_nonfinal_hops", "ps_plain_chain"):
    v, n = ex7[("s_plain", met)]
    add(run="7b", model="qwen-7b", what=f"{met} s_plain", value=100 * v, lo=np.nan, hi=np.nan, n=n, ref=np.nan,
        new=100 * v)

df = pd.DataFrame(rows)
df.round(2).to_csv(os.path.join(CF, "analysis.csv"), index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70)
print(df.round(1).to_string(index=False))
