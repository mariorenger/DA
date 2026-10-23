"""Round 10 analysis: structured-chain SFT vs the pre-registered baselines (see REVIEW.md).

    python tools/analyze_cf.py   ->  results/cftrain/analysis.csv, analysis.md

Primary endpoint: C5 (k=5-8) accuracy of s_struct minus the best of {s_plain, B0 cot_free, B0 sc_plus},
paired bootstrap over questions, at Qwen2.5-3B (seed 0 and seed 1 reported separately).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
BASE = {"c4": "C4_gender_clutrr", "c6": "C6_fresh_short", "c5": "C5_long_clutrr"}
MODELS = ["qwen-1.5b", "qwen-3b"]


def boot(diff, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(diff, len(diff)).mean() for _ in range(n)])
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load_base(model, st):
    d = pd.read_csv(os.path.join(R, "causaldec", BASE[st], model, "decisions.csv"))
    return d.pivot_table(index="uid", columns="method", values="correct")


def main():
    rows, md = [], []
    for seed in (0, 1):
        for model in MODELS:
            cf = pd.read_csv(os.path.join(R, "cftrain", f"seed{seed}", model, "decisions.csv"))
            for st in ("c4", "c6", "c5"):
                g = cf[(cf.set == st) & (cf.cond == "orig")]
                w = g.pivot_table(index="uid", columns="arm", values="correct")
                b = load_base(model, st).reindex(w.index)
                cols = {"s_struct": w.s_struct, "s_plain": w.s_plain, "cot_free": b.cot_free,
                        "sc_plus": b.sc_plus, "d2_ext_g_sc": b.ext_g_sc}
                best = max(["s_plain", "cot_free", "sc_plus"], key=lambda k: cols[k].mean())
                for name, ref in (("best_baseline:" + best, cols[best]), ("s_plain", cols["s_plain"]),
                                  ("d2_ext_g_sc (exploratory)", cols["d2_ext_g_sc"])):
                    m, lo, hi = boot((cols["s_struct"] - ref).values)
                    rows.append(dict(seed=seed, model=model, set=st, n=len(w), contrast="s_struct - " + name,
                                     s_struct=100 * cols["s_struct"].mean(), ref=100 * ref.mean(),
                                     delta=100 * m, lo=100 * lo, hi=100 * hi))
                cov = g.drop_duplicates("uid").set_index("uid").covered
                for arm in ("s_struct", "s_plain"):
                    a = g[g.arm == arm].set_index("uid")
                    for c in (1, 0):
                        idx = a.index[cov.reindex(a.index) == c]
                        rows.append(dict(seed=seed, model=model, set=st, n=len(idx),
                                         contrast=f"acc {arm} covered={c}", s_struct=np.nan,
                                         ref=100 * a.loc[idx, "correct"].mean(), delta=np.nan, lo=np.nan, hi=np.nan))
            # robustness on c6
            c6 = cf[cf.set == "c6"]
            for arm in ("s_struct", "s_plain"):
                o = c6[(c6.arm == arm) & (c6.cond == "orig")].set_index("uid")
                for cond in ("rename", "order", "distract"):
                    p = c6[(c6.arm == arm) & (c6.cond == cond)].set_index("uid")
                    rows.append(dict(seed=seed, model=model, set="c6", n=len(p), contrast=f"{arm} {cond}: acc / flip",
                                     s_struct=100 * p.correct.mean(), ref=100 * (p.pred != o.pred.reindex(p.index)).mean(),
                                     delta=100 * (p.correct.mean() - o.correct.mean()), lo=np.nan, hi=np.nan))
    df = pd.DataFrame(rows)
    df.round(2).to_csv(os.path.join(R, "cftrain", "analysis.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
