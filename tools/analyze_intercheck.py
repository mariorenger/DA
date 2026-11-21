"""Round 35 analysis: AUROC of intervention vs sampling disagreement for 'base extraction is wrong', and vote accuracy.

    python tools/analyze_intercheck.py results/intercheck/*/intercheck.csv
"""
import sys

import numpy as np
import pandas as pd


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def boot(d, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


for path in sys.argv[1:]:
    d = pd.read_csv(path)
    d["err"] = 1 - d.base_correct
    for name, sub in [("k<=4", d[d.k <= 4]), ("k>=5", d[d.k >= 5])]:
        if len(sub) == 0:
            continue
        a_i = auroc(sub.int_disagree, sub.err)
        a_s = auroc(sub.samp_disagree, sub.err)
        fi = auroc(sub.int_factmiss, sub.err)
        fs = auroc(sub.samp_factmiss, sub.err)
        ci = boot(sub, lambda x: auroc(x.int_disagree, x.err) - auroc(x.samp_disagree, x.err), n=500)
        di = boot(sub, lambda x: x.int_correct.mean() - x.samp_correct.mean())
        db = boot(sub, lambda x: x.int_correct.mean() - x.base_correct.mean())
        print(f"{path.split('/')[-2]:>22s} {name} n={len(sub):3d} base={sub.base_correct.mean():.3f} "
              f"int={sub.int_correct.mean():.3f} samp={sub.samp_correct.mean():.3f} | AUROC ans int {a_i:.3f} "
              f"samp {a_s:.3f} diff CI [{ci[0]:+.3f},{ci[1]:+.3f}] | fact int {fi:.3f} samp {fs:.3f} | "
              f"int-samp [{di[0]*100:+.1f},{di[1]*100:+.1f}] int-base [{db[0]*100:+.1f},{db[1]*100:+.1f}]")
