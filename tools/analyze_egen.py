"""Round 10b analysis (2WikiMultihopQA): s_struct vs the pre-registered baselines (REVIEW.md round 10b).

    python tools/analyze_egen.py  ->  results/egen/wiki2/analysis.csv
Primary: Qwen2.5-3B, EM of s_struct minus the best of {s_plain, s_ans, p0}, paired bootstrap (95% CI).
"""
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "egen", "wiki2")


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


rows = []
for model in ("qwen-1.5b", "qwen-3b"):
    d = pd.read_csv(os.path.join(R, model, "decisions.csv"))
    w = d.pivot_table(index="uid", columns="arm", values="em")
    kind = d.drop_duplicates("uid").set_index("uid").kind
    for arm in w.columns:
        m, lo, hi = boot(w[arm].values)
        rows.append(dict(model=model, what=f"EM {arm}", n=len(w), value=100 * m, lo=100 * lo, hi=100 * hi))
    best = max(["s_plain", "s_ans", "p0"], key=lambda a: w[a].mean())
    for name, ref in (("best:" + best, w[best]), ("s_plain", w.s_plain), ("s_ans", w.s_ans), ("p0", w.p0)):
        m, lo, hi = boot((w.s_struct - ref).values)
        rows.append(dict(model=model, what=f"s_struct - {name}", n=len(w), value=100 * m, lo=100 * lo, hi=100 * hi))
    m, lo, hi = boot((w.s_plain - w.s_ans).values)
    rows.append(dict(model=model, what="s_plain - s_ans", n=len(w), value=100 * m, lo=100 * lo, hi=100 * hi))
    for k in kind.unique():
        idx = kind.index[kind == k]
        for arm in ("p0", "s_ans", "s_plain", "s_struct"):
            rows.append(dict(model=model, what=f"EM {arm} [{k}]", n=len(idx), value=100 * w.loc[idx, arm].mean(),
                             lo=np.nan, hi=np.nan))
    s = d[d.arm == "s_struct"]
    rows.append(dict(model=model, what="cite_ok (cited titles that are gold supporting)", n=len(s),
                     value=100 * s.cite_ok.mean(), lo=np.nan, hi=np.nan))
    rows.append(dict(model=model, what="last_hop_match s_struct / s_plain", n=len(s),
                     value=100 * s.last_hop_match.mean(), lo=100 * d[d.arm == "s_plain"].last_hop_match.mean(), hi=np.nan))
df = pd.DataFrame(rows)
df.round(2).to_csv(os.path.join(R, "analysis.csv"), index=False)
pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 60)
print(df.round(1).to_string(index=False))
