"""Selection gap of the wrong-answer copy test (Round 8 data), after Pan & Pan 2026 (arXiv 2610.37070):
Delta = P(follow | solvable) - P(follow) = Cov(A, C) / P(C = 1), A = follows the wrong stated answer, C = plain chain solved."""
import glob, os
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
rows = []
for d in sorted(glob.glob("results/reliance_v2/*/estimators.csv")):
    m = d.split("/")[-2]
    e = pd.read_csv(d)
    cw = e[e.dataset == "clutrr_copywrong"].drop_duplicates("uid") if "clutrr_copywrong" in set(e.dataset) else None
    pl = e[e.dataset == "clutrr"].drop_duplicates("uid").set_index("uid").full_correct if "clutrr" in set(e.dataset) else None
    if cw is None or pl is None:
        continue
    cw = cw.assign(solv=cw.uid.map(pl)).dropna(subset=["solv", "follows_wrong"])
    A, C = cw.follows_wrong.astype(float).values, cw.solv.astype(float).values
    def stat(idx):
        a, c = A[idx], C[idx]
        if c.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        return a.mean(), a[c == 1].mean(), a[c == 0].mean() if (c == 0).any() else np.nan, a[c == 1].mean() - a.mean()
    pt = stat(np.arange(len(A)))
    bs = np.array([stat(rng.integers(0, len(A), len(A))) for _ in range(2000)])
    lo, hi = np.nanpercentile(bs[:, 3], [2.5, 97.5])
    rows.append(dict(model=m, n=len(A), p_solvable=C.mean(), follow_all=pt[0], follow_solvable=pt[1], follow_unsolvable=pt[2],
                     delta=pt[3], delta_lo=lo, delta_hi=hi))
df = pd.DataFrame(rows)
df.to_csv("results/reliance_v2/selection_gap.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
