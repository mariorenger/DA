"""Round 18 analysis (H-S1..H-S3, pre-registered in REVIEW.md). python tools/analyze_scale.py [results/scale]"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/scale"
rng = np.random.default_rng(0)
ORDER = ["q25-0.5b", "q25-1.5b", "q25-3b", "q25-7b", "q25-14b-4bit", "q25-32b-bnb4", "q3-1.7b", "q3-4b"]


def ci(v):
    v = np.asarray(v, float)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)])
    return v.mean(), *np.percentile(b, [2.5, 97.5])


rows = []
for lab in ORDER:
    f = f"{ROOT}/scale_{lab}/copy.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    r = dict(model=lab)
    for mode in ["intact", "last", "all"]:
        s = d[d["mode"] == mode]
        m, lo, hi = ci(s.follow)
        r[f"follow_{mode}"] = f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        r[f"n_{mode}"] = len(s)
        r[f"_f_{mode}"] = m
        r[f"correct_{mode}"] = 100 * s.correct.mean()
    rows.append(r)
df = pd.DataFrame(rows)
cols = [c for c in df.columns if not c.startswith("_")]
pd.set_option("display.width", 220)
print(df[cols].to_string(index=False))
df[cols].to_csv(f"{ROOT}/analysis_scale.csv", index=False)

q = df[df.model.str.startswith("q25")].set_index("model")
chain = [m for m in ["q25-1.5b", "q25-3b", "q25-7b", "q25-14b-4bit", "q25-32b-bnb4"] if m in q.index]
last = [q.loc[m, "_f_last"] for m in chain]
mono = all(b <= a + 0.03 for a, b in zip(last, last[1:]))
print("\nH-S1 monotone (rises <= 3 pts):", mono, dict(zip(chain, [round(100 * x, 1) for x in last])))
if "q25-3b" in q.index and "q25-14b-4bit" in q.index:
    print("H-S1 14B / 3B ratio = %.2f (rule: <= 0.5)" % (q.loc["q25-14b-4bit", "_f_last"] / q.loc["q25-3b", "_f_last"]))
big = "q25-32b-bnb4" if "q25-32b-bnb4" in q.index else "q25-14b-4bit"
print("H-S2 largest model", big, "follow(last) =", q.loc[big, "follow_last"], "(rule: > 5% with CI lower bound > 0)")
print("H-S3 follow(all) >= follow(last):", {m: bool(q.loc[m, "_f_all"] >= q.loc[m, "_f_last"]) for m in q.index})
