"""Round 20 analysis (H-T1..H-T3). python tools/analyze_stride.py [results/stride]"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/stride"
rows = []
for f in sorted(glob.glob(f"{ROOT}/stride_*/stride_eval.csv")):
    name = os.path.basename(os.path.dirname(f)).replace("stride_", "")
    d = pd.read_csv(f)
    for (m, k), g in d.groupby(["stride", "k"]):
        rows.append(dict(model=name, stride=m, k=k, acc=g.correct.mean(), tokens=g.gen_tokens.mean(), n=len(g)))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no results")
piv = df.pivot_table(index=["model", "stride"], columns="k", values="acc")
piv["mean"] = piv.mean(axis=1)
pd.set_option("display.width", 200)
print(piv.round(3).to_string())
piv.to_csv(f"{ROOT}/analysis_stride.csv")
res = {}
for name, g in df.groupby("model"):
    a = g.groupby("stride").acc.mean()
    s14 = [s for s in [1, 2, 3, 4] if s in a.index]
    drops = [a[s14[i]] - a[s14[i + 1]] for i in range(len(s14) - 1)]
    cap = max([s for s in s14 if a[s] >= 0.80], default=0)
    res[name] = dict(acc_stride1=a.get(1, np.nan), max_drop_1to4=max(drops) if drops else np.nan, capacity=cap)
print(pd.DataFrame(res).T.round(3).to_string())
small = [m for m in res if m in ("qwen-1.5b", "qwen3-1.7b", "qwen-3b")]
print("H-T1:", {m: bool(res[m]["acc_stride1"] >= 0.85) for m in small})
print("H-T2 knee (>=0.30 between consecutive strides in 1..4):", {m: bool(res[m]["max_drop_1to4"] >= 0.30) for m in small})
cap = {m: res[m]["capacity"] for m in res}
print("H-T3 capacity:", cap)
