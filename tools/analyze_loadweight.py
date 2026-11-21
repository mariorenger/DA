"""Round 34 analysis: paired bootstrap of arm - arm accuracy differences (same 500 test questions).

    python tools/analyze_loadweight.py results/loadweight/qwen-1.5b/sft_eval.csv ...
"""
import sys
import numpy as np
import pandas as pd

PAIRS = [("struct", "uniform"), ("struct", "loo"), ("struct", "random"), ("anti", "uniform"), ("loo", "uniform"),
         ("random", "uniform"), ("uniform", "base")]
rng = np.random.default_rng(0)
for path in sys.argv[1:]:
    d = pd.read_csv(path)
    w = d.pivot(index="uid", columns="arm", values="correct")
    print(path.split("/")[-2], "n =", len(w), " acc:", {a: round(float(w[a].mean()), 3) for a in w.columns})
    for a, b in PAIRS:
        if a not in w or b not in w:
            continue
        x = (w[a] - w[b]).values
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
        print(f"  {a:>8s} - {b:<8s} {100*x.mean():+6.1f} [{100*np.percentile(bs,2.5):+5.1f}, {100*np.percentile(bs,97.5):+5.1f}]")
