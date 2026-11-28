"""Round 41 analysis: accuracy per method, paired bootstrap against greedy and sc5, repair / break rates, error split.

    python tools/analyze_ruleverify.py results/ruleverify/*/ruleverify.csv
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
ORDER = ["greedy", "sc5", "sym_full", "v_resample_full", "v_override_full", "v_override_small", "v_override_noisy"]


def boot(x, n=3000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


for path in sys.argv[1:]:
    d = pd.read_csv(path)
    print("==", path.split("/")[-2])
    for name, g in d.groupby("set"):
        w = g.pivot_table(index="uid", columns="method", values="correct")
        g0 = g[g.method == "greedy"].set_index("uid")
        wrong = g0.index[g0.correct == 0]
        print(f" set {name}: n={len(w)}  greedy errors {len(wrong)}  "
              f"(composition error {int((g0.loc[wrong].comp_ok == 0).sum())}, "
              f"hop error with consistent composition {int(((g0.loc[wrong].comp_ok == 1) & (g0.loc[wrong].hops_ok == 0)).sum())})")
        for m in ORDER:
            if m not in w:
                continue
            row = f"   {m:18s} acc {100 * w[m].mean():5.1f}"
            if m != "greedy":
                x = (w[m] - w["greedy"]).values
                lo, hi = boot(x)
                row += f"  vs greedy {100 * x.mean():+5.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]"
                rep = float(w.loc[wrong, m].mean()) if len(wrong) else float("nan")
                right = w.index.difference(wrong)
                brk = float(1 - w.loc[right, m].mean()) if len(right) else float("nan")
                row += f"  repair {100 * rep:4.1f}%  break {100 * brk:4.1f}%"
            if m not in ("greedy", "sc5") and "sc5" in w:
                x = (w[m] - w["sc5"]).values
                lo, hi = boot(x)
                row += f"  vs sc5 {100 * x.mean():+5.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]"
            print(row)
