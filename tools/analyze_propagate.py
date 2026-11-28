"""Round 43 analysis: propagate / repair / other rates per arm, task and position, with bootstrap CIs.

    python tools/analyze_propagate.py results/propagate/*/propagate.csv
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def ci(x, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


for path in sys.argv[1:]:
    d = pd.read_csv(path)
    print("==", path.split("/")[-2])
    ctl = d[d.cond == "intact"]
    iv = d[d.cond == "intervened"]
    for (arm, task), g in iv.groupby(["arm", "task"]):
        acc = (ctl[(ctl.arm == arm) & (ctl.task == task)].cls == "right").mean()
        line = f" {arm:3s} {task:8s} intact-continuation acc {100 * acc:5.1f} |"
        for pos in ["early", "mid", "late"]:
            h = g[g.pos == pos]
            if not len(h):
                continue
            p = (h.cls == "propagate").values.astype(float)
            r = (h.cls == "repair").values.astype(float)
            lo, hi = ci(p)
            line += f" {pos}: prop {100 * p.mean():5.1f} [{lo:4.1f},{hi:5.1f}] rep {100 * r.mean():4.1f} oth {100 * (1 - p.mean() - r.mean()):4.1f} |"
        print(line)
