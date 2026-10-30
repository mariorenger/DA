"""Round 13 (TN9): standard abstention metrics from the saved decisions.

Positive class = unanswerable question. Abstain precision / recall / F1, coverage on answerable questions
(share answered) and precision of the answers that are given.
    python tools/analyze_hall_abstain.py  ->  results/hall/abstain_metrics.csv
"""
import glob
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hall")
RUNS = {"qwen2.5-3b_s0": ["q25_3b_s0"], "qwen2.5-3b_s1": ["q25_3b_s1"], "qwen3-4b": ["q3_4b_run1", "q3_4b_fix"]}
rows = []
for run, dirs in RUNS.items():
    d = pd.concat([pd.read_csv(f) for x in dirs for f in sorted(glob.glob(os.path.join(R, x, "*", "decisions.csv")))])
    for arm in sorted(set(d.arm)):
        g = d[(d.arm == arm) & d.cond.isin(["ans", "unans"])].drop_duplicates(["cond", "uid"], keep="last")
        a, u = g[g.cond == "ans"], g[g.cond == "unans"]
        if a.empty or u.empty:
            continue
        tp, fp = u.abstain.sum(), a.abstain.sum()
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / len(u)
        f1 = 2 * p * r / (p + r) if tp else float("nan")
        answered = a[a.abstain == 0]
        rows.append(dict(run=run, arm=arm, abstain_precision=p, abstain_recall=r, abstain_f1=f1,
                         coverage_answerable=len(answered) / len(a), answer_precision=answered.correct.mean(),
                         n_ans=len(a), n_unans=len(u)))
df = pd.DataFrame(rows).round(3)
df.to_csv(os.path.join(R, "abstain_metrics.csv"), index=False)
print(df.to_string(index=False))
