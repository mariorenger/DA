"""R1 sanity check: is do_total on a non-final step just 'removing the answer from the chain'?

Compares, per model and dataset (results/reliance_v2/<model>/estimators.csv):
  d_total on non-final gold steps  vs  d_naive on the final step (masks only the hop that states the answer)
  vs  the whole-chain effect p(gold | X, chain) - p(gold | X, no chain).
    python tools/check_r1_triviality.py  ->  results/reliance_v2/r1_triviality.csv
"""
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reliance_v2")
rows = []
for m in ["qwen-0.5b", "qwen-1.5b", "smollm2-1.7b", "qwen-3b", "qwen-7b"]:
    d = pd.read_csv(os.path.join(R, m, "estimators.csv"))
    for ds, g in d.groupby("dataset"):
        nf, fi, irr = g[(g.label == 1) & (g.is_final == 0)], g[(g.label == 1) & (g.is_final == 1)], g[g.label == 0]
        u = g.drop_duplicates("uid")
        rows.append(dict(model=m, dataset=ds, n_questions=len(u),
                         d_total_nonfinal=100 * nf.d_total.mean(), d_naive_nonfinal=100 * nf.d_naive.mean(),
                         d_naive_final_only=100 * fi.d_naive.mean(),
                         whole_chain=100 * (u.p_full_gold - u.p_gold_nochain).mean(),
                         d_total_irrelevant=100 * irr.d_total.mean()))
df = pd.DataFrame(rows).round(1)
df.to_csv(os.path.join(R, "r1_triviality.csv"), index=False)
print(df.to_string(index=False))
