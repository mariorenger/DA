"""C7 ablation: same extracted facts, rule table (ext_g) vs LLM composition (ext_llm)."""
import sys
import pandas as pd
sys.path.insert(0, ".")
from hgcd import metrics as M

R = sys.argv[1]
for lab in sys.argv[2:]:
    try:
        d = pd.read_csv(f"{R}/causaldec_clutrr_{lab}/decisions.csv")
    except FileNotFoundError:
        continue
    w = d.pivot_table(index="uid", columns="method", values="correct")
    print(lab, len(w), {k: round(w[k].mean(), 3) for k in ["cot_free", "sc", "ext_llm", "ext_g", "ext_g_sc"] if k in w})
    for a, b in [("ext_g", "ext_llm"), ("ext_llm", "cot_free"), ("ext_g", "cot_free")]:
        dm, lo, hi = M.boot_ci((w[a] - w[b]).values)
        print(f"   {a} - {b}: {100*dm:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={M.mcnemar_p(w[a].values > .5, w[b].values > .5):.4f}")
