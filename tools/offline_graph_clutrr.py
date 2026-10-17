import sys; sys.path.insert(0, '/home/user/DA')
import pandas as pd, numpy as np
from hgcd import data as D, metrics as M
from hgcd.causaldec import induce_table, parse_triples, graph_compose, chain_compose
from collections import Counter
O = sys.argv[1]
train, test = D.load_clutrr("train"), D.load_clutrr("test")
table = induce_table(train); vocab = set(D.clutrr_relations(train))
ex = {e.uid: e for e in test}
for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
    d = pd.read_csv(f"{O}/causaldec_clutrr_{m}/decisions.csv")
    w = d.pivot_table(index="uid", columns="method", values="correct")
    c = d[d.method == "cot_free"].set_index("uid")
    g, st = {}, Counter()
    for uid, r in c.iterrows():
        e = ex[uid]
        p, s = graph_compose(table, parse_triples(str(r.chain)), e.source, e.target, vocab)
        st[s] += 1
        g[uid] = float(p == e.answer) if p else np.nan
    gs = pd.Series(g)
    w["sym_graph"] = gs.fillna(w["cot_free"])
    w["graph_only_ok"] = gs
    ok = gs.notna()
    print(f"== {m}: status {dict(st)}")
    print(f"   graph coverage {ok.mean():.3f}; acc when composable {gs[ok].mean():.3f} vs cot_free on same {w.loc[ok,'cot_free'].mean():.3f}")
    for a in ["cot_free", "sc", "sym_sc", "sym_graph"]:
        mu, lo, hi = M.boot_ci(w[a].values)
        dd = M.boot_ci((w[a] - w["cot_free"]).values)
        print(f"   {a:10s} {mu:.3f} [{lo:.3f},{hi:.3f}]  vs cot_free {dd[0]:+.3f} [{dd[1]:+.3f},{dd[2]:+.3f}] p={M.mcnemar_p(w[a].values>0.5, w['cot_free'].values>0.5):.3f}")
