import sys; sys.path.insert(0, '/home/user/DA')
import pandas as pd, numpy as np
from hgcd import data as D, chain as C, metrics as M
from hgcd import causaldec as K
R = sys.argv[1]
train, test = D.load_clutrr("train"), D.load_clutrr("test")
table = K.induce_table(train); vocab = set(D.clutrr_relations(train)); ex = {e.uid: e for e in test}
glex = K.induce_gender_lexicon(train)
for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
    d = pd.read_csv(f"{R}/causaldec_clutrr_{m}/decisions.csv")
    w = d.pivot_table(index="uid", columns="method", values="correct")
    c = d[d.method == "cot_free"].set_index("uid")
    cov = np.mean([C.norm(n) in glex for u in w.index for n in ex[u].names])
    lex, ok = {}, 0
    for u in w.index:
        e = ex[u]; f = K.parse_facts(c.loc[u, "extract"] if isinstance(c.loc[u, "extract"], str) else "")
        p, s = K.graph_compose(table, f, e.source, e.target, vocab, glex)
        ok += s == "ok"
        lex[u] = float(p == e.answer) if p else w.loc[u, "cot_free"]
    w["ext_lex"] = pd.Series(lex)
    base = "sc_plus" if w.sc_plus.mean() >= w.cot_free.mean() else "cot_free"
    dm, lo, hi = M.boot_ci((w.ext_lex - w[base]).values)
    print(f"{m}: lexicon covers {cov:.2f} of people; composable {ok}/{len(w)}; ext_lex {w.ext_lex.mean():.3f} vs {base} {w[base].mean():.3f}: {100*dm:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={M.mcnemar_p(w.ext_lex.values>.5, w[base].values>.5):.3f}")
    print("   by k:", w.assign(k=[ex[u].k for u in w.index]).groupby("k")[["cot_free", "sc_plus", "ext_lex"]].mean().round(3).to_dict("index"))
