import sys; sys.path.insert(0, '/home/user/DA')
import pandas as pd, numpy as np
from collections import Counter
from hgcd import data as D, chain as C
from hgcd import causaldec as K
R = sys.argv[1]; train = D.load_clutrr("train"); test = {e.uid: e for e in D.load_clutrr("test")}
table = K.induce_table(train); vocab = set(D.clutrr_relations(train))
c = pd.read_csv(f"{R}/causaldec_clutrr_qwen-7b/decisions.csv"); c = c[c.method == "cot_free"]
st = Counter(); shown = 0
for _, r in c.iterrows():
    e = test[r.uid]; f = K.parse_facts(r.extract if isinstance(r.extract, str) else "")
    got = {frozenset((C.norm(a), C.norm(b))) for a, _, b in f}
    if not all(frozenset((C.norm(h.head), C.norm(h.tail))) in got for h in e.gold_hops): continue
    go = {C.norm(k): ("f" if v == "female" else "m") for k, v in e.genders.items()}
    p0, s0 = K.graph_compose(table, f, e.source, e.target, vocab)
    p1, s1 = K.graph_compose(table, f, e.source, e.target, vocab, go)
    oov = [x[1] for x in f if x[1] and x[1] not in vocab]
    st[(s0, s1, p1 == e.answer if p1 else None, bool(oov))] += 1
    if s1 != "ok" and shown < 3:
        shown += 1; print(e.k, e.source, "->", e.target, "| oov:", oov); print(r.extract[:500]); print("gold:", [(h.head, h.rel, h.tail) for h in e.gold_hops]); print()
print(st)
