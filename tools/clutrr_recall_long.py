import sys; sys.path.insert(0, '/home/user/DA')
import pandas as pd, numpy as np
from hgcd import data as D, chain as C
from hgcd import causaldec as K
R = sys.argv[1]; test = {e.uid: e for e in D.load_clutrr("test")}
for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
    c = pd.read_csv(f"{R}/causaldec_clutrr_{m}/decisions.csv"); c = c[c.method == "cot_free"]
    rec, allr, nf = [], [], []
    for _, r in c.iterrows():
        e = test[r.uid]; f = K.parse_facts(r.extract if isinstance(r.extract, str) else "")
        got = {frozenset((C.norm(a), C.norm(b))) for a, _, b in f}
        g = [frozenset((C.norm(h.head), C.norm(h.tail))) in got for h in e.gold_hops]
        rec.append(np.mean(g)); allr.append(all(g)); nf.append(len([x for x in f if x[1]]))
    c = c.assign(rec=rec, allr=allr, nf=nf)
    print(m, c.groupby("k")[["rec", "allr", "nf"]].mean().round(2).to_dict("index"))
