import sys; sys.path.insert(0, '/home/user/DA')
import pandas as pd, numpy as np
from collections import Counter, defaultdict
from hgcd import data as D, chain as C
from hgcd import causaldec as K
train, test = D.load_clutrr("train"), D.load_clutrr("test")
table = K.induce_table(train); vocab = set(D.clutrr_relations(train)); ex = {e.uid: e for e in test}

def gc(triples, src, tgt, gender_extra=None):
    n = C.norm
    facts = [(n(a), r, n(b)) for a, r, b in triples if r in vocab and a and b and n(a) != n(b)]
    if not facts: return None, "no_facts"
    gender = dict(gender_extra or {})
    for a, r, b in facts: gender[b] = "f" if r in K._FEMALE else "m"
    adj = defaultdict(list)
    for a, r, b in facts:
        adj[a].append((r, b))
        if a in gender and r in K._INV: adj[b].append((K._INV[r][gender[a] == "f"], a))
    s, t = n(src), n(tgt); prev, fr = {s: None}, [s]
    while fr and t not in prev:
        nx = []
        for u in fr:
            for r, v in adj[u]:
                if v not in prev: prev[v] = (u, r); nx.append(v)
        fr = nx
    if t not in prev: return None, "no_path"
    rels, v = [], t
    while prev[v] is not None: u, r = prev[v]; rels.append(r); v = u
    r = K.compose(table, rels[::-1]); return (r, "ok") if r else (None, "no_rule")

for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
    d = pd.read_csv(f"results/causaldec/C2_extract_clutrr/{m}/decisions.csv")
    c = d[d.method == "cot_free"]
    st = Counter(); acc = Counter(); miss = Counter(); recall = []
    for _, r in c.iterrows():
        e = ex[r.uid]; facts = K.parse_facts(r.extract if isinstance(r.extract, str) else "")
        gold = {(C.norm(h.head), h.rel, C.norm(h.tail)) for h in e.gold_hops}
        got = {(C.norm(a), rr, C.norm(b)) for a, rr, b in facts}
        # a gold edge counts as recovered if stated either way round
        rec = np.mean([(g in got) or any((b2, a2) == (g[0], g[2]) for a2, _, b2 in got) for g in gold])
        recall.append(rec)
        p0, s0 = gc(facts, e.source, e.target)
        g_or = {C.norm(k): ("f" if v == "female" else "m") for k, v in e.genders.items()}
        p1, s1 = gc(facts, e.source, e.target, g_or)
        st[(s0, s1)] += 1; acc["base"] += p0 == e.answer; acc["oracle_gender"] += p1 == e.answer
    print(m, "gold-edge recall in extraction:", round(np.mean(recall), 3), "| all gold edges:", round(np.mean(np.array(recall) == 1), 3))
    print("   composable base vs oracle-gender:", sum(v for k, v in st.items() if k[0] == "ok"), sum(v for k, v in st.items() if k[1] == "ok"), "/", len(c), "| correct", dict(acc))
