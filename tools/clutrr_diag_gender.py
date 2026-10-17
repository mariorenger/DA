import sys; sys.path.insert(0, '/home/user/DA')
exec(open('/tmp/claude-0/-home-user-DA/153a9331-9349-5ae8-a928-f4cb3c52c859/scratchpad/exp/diag_ext.py').read().split("for m in")[0])
lex = defaultdict(Counter)
for e in train:
    for k, v in e.genders.items(): lex[C.norm(k)][v] += 1
glex = {k: ("f" if c.most_common(1)[0][0] == "female" else "m") for k, c in lex.items() if c.most_common(1)[0][1] >= 0.95 * sum(c.values())}
tot = hit = right = 0
for e in test:
    for k, v in e.genders.items():
        tot += 1; g = glex.get(C.norm(k))
        if g: hit += 1; right += g == ("f" if v == "female" else "m")
print("lexicon names", len(glex), "test coverage", round(hit / tot, 3), "accuracy when covered", round(right / max(1, hit), 3))
for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
    d = pd.read_csv(f"results/causaldec/C2_extract_clutrr/{m}/decisions.csv")
    c = d[d.method == "cot_free"]; w = d.pivot_table(index="uid", columns="method", values="correct")
    res = {}
    for _, r in c.iterrows():
        e = ex[r.uid]; facts = K.parse_facts(r.extract if isinstance(r.extract, str) else "")
        res[r.uid] = gc(facts, e.source, e.target, glex)[0]
    ok = {u: p for u, p in res.items() if p}
    corr = pd.Series({u: float(p == ex[u].answer) for u, p in ok.items()})
    ext_lex = pd.Series({u: (float(res[u] == ex[u].answer) if res[u] else w.loc[u, "cot_free"]) for u in w.index})
    print(m, "composable", len(ok), "acc|composable", round(corr.mean(), 3), "| ext_lex (fallback cot_free)", round(ext_lex.mean(), 3), "vs ext", round(w.ext.mean(), 3), "cot_free", round(w.cot_free.mean(), 3), "sc", round(w.sc.mean(), 3))
