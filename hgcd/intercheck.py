"""Round 35: do interventions on the source text detect and repair errors of fact extraction? (CLUTRR)

Setting (Rounds 9-10, REVIEW.md): the rule table composes exactly, so the accuracy of "extract facts, then compose with
the table" is limited by the extraction (recall and precision of the facts the model lists, worst on long chains).
Idea: use the interventions of the thesis as an unsupervised check of the extraction. Three interventions have a known
effect on the set of stated facts, namely none:
  rename   every person gets a fresh name of the same gender (facts must map one-to-one onto the original ones)
  reorder  the sentences are shuffled (the facts are the same)
  placebo  two neutral sentences are inserted (no new facts)
The model extracts the facts from the original story and from each intervened story (names mapped back). A fact the
extraction loses or changes under a null intervention was not read reliably.

Per example the file records the prediction of the rule table from
  base    the greedy extraction of the original story
  int     a vote over {base, rename, reorder, placebo}                      (3 extra extractions, interventions)
  samp    a vote over {base, 3 samples at T=0.7 of the original story}     (3 extra extractions, sampling)
and two instability scores of the base prediction (fraction of the 3 extra extractions that disagree with it at the
answer level, and fraction of base facts that are missing from them at the fact level), once from the interventions and
once from sampling. AUROC of these scores against "base is wrong" is computed by tools/analyze_intercheck.py.

    python -m hgcd.intercheck --model Qwen/Qwen2.5-3B-Instruct --outdir out --ks 2 3 4 5 6 7 8 --n_per_k 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import chain as C
from . import data as D
from .causaldec import (EXTRACT_SYSTEM, _vote, extract_messages, graph_compose, induce_table, parse_facts,
                        parse_genders)
from .lm import LM

T0 = time.time()


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def facts_of(text):
    """Facts of an extraction (the Genders line is not a fact)."""
    return parse_facts("\n".join(l for l in (text or "").splitlines() if not l.strip().lower().startswith("genders")))


def norm_facts(triples, mp: Optional[Dict[str, str]] = None):
    n = C.norm
    mp = mp or {}
    out = set()
    for a, r, b in triples:
        a, b = n(a), n(b)
        out.add((mp.get(a, a), r, mp.get(b, b)))
    return out


def predict(table, vocab, triples, genders, src, tgt, mp):
    n = C.norm
    t2 = [(mp.get(n(a), n(a)), r, mp.get(n(b), n(b))) for a, r, b in triples]
    g2 = {mp.get(k, k): v for k, v in genders.items()}
    return graph_compose(table, t2, src, tgt, vocab, g2)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--n_per_k", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_samp", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--task_name", default=D.CLUTRR_TASK)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    train = D.load_clutrr("train")
    table = induce_table(train)
    vocab_l = D.clutrr_relations(train)
    vocab = set(vocab_l)
    pool = D.load_clutrr("test", task=args.task_name)
    ex = D.sample_by_k(pool, args.ks, args.n_per_k, args.seed)
    lm = LM(args.model, device_map=args.device_map)
    _log(f"loaded {args.model}; {len(ex)} examples, ks {sorted(set(e.k for e in ex))}")
    n = C.norm
    rows = []
    for s in range(0, len(ex), 24):
        part = ex[s:s + 24]
        variants = {"orig": part, "rename": [D.rename(e, args.seed) for e in part],
                    "reorder": [D.reorder(e, args.seed) for e in part],
                    "placebo": [D.placebo_input(e, args.seed) for e in part]}
        texts: Dict[str, List[str]] = {}
        for name, vs in variants.items():
            ps = [lm.prompt(extract_messages(v.story_text, vocab_l, genders=True)) for v in vs]
            out: List[Optional[str]] = [None] * len(vs)
            for k in sorted(set(e.k for e in vs)):
                idx = [i for i, e in enumerate(vs) if e.k == k]
                g = lm.generate([ps[i] for i in idx], max_new_tokens=20 * (k + 3) + 40, batch_size=args.bs,
                                stop="\n\n")
                for t, i in enumerate(idx):
                    out[i] = g[t]
            texts[name] = out
        ps0 = [lm.prompt(extract_messages(e.story_text, vocab_l, genders=True)) for e in part]
        samp: List[List[str]] = [[] for _ in part]
        for k in sorted(set(e.k for e in part)):
            idx = [i for i, e in enumerate(part) if e.k == k]
            sm = lm.generate([ps0[i] for i in idx], max_new_tokens=20 * (k + 3) + 40, do_sample=True,
                             n=args.n_samp, batch_size=max(1, args.bs // 2), stop="\n\n")
            for t, i in enumerate(idx):
                samp[i] = sm[t * args.n_samp:(t + 1) * args.n_samp]
        for i, e in enumerate(part):
            def pred_from(text, mp):
                return predict(table, vocab, facts_of(text), parse_genders(text), n(e.source), n(e.target), mp)
            maps = {"orig": {}, "reorder": {}, "placebo": {},
                    "rename": {n(a): n(b) for a, b in zip(variants["rename"][i].names, e.names)}}
            ident = {"orig": e, "rename": variants["rename"][i], "reorder": variants["reorder"][i],
                     "placebo": variants["placebo"][i]}
            def pred_variant(name):
                v = ident[name]
                t = texts[name][i]
                return predict(table, vocab, facts_of(t), parse_genders(t), n(e.source), n(e.target), maps[name])
            base = pred_variant("orig")
            ivs = [pred_variant(x) for x in ["rename", "reorder", "placebo"]]
            sps = [pred_from(t, {}) for t in samp[i]]
            base_f = norm_facts(facts_of(texts["orig"][i]))
            def fact_miss(texts_list, mps):
                if not base_f:
                    return 1.0
                miss = []
                for t, mp in zip(texts_list, mps):
                    f = norm_facts(facts_of(t), mp)
                    miss.append(sum(x not in f for x in base_f) / len(base_f))
                return float(sum(miss) / len(miss)) if miss else 0.0
            iv_texts = [texts[x][i] for x in ["rename", "reorder", "placebo"]]
            iv_maps = [maps[x] for x in ["rename", "reorder", "placebo"]]
            gold_f = {(n(h.head), h.rel, n(h.tail)) for h in e.gold_hops}
            rows.append(dict(
                uid=e.uid, k=e.k, gold=e.answer, base=base, int_vote=_vote([base] + ivs), samp_vote=_vote([base] + sps),
                int_disagree=float(sum(p != base for p in ivs) / max(1, len(ivs))),
                samp_disagree=float(sum(p != base for p in sps) / max(1, len(sps))),
                int_factmiss=fact_miss(iv_texts, iv_maps), samp_factmiss=fact_miss(samp[i], [{}] * len(samp[i])),
                base_correct=float(base == e.answer), int_correct=float(_vote([base] + ivs) == e.answer),
                samp_correct=float(_vote([base] + sps) == e.answer),
                base_gold_recall=float(len(base_f & gold_f) / max(1, len(gold_f))), n_base_facts=len(base_f),
                iv_preds="|".join(str(p) for p in ivs), samp_preds="|".join(str(p) for p in sps)))
        pd.DataFrame(rows).to_csv(f"{args.outdir}/intercheck.csv", index=False)
        d = pd.DataFrame(rows)
        _log(f"  {len(rows)}/{len(ex)} base {d.base_correct.mean():.3f} int {d.int_correct.mean():.3f} "
             f"samp {d.samp_correct.mean():.3f}")
    _log("done")


if __name__ == "__main__":
    main()
