"""Round 41: an external rule table as a step verifier for the model's OWN mistakes (CLUTRR, structured chains).

Round 40 measured that a model asked to recompute repairs only 7-14% of its own wrong chains: it cannot check its own
reasoning with the information it has. The advisor's hypothesis is that external knowledge is needed. Here the external
knowledge is the composition table induced from the TRAINING stories (hgcd.causaldec.induce_table + hgcd.cftrain.extend_table),
used as a verifier at every step of a fine-tuned structured chain ("Hop i: X's r is Y. => Y" then the running relation
"So Y is S's c."):  the running relation c_i the model writes must equal table[(c_{i-1}, r_i)] (c_1 = r_1).

Methods, per story (the same stories and the same fine-tuned model for all):
  greedy           the model's chain and answer (reference)
  sc5              majority of 5 sampled chains (same number of extra chains as the retries below)
  sym_full         the model's own hop relations composed by the table (external composition, TN6 style)
  v_resample_full  step by step; a running relation that contradicts the table is resampled (up to R tries), the original
                   is kept if no try agrees
  v_override_full  as above, and if no try agrees the running relation is replaced by the table value; the answer is the
                   last verified running relation
  v_override_small / v_override_noisy   the same with a table induced from 5% of the training stories, or with 10% of the
                   rules of the full table corrupted (how much the quality of the external knowledge matters)
For the greedy chains the errors are split: composition error (some running relation contradicts the full table) and
hop error (a stated hop is not a gold hop), the part of the errors that a composition table can and cannot repair.

    python -m hgcd.ruleverify --model Qwen/Qwen2.5-3B-Instruct --outdir out
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import chain as C
from . import data as D
from .causaldec import _first_rel, _rel_norm, _vote, chain_compose, induce_table, parse_triples
from .cftrain import SETS, STRUCT_SYSTEM, _SO, chain_text, covered, extend_table, msgs_zero, train_sft
from .lm import LM

T0 = time.time()


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def corrupt_table(table: Dict, vocab: List[str], frac: float, rng: random.Random) -> Dict:
    t = dict(table)
    keys = sorted(t)
    for k in rng.sample(keys, int(frac * len(keys))):
        t[k] = rng.choice([r for r in vocab if r != t[k]])
    return t


def parse_step(text: str):
    """(hop_line, hop_relation, running_relation, so_line, is_final) of one generated step."""
    final = "Answer:" in text
    text = text.split("Answer:")[0]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hop = next((ln.strip() for ln in lines if ln.lstrip().startswith("Hop")), None)
    so = next((ln.strip() for ln in lines if _SO.match(ln)), None)
    rel = run = None
    if hop:
        t = parse_triples(hop)
        rel = t[0][1] if t and t[0][1] else None
    if so:
        run = _rel_norm(_SO.match(so).group(3))
    return hop, rel, run, so, final


def expected_running(table: Dict, prev: Optional[str], rel: Optional[str]) -> Optional[str]:
    if rel is None:
        return None
    return rel if prev is None else table.get((prev, rel))


def _line(lm: LM, prompts: List[str], args, sample: bool = False, n: int = 1) -> List[str]:
    """One generated line per prompt (stop at the first newline).  NOTE: a stop string such as "\\nHop" cannot be used here:
    the stopping criterion of generate() looks at the whole sequence, the prompt included, and the assembled chain ends
    with a newline, so "\\nHop" matches after the first generated token and cuts every step after the first one (Round 41,
    first version)."""
    return lm.generate(prompts, max_new_tokens=60, do_sample=sample, temperature=0.7, n=n,
                       batch_size=max(1, args.bs // 2) if sample else args.bs, stop="\n")


def _gen_steps(lm: LM, prompts: List[str], args, sample: bool = False, n: int = 1):
    """(hop_line, so_line, is_answer) for each prompt (and each of the n samples): hop line, then running-relation line."""
    hops = [h.strip() for h in _line(lm, prompts, args, sample, n)]
    rep = [p for p in prompts for _ in range(n)] if n > 1 else prompts
    is_ans = [h.startswith("Answer") or not h.startswith("Hop") for h in hops]
    nxt = [p + h + "\n" for p, h, a in zip(rep, hops, is_ans)]
    sos = [x.strip() for x in _line(lm, nxt, args)]
    return list(zip(hops, sos, is_ans))


def decode_verified(lm: LM, exs: List[D.ClutrrExample], table: Dict, override: bool, args, vocab_l: List[str],
                    tag: str) -> Tuple[List[Optional[str]], List[Dict]]:
    n = len(exs)
    prompts = [lm.prompt(msgs_zero(STRUCT_SYSTEM, e)) for e in exs]
    chain, prev, done = [""] * n, [None] * n, [False] * n
    st = [Counter() for _ in range(n)]

    def analyse(hop, so):
        rel = None
        t = parse_triples(hop)
        if t and t[0][1]:
            rel = t[0][1]
        m = _SO.match(so)
        run = _rel_norm(m.group(3)) if m else None
        return rel, run

    for _ in range(max(e.k for e in exs) + 2):
        idx = [i for i in range(n) if not done[i]]
        if not idx:
            break
        base = [prompts[i] + chain[i] for i in idx]
        cur = dict(zip(idx, _gen_steps(lm, base, args)))
        flagged = []
        for i in idx:
            hop, so, isans = cur[i]
            if isans or not hop.startswith("Hop") or not _SO.match(so):
                continue
            rel, run = analyse(hop, so)
            exp = expected_running(table, prev[i], rel)
            if exp is not None and run != exp:
                flagged.append(i)
                st[i]["flagged"] += 1
        if flagged:
            alts = _gen_steps(lm, [prompts[i] + chain[i] for i in flagged], args, sample=True, n=args.retries)
            for t, i in enumerate(flagged):
                chosen = None
                for hop, so, isans in alts[t * args.retries:(t + 1) * args.retries]:
                    st[i]["retries"] += 1
                    if isans or not hop.startswith("Hop") or not _SO.match(so):
                        continue
                    rel, run = analyse(hop, so)
                    exp = expected_running(table, prev[i], rel)
                    if exp is None or run == exp:
                        chosen = (hop, so, isans)
                        st[i]["repaired"] += 1
                        break
                if chosen is not None:
                    cur[i] = chosen
                elif override:
                    hop, so, isans = cur[i]
                    rel, run = analyse(hop, so)
                    exp = expected_running(table, prev[i], rel)
                    if exp is not None:
                        so = re.sub(r"(['’]s\s+)[A-Za-z\- ]+?(\s*\.?\s*)$", lambda m: m.group(1) + exp + ".", so)
                        cur[i] = (hop, so, isans)
                        st[i]["overridden"] += 1
        for i in idx:
            hop, so, isans = cur[i]
            if isans or not hop.startswith("Hop") or not _SO.match(so):
                done[i] = True          # the model wrote the answer line (or something unparsable): the chain is complete
                continue
            rel, run = analyse(hop, so)
            chain[i] += hop + "\n" + so + "\n"
            prev[i] = run if run is not None else prev[i]
    ans = lm.generate([p + c + e.answer_prefix() for p, c, e in zip(prompts, chain, exs)], max_new_tokens=8,
                      batch_size=args.bs * 2, stop="\n")
    preds = []
    for i, a in enumerate(ans):
        p = _first_rel(a, vocab_l)
        if override and prev[i] in vocab_l:
            p = prev[i]
        preds.append(p)
    return preds, [dict(s) for s in st]


def greedy_and_sc(lm: LM, exs, args, vocab_l):
    prompts = [lm.prompt(msgs_zero(STRUCT_SYSTEM, e)) for e in exs]
    mx = lambda k: 44 * (k + 1) + 16
    chains, samples = [""] * len(exs), [[] for _ in exs]
    for k in sorted(set(e.k for e in exs)):
        idx = [i for i, e in enumerate(exs) if e.k == k]
        g = lm.generate([prompts[i] for i in idx], max_new_tokens=mx(k), batch_size=args.bs)
        sm = lm.generate([prompts[i] for i in idx], max_new_tokens=mx(k), do_sample=True, n=args.sc_n,
                         batch_size=max(1, args.bs // 2))
        for t, i in enumerate(idx):
            chains[i] = g[t].rstrip() + "\n"
            samples[i] = [x.rstrip() + "\n" for x in sm[t * args.sc_n:(t + 1) * args.sc_n]]
    flat, owner = [], []
    for i, e in enumerate(exs):
        for j, c in enumerate([chains[i]] + samples[i]):
            flat.append(prompts[i] + c + e.answer_prefix())
            owner.append((i, j))
    outs = lm.generate(flat, max_new_tokens=8, batch_size=args.bs * 2, stop="\n")
    by = {}
    for (i, j), o in zip(owner, outs):
        by.setdefault(i, {})[j] = _first_rel(o, vocab_l)
    greedy = [by[i][0] for i in range(len(exs))]
    sc = [_vote([by[i][j] for j in range(1, args.sc_n + 1)]) or greedy[i] for i in range(len(exs))]
    return chains, greedy, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sets", nargs="+", default=["c6", "c5"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sc_n", type=int, default=5)
    ap.add_argument("--small_frac", type=float, default=0.05)
    ap.add_argument("--noise_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train = D.load_clutrr("train")
    vocab_l = D.clutrr_relations(train)
    t_full = extend_table(train, induce_table(train), vocab_l)
    small = random.Random(args.seed).sample(train, max(50, int(args.small_frac * len(train))))
    t_small = extend_table(small, induce_table(small), vocab_l)
    t_noisy = corrupt_table(t_full, vocab_l, args.noise_frac, random.Random(args.seed + 1))
    agree = lambda a, b: float(np.mean([a.get(k) == v for k, v in b.items()]))
    _log(f"rules: full {len(t_full)}, small {len(t_small)} (agree with full on its rules: {agree(t_small, t_full):.2f}), "
         f"noisy {len(t_noisy)} (agree {agree(t_noisy, t_full):.2f})")
    pool = [e for e in train if 2 <= e.k <= 4]
    sample = D.sample_by_k(pool, [2, 3, 4], args.n_train // 3, args.seed)
    lm = LM(args.model, device_map=args.device_map)
    data = []
    for e in sample:
        t = chain_text(e, t_full, True)
        if t is not None:
            data.append((lm.prompt(msgs_zero(STRUCT_SYSTEM, e)), t))
    info = train_sft(lm, data, args)
    _log(f"sft {info}")
    json.dump(dict(info=info, n_rules=dict(full=len(t_full), small=len(t_small), noisy=len(t_noisy))),
              open(f"{args.outdir}/info.json", "w"))
    rows = []
    for name in args.sets:
        task, split, ks, npk, seed = SETS[name]
        exs = D.sample_by_k(D.load_clutrr(split, task=task), ks, npk, seed)
        cov = {nm: float(np.mean([covered(e, t) for e in exs])) for nm, t in
               [("full", t_full), ("small", t_small), ("noisy", t_noisy)]}
        _log(f"[{name}] {len(exs)} stories; gold chains covered by the table: {cov}")
        chains, greedy, sc = greedy_and_sc(lm, exs, args, vocab_l)
        vocab = set(vocab_l)
        sym = []
        for e, c, g in zip(exs, chains, greedy):
            r = chain_compose(t_full, parse_triples(c), e.source, e.target, vocab)[0]
            sym.append(r or g)
        base = {"greedy": greedy, "sc5": sc, "sym_full": sym}
        stats = {}
        for mth, table, ov in [("v_resample_full", t_full, False), ("v_override_full", t_full, True),
                               ("v_override_small", t_small, True), ("v_override_noisy", t_noisy, True)]:
            preds, st = decode_verified(lm, exs, table, ov, args, vocab_l, mth)
            base[mth] = preds
            stats[mth] = st
            _log(f"[{name}] {mth}: acc {100 * np.mean([p == e.answer for p, e in zip(preds, exs)]):.1f}")
        for i, e in enumerate(exs):
            trip = parse_triples(chains[i])
            gold_t = {(C.norm(h.head), h.rel, C.norm(h.tail)) for h in e.gold_hops}
            hops_ok = float(all((C.norm(x), r, C.norm(y)) in gold_t for x, r, y in trip)) if trip else 0.0
            runs = [_rel_norm(m.group(3)) for ln in chains[i].splitlines() if (m := _SO.match(ln))]
            rels = [t[1] for t in trip]
            comp_ok, prevr = 1.0, None
            for r, c in zip(rels, runs):
                ex_ = expected_running(t_full, prevr, r)
                if ex_ is not None and c != ex_:
                    comp_ok = 0.0
                prevr = c
            for mth, preds in base.items():
                s = stats.get(mth, [None] * len(exs))[i] or {}
                rows.append(dict(set=name, uid=e.uid, k=e.k, method=mth, pred=preds[i], gold=e.answer,
                                 correct=float(preds[i] == e.answer), hops_ok=hops_ok, comp_ok=comp_ok,
                                 flagged=s.get("flagged", 0), repaired=s.get("repaired", 0),
                                 overridden=s.get("overridden", 0)))
        pd.DataFrame(rows).to_csv(f"{args.outdir}/ruleverify.csv", index=False)
        d = pd.DataFrame(rows)
        print(d[d.set == name].groupby("method").correct.mean().round(3).to_string())
    _log("done")


if __name__ == "__main__":
    main()
