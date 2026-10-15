"""Experiment runners. Each runner returns plain-dict records that store the candidate
distributions, so every decoding rule / alpha is evaluated post hoc."""

from __future__ import annotations

import random
import re
import time
from typing import Dict, List, Optional

import numpy as np

from . import chain as C
from . import data as D
from .decoding import softmax_logp
from .prompts import (clutrr_messages, clutrr_question_only, musique_messages,
                      musique_question_only)


def _log(msg: str):
    print(time.strftime("[%H:%M:%S] "), msg, flush=True)


def _dist(lm, prefix: str, cands: List[str]) -> np.ndarray:
    return softmax_logp(lm.score(prefix, cands))


def _score_chain_contexts(lm, base_prompt: str, hops: List[C.Hop], ans_prefix: str,
                          cands: List[str], full: bool, marker: bool = False) -> Dict:
    """p(a | context, intervened chain) for every hop and intervention type."""
    def ctx(hs):
        return base_prompt + C.render_chain(hs) + ans_prefix
    rec = {"p_cot": _dist(lm, ctx(hops), cands),
           "p_tot": [_dist(lm, ctx(C.do_total(hops, j)), cands) for j in range(len(hops))]}
    if full:
        rec["p_nai"] = [_dist(lm, ctx(C.do_naive(hops, j)), cands) for j in range(len(hops))]
        rec["p_pla"] = [_dist(lm, ctx(C.placebo(hops, j)), cands) for j in range(len(hops))]
        if marker:
            rec["p_mm"] = [_dist(lm, ctx(C.mask_marker(hops, j)), cands) for j in range(len(hops))]
    else:
        rec["p_nai"] = rec["p_pla"] = None
    return rec


# ============================================================================ CLUTRR

def clutrr_score(lm, ex: D.ClutrrExample, chain_text: str, rels: List[str],
                 full: bool = True) -> Dict:
    cands = [" " + r for r in rels]
    hops = C.parse_chain(chain_text)
    base = lm.prompt(clutrr_messages(ex.story_text, ex.source, ex.target))
    rec = _score_chain_contexts(lm, base, hops, ex.answer_prefix(), cands, full)
    rec["p_dir"] = _dist(lm, base + ex.answer_prefix(), cands)
    rec["p_noctx"] = _dist(lm, lm.prompt(clutrr_question_only(ex.source, ex.target))
                           + ex.answer_prefix(), cands)
    rec["rel"] = C.relevant_hops(hops, target=ex.target)
    rec["drg_connected"] = bool(hops) and len(rec["rel"]) == len(hops)
    gold_tails = [h.tail for h in ex.gold_hops]
    res = [C.norm(h.result) for h in hops]
    rec.update(uid=ex.uid, k=ex.k, cond=ex.condition, gold=rels.index(ex.answer),
               n_hops=len(hops), fmt_ok=len(hops) > 0, chain=chain_text,
               hop_fid=float(np.mean([C.norm(t) in res for t in gold_tails])),
               cands=rels)
    return rec


def _gen_by_k(lm, prompts: List[str], ks: List[int], batch_size: int) -> List[str]:
    out = [None] * len(prompts)
    for k in sorted(set(ks)):
        idx = [i for i, kk in enumerate(ks) if kk == k]
        gens = lm.generate([prompts[i] for i in idx], max_new_tokens=26 * (k + 2) + 16,
                           batch_size=batch_size)
        for i, g in zip(idx, gens):
            out[i] = g
    return out


def run_clutrr(lm, examples: List[D.ClutrrExample], pool: List[D.ClutrrExample],
               rels: List[str], conditions=("orig", "rename", "order", "distract", "placebo_input"),
               batch_size: int = 8, l1: bool = False, l1_alpha: float = 1.0,
               l1_gate: str = "result", sc_n: int = 0) -> List[Dict]:
    """sc_n > 0 adds a self-consistency baseline (sc_n sampled chains, T=0.7, majority vote)
    on the 'orig' condition. It costs sc_n extra generations per question, i.e. strictly more
    compute than HGCD, which needs no extra generation."""
    recs = []
    for cond in conditions:
        exs = [D.apply_condition(e, cond, pool) for e in examples]
        prompts = [lm.prompt(clutrr_messages(e.story_text, e.source, e.target)) for e in exs]
        _log(f"CLUTRR [{cond}] generating {len(exs)} chains")
        gens = _gen_by_k(lm, prompts, [e.k for e in exs], batch_size)
        _log(f"CLUTRR [{cond}] scoring")
        votes = [None] * len(exs)
        if sc_n > 0 and cond == "orig":
            _log(f"CLUTRR [sc] sampling {sc_n} chains per question")
            cands = [" " + r for r in rels]
            for k in sorted(set(e.k for e in exs)):
                idx = [i for i, e in enumerate(exs) if e.k == k]
                samples = lm.generate([prompts[i] for i in idx], max_new_tokens=26 * (k + 2) + 16,
                                      do_sample=True, n=sc_n, batch_size=max(1, batch_size // 2))
                for t, i in enumerate(idx):
                    vs = []
                    for c in samples[t * sc_n:(t + 1) * sc_n]:
                        pre = prompts[i] + C.render_chain(C.parse_chain(c)) + exs[i].answer_prefix()
                        vs.append(int(np.argmax(lm.score(pre, cands))))
                    votes[i] = vs
        for e, g, v in zip(exs, gens, votes):
            r = clutrr_score(lm, e, g, rels, full=(cond == "orig"))
            r["chain_src"] = "greedy"
            r["sc_votes"] = v
            recs.append(r)
        if l1 and cond == "orig":
            _log(f"CLUTRR [L1] contrastive generation ({l1_gate}, a={l1_alpha})")
            cf = [lm.prompt(clutrr_messages(D.counterfactual_story(e), e.source, e.target))
                  for e in exs]
            order = sorted(range(len(exs)), key=lambda i: exs[i].k)
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                kmax = max(exs[i].k for i in idx)
                gl1 = lm.generate_contrastive([prompts[i] for i in idx], [cf[i] for i in idx],
                                              alpha=l1_alpha, gate=l1_gate,
                                              max_new_tokens=26 * (kmax + 2) + 16,
                                              batch_size=batch_size)
                for i, g in zip(idx, gl1):
                    r = clutrr_score(lm, exs[i], g, rels, full=False)
                    r["chain_src"] = "l1"
                    recs.append(r)
    return recs


# ============================================================================ estimator validation

def _corrupt_clutrr_hop(h: C.Hop, rels: List[str], rng: random.Random) -> C.Hop:
    m = C.CLUTRR_HOP_RE.match(h.body)
    if not m:
        return C.Hop(h.idx, C.MASK, h.result)
    alt = rng.choice([r for r in rels if r != m.group(2)])
    return C.Hop(h.idx, f"{m.group(1)}'s {alt} is {m.group(3)}.", h.result)


def run_estimators_clutrr(lm, examples: List[D.ClutrrExample], pool: List[D.ClutrrExample],
                          rels: List[str], rollouts: int = 0, batch_size: int = 8,
                          seed: int = 0) -> List[Dict]:
    """Gold chains + one inserted true-but-redundant hop (true necessity 0)."""
    cands = [" " + r for r in rels]
    rows = []
    for n, ex in enumerate(examples):
        rng = random.Random(f"est-{ex.uid}-{seed}")
        ex2, red = D.with_redundant_fact(ex, pool, seed)
        hops = [C.Hop(i + 1, h.text(), h.tail) for i, h in enumerate(ex.gold_hops)]
        rpos = rng.randint(0, len(hops) - 1)
        hops.insert(rpos, C.Hop(0, red.text(), red.tail))
        base = lm.prompt(clutrr_messages(ex2.story_text, ex2.source, ex2.target))
        pre = ex2.answer_prefix()
        g = rels.index(ex.answer)
        recs = _score_chain_contexts(lm, base, hops, pre, cands, full=True, marker=True)
        p_full = recs["p_cot"]
        rel = C.relevant_hops(hops, target=ex.target)
        pns = [np.nan] * len(hops)
        if rollouts > 0:
            parts = [hops[:j] + [_corrupt_clutrr_hop(hops[j], rels, rng)] for j in range(len(hops))]
            prompts = [base + C.render_chain(part) for part in parts]
            conts = lm.generate(prompts, max_new_tokens=26 * (len(hops) + 1) + 16,
                                do_sample=True, n=rollouts, batch_size=max(1, batch_size // 2))
            y_full = float(np.argmax(p_full) == g)
            for j in range(len(hops)):
                ys = []
                for c in conts[j * rollouts:(j + 1) * rollouts]:
                    allh = parts[j] + C.parse_chain(c)
                    p = _dist(lm, base + C.render_chain(allh) + pre, cands)
                    ys.append(float(np.argmax(p) == g))
                pns[j] = y_full - float(np.mean(ys))
        for j in range(len(hops)):
            rows.append(dict(
                dataset="clutrr", uid=ex.uid, k=ex.k, hop=j, n_hops=len(hops),
                label=int(j != rpos), is_final=int(j == len(hops) - 1),
                drg_relevant=int(j in rel), p_full_gold=float(p_full[g]),
                full_correct=int(np.argmax(p_full) == g),
                d_total=float(p_full[g] - recs["p_tot"][j][g]),
                d_naive=float(p_full[g] - recs["p_nai"][j][g]),
                d_placebo=float(p_full[g] - recs["p_pla"][j][g]),
                d_maskmark=float(p_full[g] - recs["p_mm"][j][g]),
                pns_rollout=pns[j]))
        if (n + 1) % 10 == 0:
            _log(f"estimators clutrr {n + 1}/{len(examples)}")
    return rows


def _musique_cands(ex: D.MusiqueExample, extra: List[str]) -> List[str]:
    seen, out = set(), []
    for c in extra + [ex.answer] + [h[1] for h in ex.gold_hops] + \
            [p["title"] for p in ex.context()]:
        k = C.norm(c)
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def run_estimators_musique(lm, examples: List[D.MusiqueExample], seed: int = 0) -> List[Dict]:
    """MuSiQue gold chains (bridge-entity graphs) + an inserted redundant hop."""
    rows = []
    rng = random.Random(seed)
    for n, ex in enumerate(examples):
        other = rng.choice([o for o in examples if o.uid != ex.uid])
        hops = [C.Hop(i + 1, q, a) for i, (q, a) in enumerate(ex.gold_hops)]
        rpos = rng.randint(0, len(hops) - 1)
        hops.insert(rpos, C.Hop(0, other.gold_hops[0][0], other.gold_hops[0][1]))
        cands = [" " + c for c in _musique_cands(ex, [])]
        base = lm.prompt(musique_messages(ex.context_text(), ex.question))
        recs = _score_chain_contexts(lm, base, hops, ex.answer_prefix(), cands, full=True,
                                     marker=True)
        g = [C.norm(c) for c in cands].index(C.norm(ex.answer))
        p_full = recs["p_cot"]
        rel = C.relevant_hops(hops)
        for j in range(len(hops)):
            rows.append(dict(
                dataset="musique", uid=ex.uid, k=ex.k, hop=j, n_hops=len(hops),
                label=int(j != rpos), is_final=int(j == len(hops) - 1),
                drg_relevant=int(j in rel), p_full_gold=float(p_full[g]),
                full_correct=int(np.argmax(p_full) == g),
                d_total=float(p_full[g] - recs["p_tot"][j][g]),
                d_naive=float(p_full[g] - recs["p_nai"][j][g]),
                d_placebo=float(p_full[g] - recs["p_pla"][j][g]),
                d_maskmark=float(p_full[g] - recs["p_mm"][j][g]),
                pns_rollout=np.nan))
        if (n + 1) % 10 == 0:
            _log(f"estimators musique {n + 1}/{len(examples)}")
    return rows


# ============================================================================ MuSiQue decoding

def run_musique(lm, examples: List[D.MusiqueExample], conditions=("orig", "dist8", "order"),
                batch_size: int = 4) -> List[Dict]:
    recs = []
    for cond in conditions:
        exs = [D.musique_condition(e, cond) for e in examples]
        prompts = [lm.prompt(musique_messages(e.context_text(), e.question)) for e in exs]
        _log(f"MuSiQue [{cond}] generating {len(exs)} chains")
        gens = lm.generate(prompts, max_new_tokens=200, batch_size=batch_size)
        p2 = [p + C.render_chain(C.parse_chain(g)) + "Answer:" for p, g in zip(prompts, gens)]
        free = lm.generate(p2, max_new_tokens=24, batch_size=batch_size, stop="\n")
        _log(f"MuSiQue [{cond}] scoring")
        for e, p, g, fa in zip(exs, prompts, gens, free):
            hops = C.parse_chain(g)
            fa = fa.strip().strip(".")
            cand_txt = _musique_cands_model(e, [fa] + [h.result for h in hops])
            cands = [" " + c for c in cand_txt]
            r = _score_chain_contexts(lm, p, hops, e.answer_prefix(), cands, full=(cond == "orig"))
            r["sc_votes"] = None
            r["p_dir"] = _dist(lm, p + e.answer_prefix(), cands)
            r["p_noctx"] = _dist(lm, lm.prompt(musique_question_only(e.question))
                                 + e.answer_prefix(), cands)
            r["rel"] = C.relevant_hops(hops)
            r.update(uid=e.uid, k=e.k, cond=cond, cands=cand_txt, gold_answer=e.answer,
                     aliases=e.aliases, free_answer=fa, n_hops=len(hops),
                     fmt_ok=len(hops) > 0, chain=g, chain_src="greedy")
            recs.append(r)
    return recs


_CAPS = re.compile(r"[A-Z][\w'\u2019\-\.]*(?:\s+(?:of|the|de|la|and|du|von|van|&)?\s*[A-Z][\w'\u2019\-\.]*)*")
_NUMS = re.compile(r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|"
                   r"October|November|December)?\s*\d{3,4}\b|\b\d+(?:[.,]\d+)?\b")


_STOP = set("""a an the in on at for of to by from with as after before during while when where
which who whom whose what why how it its he she they his her their this that these those there here
most many some other others also however although though but and or if then thus since because
both each every all any one first last later early during despite following according""".split())


def _musique_cands_model(ex: D.MusiqueExample, model_strings: List[str], cap: int = 64) -> List[str]:
    """Candidates without access to gold: model answers and hop results first, then paragraph
    titles, then capitalised spans and numbers/dates from the context (order of appearance)."""
    pool = list(model_strings) + [p["title"] for p in ex.context()]
    for p in ex.context():
        t = p["paragraph_text"]
        pool += [m.strip() for m in _CAPS.findall(t) + _NUMS.findall(t)]
    seen, out = set(), []
    for c in pool:
        k = C.norm(c)
        if k and k not in seen and c != C.MASK and k not in _STOP:
            seen.add(k)
            out.append(c)
    return out[:cap]
