"""Decoding that injects causal / structural information *before or during* chain generation.

Motivation (paper, R3-R5): answers are read from the chain, not computed over it, and distractor
harm is mediated by the regenerated chain. Readout-level contrast cannot help, so the methods here
act on the two places the causal analysis points to:

CLUTRR  (make the answer a deterministic function of the chain, i.e. force use of the chain)
  sym        parse the model's own hop chain into (head, relation, tail) triples and compose the
             relations with a kinship composition table induced from the *training* split
  sym_sc     the same over the sampled chains (compute-matched with sc), majority vote over valid
             compositions, falling back to sc when no sampled chain composes
MuSiQue (cut the spurious path distractor -> chain before the chain is generated)
  attn_sel   rank paragraphs by the model's own attention from the question to each paragraph,
             calibrated by a content-free question (ICR-style), keep the top K, then CoT
  self_sel   the model lists the relevant paragraph numbers, keep those, then CoT
  last_hop   answer = result of the chain's last hop (answer depends on the chain by construction)

Baselines: greedy CoT with free-form answer (cot_free), self-consistency over sampled chains (sc),
label scoring after CoT on CLUTRR (cot_score); oracle ceilings (sym_gold, gold_only) are reported
but are not methods.

    python -m hgcd.causaldec --task clutrr  --model Qwen/Qwen2.5-3B-Instruct --outdir out/c3b
    python -m hgcd.causaldec --task musique --model Qwen/Qwen2.5-3B-Instruct --outdir out/m3b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import chain as C
from . import data as D
from . import metrics as M
from .prompts import clutrr_messages, musique_messages

T0 = time.time()


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ CLUTRR: composition

def compose_counts(table: Dict[Tuple[str, str], str], rels: Tuple[str, ...]) -> Counter:
    """Results of composing `rels` over every binary bracketing (count = #bracketings)."""
    n = len(rels)
    memo: Dict[Tuple[int, int], Counter] = {}

    def f(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        if j - i == 1:
            out = Counter({rels[i]: 1})
        else:
            out = Counter()
            for m in range(i + 1, j):
                for a, ca in f(i, m).items():
                    for b, cb in f(m, j).items():
                        r = table.get((a, b))
                        if r is not None:
                            out[r] += ca * cb
        memo[(i, j)] = out
        return out
    return f(0, n) if n else Counter()


def compose(table, rels) -> Optional[str]:
    c = compose_counts(table, tuple(rels))
    return c.most_common(1)[0][0] if c else None


def induce_table(train: List[D.ClutrrExample], iters: int = 8, purity: float = 0.9) -> Dict:
    """Pairwise composition rules (r1, r2) -> r induced from training stories only: a split of a
    training chain whose two halves each compose to a single relation yields one observation."""
    table: Dict[Tuple[str, str], str] = {}
    for _ in range(iters):
        obs = defaultdict(Counter)
        for e in train:
            rels = tuple(h.rel for h in e.gold_hops)
            for m in range(1, len(rels)):
                left, right = compose_counts(table, rels[:m]), compose_counts(table, rels[m:])
                if len(left) == 1 and len(right) == 1:
                    obs[(next(iter(left)), next(iter(right)))][e.answer] += 1
        new = {k: v.most_common(1)[0][0] for k, v in obs.items()
               if v.most_common(1)[0][1] >= purity * sum(v.values())}
        if all(table.get(k) == r for k, r in new.items()):
            break
        table.update(new)
    return table


_HOP_REL = re.compile(r"^\s*(.+?)['’]s?\s+([A-Za-z][A-Za-z\- ]*?)\s+is\s+(.+?)\s*\.?\s*$")
# "X has a <relation> named Y."
_HOP_HAS = re.compile(r"^\s*([A-Z][\w\-]*)\s+has\s+an?\s+([A-Za-z][A-Za-z\- ]*?)\s+(?:named|called)\s+"
                      r"([A-Z][\w\-]*)\s*\.?\s*$")
# the same fact written the other way round: "Y is X's <relation>."
_HOP_REL_INV = re.compile(r"^\s*([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+is\s+(.+?)['’]s?\s+"
                          r"([A-Za-z][A-Za-z\- ]*?)\s*\.?\s*$")


def _rel_norm(r: str) -> str:
    r = r.lower().strip()
    r = re.sub(r"\s*-\s*", "-", r)
    r = re.sub(r"\s+in\s+law$", "-in-law", r)
    r = r.replace("grand ", "grand")
    return r


def parse_triples(chain_text: str) -> List[Tuple[str, str, str]]:
    out = []
    for h in C.parse_chain(chain_text):
        m = _HOP_REL.match(h.body)
        mi = _HOP_REL_INV.match(h.body)
        mh = _HOP_HAS.match(h.body)
        if mh:
            out.append((mh.group(1).strip(), _rel_norm(mh.group(2)), mh.group(3).strip()))
        elif mi:
            out.append((mi.group(2).strip(), _rel_norm(mi.group(3)), mi.group(1).strip()))
        elif m:
            out.append((m.group(1).strip(), _rel_norm(m.group(2)), m.group(3).strip().strip(".")))
        else:
            out.append(("", "", ""))
    return out


def chain_compose(table, triples, src: str, tgt: str, vocab: set) -> Tuple[Optional[str], str]:
    """Answer computed from the chain alone, or None with the reason it is not computable."""
    if not triples:
        return None, "no_hops"
    if any(not t[1] for t in triples):
        return None, "unparsed"
    if any(t[1] not in vocab for t in triples):
        return None, "rel_oov"
    n = C.norm
    if n(triples[0][0]) != n(src):
        return None, "bad_start"
    for a, b in zip(triples, triples[1:]):
        if n(a[2]) != n(b[0]):
            return None, "broken"
    if n(triples[-1][2]) != n(tgt):
        return None, "bad_end"
    r = compose(table, [t[1] for t in triples])
    return (r, "ok") if r else (None, "no_rule")


_FEMALE = {"mother", "daughter", "sister", "wife", "grandmother", "granddaughter", "aunt", "niece",
           "mother-in-law", "daughter-in-law"}
# inverse relation by the gender of the person it now describes: (rel) -> (male, female)
_INV = {"father": ("son", "daughter"), "mother": ("son", "daughter"),
        "son": ("father", "mother"), "daughter": ("father", "mother"),
        "brother": ("brother", "sister"), "sister": ("brother", "sister"),
        "husband": ("husband", "wife"), "wife": ("husband", "wife"),
        "grandfather": ("grandson", "granddaughter"), "grandmother": ("grandson", "granddaughter"),
        "grandson": ("grandfather", "grandmother"), "granddaughter": ("grandfather", "grandmother"),
        "uncle": ("nephew", "niece"), "aunt": ("nephew", "niece"),
        "nephew": ("uncle", "aunt"), "niece": ("uncle", "aunt"),
        "father-in-law": ("son-in-law", "daughter-in-law"), "mother-in-law": ("son-in-law", "daughter-in-law"),
        "son-in-law": ("father-in-law", "mother-in-law"),
        "daughter-in-law": ("father-in-law", "mother-in-law")}


def graph_compose(table, triples, src: str, tgt: str, vocab: set,
                  gender_prior: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
    """Treat every fact the model wrote as an edge of a kinship graph (order-free), add inverse
    edges when the gender they need is stated by the model's own relation words, find the shortest
    path src -> tgt and compose its relations with the rule table."""
    n = C.norm
    facts = [(n(a), r, n(b)) for a, r, b in triples if r in vocab and a and b and n(a) != n(b)]
    if not facts:
        return None, "no_facts"
    gender = dict(gender_prior or {})       # 'm' / 'f' by normalised name; stated facts override
    for a, r, b in facts:
        gender[b] = "f" if r in _FEMALE else "m"
    adj = defaultdict(list)
    for a, r, b in facts:
        adj[a].append((r, b))
        if a in gender and r in _INV:      # "a's r is b"  =>  "b's r' is a", r' gendered by a
            adj[b].append((_INV[r][gender[a] == "f"], a))
    s, t = n(src), n(tgt)
    prev, frontier = {s: None}, [s]
    while frontier and t not in prev:
        nxt = []
        for u in frontier:
            for r, v in adj[u]:
                if v not in prev:
                    prev[v] = (u, r)
                    nxt.append(v)
        frontier = nxt
    if t not in prev:
        return None, "no_path"
    rels, v = [], t
    while prev[v] is not None:
        u, r = prev[v]
        rels.append(r)
        v = u
    r = compose(table, rels[::-1])
    return (r, "ok") if r else (None, "no_rule")


EXTRACT_SYSTEM = ("List every family relation stated in the story, one per line, in the form "
                  "\"X's <relation> is Y.\" Use only these relation words: {rels}. "
                  "Write only facts the story states; do not infer or answer anything.")
EXTRACT_SHOTS = [
    ("Mabel took her son Rupert to the zoo. Rupert and his sister Agatha baked a cake for their "
     "grandmother.", ["Mabel's son is Rupert.", "Rupert's sister is Agatha."]),
    ("Clement went fishing with his father Bertram. Bertram called his sister Winifred to wish her a "
     "happy birthday. Winifred was proud of her daughter Odette.",
     ["Clement's father is Bertram.", "Bertram's sister is Winifred.", "Winifred's daughter is Odette."]),
]


EXTRACT_GENDER_NOTE = (" Then write one last line 'Genders:' giving every person's gender as stated "
                       "by the story's pronouns, e.g. 'Genders: Ann=female, Bob=male'.")
EXTRACT_SHOT_GENDERS = ["Genders: Mabel=female, Rupert=male, Agatha=female",
                        "Genders: Clement=male, Bertram=male, Winifred=female, Odette=female"]


def extract_messages(story: str, rels: List[str], genders: bool = False) -> List[dict]:
    sysm = EXTRACT_SYSTEM.format(rels=", ".join(rels)) + (EXTRACT_GENDER_NOTE if genders else "")
    msgs = [{"role": "system", "content": sysm}]
    for (st, facts), gl in zip(EXTRACT_SHOTS, EXTRACT_SHOT_GENDERS):
        msgs.append({"role": "user", "content": f"Story: {st}"})
        msgs.append({"role": "assistant", "content": "\n".join(facts + ([gl] if genders else []))})
    msgs.append({"role": "user", "content": f"Story: {story}"})
    return msgs


def parse_genders(text: Optional[str]) -> Dict[str, str]:
    out = {}
    for ln in (text or "").splitlines():
        if ln.strip().lower().startswith("genders"):
            for name, g in re.findall(r"([A-Z][\w\-]*)\s*[=:]\s*(male|female|man|woman|boy|girl)", ln, re.I):
                out[C.norm(name)] = "f" if g.lower() in ("female", "woman", "girl") else "m"
    return out


def induce_gender_lexicon(train: List[D.ClutrrExample], purity: float = 0.95) -> Dict[str, str]:
    """Name -> gender learned from the training split only (CLUTRR reuses a fixed name pool)."""
    lex = defaultdict(Counter)
    for e in train:
        for k, v in e.genders.items():
            lex[C.norm(k)][v] += 1
    return {k: ("f" if c.most_common(1)[0][0] == "female" else "m") for k, c in lex.items()
            if c.most_common(1)[0][1] >= purity * sum(c.values())}


def parse_facts(text: Optional[str]) -> List[Tuple[str, str, str]]:
    """Facts written one per line (no 'Hop i:' prefix needed)."""
    lines = [ln.strip().lstrip("-*0123456789. ").strip() for ln in (text or "").splitlines()]
    return parse_triples("".join(f"Hop {i + 1}: {ln} => x\n" for i, ln in enumerate(lines) if ln))


def _first_rel(text: str, vocab: List[str]) -> Optional[str]:
    t = " " + text.lower() + " "
    best = None
    for r in vocab:
        m = re.search(r"(?<![a-z\-])" + re.escape(r) + r"(?![a-z\-])", t)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(r) > len(best[1]))):
            best = (m.start(), r)
    return best[1] if best else None


def _vote(xs: List[Optional[str]], key=lambda s: s) -> Optional[str]:
    xs = [x for x in xs if x]
    if not xs:
        return None
    c = Counter(key(x) for x in xs)
    top = c.most_common(1)[0][0]
    return next(x for x in xs if key(x) == top)


def run_clutrr(lm, args) -> pd.DataFrame:
    train, test = D.load_clutrr("train"), D.load_clutrr("test")
    table = induce_table(train)
    vocab_l = D.clutrr_relations(train)
    vocab = set(vocab_l)
    _log(f"composition table: {len(table)} rules from {len(train)} training stories")
    glex = induce_gender_lexicon(train)
    pool = (test if (args.split == "test" and args.task_name == D.CLUTRR_TASK)
            else D.load_clutrr(args.split, task=args.task_name))
    ex = D.sample_by_k(pool, args.ks, args.n_per_k, args.seed)
    cov = np.mean([C.norm(n) in glex for e in ex for n in e.names])
    _log(f"gender lexicon: {len(glex)} names from train; covers {cov:.2f} of people in this sample")
    _log(f"CLUTRR test: {len(ex)} examples, k in {sorted(set(e.k for e in ex))}")
    rows = []
    for s in range(0, len(ex), args.chunk):
        if (time.time() - T0) / 60 > args.max_minutes:
            _log("time budget reached; stopping early")
            break
        part = ex[s:s + args.chunk]
        prompts = [lm.prompt(clutrr_messages(e.story_text, e.source, e.target)) for e in part]
        mx = [26 * (e.k + 2) + 16 for e in part]
        greedy, samples, extra_s = [None] * len(part), [None] * len(part), [None] * len(part)
        nsm = args.sc_n + args.sc_extra if args.sc_n else 0   # extra samples: compute-matched SC
        for k in sorted(set(e.k for e in part)):            # batch by k: similar lengths
            idx = [i for i, e in enumerate(part) if e.k == k]
            g = lm.generate([prompts[i] for i in idx], max_new_tokens=mx[idx[0]], batch_size=args.bs)
            sm = lm.generate([prompts[i] for i in idx], max_new_tokens=mx[idx[0]], do_sample=True,
                             n=nsm, batch_size=max(1, args.bs // 2)) if nsm else []
            for t, i in enumerate(idx):
                greedy[i] = g[t]
                samples[i] = sm[t * nsm:t * nsm + args.sc_n] if nsm else []
                extra_s[i] = sm[t * nsm + args.sc_n:(t + 1) * nsm] if nsm else []
        # fact extraction (no reasoning asked): greedy + sampled, then graph search + rule table
        ext_g, ext_s = [None] * len(part), [None] * len(part)
        if args.extract:
            ep = [lm.prompt(extract_messages(e.story_text, vocab_l, genders=bool(args.extract_genders)))
                  for e in part]
            mxe = [20 * (e.k + 3) for e in part]
            for k in sorted(set(e.k for e in part)):
                idx = [i for i, e in enumerate(part) if e.k == k]
                g = lm.generate([ep[i] for i in idx], max_new_tokens=mxe[idx[0]], batch_size=args.bs,
                                stop="\n\n")
                sm = lm.generate([ep[i] for i in idx], max_new_tokens=mxe[idx[0]], do_sample=True,
                                 n=args.sc_n, batch_size=max(1, args.bs // 2), stop="\n\n") if args.sc_n else []
                for t, i in enumerate(idx):
                    ext_g[i] = g[t]
                    ext_s[i] = sm[t * args.sc_n:(t + 1) * args.sc_n] if args.sc_n else []
        # free-form answer after each chain (greedy and sampled), one batched pass
        ans_prompts, owner = [], []
        for i, e in enumerate(part):
            for j, ch in enumerate([greedy[i]] + samples[i] + extra_s[i]):
                ans_prompts.append(prompts[i] + C.render_chain(C.parse_chain(ch)) + e.answer_prefix())
                owner.append((i, j))
        free = lm.generate(ans_prompts, max_new_tokens=8, batch_size=args.bs * 2, stop="\n")
        free_by = defaultdict(dict)
        for (i, j), f in zip(owner, free):
            free_by[i][j] = _first_rel(f, vocab_l)
        for i, e in enumerate(part):
            golds = [e.answer]
            chains = [greedy[i]] + samples[i]
            sym = [chain_compose(table, parse_triples(c), e.source, e.target, vocab) for c in chains]
            score_pre = prompts[i] + C.render_chain(C.parse_chain(greedy[i])) + e.answer_prefix()
            cot_score = vocab_l[int(np.argmax(lm.score(score_pre, [" " + r for r in vocab_l])))]
            gold_sym = compose(table, [h.rel for h in e.gold_hops])
            trip = parse_triples(greedy[i])
            gold_t = {(C.norm(h.head), h.rel, C.norm(h.tail)) for h in e.gold_hops}
            fid = np.mean([(C.norm(a), r, C.norm(b)) in gold_t for a, r, b in trip]) if trip else 0.0
            gsym = [graph_compose(table, parse_triples(c), e.source, e.target, vocab) for c in chains]
            sc_vote = _vote([free_by[i][j] for j in range(1, len(chains))]) if samples[i] else free_by[i][0]
            preds_ext = {}
            if args.extract:
                ef = [parse_facts(x) for x in [ext_g[i]] + ext_s[i]]
                gm = [parse_genders(x) for x in [ext_g[i]] + ext_s[i]]
                ex_res = [graph_compose(table, f, e.source, e.target, vocab) for f in ef]
                lex_res = [graph_compose(table, f, e.source, e.target, vocab, glex) for f in ef]
                g_res = [graph_compose(table, f, e.source, e.target, vocab, g) for f, g in zip(ef, gm)]
                gl_res = [graph_compose(table, f, e.source, e.target, vocab, {**glex, **g})
                          for f, g in zip(ef, gm)]
                union = [t for f in ef for t in f]
                eu = graph_compose(table, union, e.source, e.target, vocab)
                preds_ext = {"ext": ex_res[0][0] or free_by[i][0], "ext_only": ex_res[0][0],
                             "ext_sc": _vote([r for r, _ in ex_res[1:]]) or sc_vote,
                             "ext_union": eu[0] or free_by[i][0],
                             "ext_all": _vote([r for r, _ in ex_res]) or _vote([r for r, _ in gsym]) or sc_vote,
                             "ext_lex": lex_res[0][0] or free_by[i][0],
                             "ext_g": g_res[0][0] or free_by[i][0],
                             "ext_glex": gl_res[0][0] or free_by[i][0],
                             "ext_g_sc": _vote([r for r, _ in g_res[1:]]) or sc_vote,
                             "ext_glex_sc": _vote([r for r, _ in gl_res[1:]]) or sc_vote,
                             "ext_glex_all": _vote([r for r, _ in gl_res]) or sc_vote}
            preds = {
                "graph": gsym[0][0] or free_by[i][0],
                "graph_sc": _vote([r for r, _ in gsym[1:]]) or sc_vote,
                **preds_ext,
                "cot_free": free_by[i][0],
                "cot_score": cot_score,
                "sc": _vote([free_by[i][j] for j in range(1, len(chains))]) if samples[i] else None,
                "sc_plus": _vote([free_by[i][j] for j in range(1, len(chains) + len(extra_s[i]))])
                if samples[i] else None,
                "sym": sym[0][0] or free_by[i][0],
                "sym_only": sym[0][0],
                "sym_sc": _vote([r for r, _ in sym[1:]]) or (
                    _vote([free_by[i][j] for j in range(1, len(chains))]) if samples[i] else free_by[i][0]),
                "sym_gold": gold_sym,
            }
            for mth, p in preds.items():
                rows.append(dict(uid=e.uid, k=e.k, method=mth, pred=p, gold=e.answer,
                                 correct=float(p == e.answer), sym_status=sym[0][1],
                                 n_valid_sym=sum(r is not None for r, _ in sym), hop_fidelity=fid,
                                 graph_status=gsym[0][1],
                                 ext_status=(graph_compose(table, parse_facts(ext_g[i]), e.source, e.target,
                                                           vocab)[1] if args.extract else None),
                                 chain=greedy[i] if mth == "cot_free" else None,
                                 extract=ext_g[i] if mth == "cot_free" else None))
        _log(f"  {min(s + args.chunk, len(ex))}/{len(ex)} done "
             f"({(time.time() - T0) / 60:.0f} min)")
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
    return pd.DataFrame(rows)


# ============================================================================ MuSiQue: evidence selection

def _ctx_text(paras: List[dict]) -> str:
    return "\n".join(f"[{i + 1}] {p['title']}: {p['paragraph_text']}" for i, p in enumerate(paras))


def _set_attn(model, impl: str) -> str:
    old = getattr(model.config, "_attn_implementation", "sdpa")
    for m in [model] + [x for x in model.modules() if hasattr(x, "config")]:
        try:
            m.config._attn_implementation = impl
        except Exception:
            pass
    return old


@torch.no_grad()
def attention_scores(lm, paras: List[dict], question: str) -> np.ndarray:
    """Attention mass from the question tokens to each paragraph, summed over layers and heads,
    minus the same quantity for a content-free question (calibration as in ICR, ICLR 2025)."""
    ctx = _ctx_text(paras)
    full = lm.prompt(musique_messages(ctx, question))
    null = lm.prompt(musique_messages(ctx, "N/A"))
    cut = full.rfind("\nQuestion: ")
    prefix = full[:cut]
    assert null.startswith(prefix)
    enc = lm.tok(prefix, return_offsets_mapping=True, add_special_tokens=False)
    offs = np.array([o[0] for o in enc["offset_mapping"]])
    base = prefix.rfind("Paragraphs:\n")
    spans, pos = [], base
    for i, p in enumerate(paras):
        line = f"[{i + 1}] {p['title']}: {p['paragraph_text']}"
        a = prefix.index(line, pos)
        spans.append((a, a + len(line)))
        pos = a + len(line)
    tok_spans = [np.where((offs >= a) & (offs < b))[0] for a, b in spans]
    ids = torch.tensor([enc["input_ids"]], device=lm.device)
    past = lm.model(input_ids=ids, use_cache=True).past_key_values
    L = ids.shape[1]
    res = []
    old = _set_attn(lm.model, "eager")
    try:
        for text in (full[cut:], null[cut:]):
            sid = lm.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
            out = lm.model(input_ids=sid, past_key_values=copy.deepcopy(past), use_cache=True,
                           output_attentions=True,
                           cache_position=torch.arange(L, L + sid.shape[1], device=lm.device))
            if out.attentions is None or out.attentions[0] is None:
                raise RuntimeError("model returned no attentions")
            a = sum(x[0, :, :, :L].float().sum(0).mean(0).cpu() for x in out.attentions).numpy()
            res.append(np.array([a[t].sum() if len(t) else 0.0 for t in tok_spans]))
    finally:
        _set_attn(lm.model, old)
    return res[0] - res[1]


SELECT_SYSTEM = ("You are given numbered paragraphs and a multi-hop question. Do not answer the "
                 "question. List the numbers of all paragraphs needed to answer it, in the form "
                 "'Relevant: 2, 5'.")


def _chain_and_answer(lm, contexts: List[str], questions: List[str], bs: int, sample_n: int = 0):
    prompts = [lm.prompt(musique_messages(c, q)) for c, q in zip(contexts, questions)]
    if sample_n:
        chains = lm.generate(prompts, max_new_tokens=200, do_sample=True, n=sample_n,
                             batch_size=max(1, bs // 2))
        prompts = [p for p in prompts for _ in range(sample_n)]
    else:
        chains = lm.generate(prompts, max_new_tokens=200, batch_size=bs)
    p2 = [p + C.render_chain(C.parse_chain(g)) + "Answer:" for p, g in zip(prompts, chains)]
    ans = [a.strip().strip(".") for a in lm.generate(p2, max_new_tokens=24, batch_size=bs, stop="\n")]
    return chains, ans


def run_musique(lm, args) -> pd.DataFrame:
    ms = D.load_musique("dev")
    rng = np.random.default_rng(args.seed + 3)          # same draw as the mediation runs
    ex = [ms[i] for i in rng.choice(len(ms), args.n, replace=False)]
    _log(f"MuSiQue: {len(ex)} examples x {args.conditions}")
    rows = []
    for cond in args.conditions:
        exs = [D.musique_condition(e, cond) for e in ex]
        for s in range(0, len(exs), args.chunk):
            if (time.time() - T0) / 60 > args.max_minutes:
                _log("time budget reached; stopping early")
                break
            part = exs[s:s + args.chunk]
            paras = [e.context() for e in part]
            qs = [e.question for e in part]
            golds = [[e.answer] + list(e.aliases) for e in part]
            preds: Dict[str, List] = {}
            extra: Dict[str, List] = defaultdict(lambda: [None] * len(part))
            # baseline: greedy CoT + free answer; last-hop readout from the same chain
            ch, an = _chain_and_answer(lm, [_ctx_text(p) for p in paras], qs, args.bs)
            preds["cot_free"] = an
            preds["last_hop"] = [(C.parse_chain(c)[-1].result if C.parse_chain(c) else a)
                                 for c, a in zip(ch, an)]
            # self-consistency over sampled chains
            if args.sc_n:
                _, sa = _chain_and_answer(lm, [_ctx_text(p) for p in paras], qs, args.bs, args.sc_n)
                preds["sc"] = [_vote(sa[i * args.sc_n:(i + 1) * args.sc_n], M._norm_ans)
                               for i in range(len(part))]
            M_ = set(args.ms_methods)
            # attention scores (computed once, reused by every attention variant)
            scores = [None] * len(part)
            if M_ & {"attn_sel", "attn_rerank", "attn_drop2"}:
                for i, p in enumerate(paras):
                    try:
                        scores[i] = attention_scores(lm, p, qs[i])
                    except Exception as err:          # never lose the run to one example
                        _log(f"attention scoring failed ({err}); keeping all paragraphs")
                        scores[i] = np.zeros(len(p))

            def _attn_ctx(name, pick):
                ctxs = []
                for i, p in enumerate(paras):
                    keep = pick(scores[i], len(p))
                    kept = [p[j] for j in keep]
                    sup = [q for q in p if q["is_supporting"]]
                    extra[name + "_recall"][i] = sum(q["is_supporting"] for q in kept) / max(1, len(sup))
                    ctxs.append(_ctx_text(kept))
                return ctxs
            if "attn_sel" in M_:     # hard filter: top-K paragraphs, original order
                c = _attn_ctx("attn", lambda sc, n: sorted(np.argsort(-sc)[:args.top_k].tolist()))
                _, preds["attn_sel"] = _chain_and_answer(lm, c, qs, args.bs)
            if "attn_rerank" in M_:  # soft: keep every paragraph, most-attended first
                c = _attn_ctx("rerank", lambda sc, n: np.argsort(-sc).tolist())
                _, preds["attn_rerank"] = _chain_and_answer(lm, c, qs, args.bs)
            if "attn_drop2" in M_:   # mild filter: drop only the two least-attended paragraphs
                c = _attn_ctx("drop2", lambda sc, n: sorted(np.argsort(-sc)[:max(1, n - 2)].tolist()))
                _, preds["attn_drop2"] = _chain_and_answer(lm, c, qs, args.bs)
            if "self_sel" in M_:     # self-selection by prompting, then CoT
                sp = [lm.prompt([{"role": "system", "content": SELECT_SYSTEM},
                                 {"role": "user", "content": f"Paragraphs:\n{_ctx_text(p)}\nQuestion: {q}"}],
                                assistant_prefix="Relevant:") for p, q in zip(paras, qs)]
                raw = lm.generate(sp, max_new_tokens=24, batch_size=args.bs, stop="\n")
                self_ctx = []
                for i, (p, r) in enumerate(zip(paras, raw)):
                    nums = sorted({int(x) - 1 for x in re.findall(r"\d+", r) if 0 < int(x) <= len(p)})
                    kept = [p[j] for j in nums] if nums else p
                    sup = [q for q in p if q["is_supporting"]]
                    extra["self_recall"][i] = sum(q["is_supporting"] for q in kept) / max(1, len(sup))
                    extra["self_kept"][i] = len(kept)
                    self_ctx.append(_ctx_text(kept))
                _, preds["self_sel"] = _chain_and_answer(lm, self_ctx, qs, args.bs)
            if "gold_only" in M_:    # oracle ceiling: supporting paragraphs only
                _, preds["gold_only"] = _chain_and_answer(
                    lm, [_ctx_text([q for q in p if q["is_supporting"]]) for p in paras], qs, args.bs)
            for i, e in enumerate(part):
                for mth, pl in preds.items():
                    p = pl[i] or ""
                    rows.append(dict(uid=e.uid, k=e.k, cond=cond, method=mth, pred=p,
                                     em=M.em(p, golds[i]), f1=M.f1(p, golds[i]),
                                     attn_recall=extra["attn_recall"][i],
                                     drop2_recall=extra["drop2_recall"][i],
                                     self_recall=extra["self_recall"][i],
                                     self_kept=extra["self_kept"][i], n_paras=len(paras[i]),
                                     chain=ch[i] if mth == "cot_free" else None))
            _log(f"  [{cond}] {min(s + args.chunk, len(exs))}/{len(exs)} "
                 f"({(time.time() - T0) / 60:.0f} min)")
            pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
    return pd.DataFrame(rows)


# ============================================================================ analysis

def summarize(df: pd.DataFrame, metric: str, by: List[str], base: List[str]) -> pd.DataFrame:
    """Mean with bootstrap CI per method, and paired difference + McNemar against each baseline."""
    out = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        w = g.pivot_table(index="uid", columns="method", values=metric)
        for mth in w.columns:
            mu, lo, hi = M.boot_ci(w[mth].values)
            rec = dict(zip(by, key), method=mth, mean=mu, lo=lo, hi=hi, n=int(w[mth].notna().sum()))
            for b in base:
                if b in w.columns and b != mth:
                    d = (w[mth] - w[b]).dropna().values
                    dm, dlo, dhi = M.boot_ci(d)
                    rec[f"d_vs_{b}"], rec[f"d_lo_{b}"], rec[f"d_hi_{b}"] = dm, dlo, dhi
                    if metric in ("correct", "em"):
                        rec[f"p_vs_{b}"] = M.mcnemar_p(w[mth].values >= 0.5, w[b].values >= 0.5)
            out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["clutrr", "musique"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--sc_n", type=int, default=5)
    ap.add_argument("--sc_extra", type=int, default=0,
                    help="extra sampled chains for a compute-matched self-consistency baseline (sc_plus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--max_minutes", type=float, default=600)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--n_per_k", type=int, default=60)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--conditions", nargs="+", default=["orig", "dist8"])
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--extract", type=int, default=1)
    ap.add_argument("--extract_genders", type=int, default=1,
                    help="ask the extractor to also state each person's gender (from pronouns)")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--task_name", default=D.CLUTRR_TASK,
                    help="CLUTRR task folder to draw evaluation stories from (rules always come from "
                         "the gen_train234 training split)")
    ap.add_argument("--ms_methods", nargs="+",
                    default=["attn_sel", "self_sel", "gold_only"],
                    help="MuSiQue methods besides cot_free/last_hop/sc: attn_sel attn_rerank attn_drop2 "
                         "self_sel gold_only")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.outdir, "config.json"), "w"), indent=1)
    from .lm import LM
    lm = LM(args.model, device_map=args.device_map)
    _log(f"loaded {args.model} on {lm.device}")
    if args.task == "clutrr":
        df = run_clutrr(lm, args)
        summ = summarize(df[df.method != "sym_gold"].assign(cond="all"), "correct", ["cond"],
                         ["cot_free", "sc", "sc_plus", "cot_score"])
        g = df[df.method == "cot_free"]
        byk = summarize(df.assign(cond="all"), "correct", ["k"], ["cot_free"])
        byk.to_csv(os.path.join(args.outdir, "by_k.csv"), index=False)
        diag = g.sym_status.value_counts(normalize=True).to_dict()
        extra = dict(sym_status=diag, graph_status=g.graph_status.value_counts(normalize=True).to_dict(),
                     ext_status=g.ext_status.value_counts(normalize=True).to_dict(), hop_fidelity=float(df[df.method == "cot_free"].hop_fidelity.mean()),
                     sym_gold_acc=float(df[df.method == "sym_gold"].correct.mean()))
    else:
        df = run_musique(lm, args)
        summ = pd.concat([summarize(df, "em", ["cond"], ["cot_free", "sc"]).assign(metric="em"),
                          summarize(df, "f1", ["cond"], ["cot_free", "sc"]).assign(metric="f1")])
        g = df[df.method == "cot_free"]
        extra = dict(attn_recall=float(g.attn_recall.mean()), drop2_recall=float(g.drop2_recall.mean()),
                     self_recall=float(g.self_recall.mean()),
                     self_kept=float(g.self_kept.mean()), n_paras=float(g.n_paras.mean()))
    df.to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
    summ.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    extra["minutes"] = (time.time() - T0) / 60
    json.dump(dict(model=args.model, task=args.task, **extra), open(os.path.join(args.outdir, "summary.json"), "w"),
              indent=1, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        _log("\n" + summ.round(3).to_string())
    _log(f"extra: {extra}")


if __name__ == "__main__":
    main()
