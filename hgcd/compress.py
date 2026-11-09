"""Round 14: step-importance estimators for CoT compression in small models (GSM8K).

Question: which estimate of "this step is needed" should decide what to delete from a chain of thought?

Estimators (per body step s_t of a correct greedy chain):
  random    uniform score
  entropy   sum of token entropies inside the step (Li et al., ICLR 2026, step entropy)
  yu        PNS of Yu et al. (NeurIPS 2025): 1 - mean V over k rollouts that continue from (s_<t, s~_t),
            s~_t a sampled alternative step not equivalent to s_t. Assumes P(correct | keep s_t) = 1.
  cal       calibrated version: mean V over k rollouts from (s_<=t)  -  mean V over the yu rollouts
            (on-policy keep-vs-replace contrast, as in Thought Anchors / Thought Branches)
  loo       leave-one-out direct effect: log p(gold | chain) - log p(gold | chain without s_t),
            teacher-forced readout "The answer is"; one forward per step, no sampling
  loo_g     greedy iterative version of loo used for pruning (recomputed after every deletion)

Ground truth: a planted duplicate step (an earlier step repeated, "Recall: ...") is unnecessary by construction.
Pruning test: delete the m lowest-scored steps, read out the answer, compare accuracy and kept tokens.

    python -m hgcd.compress --model Qwen/Qwen2.5-1.5B-Instruct --outdir out --phase score --n 200 --k 6
    python -m hgcd.compress --model Qwen/Qwen2.5-1.5B-Instruct --outdir out --phase sft --n_train 800 --k 4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .lm import LM

T0 = time.time()
SYS = ("Solve the math word problem step by step. Write each step as a separate short paragraph, "
       "separated by a blank line. Finish with a final line such as 'The answer is 42.' giving the number.")
ANS_RE = re.compile(r"answer is[^0-9\-\n]{0,20}?(-?[\d,]*\.?\d+)", re.I)
BOX_RE = re.compile(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
METHODS = ["random", "entropy", "yu", "cal", "loo"]
STOP = "<|im_end|>"


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# ----------------------------------------------------------------------------- data

def load_gsm8k(split: str) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    out = []
    for i, r in d.iterrows():
        g = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(uid=f"{split}-{i}", q=r["question"], gold=float(g)))
    return out


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def parse_answer(text: str) -> Optional[float]:
    """Number after the last 'answer is' (or in \\boxed{}); no last-number fallback."""
    m = ANS_RE.findall(text) or BOX_RE.findall(text)
    return _num(m[-1]) if m else None


def is_correct(text: str, gold: float) -> bool:
    a = parse_answer(text)
    return a is not None and abs(a - gold) < 1e-4


def split_body(text: str) -> Optional[List[str]]:
    """Body steps before the final 'The answer is' sentence (None if there is no such sentence)."""
    m = list(re.finditer(r"the answer is", text, re.I))
    if not m:
        return None
    body = text[:m[-1].start()].strip()
    return [s.strip() for s in body.split("\n\n") if s.strip()]


def _nums(s: str):
    return sorted(x.replace(",", "").rstrip(".") for x in NUM_RE.findall(s))


def equivalent_step(a: str, b: str) -> bool:
    """Cheap stand-in for Yu et al.'s LLM equivalence judge: same text or same multiset of numbers."""
    na, nb = re.sub(r"\W+", " ", a.lower()).strip(), re.sub(r"\W+", " ", b.lower()).strip()
    if na == nb:
        return True
    xa, xb = _nums(a), _nums(b)
    return bool(xa) and xa == xb


# ----------------------------------------------------------------------------- prompts

def msgs(q: str) -> List[dict]:
    return [{"role": "system", "content": SYS}, {"role": "user", "content": q}]


def prefix(lm: LM, q: str, steps: Sequence[str]) -> str:
    return lm.prompt(msgs(q), "".join(s + "\n\n" for s in steps))


def gen(lm: LM, prompts: List[str], args, n: int = 1, sample: bool = False, temp: float = 0.7,
        max_new: int = 320, stop: str = STOP, bs: Optional[int] = None) -> List[str]:
    bs = bs or args.bs
    out: List[str] = []
    for s in range(0, len(prompts), bs):
        out += lm.generate(prompts[s:s + bs], max_new_tokens=max_new, do_sample=sample, temperature=temp,
                           n=n, batch_size=bs, stop=stop)
    return out


def readout(lm: LM, items: List[Dict], chains: List[List[str]], args) -> List[bool]:
    prompts = [prefix(lm, it["q"], ch) + "The answer is" for it, ch in zip(items, chains)]
    outs = gen(lm, prompts, args, max_new=14, stop="\n", bs=max(args.bs * 4, 8))
    return [is_correct("The answer is" + o, it["gold"]) for it, o in zip(items, outs)]


@torch.no_grad()
def logp_gold(lm: LM, it: Dict, steps: Sequence[str]) -> float:
    g = it["gold"]
    gs = str(int(g)) if float(g).is_integer() else str(g)
    return float(lm.score(prefix(lm, it["q"], steps) + "The answer is", [" " + gs])[0])


@torch.no_grad()
def step_entropy(lm: LM, it: Dict, steps: List[str]) -> List[float]:
    """Sum of next-token entropies over the tokens of each step (one forward over the whole chain)."""
    tok = lm.tok
    bounds = [len(tok(prefix(lm, it["q"], steps[:t]), add_special_tokens=False).input_ids)
              for t in range(len(steps) + 1)]
    ids = tok(prefix(lm, it["q"], steps), add_special_tokens=False).input_ids
    x = torch.tensor([ids], device=lm.device)
    logits = lm.model(input_ids=x).logits[0].float()
    lp = F.log_softmax(logits, -1)
    H = -(lp.exp() * lp).sum(-1).cpu().numpy()          # H[p] = entropy of the prediction for token p+1
    out = []
    for t in range(len(steps)):
        a, b = bounds[t], bounds[t + 1]
        out.append(float(H[max(0, a - 1):max(0, b - 1)].sum()))
    return out


# ----------------------------------------------------------------------------- chains

def make_chains(lm: LM, items: List[Dict], args, min_steps: int = 2, max_steps: int = 10) -> List[Dict]:
    outs = gen(lm, [prefix(lm, it["q"], []) for it in items], args, max_new=420)
    keep = []
    for it, o in zip(items, outs):
        body = split_body(o)
        ok = is_correct(o, it["gold"])
        if ok and body is not None and min_steps <= len(body) <= max_steps:
            keep.append(dict(it, raw=o, body=body))
    _log(f"chains: {len(items)} generated, {np.mean([is_correct(o, it['gold']) for it, o in zip(items, outs)]):.3f} "
         f"correct, {len(keep)} kept ({min_steps}-{max_steps} body steps)")
    return keep


def plant(c: Dict, rng: random.Random) -> Dict:
    body = c["body"]
    j = rng.randint(1, len(body) - 1)        # insert before position j: after >=1 step, before the last one
    i = rng.randint(0, j - 1)
    planted = body[:j] + ["Recall: " + body[i]] + body[j:]
    return dict(c, body=planted, planted=j, planted_src=i)


# ----------------------------------------------------------------------------- estimators

def alternatives(lm: LM, cs: List[Dict], args) -> Dict:
    """One alternative step per (chain, t), sampled from the prefix and checked to differ from s_t."""
    keys, prompts = [], []
    for ci, c in enumerate(cs):
        for t in range(len(c["body"])):
            keys.append((ci, t))
            prompts.append(prefix(lm, c["q"], c["body"][:t]))
    na = args.n_alt
    outs = gen(lm, prompts, args, n=na, sample=True, temp=1.0, max_new=96, stop="\n\n",
               bs=max(1, args.bs * args.k // na))
    alt, status = {}, {}
    for kidx, key in enumerate(keys):
        ci, t = key
        cands = [o.strip() for o in outs[kidx * na:(kidx + 1) * na] if o.strip()]
        s = cs[ci]["body"][t]
        good = [x for x in cands if not equivalent_step(x, s)]
        if good:
            alt[key], status[key] = good[0], "ok"
        else:
            alt[key], status[key] = None, "none"     # Yu et al. skip such steps (PN = None -> kept)
    return dict(alt=alt, status=status)


def rollouts(lm: LM, cs: List[Dict], A: Dict, args) -> Dict:
    """k continuations from keep = s_<=t and from replace = (s_<t, s~_t); V = final answer correct."""
    jobs = []
    for ci, c in enumerate(cs):
        jobs.append(("base", ci, -1, prefix(lm, c["q"], [])))
        for t in range(len(c["body"])):
            jobs.append(("keep", ci, t, prefix(lm, c["q"], c["body"][:t + 1])))
            a = A["alt"][(ci, t)]
            if a is not None:
                jobs.append(("rep", ci, t, prefix(lm, c["q"], c["body"][:t] + [a])))
    k = args.k
    res: Dict = {}
    t0 = time.time()
    chunk = 64
    for s in range(0, len(jobs), chunk):
        sub = jobs[s:s + chunk]
        outs = gen(lm, [j[3] for j in sub], args, n=k, sample=True, temp=0.7, max_new=args.roll_tokens)
        for ji, j in enumerate(sub):
            v = [is_correct(o, cs[j[1]]["gold"]) for o in outs[ji * k:(ji + 1) * k]]
            res[(j[0], j[1], j[2])] = float(np.mean(v))
        if (s // chunk) % 10 == 0:
            _log(f"  rollouts {s + len(sub)}/{len(jobs)} jobs ({(time.time() - t0) / 60:.1f} min)")
    return res


def score_all(lm: LM, cs: List[Dict], args, rng: random.Random) -> pd.DataFrame:
    A = alternatives(lm, cs, args)
    _log(f"alternatives: {sum(v == 'ok' for v in A['status'].values())}/{len(A['status'])} found")
    R = rollouts(lm, cs, A, args)
    rows = []
    for ci, c in enumerate(cs):
        ent = step_entropy(lm, c, c["body"])
        full = logp_gold(lm, c, c["body"])
        ntok = [len(lm.tok(s, add_special_tokens=False).input_ids) for s in c["body"]]
        for t, s in enumerate(c["body"]):
            loo = full - logp_gold(lm, c, c["body"][:t] + c["body"][t + 1:])
            keep = R[("keep", ci, t)]
            rep = R.get(("rep", ci, t))
            rows.append(dict(uid=c["uid"], t=t, n=len(c["body"]), planted=int(c.get("planted", -1) == t),
                             ntok=ntok[t], random=rng.random(), entropy=ent[t], loo=loo,
                             keep_acc=keep, rep_acc=rep, base_acc=R[("base", ci, -1)],
                             yu=None if rep is None else 1.0 - rep,
                             cal=None if rep is None else keep - rep,
                             alt=A["alt"][(ci, t)], step=s[:300]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pruning

def _order(scores: List[Optional[float]], rng: random.Random) -> List[int]:
    """Indices from least to most important; None (no valid alternative) counts as important (kept)."""
    key = [(math.inf if s is None or (isinstance(s, float) and math.isnan(s)) else s, rng.random(), i)
           for i, s in enumerate(scores)]
    return [i for _, _, i in sorted(key)]


def greedy_loo(lm: LM, c: Dict, m: int) -> List[int]:
    cur = list(range(len(c["body"])))
    removed = []
    for _ in range(m):
        if len(cur) <= 1:
            break
        steps = [c["body"][i] for i in cur]
        full = logp_gold(lm, c, steps)
        eff = [full - logp_gold(lm, c, steps[:j] + steps[j + 1:]) for j in range(len(cur))]
        j = int(np.argmin(eff))
        removed.append(cur.pop(j))
    return removed


def budgets(n: int) -> Dict[str, int]:
    return {"m1": 1, "m30": max(1, math.ceil(0.3 * n)), "m50": max(1, math.ceil(0.5 * n))}


def prune_eval(lm: LM, cs: List[Dict], df: pd.DataFrame, args, rng: random.Random) -> pd.DataFrame:
    jobs = []           # (chain index, method, budget name, removed set)
    for ci, c in enumerate(cs):
        d = df[df.uid == c["uid"]].sort_values("t")
        n = len(c["body"])
        g = greedy_loo(lm, c, max(budgets(n).values()))   # greedy order: every budget is a prefix of it
        for bname, m in budgets(n).items():
            jobs.append((ci, "none", bname, []))
            for meth in METHODS:
                order = _order(list(d[meth].astype(object).where(d[meth].notna(), None)), rng)
                jobs.append((ci, meth, bname, order[:m]))
            jobs.append((ci, "loo_g", bname, g[:m]))
        # Yu et al.'s own decision rule: drop steps with PN below alpha (0.3 in their code)
        yu = list(d["yu"])
        jobs.append((ci, "yu_alpha", "alpha", [t for t, v in enumerate(yu) if v is not None
                                               and not (isinstance(v, float) and math.isnan(v)) and v < args.alpha]))
    chains = [[s for t, s in enumerate(cs[ci]["body"]) if t not in set(rem)] for ci, _, _, rem in jobs]
    ok = readout(lm, [cs[j[0]] for j in jobs], chains, args)
    rows = []
    for (ci, meth, bname, rem), o in zip(jobs, ok):
        c = cs[ci]
        d = df[df.uid == c["uid"]].sort_values("t")
        ntok = list(d["ntok"])
        rows.append(dict(uid=c["uid"], method=meth, budget=bname, m=len(rem), n=len(c["body"]), acc=float(o),
                         tok_frac=1 - sum(ntok[t] for t in rem) / max(1, sum(ntok)),
                         planted_removed=float(c.get("planted", -1) in set(rem)),
                         base_acc=float(d["base_acc"].iloc[0])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- SFT phase

def sft_targets(lm: LM, cs: List[Dict], df: pd.DataFrame, args, rng: random.Random) -> Dict[str, List]:
    """Pruned training chains per method (40% of body steps); a pruned chain whose readout is wrong is replaced
    by the full chain, so every arm has the same questions."""
    out: Dict[str, List] = {"full": []}
    removed: Dict[str, List[List[int]]] = {}
    for meth in METHODS + ["loo_g"]:
        removed[meth] = []
        for c in cs:
            m = max(1, math.ceil(args.sft_prune * len(c["body"])))
            if meth == "loo_g":
                removed[meth].append(greedy_loo(lm, c, m))
            else:
                d = df[df.uid == c["uid"]].sort_values("t")
                order = _order(list(d[meth].astype(object).where(d[meth].notna(), None)), rng)
                removed[meth].append(order[:m])
    stats = {}
    for meth, rems in removed.items():
        chains = [[s for t, s in enumerate(c["body"]) if t not in set(r)] for c, r in zip(cs, rems)]
        ok = readout(lm, cs, chains, args)
        out[meth] = [ch if o else c["body"] for c, ch, o in zip(cs, chains, ok)]
        stats[meth] = float(np.mean(ok))
    out["full"] = [c["body"] for c in cs]
    _log("pruned-chain readout still correct: " + ", ".join(f"{k} {v:.3f}" for k, v in stats.items()))
    return dict(chains=out, verified=stats)


def _fmt_gold(g: float) -> str:
    return str(int(g)) if float(g).is_integer() else str(g)


def eval_test(lm: LM, items: List[Dict], args) -> pd.DataFrame:
    outs = gen(lm, [prefix(lm, it["q"], []) for it in items], args, max_new=512)
    return pd.DataFrame([dict(uid=it["uid"], correct=float(is_correct(o, it["gold"])),
                              gen_tokens=len(lm.tok(o, add_special_tokens=False).input_ids),
                              out=o[-400:]) for it, o in zip(items, outs)])


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", choices=["score", "sft"], default="score")
    ap.add_argument("--n", type=int, default=200, help="correct chains to score (phase score)")
    ap.add_argument("--n_pool", type=int, default=400, help="test questions to draw chains from")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_alt", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--roll_tokens", type=int, default=320)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--sft_prune", type=float, default=0.4)
    ap.add_argument("--arms", nargs="*", default=["base", "full", "random", "entropy", "yu", "cal", "loo_g"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    lm = LM(args.model, device_map=args.device_map)
    _log(f"loaded {args.model}")

    if args.phase == "score":
        test = load_gsm8k("test")
        random.Random(args.seed).shuffle(test)
        cs = make_chains(lm, test[:args.n_pool], args)[:args.n]
        cs = [plant(c, rng) for c in cs]
        with open(f"{args.outdir}/chains.jsonl", "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        df = score_all(lm, cs, args, rng)
        df.to_csv(f"{args.outdir}/steps.csv", index=False)
        _log("scores written; pruning")
        pr = prune_eval(lm, cs, df, args, rng)
        pr.to_csv(f"{args.outdir}/prune.csv", index=False)
        print(pr.groupby(["budget", "method"])[["acc", "tok_frac", "planted_removed"]].mean().round(3).to_string())
        pl = df[df.planted == 1]
        print("planted mean yu / cal / loo:", pl.yu.mean(), pl.cal.mean(), pl.loo.mean())
        _log("done")
        return

    # ---- phase sft
    train = load_gsm8k("train")
    random.Random(args.seed).shuffle(train)
    test = load_gsm8k("test")
    random.Random(args.seed + 1).shuffle(test)
    evals = test[:args.n_eval]
    cs = make_chains(lm, train[:args.n_train], args)
    df = score_all(lm, cs, args, rng)
    df.to_csv(f"{args.outdir}/train_steps.csv", index=False)
    T = sft_targets(lm, cs, df, args, rng)
    json.dump(dict(verified=T["verified"], n=len(cs)), open(f"{args.outdir}/sft_targets.json", "w"), indent=1)
    with open(f"{args.outdir}/sft_chains.jsonl", "w") as f:
        for i, c in enumerate(cs):
            f.write(json.dumps(dict(uid=c["uid"], **{m: T["chains"][m][i] for m in T["chains"]})) + "\n")
    from .cftrain import train_sft
    rows, summ = [], []
    for arm in args.arms:
        if arm != "base":
            del lm
            gc.collect()
            torch.cuda.empty_cache()
            lm = LM(args.model, device_map=args.device_map)
            data = [(prefix(lm, c["q"], []), "\n\n".join(ch) + f"\n\nThe answer is {_fmt_gold(c['gold'])}.")
                    for c, ch in zip(cs, T["chains"][arm])]
            info = train_sft(lm, data, args)
            _log(f"[{arm}] sft {info}")
        r = eval_test(lm, evals, args)
        r["arm"] = arm
        rows.append(r)
        s = dict(arm=arm, acc=r.correct.mean(), gen_tokens=r.gen_tokens.mean(),
                 train_tokens=np.mean([sum(len(lm.tok(x, add_special_tokens=False).input_ids) for x in ch)
                                       for ch in T["chains"].get(arm, T["chains"]["full"])]))
        summ.append(s)
        _log(f"[{arm}] acc {s['acc']:.3f} gen_tokens {s['gen_tokens']:.1f}")
        pd.concat(rows).to_csv(f"{args.outdir}/sft_eval.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{args.outdir}/sft_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    _log("done")


if __name__ == "__main__":
    main()
