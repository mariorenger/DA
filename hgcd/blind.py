"""Round 15: answer-blind attribution and placebo-calibrated pruning of chains of thought (GSM8K).

Motivation (Round 14): leave-one-out (LOO) on log p(gold answer) keeps the step that STATES the answer and deletes
the derivation, so a readout test rewards it and SFT on its chains collapses (-35...-46 points). Yu et al.'s PNS and
step entropy have no placebo either: their scores have no known zero.

Two ideas, both new relative to the estimators compared in Round 14:
  1. answer-blind LOO (bloo): redact every occurrence of the gold number in the chain, then take the LOO drop of
     log p(gold). A step now counts only if the answer can still be DERIVED with it; copying a stated answer is impossible.
  2. placebo calibration: one planted duplicate step (known redundant) is scored with the real steps of the same chain;
     a real step is removed only if its score is below the placebo's (at most `cap` of the real steps). The threshold is
     the estimator's own value for a step known to be unnecessary, so it adapts per chain and per estimator.

Phases
  audit : ground-truth AUROC (planted duplicate vs load-bearing step), fixed-budget pruning with a plain readout and
          an answer-blind readout, placebo-calibrated pruning                          -> steps.csv, prune.csv
  sft   : SFT on pruned chains (arms: full, random_m, loo_cal, bloo_cal, loo_g, bloo_g), accuracy on held-out
          GSM8K test questions and generated tokens                                      -> sft_eval.csv

    python -m hgcd.blind --model Qwen/Qwen2.5-3B-Instruct --outdir out --phase audit --n 200
    python -m hgcd.blind --model Qwen/Qwen2.5-3B-Instruct --outdir out --phase sft --n_train 800
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .compress import (NUM_RE, _fmt_gold, _num, _nums, _order, budgets, eval_test, gen, is_correct, load_gsm8k,
                       logp_gold, make_chains, plant, prefix, readout, step_entropy)
from .lm import LM

T0 = time.time()


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# ----------------------------------------------------------------------------- redaction and labels

def redact_steps(steps: List[str], gold: float) -> List[str]:
    """Replace every number equal to the gold answer by '?' (the answer can no longer be copied from the chain)."""
    def f(m):
        v = _num(m.group(0))
        return "?" if v is not None and abs(v - float(gold)) < 1e-9 else m.group(0)
    return [NUM_RE.sub(f, s) for s in steps]


def loadbearing(q: str, steps: List[str], gold: float) -> List[int]:
    """1 if the step introduces a number (absent from the question and earlier steps) that a later step, or the
    answer, uses. A planted duplicate introduces nothing, so its label is 0 by construction."""
    seen = set(_nums(q))
    g = _fmt_gold(gold)
    out = []
    for t, s in enumerate(steps):
        intro = set(_nums(s)) - seen
        later = set()
        for x in steps[t + 1:]:
            later |= set(_nums(x))
        used = bool(intro & later) or g in intro
        out.append(int(bool(intro) and used))
        seen |= set(_nums(s))
    return out


# ----------------------------------------------------------------------------- scoring

@torch.no_grad()
def loo_scores(lm: LM, c: Dict, steps: List[str], blind: bool) -> List[float]:
    s = redact_steps(steps, c["gold"]) if blind else steps
    full = logp_gold(lm, c, s)
    return [full - logp_gold(lm, c, s[:t] + s[t + 1:]) for t in range(len(s))]


def greedy_order(lm: LM, c: Dict, m: int, blind: bool) -> List[int]:
    cur = list(range(len(c["body"])))
    removed: List[int] = []
    for _ in range(m):
        if len(cur) <= 1:
            break
        s = [c["body"][i] for i in cur]
        if blind:
            s = redact_steps(s, c["gold"])
        full = logp_gold(lm, c, s)
        eff = [full - logp_gold(lm, c, s[:j] + s[j + 1:]) for j in range(len(s))]
        removed.append(cur.pop(int(np.argmin(eff))))
    return removed


def cal_remove(scores: List[float], ctrl: int, cap: float) -> List[int]:
    """Placebo calibration: real steps scoring strictly below the planted control, lowest first, at most `cap` of them."""
    real = [i for i in range(len(scores)) if i != ctrl]
    below = sorted((i for i in real if scores[i] < scores[ctrl]), key=lambda i: scores[i])
    return below[:int(math.floor(cap * len(real)))]


# ----------------------------------------------------------------------------- audit phase

def audit(lm: LM, cs: List[Dict], args, rng: random.Random):
    rows = []
    sc: Dict[str, Dict] = {}
    for ci, c in enumerate(cs):
        body = c["body"]
        ent = step_entropy(lm, c, body)
        loo = loo_scores(lm, c, body, blind=False)
        bloo = loo_scores(lm, c, body, blind=True)
        lb = loadbearing(c["q"], body, c["gold"])
        ntok = [len(lm.tok(s, add_special_tokens=False).input_ids) for s in body]
        sc[c["uid"]] = dict(entropy=ent, loo=loo, bloo=bloo)
        for t, s in enumerate(body):
            rows.append(dict(uid=c["uid"], t=t, n=len(body), planted=int(c["planted"] == t), loadbearing=lb[t],
                             ntok=ntok[t], random=rng.random(), entropy=ent[t], loo=loo[t], bloo=bloo[t], step=s[:300]))
        if ci % 25 == 0:
            _log(f"  scored {ci + 1}/{len(cs)} chains")
    st = pd.DataFrame(rows)

    jobs = []   # (chain index, method, budget, removed indices)
    for ci, c in enumerate(cs):
        n, ctrl = len(c["body"]), c["planted"]
        S = sc[c["uid"]]
        g_plain = greedy_order(lm, c, max(budgets(n).values()), blind=False)
        g_blind = greedy_order(lm, c, max(budgets(n).values()), blind=True)
        for bname, m in budgets(n).items():
            if bname == "m1":
                continue
            jobs.append((ci, "none", bname, []))
            jobs.append((ci, "random", bname, rng.sample(range(n), m)))
            for meth in ["entropy", "loo", "bloo"]:
                jobs.append((ci, meth, bname, _order(S[meth], rng)[:m]))
            jobs.append((ci, "loo_g", bname, g_plain[:m]))
            jobs.append((ci, "bloo_g", bname, g_blind[:m]))
        # placebo-calibrated rules (the control is always removed as well)
        r_loo = cal_remove(S["loo"], ctrl, args.cap)
        r_bloo = cal_remove(S["bloo"], ctrl, args.cap)
        jobs.append((ci, "loo_cal", "cal", sorted(set(r_loo + [ctrl]))))
        jobs.append((ci, "bloo_cal", "cal", sorted(set(r_bloo + [ctrl]))))
        real = [i for i in range(n) if i != ctrl]
        jobs.append((ci, "random_m", "cal", sorted(set(rng.sample(real, len(r_bloo)) + [ctrl]))))
        jobs.append((ci, "control_only", "cal", [ctrl]))
    chains = [[s for t, s in enumerate(cs[ci]["body"]) if t not in set(rem)] for ci, _, _, rem in jobs]
    items = [cs[j[0]] for j in jobs]
    ok_plain = readout(lm, items, chains, args)
    ok_blind = readout(lm, items, [redact_steps(ch, it["gold"]) for ch, it in zip(chains, items)], args)
    out = []
    for (ci, meth, bname, rem), a, b in zip(jobs, ok_plain, ok_blind):
        c = cs[ci]
        d = st[st.uid == c["uid"]].sort_values("t")
        nt = list(d["ntok"])
        n_real = len(c["body"]) - 1
        out.append(dict(uid=c["uid"], method=meth, budget=bname, m=len(rem), n=len(c["body"]),
                        acc_plain=float(a), acc_blind=float(b), tok_frac=1 - sum(nt[t] for t in rem) / max(1, sum(nt)),
                        planted_removed=float(c["planted"] in set(rem)),
                        real_removed_frac=len([t for t in rem if t != c["planted"]]) / max(1, n_real)))
    return st, pd.DataFrame(out)


# ----------------------------------------------------------------------------- sft phase

def sft_chains(lm: LM, cs: List[Dict], args, rng: random.Random):
    """Per-arm training chains (lists of steps) for the questions in cs; cs hold the ORIGINAL chains, `pc` the planted one."""
    arms: Dict[str, List[List[str]]] = {"full": [c["body"] for c in cs]}
    plants = [plant(c, rng) for c in cs]
    removed = {k: [] for k in ["loo_cal", "bloo_cal", "random_m", "loo_g", "bloo_g"]}
    for ci, (c, pc) in enumerate(zip(cs, plants)):
        ctrl = pc["planted"]
        body = pc["body"]
        loo = loo_scores(lm, pc, body, blind=False)
        bloo = loo_scores(lm, pc, body, blind=True)

        def to_orig(idx):                    # index in planted chain -> index in original chain
            return [i if i < ctrl else i - 1 for i in idx if i != ctrl]
        r_l, r_b = to_orig(cal_remove(loo, ctrl, args.cap)), to_orig(cal_remove(bloo, ctrl, args.cap))
        removed["loo_cal"].append(r_l)
        removed["bloo_cal"].append(r_b)
        removed["random_m"].append(rng.sample(range(len(c["body"])), len(r_b)))
        m = max(1, math.ceil(args.sft_prune * len(c["body"])))
        removed["loo_g"].append(greedy_order(lm, c, m, blind=False))
        removed["bloo_g"].append(greedy_order(lm, c, m, blind=True))
        if ci % 50 == 0:
            _log(f"  scored {ci + 1}/{len(cs)} training chains")
    stats = {}
    for k, rems in removed.items():
        ch = [[s for t, s in enumerate(c["body"]) if t not in set(r)] for c, r in zip(cs, rems)]
        ok = readout(lm, cs, ch, args)
        arms[k] = [x if o else c["body"] for c, x, o in zip(cs, ch, ok)]
        stats[k] = dict(readout_ok=float(np.mean(ok)), removed_frac=float(np.mean(
            [len(r) / len(c["body"]) for c, r in zip(cs, rems)])),
            removed_nonempty=float(np.mean([len(r) > 0 for r in rems])))
    _log("pruning statistics: " + json.dumps(stats))
    return arms, stats


# ----------------------------------------------------------------------------- order-reversal control (Round 16)

def order_arms(lm: LM, cs: List[Dict], args, rng: random.Random):
    """Delete the same number (ceil(sft_prune * n)) of steps per chain, chosen as the LEAST important (loo_low), the MOST
    important (loo_high) by single-pass plain LOO, or at random (random40). No readout verification: loo_high deletes the
    answer-bearing steps on purpose. If the LOO order carried information about training value, loo_high would hurt more
    than loo_low."""
    arms: Dict[str, List[List[str]]] = {"full": [c["body"] for c in cs]}
    removed = {"random40": [], "loo_low": [], "loo_high": []}
    for ci, c in enumerate(cs):
        sc = loo_scores(lm, c, c["body"], blind=False)
        n = len(sc)
        m = max(1, math.ceil(args.sft_prune * n))
        removed["loo_low"].append(_order(sc, rng)[:m])
        removed["loo_high"].append(_order([-x for x in sc], rng)[:m])
        removed["random40"].append(rng.sample(range(n), m))
        if ci % 50 == 0:
            _log(f"  scored {ci + 1}/{len(cs)} training chains")
    stats = {}
    for k, rems in removed.items():
        arms[k] = [[s for t, s in enumerate(c["body"]) if t not in set(r)] for c, r in zip(cs, rems)]
        g = [_fmt_gold(c["gold"]) for c in cs]
        stats[k] = dict(mean_steps=float(np.mean([len(x) for x in arms[k]])),
                        single_step=float(np.mean([len(x) == 1 for x in arms[k]])),
                        last_has_gold=float(np.mean([gg in set(_nums(x[-1])) for gg, x in zip(g, arms[k])])),
                        readout_ok=float(np.mean(readout(lm, cs, arms[k], args))))
    _log("order statistics: " + json.dumps(stats))
    return arms, stats


# ----------------------------------------------------------------------------- compaction test (Round 17)

import re as _re
_EQ = _re.compile(r"[\$\d\(][\d\.,\s\+\-\*/x\u00d7\u00f7\(\)\$%]*?=\s*[\$\d][\d\.,\$%]*")


def compact_step(step: str) -> str:
    """Keep only the equations of a step ('18 * 3 = 54'); a step without an equation is kept unchanged (premises, restatements)."""
    t = (step.replace("\\(", "").replace("\\)", "").replace("\\times", "\u00d7").replace("\\div", "\u00f7")
         .replace("\\[", "").replace("\\]", "").replace("\\$", "$"))
    eqs = [m.group(0).strip() for m in _EQ.finditer(t)]
    return "; ".join(eqs) if eqs else step


def compact_arms(lm: LM, cs: List[Dict], args, rng: random.Random):
    """full | compact (every step reduced to its equations, all computations kept) | random40 (40% of the steps deleted)."""
    arms: Dict[str, List[List[str]]] = {"full": [c["body"] for c in cs]}
    arms["compact"] = [[compact_step(s) for s in c["body"]] for c in cs]
    arms["random40"] = []
    for c in cs:
        n = len(c["body"])
        rem = set(rng.sample(range(n), max(1, math.ceil(args.sft_prune * n))))
        arms["random40"].append([s for t, s in enumerate(c["body"]) if t not in rem])
    tl = lambda ch: float(np.mean([sum(len(lm.tok(x, add_special_tokens=False).input_ids) for x in y) for y in ch]))
    full_t = tl(arms["full"])
    stats = {k: dict(train_token_ratio=tl(v) / full_t, mean_steps=float(np.mean([len(y) for y in v])),
                     all_numbers_kept=float(np.mean([set(_nums(" ".join(c["body"]))) <= set(_nums(" ".join(y)))
                                                     for c, y in zip(cs, v)]))) for k, v in arms.items()}
    _log("compact statistics: " + json.dumps(stats))
    return arms, stats


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", choices=["audit", "sft", "order", "compact"], default="audit")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_pool", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--sft_prune", type=float, default=0.4)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()
    if args.arms is None:
        args.arms = (["full", "compact", "random40"] if args.phase == "compact" else
                     ["full", "random40", "loo_low", "loo_high"] if args.phase == "order" else
                     ["base", "full", "random_m", "loo_cal", "bloo_cal", "loo_g", "bloo_g"])
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    lm = LM(args.model, device_map=args.device_map)
    _log(f"loaded {args.model}")

    if args.phase == "audit":
        test = load_gsm8k("test")
        random.Random(args.seed).shuffle(test)
        cs = [plant(c, rng) for c in make_chains(lm, test[:args.n_pool], args)[:args.n]]
        with open(f"{args.outdir}/chains.jsonl", "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        st, pr = audit(lm, cs, args, rng)
        st.to_csv(f"{args.outdir}/steps.csv", index=False)
        pr.to_csv(f"{args.outdir}/prune.csv", index=False)
        print(pr.groupby(["budget", "method"])[["acc_plain", "acc_blind", "tok_frac", "planted_removed",
                                                 "real_removed_frac"]].mean().round(3).to_string())
        _log("done")
        return

    train = load_gsm8k("train")
    random.Random(args.seed).shuffle(train)
    test = load_gsm8k("test")
    random.Random(args.seed + 1).shuffle(test)
    evals = test[:args.n_eval]
    cs = make_chains(lm, train[:args.n_train], args)
    if args.phase == "compact":
        arms, stats = compact_arms(lm, cs, args, rng)
    else:
        arms, stats = order_arms(lm, cs, args, rng) if args.phase == "order" else sft_chains(lm, cs, args, rng)
    json.dump(dict(stats=stats, n=len(cs)), open(f"{args.outdir}/sft_targets.json", "w"), indent=1)
    with open(f"{args.outdir}/sft_chains.jsonl", "w") as f:
        for i, c in enumerate(cs):
            f.write(json.dumps(dict(uid=c["uid"], **{k: arms[k][i] for k in arms})) + "\n")
    from .cftrain import train_sft
    rows, summ = [], []
    for arm in args.arms:
        if arm != "base":
            del lm
            gc.collect()
            torch.cuda.empty_cache()
            lm = LM(args.model, device_map=args.device_map)
            data = [(prefix(lm, c["q"], []), "\n\n".join(ch) + f"\n\nThe answer is {_fmt_gold(c['gold'])}.")
                    for c, ch in zip(cs, arms[arm])]
            info = train_sft(lm, data, args)
            _log(f"[{arm}] sft {info}")
        r = eval_test(lm, evals, args)
        r["arm"] = arm
        rows.append(r)
        tr = arms.get(arm, arms["full"])
        summ.append(dict(arm=arm, acc=r.correct.mean(), gen_tokens=r.gen_tokens.mean(),
                         train_steps=np.mean([len(x) for x in tr])))
        _log(f"[{arm}] acc {summ[-1]['acc']:.3f} gen_tokens {summ[-1]['gen_tokens']:.1f}")
        pd.concat(rows).to_csv(f"{args.outdir}/sft_eval.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{args.outdir}/sft_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    _log("done")


if __name__ == "__main__":
    main()
