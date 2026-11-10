"""Round 18: scale curve of 'read the stated result instead of recomputing' on GSM8K.

For every question the model solves by itself (greedy chain is correct), the stated final result is replaced by a wrong
number w = gold + d (d in {-3,-2,-1,1,2,3}) and the answer is read out after "The answer is".
  intact  : the chain unchanged (sanity: should be ~1.0)
  last    : only the last stated occurrence of the gold number is replaced (one wrong claim at the end of the chain)
  all     : every occurrence of the gold number in the chain is replaced (a consistently wrong result; the arithmetic that
            leads to it no longer adds up)
follow = readout equals w; correct = readout equals gold. Conditioned on 'solvable' by construction. Same family
(Qwen2.5-Instruct) from 0.5B to 32B; 14B/32B in 4 bit (NF4).

    python -m hgcd.scalecopy --model Qwen/Qwen2.5-14B-Instruct --outdir out --load_in_4bit --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .compress import NUM_RE, _fmt_gold, _num, gen, is_correct, load_gsm8k, make_chains, parse_answer, prefix
from .lm import LM

T0 = time.time()


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def corrupt(body: List[str], gold: float, w: float, mode: str) -> Optional[List[str]]:
    """Replace the gold number by w: 'last' = last occurrence in the last step that contains it; 'all' = everywhere."""
    def spans(step):
        return [(m.start(), m.end()) for m in NUM_RE.finditer(step)
                if _num(m.group(0)) is not None and abs(_num(m.group(0)) - gold) < 1e-9]
    out = list(body)
    if mode == "all":
        hit = False
        for i, s in enumerate(out):
            sp = spans(s)
            for a, b in reversed(sp):
                out[i] = out[i][:a] + _fmt(w) + out[i][b:]
                hit = True
        return out if hit else None
    for i in range(len(out) - 1, -1, -1):
        sp = spans(out[i])
        if sp:
            a, b = sp[-1]
            out[i] = out[i][:a] + _fmt(w) + out[i][b:]
            return out
    return None


def read(lm: LM, items: List[Dict], chains: List[List[str]], args) -> List[Optional[float]]:
    prompts = [prefix(lm, it["q"], ch) + "The answer is" for it, ch in zip(items, chains)]
    outs = gen(lm, prompts, args, max_new=14, stop="\n", bs=max(args.bs * 2, 4))
    return [parse_answer("The answer is" + o) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_pool", type=int, default=450)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    lm = LM(args.model, device_map=args.device_map, load_in_4bit=args.load_in_4bit)
    _log(f"loaded {args.model} (4bit={args.load_in_4bit})")
    test = load_gsm8k("test")
    random.Random(args.seed).shuffle(test)
    pool = test[:args.n_pool]
    cs = make_chains(lm, pool, args, min_steps=1, max_steps=12)[:args.n]
    n_correct = len(cs)
    ds = []
    for c in cs:
        w = None
        for _ in range(20):
            cand = c["gold"] + rng.choice([-3, -2, -1, 1, 2, 3])
            if cand >= 0 and cand != c["gold"]:
                w = float(cand)
                break
        ds.append(w)
    rows = []
    for mode in ["intact", "last", "all"]:
        keep, chains, ws = [], [], []
        for c, w in zip(cs, ds):
            ch = c["body"] if mode == "intact" else corrupt(c["body"], c["gold"], w, mode)
            if ch is not None and w is not None:
                keep.append(c); chains.append(ch); ws.append(w)
        ans = read(lm, keep, chains, args)
        for c, w, a in zip(keep, ws, ans):
            rows.append(dict(model=args.model, mode=mode, uid=c["uid"], n_steps=len(c["body"]), gold=c["gold"], wrong=w,
                             ans=a, follow=float(a is not None and abs(a - w) < 1e-4),
                             correct=float(a is not None and abs(a - c["gold"]) < 1e-4)))
        r = [x for x in rows if x["mode"] == mode]
        _log(f"[{mode}] n={len(r)} follow={100 * np.mean([x['follow'] for x in r]):.1f} "
             f"correct={100 * np.mean([x['correct'] for x in r]):.1f}")
        pd.DataFrame(rows).to_csv(f"{args.outdir}/copy.csv", index=False)
    json.dump(dict(n_chains=n_correct, n_pool=args.n_pool), open(f"{args.outdir}/info.json", "w"))
    _log("done")


if __name__ == "__main__":
    main()
