"""Round 33: does a training-free recheck cue remove the copying of a wrong stated result (GSM8K)?

Background (Rounds 18, TN3, TN13): given a chain whose stated final result is wrong, models read the wrong number out
("follow"). Self-Correction Bench (arXiv 2507.02778) reports that appending "Wait" removes most of the self-correction
blind spot of non-reasoning models without any training. Here the same cue is tested on OUR corruption protocol, whose
wrong number is planted exactly, so follow / correct are exact (no judge).

For each question the model solves by itself, the stated result is corrupted as in hgcd.scalecopy (modes last and all;
intact is the sanity arm). The corrupted chain is then continued under four conditions:
  readout   the chain is followed directly by "The answer is" (replicates Round 18)
  neutral   chain + "Let me restate the result." + free continuation, then forced readout (control for extra tokens)
  wait      chain + "Wait, let me double-check the calculation." + free continuation, then forced readout
  recompute chain + "Let me recompute each step of the calculation from the start." + free continuation, then readout
follow = readout equals the planted wrong number; correct = readout equals gold. The harm of a cue is the drop of
`correct` on the intact arm (a cue must not make the model distrust a correct chain).

    python -m hgcd.waitfix --model Qwen/Qwen2.5-3B-Instruct --outdir out --n 200
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

from .compress import gen, load_gsm8k, make_chains, parse_answer, prefix
from .lm import LM
from .scalecopy import corrupt

T0 = time.time()
CUES = {
    "readout": None,
    "neutral": "Let me restate the result.",
    "wait": "Wait, let me double-check the calculation.",
    "recompute": "Let me recompute each step of the calculation from the start.",
}


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def answers(lm: LM, items: List[Dict], chains: List[List[str]], cue: Optional[str], args) -> List[Optional[float]]:
    base = [prefix(lm, it["q"], ch) for it, ch in zip(items, chains)]
    if cue is None:
        prompts = [p + "The answer is" for p in base]
    else:
        pre = [p + cue + "\n\n" for p in base]
        cont = gen(lm, pre, args, max_new=args.max_cont, stop="The answer is", bs=args.bs)
        prompts = [p + c.rstrip() + "\n\nThe answer is" for p, c in zip(pre, cont)]
    outs = gen(lm, prompts, args, max_new=14, stop="\n", bs=max(args.bs, 4))
    return [parse_answer("The answer is" + o) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_pool", type=int, default=450)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_cont", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    lm = LM(args.model, device_map=args.device_map, load_in_4bit=args.load_in_4bit)
    _log(f"loaded {args.model}")
    test = load_gsm8k("test")
    random.Random(args.seed).shuffle(test)
    cs = make_chains(lm, test[:args.n_pool], args, min_steps=1, max_steps=12)[:args.n]
    ws = []
    for c in cs:
        w = None
        for _ in range(20):
            cand = c["gold"] + rng.choice([-3, -2, -1, 1, 2, 3])
            if cand >= 0 and cand != c["gold"]:
                w = float(cand)
                break
        ws.append(w)
    rows = []
    for mode in ["intact", "last", "all"]:
        keep, chains, wr = [], [], []
        for c, w in zip(cs, ws):
            ch = c["body"] if mode == "intact" else corrupt(c["body"], c["gold"], w, mode)
            if ch is not None and w is not None:
                keep.append(c)
                chains.append(ch)
                wr.append(w)
        for cond, cue in CUES.items():
            ans = answers(lm, keep, chains, cue, args)
            for c, w, a in zip(keep, wr, ans):
                rows.append(dict(model=args.model, mode=mode, cond=cond, uid=c["uid"], gold=c["gold"], wrong=w, ans=a,
                                 follow=float(a is not None and abs(a - w) < 1e-4),
                                 correct=float(a is not None and abs(a - c["gold"]) < 1e-4)))
            r = [x for x in rows if x["mode"] == mode and x["cond"] == cond]
            _log(f"[{mode:6s} {cond:9s}] n={len(r)} follow={100 * np.mean([x['follow'] for x in r]):5.1f} "
                 f"correct={100 * np.mean([x['correct'] for x in r]):5.1f}")
            pd.DataFrame(rows).to_csv(f"{args.outdir}/wait.csv", index=False)
    json.dump(dict(n_chains=len(cs), n_pool=args.n_pool), open(f"{args.outdir}/info.json", "w"))
    _log("done")


if __name__ == "__main__":
    main()
