"""Round 36: distil the recompute cue into the default behaviour (GSM8K), self-distillation with ground-truth filtering.

Round 33 found that one sentence ("Let me recompute each step of the calculation from the start.") placed after a chain
with a wrong stated result makes the model recompute and recover (follow 8-49% instead of 85-100%), while a neutral
continuation does not. The failure is therefore a default (immediate readout), not an incapacity. Here the default is
changed by SFT instead of by a cue at inference, without a teacher and without a verifier at inference time:

  1. the base model writes correct chains for training questions (as in Rounds 14-19);
  2. each chain is intact or, with probability 1/2, corrupted (the stated result replaced by a wrong number, hgcd.scalecopy);
  3. the base model continues the chain under the recompute cue; a continuation is kept only if its final answer equals the
     gold answer (ground truth, so corrupted chains are paired with a recomputation that corrects them);
  4. SFT on  chain (not trained on)  ->  cue + recomputation + answer, i.e. the model learns to recompute after a chain on
     its own. Loss on the post-chain part only.

Arms: base (no training); base_cue (no training, the cue appended at inference: the reference of Round 33);
dist_mix (corrupted and intact chains); dist_intact (intact chains only: tests whether exposure to wrong results matters);
cond_mix (Round 37: recompute only after a corrupted chain, answer directly after an intact one: the model must decide).
Evaluation on held-out test questions (the model solves them by itself): the chain is given as a prefix, the model
continues freely (no forced readout, no cue), the last stated answer is parsed:
  copy_last / copy_all  corrupted prefix  -> follow (= planted wrong number), correct
  intact                correct prefix    -> correct (harm check)
  clean                 free generation from the question -> accuracy and generated tokens (cost)

    python -m hgcd.cuedistill --model Qwen/Qwen2.5-3B-Instruct --outdir out --n_train 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .cftrain import train_sft
from .compress import _fmt_gold, eval_test, gen, load_gsm8k, make_chains, parse_answer, prefix
from .lm import LM
from .scalecopy import corrupt

T0 = time.time()
CUE = "Let me recompute each step of the calculation from the start."


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def wrong_of(gold: float, rng: random.Random) -> Optional[float]:
    for _ in range(20):
        cand = gold + rng.choice([-3, -2, -1, 1, 2, 3])
        if cand >= 0 and cand != gold:
            return float(cand)
    return None


def last_answer(text: str) -> Optional[float]:
    return parse_answer(text)


def build_targets(lm: LM, cs: List[Dict], args, rng: random.Random):
    """(chain variant, corrupted?) -> verified continuation under the cue."""
    items = []
    for c in cs:
        w = wrong_of(c["gold"], rng)
        mode = rng.choice(["last", "all"])
        for corrupted in (False, True):
            body = c["body"]
            if corrupted:
                body = corrupt(c["body"], c["gold"], w, mode) if w is not None else None
            if body is not None:
                items.append(dict(c=c, body=body, corrupted=corrupted))
    prompts = [prefix(lm, it["c"]["q"], it["body"]) + CUE + "\n\n" for it in items]
    outs = gen(lm, prompts, args, max_new=args.max_cont, bs=args.bs)
    keep = []
    for it, o in zip(items, outs):
        a = last_answer(o)
        if a is not None and abs(a - it["c"]["gold"]) < 1e-4 and "answer is" in o.lower():
            it["target"] = CUE + "\n\n" + o.strip()
            keep.append(it)
    _log(f"targets: {len(items)} candidates, {len(keep)} verified (corrupted kept "
         f"{sum(i['corrupted'] for i in keep)}/{sum(i['corrupted'] for i in items)}, intact kept "
         f"{sum(not i['corrupted'] for i in keep)}/{sum(not i['corrupted'] for i in items)})")
    return keep


def evaluate_copy(lm: LM, cs: List[Dict], ws: List[Optional[float]], args, cue: bool) -> List[Dict]:
    rows = []
    for mode in ["intact", "last", "all"]:
        keep, chains, wr = [], [], []
        for c, w in zip(cs, ws):
            ch = c["body"] if mode == "intact" else (corrupt(c["body"], c["gold"], w, mode) if w is not None else None)
            if ch is not None:
                keep.append(c)
                chains.append(ch)
                wr.append(w)
        pre = [prefix(lm, c["q"], ch) + ((CUE + "\n\n") if cue else "") for c, ch in zip(keep, chains)]
        outs = gen(lm, pre, args, max_new=args.max_cont, bs=args.bs)
        for c, w, o in zip(keep, wr, outs):
            a = last_answer(o)
            rows.append(dict(mode=mode, uid=c["uid"], gold=c["gold"], wrong=w, ans=a,
                             follow=float(a is not None and w is not None and abs(a - w) < 1e-4),
                             correct=float(a is not None and abs(a - c["gold"]) < 1e-4),
                             triggered=float("recompute each step" in o),
                             gen_tokens=len(lm.tok(o, add_special_tokens=False).input_ids)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--n_copy", type=int, default=200)
    ap.add_argument("--n_pool", type=int, default=450)
    ap.add_argument("--max_cont", type=int, default=260)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--arms", nargs="*", default=["base", "base_cue", "dist_mix", "dist_intact"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    lm = LM(args.model, device_map=args.device_map)
    _log(f"loaded {args.model}")
    train = load_gsm8k("train")
    random.Random(args.seed).shuffle(train)
    test = load_gsm8k("test")
    random.Random(args.seed + 1).shuffle(test)
    evals = test[:args.n_eval]
    cs_train = make_chains(lm, train[:args.n_train], args, min_steps=1, max_steps=12)
    cs_test = make_chains(lm, test[args.n_eval:args.n_eval + args.n_pool], args, min_steps=1, max_steps=12)[:args.n_copy]
    ws = [wrong_of(c["gold"], rng) for c in cs_test]
    T = build_targets(lm, cs_train, args, rng)
    json.dump(dict(n_train_chains=len(cs_train), n_targets=len(T), n_copy=len(cs_test)),
              open(f"{args.outdir}/info.json", "w"))
    rows, summ = [], []
    for arm in args.arms:
        if arm.startswith("dist") or arm.startswith("cond"):
            del lm
            gc.collect()
            torch.cuda.empty_cache()
            lm = LM(args.model, device_map=args.device_map)
            corr = [t for t in T if t["corrupted"]]
            inta = [t for t in T if not t["corrupted"]]
            n_c = min(len(corr), len(inta))
            r2 = random.Random(args.seed + 7)
            r2.shuffle(corr)
            r2.shuffle(inta)
            if arm in ("dist_mix", "cond_mix"):
                sel = corr[:n_c] + inta[:n_c]
            else:       # dist_intact: same number of items, intact chains only (repeated if there are too few)
                sel = [inta[i % len(inta)] for i in range(2 * n_c)]
            data = [(prefix(lm, t["c"]["q"], t["body"]),
                     (t["target"] if (arm != "cond_mix" or t["corrupted"])
                      else f"The answer is {_fmt_gold(t['c']['gold'])}.")) for t in sel]
            info = train_sft(lm, data, args)
            _log(f"[{arm}] sft on {len(data)} items {info}")
        r = evaluate_copy(lm, cs_test, ws, args, cue=(arm == "base_cue"))
        for x in r:
            x["arm"] = arm
        rows += r
        clean = eval_test(lm, evals, args)
        clean["arm"] = arm
        clean.to_csv(f"{args.outdir}/clean_{arm}.csv", index=False)
        d = pd.DataFrame(r)
        s = dict(arm=arm, clean_acc=clean.correct.mean(), clean_tokens=clean.gen_tokens.mean())
        for mode in ["intact", "last", "all"]:
            m = d[d["mode"] == mode]
            s[f"{mode}_follow"] = m.follow.mean()
            s[f"{mode}_correct"] = m.correct.mean()
            s[f"{mode}_trig"] = m.triggered.mean()
        summ.append(s)
        _log(f"[{arm}] " + " ".join(f"{k}={v:.3f}" for k, v in s.items() if k != "arm"))
        pd.DataFrame(rows).to_csv(f"{args.outdir}/copy.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{args.outdir}/summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    _log("done")


if __name__ == "__main__":
    main()
