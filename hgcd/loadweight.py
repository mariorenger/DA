"""Round 34: weight the SFT loss by a known structural label instead of an estimated step importance (GSM8K).

Motivation. Credit-assignment papers (Beyond Uniform Credit arXiv 2602.09331, SSVPO ICLR 2026) weight reasoning steps by
how much masking them lowers the probability of the answer, i.e. a leave-one-out score. Rounds 14-19 found that such
scores rank load-bearing steps against planted redundant ones with AUROC 0.66-0.81 only. A load-bearing label can instead
be computed from the chain alone (hgcd.blind.loadbearing: the step introduces a number that a later step or the answer
uses), with no model. This round trains with per-token loss weights and compares the weighting rules.

Arms (the same chains, the same number of steps, the same weight budget w for the up-weighted steps, w=3, others 1):
  base     no training (reference)
  uniform  standard SFT (all weights 1)
  struct   up-weight the load-bearing steps
  loo      up-weight the same NUMBER of steps per chain, chosen by the largest leave-one-out drop of log p(gold)
  random   up-weight the same number of randomly chosen steps
  anti     up-weight the steps that are NOT load-bearing (negative control: must not beat struct)
The final line "The answer is N." keeps weight 1 in every arm. Loss = sum(w * ce) / sum(w).

    python -m hgcd.loadweight --model Qwen/Qwen2.5-1.5B-Instruct --outdir out --n_train 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .blind import loadbearing, loo_scores
from .cftrain import LORA_TARGETS
from .compress import _fmt_gold, eval_test, load_gsm8k, make_chains, prefix
from .lm import LM

T0 = time.time()
ARMS = ["base", "uniform", "struct", "loo", "random", "anti"]


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def target_and_spans(c: Dict, steps: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Target text and the character span of every body step inside it."""
    text, spans = "", []
    for s in steps:
        spans.append((len(text), len(text) + len(s)))
        text += s + "\n\n"
    text += f"The answer is {_fmt_gold(c['gold'])}."
    return text, spans


def weights_for(arm: str, c: Dict, lb: List[int], loo: List[float], rng: random.Random, w: float) -> List[float]:
    n = len(c["body"])
    k = max(1, sum(lb))
    if arm == "uniform":
        hot = set()
    elif arm == "struct":
        hot = {t for t in range(n) if lb[t]}
    elif arm == "anti":
        hot = {t for t in range(n) if not lb[t]}
    elif arm == "loo":
        hot = set(np.argsort(-np.asarray(loo))[:k].tolist())
    else:  # random
        hot = set(rng.sample(range(n), min(k, n)))
    return [w if t in hot else 1.0 for t in range(n)]


def encode(tok, prompt: str, target: str, spans, step_w: List[float], max_len: int = 2048):
    p = tok(prompt, add_special_tokens=False).input_ids
    enc = tok(target + tok.eos_token, add_special_tokens=False, return_offsets_mapping=True)
    t_ids, offs = enc.input_ids, enc.offset_mapping
    tw = []
    for a, b in offs:
        wt = 1.0
        for (s0, s1), sw in zip(spans, step_w):
            if a < s1 and b > s0:
                wt = sw
                break
        tw.append(wt)
    ids = (p + t_ids)[:max_len]
    labels = ([-100] * len(p) + t_ids)[:max_len]
    weights = ([0.0] * len(p) + tw)[:max_len]
    return ids, labels, weights


def train_weighted(lm: LM, data: List[Tuple[List[int], List[int], List[float]]], args) -> Dict:
    """LoRA SFT identical to hgcd.cftrain.train_sft except the per-token loss weights."""
    from peft import LoraConfig, get_peft_model
    tok, model = lm.tok, lm.model
    cuda = torch.cuda.is_available()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    n_steps = math.ceil(len(data) * args.epochs / (args.mb * args.accum))
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pm.train()
    dev = lm.device
    order: List[int] = []
    for _ in range(math.ceil(args.epochs)):
        o = list(range(len(data)))
        rng.shuffle(o)
        order += o
    order = order[:int(len(data) * args.epochs)]
    losses, step, micro = [], 0, 0
    t_start = time.time()
    for b in range(0, len(order), args.mb):
        batch = [data[i] for i in order[b:b + args.mb]]
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        wts = torch.zeros((len(batch), L))
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y, w) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            wts[i, :len(w)] = torch.tensor(w)
            att[i, :len(x)] = 1
        ids, lab, wts, att = ids.to(dev), lab.to(dev), wts.to(dev), att.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            logits = pm(input_ids=ids, attention_mask=att).logits
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
            ignore_index=-100, reduction="none").reshape(len(batch), L - 1)
        w = wts[:, 1:] * (lab[:, 1:] != -100).float()
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        scaler.scale(loss / args.accum).backward()
        losses.append(loss.item())
        micro += 1
        if micro % args.accum == 0 or b + args.mb >= len(order):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 10 == 0 or step == n_steps:
                _log(f"  sft step {step}/{n_steps} loss {np.mean(losses[-10 * args.accum:]):.4f} "
                     f"({(time.time() - t_start) / 60:.1f} min)")
    pm.eval()
    merged = pm.merge_and_unload()
    merged.gradient_checkpointing_disable()
    merged.config.use_cache = True
    merged.eval()
    lm.model = merged
    return dict(steps=step, final_loss=float(np.mean(losses[-20:])), n_train=len(data),
                minutes=(time.time() - t_start) / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--w", type=float, default=3.0)
    ap.add_argument("--arms", nargs="*", default=ARMS)
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
    cs = make_chains(lm, train[:args.n_train], args)
    lb = [loadbearing(c["q"], c["body"], c["gold"]) for c in cs]
    loo = [loo_scores(lm, c, c["body"], False) for c in cs]
    _log(f"load-bearing fraction {np.mean([np.mean(x) for x in lb]):.3f} over {len(cs)} chains")
    json.dump(dict(n=len(cs), lb_frac=float(np.mean([np.mean(x) for x in lb]))), open(f"{args.outdir}/info.json", "w"))
    rows, summ = [], []
    for arm in args.arms:
        if arm != "base":
            del lm
            gc.collect()
            torch.cuda.empty_cache()
            lm = LM(args.model, device_map=args.device_map)
            data = []
            for c, l, s in zip(cs, lb, loo):
                sw = weights_for(arm, c, l, s, rng, args.w)
                text, spans = target_and_spans(c, c["body"])
                data.append(encode(lm.tok, prefix(lm, c["q"], []), text, spans, sw))
            info = train_weighted(lm, data, args)
            _log(f"[{arm}] sft {info}")
        r = eval_test(lm, evals, args)
        r["arm"] = arm
        rows.append(r)
        summ.append(dict(arm=arm, acc=r.correct.mean(), gen_tokens=r.gen_tokens.mean()))
        _log(f"[{arm}] acc {summ[-1]['acc']:.3f} gen_tokens {summ[-1]['gen_tokens']:.1f}")
        pd.concat(rows).to_csv(f"{args.outdir}/sft_eval.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{args.outdir}/sft_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    _log("done")


if __name__ == "__main__":
    main()
