"""Round 20: hop capacity per written step on a synthetic pointer-chasing task with a known minimal chain.

Task: a random function f over N nodes is given as N statements "37 -> 52". Question: start at s and follow the arrows k times.
The answer f^k(s) needs k lookups. A chain written with stride m shows only every m-th node (m = 1: all nodes; m >= k: the answer
only), so each written step has to perform m lookups inside one forward pass. SFT is run separately for every stride; the accuracy
as a function of stride and depth k measures how many lookups a model of a given size can compose per written step, and whether
the loss of accuracy with fewer written tokens is gradual or has a knee (the information-flow bound of BAPO, arXiv 2602.02909, would
predict a threshold where the stride exceeds the capacity).

    python -m hgcd.stride --model Qwen/Qwen2.5-1.5B-Instruct --outdir out --strides 1 2 3 4 12
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .lm import LM

T0 = time.time()
KS = [4, 6, 8, 12]


def _log(msg: str):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def make_sum_item(rng: random.Random, k: int) -> Dict:
    """Running total modulo 10 of k single digits; path[i] = total after i additions, so stride m shows every m-th total."""
    d = [rng.randint(0, 9) for _ in range(k)]
    path = [0]
    for x in d:
        path.append((path[-1] + x) % 10)
    q = ("Numbers: " + " ".join(str(x) for x in d) + ".\nAdd the numbers one by one starting from 0, keeping only the last digit "
         "of the running total each time (total modulo 10). What is the final total?")
    return dict(q=q, k=k, path=path, answer=path[-1])


def make_cumsum_item(rng: random.Random, k: int) -> Dict:
    """Running total WITHOUT modulo of k numbers in 1..5 (path[i] = total after i additions)."""
    d = [rng.randint(1, 5) for _ in range(k)]
    path = [0]
    for x in d:
        path.append(path[-1] + x)
    q = ("Numbers: " + " ".join(str(x) for x in d) + ".\nAdd the numbers one by one starting from 0. What is the final total?")
    return dict(q=q, k=k, path=path, answer=path[-1])


def make_item(rng: random.Random, k: int, n_nodes: int = 24) -> Dict:
    nodes = rng.sample(range(10, 100), n_nodes)
    f = {a: rng.choice([b for b in nodes if b != a]) for a in nodes}
    s = rng.choice(nodes)
    path = [s]
    for _ in range(k):
        path.append(f[path[-1]])
    facts = list(f.items())
    rng.shuffle(facts)
    q = ("Facts: " + "; ".join(f"{a} -> {b}" for a, b in facts) + f".\nStart at {s} and follow the arrows {k} times. "
         "Which node do you reach?")
    return dict(q=q, k=k, path=path, answer=path[-1])


def chain_text(path: List[int], stride: int) -> str:
    """Nodes after hops stride, 2*stride, ... and the last one; the answer line closes the chain."""
    k = len(path) - 1
    idx = list(range(stride, k + 1, stride))
    if not idx or idx[-1] != k:
        idx.append(k)
    shown = [path[i] for i in idx]
    inter = shown[:-1]
    body = (" , ".join(str(x) for x in inter) + "\n") if inter else ""
    return f"{body}Answer: {path[-1]}"


def parse(text: str):
    m = re.findall(r"Answer:\s*(\d+)", text)
    return int(m[-1]) if m else None


def prompt(lm: LM, q: str) -> str:
    return lm.prompt([{"role": "user", "content": q}])


def evaluate(lm: LM, items: List[Dict], bs: int) -> pd.DataFrame:
    outs: List[str] = []
    ps = [prompt(lm, it["q"]) for it in items]
    for i in range(0, len(ps), bs):
        outs += lm.generate(ps[i:i + bs], max_new_tokens=72, batch_size=bs, stop="<|im_end|>")
    rows = []
    for it, o in zip(items, outs):
        a = parse(o)
        rows.append(dict(k=it["k"], correct=float(a == it["answer"]), gen_tokens=len(lm.tok(o, add_special_tokens=False).input_ids)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4, 12])
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=150, help="per depth k")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--task", choices=["pointer", "sum", "cumsum"], default="pointer")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--no_merge", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(f"{args.outdir}/args.json", "w"), indent=1)
    rng = random.Random(args.seed)
    mk = {"sum": make_sum_item, "cumsum": make_cumsum_item}.get(args.task, make_item)
    train = [mk(rng, rng.choice(KS)) for _ in range(args.n_train)]
    evals = [mk(random.Random(10_000 + i * 7 + k), k) for k in KS for i in range(args.n_eval)]
    from .cftrain import train_sft
    rows, base_done = [], False
    lm = LM(args.model, device_map=args.device_map, load_in_4bit=args.load_in_4bit)
    _log(f"loaded {args.model}")
    for m in args.strides:
        if m != args.strides[0] or base_done:
            del lm
            gc.collect()
            torch.cuda.empty_cache()
            lm = LM(args.model, device_map=args.device_map, load_in_4bit=args.load_in_4bit)
        base_done = True
        data = [(prompt(lm, it["q"]), chain_text(it["path"], m)) for it in train]
        info = train_sft(lm, data, args)
        _log(f"[stride {m}] sft {info}")
        r = evaluate(lm, evals, args.bs)
        r["stride"] = m
        rows.append(r)
        g = r.groupby("k").correct.mean().round(3).to_dict()
        _log(f"[stride {m}] acc by k {g}  mean gen tokens {r.gen_tokens.mean():.1f}")
        pd.concat(rows).to_csv(f"{args.outdir}/stride_eval.csv", index=False)
    _log("done")


if __name__ == "__main__":
    main()
