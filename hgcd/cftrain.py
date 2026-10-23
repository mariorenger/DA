"""Round 10: train reading-friendly chains on CLUTRR and evaluate them.

Diagnosis (R3, D3): models copy what is written in a chain but do not compose k relations in one
step. Here every hop is followed by a *running relation* line ("So Y is <source>'s <c_i>."), so each
step is a one-step lookup and the final answer is a copy of the last line. A LoRA model is trained on
k in {2,3,4} only and evaluated on the round-9 sets (same sampling, so baselines join on uid), on
longer chains (k=5-8) and under surface perturbations. No rule table is used at inference.

    python -m hgcd.cftrain --model Qwen/Qwen2.5-3B-Instruct --outdir out --arms p1 s_plain s_struct
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import chain as C
from . import data as D
from . import metrics as M
from .causaldec import _first_rel, induce_table, parse_triples
from .prompts import CLUTRR_SHOTS, CLUTRR_SYSTEM, _clutrr_user
from .lm import LM

T0 = time.time()

STRUCT_SYSTEM = (
    "You solve family-relation puzzles. Walk from the first person to the second person one "
    "relation at a time, using only facts stated in the story. For every hop write "
    "'Hop i: X's <relation> is Y. => Y', then on the next line the running relation "
    "'So Y is <first person>'s <relation>.' obtained by combining the previous running relation "
    "with the new hop. End with the final answer.")

# running relations for the two few-shot examples of prompts.CLUTRR_SHOTS
_SHOT_RUN = [["son", "daughter"], ["father", "aunt", "cousin"]]

SETS = {  # name -> (task, split, ks, n_per_k, seed)  identical to rounds 9 (C4, C6, C5)
    "c4": (D.CLUTRR_TASK, "validation", [2, 3, 4], 100, 2),
    "c6": ("gen_train23_test2to10", "test", [2, 3, 4], 60, 7),
    "c5": ("gen_train23_test2to10", "test", [5, 6, 7, 8], 50, 5),
}
ROBUST_SET = "c6"
ROBUST_CONDS = ["rename", "order", "distract"]
_SO = re.compile(r"^\s*So\s+(.+?)\s+is\s+(.+?)['’]s\s+([A-Za-z\- ]+?)\s*\.?\s*$")


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- chains / prompts

def chain_text(ex: D.ClutrrExample, table: Dict, structured: bool) -> Optional[str]:
    """Gold chain (plain) or gold chain with a running relation after each hop (structured).
    None if the stepwise composition is undefined or disagrees with the gold answer."""
    lines, c = [], None
    for i, h in enumerate(ex.gold_hops):
        lines.append(f"Hop {i + 1}: {h.text()} => {h.tail}")
        if structured:
            c = h.rel if i == 0 else table.get((c, h.rel))
            if c is None:
                return None
            lines.append(f"So {h.tail} is {ex.source}'s {c}.")
    if structured and c != ex.answer:
        return None
    return "\n".join(lines) + f"\nAnswer: {ex.target} is {ex.source}'s {ex.answer}."


def extend_table(train: List[D.ClutrrExample], table: Dict, vocab: List[str], iters: int = 10,
                 purity: float = 0.9, min_count: int = 3) -> Dict:
    """Abductive extension of the pairwise table (train gold only): for a chain whose running relation
    is undefined at exactly one step, the answer pins down the missing relation if one candidate
    reproduces it."""
    table = dict(table)

    def first_missing(e):
        c = None
        for i, h in enumerate(e.gold_hops):
            n = h.rel if c is None else table.get((c, h.rel))
            if n is None:
                return i, c
            c = n
        return None, c

    def finish(e, i, x):
        c = x
        for h in e.gold_hops[i + 1:]:
            c = table.get((c, h.rel))
            if c is None:
                return None
        return c

    for _ in range(iters):
        obs: Dict[Tuple[str, str], Dict[str, int]] = {}
        for e in train:
            i, c = first_missing(e)
            if i is None or i == 0:
                continue
            ok = [x for x in vocab if finish(e, i, x) == e.answer]
            if len(ok) == 1:
                d = obs.setdefault((c, e.gold_hops[i].rel), {})
                d[ok[0]] = d.get(ok[0], 0) + 1
        new = {k: max(v, key=v.get) for k, v in obs.items() if k not in table and sum(v.values()) >= min_count
               and max(v.values()) >= purity * sum(v.values())}
        if not new:
            break
        table.update(new)
    return table


def covered(ex: D.ClutrrExample, table: Dict) -> bool:
    run = gold_running(ex, table)
    return all(r is not None for r in run) and run[-1] == ex.answer


def gold_running(ex: D.ClutrrExample, table: Dict) -> List[Optional[str]]:
    out, c = [], None
    for i, h in enumerate(ex.gold_hops):
        c = h.rel if i == 0 else (table.get((c, h.rel)) if c is not None else None)
        out.append(c)
    return out


def msgs_zero(system: str, ex: D.ClutrrExample) -> List[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": _clutrr_user(ex.story_text, ex.source, ex.target)}]


def msgs_struct_fewshot(ex: D.ClutrrExample) -> List[dict]:
    m = [{"role": "system", "content": STRUCT_SYSTEM}]
    for s, runs in zip(CLUTRR_SHOTS, _SHOT_RUN):
        lines = []
        for i, ((body, res), c) in enumerate(zip(s["hops"], runs)):
            lines.append(f"Hop {i + 1}: {body} => {res}")
            lines.append(f"So {res} is {s['src']}'s {c}.")
        m.append({"role": "user", "content": _clutrr_user(s["story"], s["src"], s["tgt"])})
        m.append({"role": "assistant", "content": "\n".join(lines) +
                  f"\nAnswer: {s['tgt']} is {s['src']}'s {s['ans']}."})
    m.append({"role": "user", "content": _clutrr_user(ex.story_text, ex.source, ex.target)})
    return m


# ----------------------------------------------------------------------------- training

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tokenize(tok, prompt: str, target: str, max_len: int = 1024):
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(target + tok.eos_token, add_special_tokens=False).input_ids
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    return ids, labels


def train_sft(lm: LM, data: List[Tuple[str, str]], args) -> Dict:
    """LoRA SFT on (prompt, target) pairs; loss on the target only. Merges the adapter into lm.model."""
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
    enc = [_tokenize(tok, p, t) for p, t in data]
    rng = random.Random(args.seed)
    n_steps = math.ceil(len(enc) * args.epochs / (args.mb * args.accum))
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pm.train()
    dev = lm.device
    losses, step, micro = [], 0, 0
    order: List[int] = []
    for _ in range(math.ceil(args.epochs)):
        o = list(range(len(enc)))
        rng.shuffle(o)
        order += o
    order = order[:int(len(enc) * args.epochs)]
    t_start = time.time()
    for b in range(0, len(order), args.mb):
        batch = [enc[i] for i in order[b:b + args.mb]]
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            att[i, :len(x)] = 1
        ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            logits = pm(input_ids=ids, attention_mask=att).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
            ignore_index=-100)
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
    return dict(steps=step, final_loss=float(np.mean(losses[-20:])), n_train=len(enc),
                minutes=(time.time() - t_start) / 60)


# ----------------------------------------------------------------------------- evaluation

def _step_acc(chain: str, ex: D.ClutrrExample, table: Dict) -> Tuple[float, float]:
    """(fraction of running-relation lines that are right, 1 if the last one is right)."""
    gold = gold_running(ex, table)
    got = [m.group(3).strip().lower() for ln in chain.splitlines() if (m := _SO.match(ln))]
    if not got:
        return float("nan"), float("nan")
    ok = [i < len(gold) and g == gold[i] for i, g in enumerate(got)]
    return float(np.mean(ok)), float(len(got) == len(gold) and got[-1] == ex.answer)


def evaluate(lm: LM, arm: str, msg_fn, args, table: Dict, vocab_l: List[str], sets, robust: bool,
             max_new_fn) -> List[Dict]:
    rows = []
    for name in sets:
        task, split, ks, npk, seed = SETS[name]
        npk = max(1, int(round(npk * args.eval_scale)))
        pool = D.load_clutrr(split, task=task)
        base = D.sample_by_k(pool, ks, npk, seed)
        conds = ["orig"] + (ROBUST_CONDS if (robust and name == ROBUST_SET) else [])
        for cond in conds:
            ex = [D.apply_condition(e, cond, pool, seed) for e in base]
            prompts = [lm.prompt(msg_fn(e)) for e in ex]
            chains: List[str] = [""] * len(ex)
            for k in sorted(set(e.k for e in ex)):
                idx = [i for i, e in enumerate(ex) if e.k == k]
                out = lm.generate([prompts[i] for i in idx], max_new_tokens=max_new_fn(k),
                                  batch_size=args.bs)
                for t, i in enumerate(idx):
                    chains[i] = out[t].rstrip() + "\n"
            ans = lm.generate([p + c + e.answer_prefix() for p, c, e in zip(prompts, chains, ex)],
                              max_new_tokens=8, batch_size=args.bs * 2, stop="\n")
            for e, c, a in zip(ex, chains, ans):
                trip = parse_triples(c)
                gold_t = {(C.norm(h.head), h.rel, C.norm(h.tail)) for h in e.gold_hops}
                fid = float(np.mean([(C.norm(x), r, C.norm(y)) in gold_t for x, r, y in trip])) if trip else 0.0
                sa, last_ok = _step_acc(c, e, table)
                pred = _first_rel(a, vocab_l)
                rows.append(dict(arm=arm, set=name, cond=cond, uid=e.uid, k=e.k, pred=pred, gold=e.answer,
                                 correct=float(pred == e.answer), hop_fidelity=fid, step_acc=sa,
                                 last_step_ok=last_ok, covered=int(covered(e, table)), chain=c))
            acc = np.mean([r["correct"] for r in rows if r["set"] == name and r["cond"] == cond
                           and r["arm"] == arm])
            _log(f"  [{arm}] {name}/{cond}: n={len(ex)} acc={100 * acc:.1f}")
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, st, cond), g in df.groupby(["arm", "set", "cond"]):
        m, lo, hi = M.boot_ci(g.correct.values)
        row = dict(arm=arm, set=st, cond=cond, n=len(g), acc=100 * m, lo=100 * lo, hi=100 * hi,
                   hop_fidelity=g.hop_fidelity.mean(), step_acc=g.step_acc.mean(),
                   last_step_ok=g.last_step_ok.mean())
        if cond != "orig":  # flip rate against the same model's unperturbed answer
            o = df[(df.arm == arm) & (df.set == st) & (df.cond == "orig")].set_index("uid").pred
            p = g.set_index("uid").pred
            common = p.index.intersection(o.index)
            row["flip_rate"] = float((p[common] != o[common]).mean())
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arms", nargs="+", default=["p1", "s_plain", "s_struct"])
    ap.add_argument("--sets", nargs="+", default=["c4", "c6", "c5"])
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_scale", type=float, default=1.0)
    ap.add_argument("--robust", type=int, default=1)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--max_minutes", type=float, default=600.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.outdir, "config.json"), "w"), indent=1)

    train = D.load_clutrr("train")
    vocab_l = D.clutrr_relations(train)
    table = extend_table(train, induce_table(train), vocab_l)
    pool = [e for e in train if 2 <= e.k <= 4]
    sample = D.sample_by_k(pool, [2, 3, 4], args.n_train // 3, args.seed)
    _log(f"train pool {len(train)} stories, {len(table)} rules; sampled {len(sample)} (k=2-4)")

    all_rows: List[Dict] = []
    info: Dict[str, Dict] = {}
    for arm in args.arms:
        if (time.time() - T0) / 60 > args.max_minutes:
            _log(f"[{arm}] SKIPPED: past max_minutes")
            continue
        _log(f"=== arm {arm}")
        lm = LM(args.model, device_map=args.device_map)
        if arm == "p1":
            msg_fn, mx = msgs_struct_fewshot, lambda k: 44 * (k + 1) + 16
        else:
            structured = arm == "s_struct"
            data, dropped = [], 0
            for e in sample:
                t = chain_text(e, table, structured)
                if t is None:
                    dropped += 1
                    continue
                data.append((lm.prompt(msgs_zero(STRUCT_SYSTEM if structured else CLUTRR_SYSTEM, e)), t))
            _log(f"  {len(data)} training chains ({dropped} dropped: stepwise rule undefined/inconsistent)")
            info[arm] = train_sft(lm, data, args)
            info[arm]["dropped"] = dropped
            system = STRUCT_SYSTEM if structured else CLUTRR_SYSTEM
            msg_fn = lambda e, s=system: msgs_zero(s, e)
            mx = (lambda k: 44 * (k + 1) + 16) if structured else (lambda k: 26 * (k + 2) + 16)
        all_rows += evaluate(lm, arm, msg_fn, args, table, vocab_l, args.sets, bool(args.robust), mx)
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
        summarize(df).round(3).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
        json.dump(info, open(os.path.join(args.outdir, "summary.json"), "w"), indent=1)
        del lm
        gc.collect()
        torch.cuda.empty_cache()
    if all_rows:
        print(summarize(pd.DataFrame(all_rows)).round(3).to_string())


if __name__ == "__main__":
    main()
