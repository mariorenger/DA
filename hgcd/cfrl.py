"""Round 11: causal training methods on CLUTRR, on top of supervised fine-tuning (hgcd/cftrain.py).

Arms (each trains its own SFT stage first; the stage-1 model is evaluated too):
  s_plain   SFT on gold chains (control, as in round 10)
  s_ans     SFT on the answer only (does the chain matter at all after fine-tuning?)
  s_dpo     SFT, then DPO with programmatic counterfactual chains (FRODO-style inference module):
            rejected = gold hops followed by a hop that states a WRONG answer / a corrupted hop / a wrong answer
  s_grpo    SFT, then GRPO with a verifiable reward = answer correctness + 0.3 * hop fidelity (process reward
            from the gold reasoning graph), token-level loss and dynamic sampling (DAPO ingredients)
Extra causal measurements on the chain-writing models (n=120 questions of the unseen k=2-4 task):
  wrong-copy follow rate (round-3 protocol T_copy-wrong) and the do_total effect of non-final hops.

    python -m hgcd.cfrl --model Qwen/Qwen2.5-3B-Instruct --outdir out --arms s_ans s_dpo
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
from .causaldec import _first_rel, induce_table, parse_triples
from .cftrain import (LORA_TARGETS, SETS, _tokenize, chain_text, evaluate, extend_table, msgs_zero,
                      summarize, train_sft)
from .experiments import _dist
from .lm import LM
from .prompts import CLUTRR_SYSTEM

T0 = time.time()
ANS_SYSTEM = ("You solve family-relation puzzles. Reply with the final answer only, in the form "
              "'Answer: X is Y's <relation>'.")


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- data

def ans_text(e: D.ClutrrExample) -> str:
    return f"Answer: {e.target} is {e.source}'s {e.answer}."


def _hops(e: D.ClutrrExample, rel_override: Optional[Dict[int, str]] = None) -> List[str]:
    out = []
    for i, h in enumerate(e.gold_hops):
        rel = (rel_override or {}).get(i, h.rel)
        out.append(f"Hop {i + 1}: {h.head}'s {rel} is {h.tail}. => {h.tail}")
    return out


def cf_pair(e: D.ClutrrExample, rels: List[str], rng: random.Random) -> Tuple[str, str]:
    """(chosen, rejected): chosen is the gold chain; rejected is a counterfactual chain."""
    chosen = chain_text(e, {}, False)
    wrong = rng.choice([r for r in rels if r != e.answer])
    kind = rng.random()
    if kind < 0.4:      # copies a wrong stated answer (mirrors the T_copy-wrong test)
        stated = f"Hop {e.k + 1}: So {e.target} is {e.source}'s relative: => {wrong}"
        rej = "\n".join(_hops(e) + [stated, f"Answer: {e.target} is {e.source}'s {wrong}."])
    elif kind < 0.7:    # one corrupted hop and an answer that follows it
        j = rng.randrange(e.k)
        alt = rng.choice([r for r in rels if r != e.gold_hops[j].rel])
        rej = "\n".join(_hops(e, {j: alt}) + [f"Answer: {e.target} is {e.source}'s {wrong}."])
    else:               # right hops, wrong answer
        rej = "\n".join(_hops(e) + [f"Answer: {e.target} is {e.source}'s {wrong}."])
    return chosen, rej


# ----------------------------------------------------------------------------- DPO

def _seq_logp(pm, ids, lab, att, cuda):
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        logits = pm(input_ids=ids, attention_mask=att).logits
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    tgt = lab[:, 1:]
    tok_lp = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    m = (tgt != -100).float()
    return (tok_lp * m).sum(-1), m.sum(-1)


def _pad(batch, pad_id, dev):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)


def train_dpo(lm: LM, pairs: List[Tuple[str, str, str]], args) -> Dict:
    """LoRA DPO on top of the (merged) SFT model; the reference is the SFT model (adapter disabled).
    A small NLL term on the chosen chain (RPO) keeps its likelihood from collapsing."""
    from peft import LoraConfig, get_peft_model
    tok, model = lm.tok, lm.model
    cuda = torch.cuda.is_available()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.dpo_lr, weight_decay=0.0)
    enc = [(_tokenize(tok, p, c), _tokenize(tok, p, r)) for p, c, r in pairs]
    n_steps = math.ceil(len(enc) / args.dpo_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 5) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pm.train()
    rng = random.Random(args.seed + 1)
    order = list(range(len(enc)))
    rng.shuffle(order)
    stats, step, t_start = [], 0, time.time()
    for n, i in enumerate(order):
        batch = [enc[i][0], enc[i][1]]
        ids, lab, att = _pad(batch, tok.pad_token_id, lm.device)
        with torch.no_grad(), pm.disable_adapter():
            ref, _ = _seq_logp(pm, ids, lab, att, cuda)
        pol, ntok = _seq_logp(pm, ids, lab, att, cuda)
        margin = args.dpo_beta * ((pol[0] - pol[1]) - (ref[0] - ref[1]))
        loss = -torch.nn.functional.logsigmoid(margin) + args.dpo_nll * (-pol[0] / ntok[0])
        scaler.scale(loss / args.dpo_accum).backward()
        stats.append((loss.item(), float(margin.item() > 0)))
        if (n + 1) % args.dpo_accum == 0 or n + 1 == len(order):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 10 == 0 or step == n_steps:
                s = np.array(stats[-10 * args.dpo_accum:])
                _log(f"  dpo step {step}/{n_steps} loss {s[:, 0].mean():.4f} pref-acc {s[:, 1].mean():.2f} "
                     f"({(time.time() - t_start) / 60:.1f} min)")
    pm.eval()
    merged = pm.merge_and_unload()
    merged.gradient_checkpointing_disable()
    merged.config.use_cache = True
    merged.eval()
    lm.model = merged
    return dict(steps=step, n_pairs=len(enc), minutes=(time.time() - t_start) / 60)


# ----------------------------------------------------------------------------- GRPO

def reward_fn(e: D.ClutrrExample, text: str, vocab: List[str]) -> Tuple[float, float]:
    m = re.search(r"Answer:\s*(.*)", text)
    ans = _first_rel(m.group(1), vocab) if m else None
    gold_t = {(C.norm(h.head), h.rel, C.norm(h.tail)) for h in e.gold_hops}
    trip = parse_triples(text)
    fid = float(np.mean([(C.norm(x), r, C.norm(y)) in gold_t for x, r, y in trip])) if trip else 0.0
    correct = float(ans == e.answer)
    return correct + 0.3 * fid - (0.0 if m else 0.2), correct


def train_grpo(lm: LM, sample: List[D.ClutrrExample], vocab: List[str], args) -> Dict:
    """On-policy group-relative policy gradient (one update per rollout batch, so the importance ratio is 1)
    with token-level loss, no KL term and dynamic sampling of groups with non-zero reward variance."""
    from peft import LoraConfig, get_peft_model
    tok, model = lm.tok, lm.model
    cuda = torch.cuda.is_available()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.rl_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    rng = random.Random(args.seed + 2)
    by_k = {k: [e for e in sample if e.k == k] for k in (2, 3, 4)}
    gen_eos = lm.model.generation_config.eos_token_id
    stop_ids = {tok.eos_token_id, tok.pad_token_id} | set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    stop_ids.discard(None)
    log, t_start = [], time.time()
    P, G = args.rl_p, args.rl_g
    for step in range(1, args.rl_steps + 1):
        if (time.time() - t_start) / 60 > args.rl_minutes:
            _log(f"  grpo: time budget reached at step {step - 1}")
            break
        exs = [rng.choice(by_k[rng.choices([2, 3, 4], [0.2, 0.3, 0.5])[0]]) for _ in range(P)]
        prompts = [lm.prompt(msgs_zero(CLUTRR_SYSTEM, e)) for e in exs]
        enc = lm._enc(prompts)
        pm.eval()
        model.config.use_cache = True
        with torch.no_grad():
            g = pm.generate(**enc, max_new_tokens=26 * (max(e.k for e in exs) + 2) + 30, do_sample=True,
                            temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=G,
                            pad_token_id=tok.pad_token_id)
        Lp = enc.input_ids.shape[1]
        comp = g[:, Lp:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        R = np.zeros(P * G)
        ok = np.zeros(P * G)
        for i, t in enumerate(texts):
            R[i], ok[i] = reward_fn(exs[i // G], t, vocab)
        A = np.zeros(P * G)
        keep = []
        for p in range(P):
            r = R[p * G:(p + 1) * G]
            if r.std() > 1e-6:
                A[p * G:(p + 1) * G] = (r - r.mean()) / (r.std() + 1e-4)
                keep += list(range(p * G, (p + 1) * G))
        log.append((R.mean(), ok.mean(), len(keep) // G))
        if keep:
            mask = torch.zeros_like(comp, dtype=torch.float32)
            for i in range(comp.shape[0]):
                pos = [j for j, t in enumerate(comp[i].tolist()) if t in stop_ids]
                mask[i, :(pos[0] + 1 if pos else comp.shape[1])] = 1.0   # the stop token is a generated token
            attn = torch.cat([enc.attention_mask.repeat_interleave(G, 0),
                              (mask > 0).long().to(enc.attention_mask.device)], 1)
            pm.train()
            model.config.use_cache = False
            n_tok = mask[keep].sum().clamp(min=1.0)
            for s0 in range(0, len(keep), args.rl_mb):
                idx = keep[s0:s0 + args.rl_mb]
                ids = g[idx].to(lm.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    logits = pm(input_ids=ids, attention_mask=attn[idx].to(lm.device)).logits
                lp = torch.log_softmax(logits[:, Lp - 1:-1].float(), -1)
                tl = lp.gather(-1, comp[idx].to(lm.device).unsqueeze(-1)).squeeze(-1)
                adv = torch.tensor(A[idx], dtype=torch.float32, device=lm.device).unsqueeze(1)
                loss = -(adv * tl * mask[idx].to(lm.device)).sum() / n_tok.to(lm.device)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            a = np.array(log[-10:])
            _log(f"  grpo step {step}/{args.rl_steps} reward {a[:, 0].mean():.3f} acc {a[:, 1].mean():.3f} "
                 f"kept {a[:, 2].mean():.1f}/{P} ({(time.time() - t_start) / 60:.1f} min)")
    pm.eval()
    merged = pm.merge_and_unload()
    merged.gradient_checkpointing_disable()
    merged.config.use_cache = True
    merged.eval()
    lm.model = merged
    a = np.array(log) if log else np.zeros((1, 3))
    return dict(steps=len(log), reward_first10=float(a[:10, 0].mean()), reward_last10=float(a[-10:, 0].mean()),
                acc_first10=float(a[:10, 1].mean()), acc_last10=float(a[-10:, 1].mean()),
                minutes=(time.time() - t_start) / 60)


# ----------------------------------------------------------------------------- causal measurements

def causal_extras(lm: LM, arm: str, rels: List[str], n: int, seed: int = 7) -> List[Dict]:
    """T_copy-wrong follow rate and mean do_total effect of non-final gold hops (chain-writing models)."""
    task, split, ks, npk, sd = SETS["c6"]
    pool = D.load_clutrr(split, task=task)
    exs = D.sample_by_k(pool, ks, npk, sd)
    random.Random(0).shuffle(exs)
    exs = exs[:n]
    cands = [" " + r for r in rels]
    rng = random.Random(f"wc-{seed}")
    follow, ps, follow_ps, eff = [], [], [], []
    for e in exs:
        base = lm.prompt(msgs_zero(CLUTRR_SYSTEM, e))
        pre = e.answer_prefix()
        hops = [C.Hop(i + 1, h.text(), h.tail) for i, h in enumerate(e.gold_hops)]
        g = rels.index(e.answer)
        p_full = _dist(lm, base + C.render_chain(hops) + pre, cands)
        ps.append(float(np.argmax(p_full) == g))
        wrong = rng.choice([r for r in rels if r != e.answer])
        w_hops = hops + [C.Hop(len(hops) + 1, f"So {e.target} is {e.source}'s relative:", wrong)]
        p_w = _dist(lm, base + C.render_chain(w_hops) + pre, cands)
        f = float(np.argmax(p_w) == rels.index(wrong))
        follow.append(f)
        if ps[-1]:
            follow_ps.append(f)
        for j in range(len(hops) - 1):
            p_m = _dist(lm, base + C.render_chain(C.do_total(hops, j)) + pre, cands)
            eff.append(p_full[g] - p_m[g])
    mk = lambda metric, v: dict(arm=arm, metric=metric, value=float(np.mean(v)) if len(v) else float("nan"),
                                n=len(v))
    out = [mk("wrong_copy_follow_all", follow), mk("wrong_copy_follow_given_ps1", follow_ps),
           mk("ps_plain_chain", ps), mk("do_total_effect_nonfinal_hops", eff)]
    _log(f"  [{arm}] wrong-copy follow {100 * out[0]['value']:.1f}% (PS=1: {100 * out[1]['value']:.1f}%, "
         f"n={out[1]['n']}); do_total effect {100 * out[3]['value']:.1f} pts")
    return out


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arms", nargs="+", default=["s_ans", "s_dpo"])
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
    ap.add_argument("--extras_n", type=int, default=120)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--max_minutes", type=float, default=600.0)
    # DPO
    ap.add_argument("--dpo_pairs", type=int, default=1500)
    ap.add_argument("--dpo_beta", type=float, default=0.1)
    ap.add_argument("--dpo_lr", type=float, default=2e-5)
    ap.add_argument("--dpo_accum", type=int, default=8)
    ap.add_argument("--dpo_nll", type=float, default=0.2)
    # GRPO
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--rl_p", type=int, default=8)
    ap.add_argument("--rl_g", type=int, default=8)
    ap.add_argument("--rl_mb", type=int, default=8)
    ap.add_argument("--rl_lr", type=float, default=2e-5)
    ap.add_argument("--rl_minutes", type=float, default=150.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.outdir, "config.json"), "w"), indent=1)

    train = D.load_clutrr("train")
    vocab_l = D.clutrr_relations(train)
    table = extend_table(train, induce_table(train), vocab_l)
    pool = [e for e in train if 2 <= e.k <= 4]
    sample = D.sample_by_k(pool, [2, 3, 4], args.n_train // 3, args.seed)
    _log(f"{len(sample)} train chains (k=2-4)")

    all_rows: List[Dict] = []
    extras: List[Dict] = []
    info: Dict[str, Dict] = {}
    plain_mx = lambda k: 26 * (k + 2) + 16
    zero = lambda e: msgs_zero(CLUTRR_SYSTEM, e)

    def flush():
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
        summarize(df).round(3).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
        pd.DataFrame(extras).to_csv(os.path.join(args.outdir, "extras.csv"), index=False)
        json.dump(info, open(os.path.join(args.outdir, "summary.json"), "w"), indent=1)

    for arm in args.arms:
        if (time.time() - T0) / 60 > args.max_minutes:
            _log(f"[{arm}] SKIPPED: past max_minutes")
            continue
        _log(f"=== arm {arm}")
        lm = LM(args.model, device_map=args.device_map)
        if arm == "s_ans":
            data = [(lm.prompt(msgs_zero(ANS_SYSTEM, e)), ans_text(e)) for e in sample]
            info[arm] = train_sft(lm, data, args)
            all_rows += evaluate(lm, arm, lambda e: msgs_zero(ANS_SYSTEM, e), args, table, vocab_l, args.sets,
                                 bool(args.robust), lambda k: 0)
        else:
            data = [(lm.prompt(zero(e)), chain_text(e, table, False)) for e in sample]
            info[arm] = train_sft(lm, data, args)
            stage1 = "s_plain" if arm == "s_plain" else "sft_before_" + arm.split("_")[1]
            all_rows += evaluate(lm, stage1, zero, args, table, vocab_l, args.sets, bool(args.robust), plain_mx)
            extras += causal_extras(lm, stage1, vocab_l, args.extras_n)
            flush()
            if arm == "s_dpo":
                rng = random.Random(args.seed + 5)
                pairs = []
                for e in rng.sample(sample, min(args.dpo_pairs, len(sample))):
                    c, r = cf_pair(e, vocab_l, rng)
                    pairs.append((lm.prompt(zero(e)), c, r))
                info[arm]["dpo"] = train_dpo(lm, pairs, args)
            elif arm == "s_grpo":
                info[arm]["grpo"] = train_grpo(lm, sample, vocab_l, args)
            if arm != "s_plain":
                all_rows += evaluate(lm, arm, zero, args, table, vocab_l, args.sets, bool(args.robust), plain_mx)
                extras += causal_extras(lm, arm, vocab_l, args.extras_n)
        flush()
        del lm
        gc.collect()
        torch.cuda.empty_cache()
    if all_rows:
        print(summarize(pd.DataFrame(all_rows)).round(3).to_string())
        print(pd.DataFrame(extras).round(3).to_string())


if __name__ == "__main__":
    main()
