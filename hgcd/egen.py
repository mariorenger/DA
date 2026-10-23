"""Round 10b: does the structured-chain recipe generalise beyond CLUTRR? (MuSiQue, 2WikiMultihopQA)

Recipe: every step is a one-step lookup whose result is written down and read by the next step; the
model is LoRA-trained on such chains built automatically from annotated decompositions, and compared
with (i) answer-only SFT, (ii) SFT on plain hop chains, (iii) the base model with a few-shot chain.

    python -m hgcd.egen --dataset musique --model Qwen/Qwen2.5-3B-Instruct --outdir out \
        --arms p0 s_ans s_plain s_struct
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import metrics as M
from .cftrain import train_sft
from .lm import LM
from .prompts import musique_messages

T0 = time.time()

SYS_ANS = ("You answer multi-hop questions using the given paragraphs. Reply with the final answer "
           "only, in the form 'Answer: <answer>'.")
SYS_PLAIN = ("You answer multi-hop questions using the given paragraphs. Break the question into "
             "single-hop sub-questions, answer each in one line of the form "
             "'Hop i: <sub-question> => <answer>', then give the final answer as 'Answer: <answer>'.")
SYS_STRUCT = ("You answer multi-hop questions using the given paragraphs. Break the question into "
              "single-hop sub-questions. For each hop write 'Hop i: <sub-question> => <answer> "
              "[from: <title of the paragraph that states it>]'. The next sub-question must use the "
              "answer of the previous hop. Then give the final answer as 'Answer: <answer>'.")
_FROM = re.compile(r"\[from:\s*(.+?)\s*\]\s*$")


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class QA:
    uid: str
    question: str
    answer: str
    aliases: List[str]
    context: str                      # "[i] Title: text" lines
    titles: List[str]                 # titles shown in the context
    hops: List[tuple]                 # (hop text, result, source title or None)
    support: List[str]                # gold supporting titles
    kind: str = ""
    cond: str = "orig"

    @property
    def k(self) -> int:
        return len(self.hops)


def _norm_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", " ", t.lower()).strip()


def _find_title(name: str, titles: List[str]) -> Optional[str]:
    n = _norm_title(name)
    for t in titles:
        if _norm_title(t) == n:
            return t
    for t in titles:
        tn = _norm_title(t)
        if n and (n in tn or tn in n) and len(tn) > 2:
            return t
    return None


# ----------------------------------------------------------------------------- loaders

def load_musique_qa(split: str, n_dist: int = 4) -> List[QA]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("bdsaglam/musique", f"musique_ans_v1.0_{split}.jsonl", repo_type="dataset")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            paras = d["paragraphs"]
            sup = [x for x in paras if x["is_supporting"]]
            dis = [x for x in paras if not x["is_supporting"]]
            rng = random.Random(f"ms-dist-{d['id']}")
            rng.shuffle(dis)
            shown = sup + dis[:n_dist]
            random.Random(f"ms-order-{d['id']}-0").shuffle(shown)
            steps = d["question_decomposition"]
            answers = [s["answer"] for s in steps]
            hops = []
            for i, s in enumerate(steps):
                idx = s.get("paragraph_support_idx")
                title = paras[idx]["title"] if isinstance(idx, int) and 0 <= idx < len(paras) else None
                hops.append((D._render_subq(s["question"], answers[:i]), s["answer"], title))
            out.append(QA(uid=d["id"], question=d["question"], answer=d["answer"],
                          aliases=d.get("answer_aliases") or [],
                          context="\n".join(f"[{i + 1}] {x['title']}: {x['paragraph_text']}"
                                            for i, x in enumerate(shown)),
                          titles=[x["title"] for x in shown], hops=hops,
                          support=[x["title"] for x in sup], kind=f"{len(steps)}hop"))
    return out


def load_wiki2_qa(split: str, types=("compositional", "inference")) -> List[QA]:
    from huggingface_hub import hf_hub_download
    files = {"train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
             "validation": ["data/validation-00000-of-00001.parquet"]}[split]
    df = pd.concat([pd.read_parquet(hf_hub_download("framolfese/2WikiMultihopQA", f, repo_type="dataset"))
                    for f in files])
    out = []
    for r in df.itertuples():
        if r.type not in types or len(r.evidences) < 2:
            continue
        titles = list(r.context["title"])
        sents = [" ".join(s) for s in r.context["sentences"]]
        hops = []
        for ev in r.evidences:
            head, rel, tail = (str(x) for x in ev)
            hops.append((f"What is the {rel} of {head}?", tail, _find_title(head, titles)))
        out.append(QA(uid=str(r.id), question=r.question, answer=str(r.answer), aliases=[],
                      context="\n".join(f"[{i + 1}] {t}: {s}" for i, (t, s) in enumerate(zip(titles, sents))),
                      titles=titles, hops=hops, support=list(dict.fromkeys(r.supporting_facts["title"])),
                      kind=r.type))
    return out


# ----------------------------------------------------------------------------- prompts / targets

def user_msg(e: QA) -> str:
    return f"Paragraphs:\n{e.context}\nQuestion: {e.question}"


def msgs_zero(system: str, e: QA) -> List[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user_msg(e)}]


def target(e: QA, arm: str) -> Optional[str]:
    if arm == "s_ans":
        return f"Answer: {e.answer}"
    lines = []
    for i, (q, a, t) in enumerate(e.hops):
        if arm == "s_struct":
            if t is None:
                return None            # every step must cite the paragraph it reads
            lines.append(f"Hop {i + 1}: {q} => {a} [from: {t}]")
        else:
            lines.append(f"Hop {i + 1}: {q} => {a}")
    return "\n".join(lines) + f"\nAnswer: {e.answer}"


# ----------------------------------------------------------------------------- evaluation

def evaluate(lm: LM, arm: str, dataset: str, exs: Dict[str, List[QA]], args) -> List[Dict]:
    rows = []
    system = {"s_ans": SYS_ANS, "s_plain": SYS_PLAIN, "s_struct": SYS_STRUCT}.get(arm)
    for cond, items in exs.items():
        prompts = [lm.prompt(musique_messages(e.context, e.question) if arm == "p0"
                             else msgs_zero(system, e)) for e in items]
        if arm == "s_ans":
            chains = [""] * len(items)
        else:
            chains = []
            for s in range(0, len(items), args.bs):
                chains += lm.generate(prompts[s:s + args.bs], max_new_tokens=60 * 4 + 40,
                                      batch_size=args.bs)
        chains = [c.rstrip() + "\n" if c.strip() else "" for c in chains]
        ans = lm.generate([p + c + "Answer:" for p, c in zip(prompts, chains)], max_new_tokens=24,
                          batch_size=args.bs, stop="\n")
        for e, c, a in zip(items, chains, ans):
            pred = a.strip().strip(".")
            golds = [e.answer] + list(e.aliases)
            lines = [ln for ln in c.splitlines() if ln.startswith("Hop")]
            cites = [m.group(1) for ln in lines if (m := _FROM.search(ln))]
            last = lines[-1].split("=>")[-1] if lines else ""
            last = _FROM.sub("", last).strip().strip(".")
            rows.append(dict(arm=arm, dataset=dataset, cond=cond, uid=e.uid, kind=e.kind, k=e.k,
                             pred=pred, gold=e.answer, em=M.em(pred, golds), f1=M.f1(pred, golds),
                             cite_ok=float(np.mean([t in e.support for t in cites])) if cites else float("nan"),
                             last_hop_match=float(M._norm_ans(last) == M._norm_ans(pred)) if lines else float("nan"),
                             chain=c))
        _log(f"  [{arm}] {dataset}/{cond}: n={len(items)} EM={100 * np.mean([r['em'] for r in rows if r['cond'] == cond and r['arm'] == arm]):.1f}")
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, ds, cond), g in df.groupby(["arm", "dataset", "cond"]):
        m, lo, hi = M.boot_ci(g.em.values)
        out.append(dict(arm=arm, dataset=ds, cond=cond, n=len(g), em=100 * m, lo=100 * lo, hi=100 * hi,
                        f1=100 * g.f1.mean(), cite_ok=g.cite_ok.mean(), last_hop_match=g.last_hop_match.mean()))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["musique", "wiki2"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arms", nargs="+", default=["p0", "s_ans", "s_plain", "s_struct"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_prompt_tokens", type=int, default=1500)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", nargs="+", default=["orig"])
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--max_minutes", type=float, default=600.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.outdir, "config.json"), "w"), indent=1)

    if args.dataset == "musique":
        train = load_musique_qa("train")
        dev = {c: load_musique_qa("dev", n_dist={"orig": 4, "dist8": 8}[c]) for c in args.conds}
        rng = np.random.default_rng(3)   # same draw as the round-9 MuSiQue runs (seed 0 + 3)
        pick = rng.choice(len(dev["orig"] if "orig" in dev else next(iter(dev.values()))), args.n_eval,
                          replace=False)
        evals = {c: [v[i] for i in pick] for c, v in dev.items()}
    else:
        train = load_wiki2_qa("train")
        val = load_wiki2_qa("validation")
        pick = np.random.default_rng(3).choice(len(val), min(args.n_eval, len(val)), replace=False)
        evals = {"orig": [val[i] for i in pick]}
    _log(f"{args.dataset}: {len(train)} train, eval {[len(v) for v in evals.values()]}")

    rng = random.Random(args.seed)
    rng.shuffle(train)
    all_rows: List[Dict] = []
    info: Dict[str, Dict] = {}
    for arm in args.arms:
        if (time.time() - T0) / 60 > args.max_minutes:
            _log(f"[{arm}] SKIPPED: past max_minutes")
            continue
        _log(f"=== arm {arm}")
        lm = LM(args.model, device_map=args.device_map)
        if arm != "p0":
            system = {"s_ans": SYS_ANS, "s_plain": SYS_PLAIN, "s_struct": SYS_STRUCT}[arm]
            data, dropped = [], 0
            for e in train:
                t = target(e, arm)
                if t is None:
                    dropped += 1
                    continue
                p = lm.prompt(msgs_zero(system, e))
                if len(lm.tok(p, add_special_tokens=False).input_ids) > args.max_prompt_tokens:
                    dropped += 1
                    continue
                data.append((p, t))
                if len(data) >= args.n_train:
                    break
            _log(f"  {len(data)} training examples ({dropped} skipped: no citation / too long)")
            info[arm] = train_sft(lm, data, args)
            info[arm]["skipped"] = dropped
        all_rows += evaluate(lm, arm, args.dataset, evals, args)
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(args.outdir, "decisions.csv"), index=False)
        summarize(df).round(3).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
        json.dump(info, open(os.path.join(args.outdir, "summary.json"), "w"), indent=1)
        del lm
        gc.collect()
        torch.cuda.empty_cache()
    if all_rows:
        print(summarize(pd.DataFrame(all_rows)).round(2).to_string())


if __name__ == "__main__":
    main()
