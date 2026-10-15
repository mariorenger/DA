"""Dataset loading and controlled (binary) interventions.

CLUTRR  : kinship chains with a gold path graph (controlled domain).
MuSiQue : real multi-hop QA with gold decompositions (bridge-entity chains).

Every intervention is a deterministic, seeded program (no LLM involved), so the
ground-truth effect of each intervention is known by construction:
  rename / order / distract / placebo_input  ->  true effect on the answer = 0.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

CLUTRR_URL = ("https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/"
              "main/{task}/{split}.csv")
CLUTRR_TASK = "gen_train234_test2to10"

# Fictional names (not used by CLUTRR) for renaming / distractor families.
FRESH_NAMES = {
    "male": ["Orrin", "Thadeus", "Brannock", "Celwyn", "Doran", "Eldric", "Fennick", "Garrow",
             "Halvard", "Ivor", "Jorvik", "Kestrel", "Lorcan", "Madoc", "Norris", "Osric",
             "Perrin", "Quillon", "Roderick", "Soren", "Tobias", "Ulric", "Varek", "Wystan",
             "Yorick", "Zander", "Alaric", "Bastian", "Corwin", "Dashiel"],
    "female": ["Aveline", "Brisa", "Calla", "Delphine", "Elowen", "Fenna", "Giselle", "Hesper",
               "Isolde", "Junia", "Kerensa", "Liora", "Maelis", "Nerys", "Oriel", "Perpetua",
               "Rosalind", "Sabine", "Talitha", "Undine", "Verity", "Winsome", "Xanthe", "Yseult",
               "Zinnia", "Amabel", "Briallen", "Cressida", "Damaris", "Evadne"],
}

NEUTRAL_SENTENCES = [
    "It was a sunny afternoon.",
    "The town had a small library near the river.",
    "Everyone agreed that the soup was delicious.",
    "The bus arrived a few minutes late that day.",
    "A light rain fell during the evening.",
]


# ----------------------------------------------------------------------------- utils

def _cache_dir() -> str:
    d = os.environ.get("HGCD_DATA", os.path.join(os.path.expanduser("~"), ".cache", "hgcd"))
    os.makedirs(d, exist_ok=True)
    return d


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def strip_brackets(text: str) -> str:
    return text.replace("[", "").replace("]", "")


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _sub_names(text: str, mapping: Dict[str, str]) -> str:
    """Replace bracketed CLUTRR names ([Name]) according to mapping."""
    def rep(m):
        n = m.group(1)
        return "[" + mapping.get(n, n) + "]"
    return re.sub(r"\[([^\]]+)\]", rep, text)


# ----------------------------------------------------------------------------- CLUTRR

@dataclass
class GoldHop:
    head: str
    rel: str
    tail: str

    def text(self) -> str:
        return f"{self.head}'s {self.rel} is {self.tail}."


@dataclass
class ClutrrExample:
    uid: str
    story: str                    # raw story with [Name] brackets
    names: List[str]              # entity names, index order
    genders: Dict[str, str]
    source: str                   # query[0]
    target: str                   # query[1]
    answer: str                   # relation: target is source's <answer>
    gold_hops: List[GoldHop]
    condition: str = "orig"
    extra_facts: List[GoldHop] = field(default_factory=list)  # true facts off the path

    @property
    def k(self) -> int:
        return len(self.gold_hops)

    @property
    def story_text(self) -> str:
        return strip_brackets(self.story)

    def question(self) -> str:
        return f"How is {self.target} related to {self.source}?"

    def answer_prefix(self) -> str:
        return f"Answer: {self.target} is {self.source}'s"


def _parse_clutrr_row(row: dict) -> Optional[ClutrrExample]:
    edges = ast.literal_eval(row["story_edges"])
    qedge = ast.literal_eval(row["query_edge"])
    if edges != [(i, i + 1) for i in range(len(edges))] or qedge != (0, len(edges)):
        return None  # keep only clean simple-path stories (documented filter)
    rels = ast.literal_eval(row["edge_types"])
    gpairs = [g.split(":") for g in row["genders"].split(",")]
    names = [g[0] for g in gpairs]
    genders = {g[0]: g[1] for g in gpairs}
    if len(names) != len(edges) + 1:
        return None
    hops = [GoldHop(names[i], rels[i], names[i + 1]) for i in range(len(edges))]
    return ClutrrExample(uid=row["id"], story=row["story"], names=names, genders=genders,
                         source=names[0], target=names[-1], answer=row["target_text"],
                         gold_hops=hops)


def load_clutrr(split: str = "test", task: str = CLUTRR_TASK) -> List[ClutrrExample]:
    import csv
    path = os.path.join(_cache_dir(), f"clutrr_{task}_{split}.csv")
    _download(CLUTRR_URL.format(task=task, split=split), path)
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ex = _parse_clutrr_row(row)
            if ex is not None:
                out.append(ex)
    return out


def clutrr_relations(examples: List[ClutrrExample]) -> List[str]:
    rels = set()
    for ex in examples:
        rels.add(ex.answer)
        rels.update(h.rel for h in ex.gold_hops)
    return sorted(rels)


def sample_by_k(examples: List[ClutrrExample], ks: List[int], n_per_k: int,
                seed: int = 0) -> List[ClutrrExample]:
    rng = random.Random(seed)
    out = []
    for k in ks:
        pool = [e for e in examples if e.k == k]
        rng.shuffle(pool)
        out.extend(pool[:n_per_k])
    return out


# --- interventions (all binary, seeded, ground-truth effect on the answer = 0)

def _fresh_mapping(names: List[str], genders: Dict[str, str], rng: random.Random,
                   avoid: Optional[set] = None) -> Dict[str, str]:
    avoid = set(avoid or ())
    pools = {g: [n for n in FRESH_NAMES[g] if n not in avoid] for g in FRESH_NAMES}
    for g in pools:
        rng.shuffle(pools[g])
    mapping = {}
    for n in names:
        g = genders.get(n, "male")
        mapping[n] = pools[g].pop() if pools[g] else n + "x"
    return mapping


def rename(ex: ClutrrExample, seed: int = 0) -> ClutrrExample:
    rng = random.Random(f"rename-{ex.uid}-{seed}")
    m = _fresh_mapping(ex.names, ex.genders, rng)
    return replace(ex, story=_sub_names(ex.story, m), names=[m[n] for n in ex.names],
                   genders={m[n]: g for n, g in ex.genders.items()},
                   source=m[ex.source], target=m[ex.target],
                   gold_hops=[GoldHop(m[h.head], h.rel, m[h.tail]) for h in ex.gold_hops],
                   condition="rename")


def reorder(ex: ClutrrExample, seed: int = 0) -> ClutrrExample:
    rng = random.Random(f"order-{ex.uid}-{seed}")
    sents = split_sentences(ex.story)
    if len(sents) > 1:
        orig = list(sents)
        for _ in range(10):
            rng.shuffle(sents)
            if sents != orig:
                break
    return replace(ex, story=" ".join(sents), condition="order")


def _distractor_family(ex: ClutrrExample, pool: List[ClutrrExample], rng: random.Random):
    """A disconnected family from another story, renamed with fresh names."""
    other = rng.choice([o for o in pool if o.uid != ex.uid and o.k <= 3] or pool)
    m = _fresh_mapping(other.names, other.genders, rng, avoid=set(ex.names))
    story = _sub_names(other.story, m)
    facts = [GoldHop(m[h.head], h.rel, m[h.tail]) for h in other.gold_hops]
    return story, facts, {m[n]: g for n, g in other.genders.items()}


def distract(ex: ClutrrExample, pool: List[ClutrrExample], seed: int = 0) -> ClutrrExample:
    rng = random.Random(f"distract-{ex.uid}-{seed}")
    dstory, facts, dgen = _distractor_family(ex, pool, rng)
    sents = split_sentences(ex.story)
    for s in split_sentences(dstory):
        sents.insert(rng.randint(0, len(sents)), s)
    g = dict(ex.genders)
    g.update(dgen)
    return replace(ex, story=" ".join(sents), genders=g, extra_facts=facts, condition="distract")


def placebo_input(ex: ClutrrExample, seed: int = 0) -> ClutrrExample:
    rng = random.Random(f"placebo-{ex.uid}-{seed}")
    sents = split_sentences(ex.story)
    for s in rng.sample(NEUTRAL_SENTENCES, 2):
        sents.insert(rng.randint(0, len(sents)), s)
    return replace(ex, story=" ".join(sents), condition="placebo_input")


def counterfactual_story(ex: ClutrrExample, seed: int = 0) -> str:
    """Type-matched counterfactual context for L1 contrast.

    Same sentences, same length, same entity *set* and genders, but names are permuted
    within each gender group (cyclic shift), so who-is-related-to-whom changes. Singleton
    gender groups are mapped to a fresh name of the same gender.
    """
    rng = random.Random(f"cf-{ex.uid}-{seed}")
    groups: Dict[str, List[str]] = {}
    for n in ex.names:
        groups.setdefault(ex.genders.get(n, "male"), []).append(n)
    m: Dict[str, str] = {}
    for g, ns in groups.items():
        if len(ns) >= 2:
            sh = list(ns)
            rng.shuffle(sh)
            for a, b in zip(sh, sh[1:] + sh[:1]):
                m[a] = b
        else:
            m[ns[0]] = rng.choice([x for x in FRESH_NAMES[g] if x not in ex.names])
    return strip_brackets(_sub_names(ex.story, m))


def apply_condition(ex: ClutrrExample, cond: str, pool: List[ClutrrExample],
                    seed: int = 0) -> ClutrrExample:
    if cond == "orig":
        return ex
    if cond == "rename":
        return rename(ex, seed)
    if cond == "order":
        return reorder(ex, seed)
    if cond == "distract":
        return distract(ex, pool, seed)
    if cond == "placebo_input":
        return placebo_input(ex, seed)
    raise ValueError(cond)


def with_redundant_fact(ex: ClutrrExample, pool: List[ClutrrExample],
                        seed: int = 0) -> Tuple[ClutrrExample, GoldHop]:
    """Add a disconnected family to the story and return one true-but-irrelevant fact.

    Used to build gold chains with an inserted redundant hop whose true necessity is 0.
    """
    ex2 = distract(ex, pool, seed)
    return ex2, ex2.extra_facts[0]


# ----------------------------------------------------------------------------- MuSiQue

@dataclass
class MusiqueExample:
    uid: str
    question: str
    answer: str
    aliases: List[str]
    paragraphs: List[dict]        # {'title','paragraph_text','is_supporting'}
    gold_hops: List[Tuple[str, str]]   # (sub-question rendered, sub-answer)
    condition: str = "orig"
    n_dist: int = 4
    order_seed: int = 0

    @property
    def k(self) -> int:
        return len(self.gold_hops)

    def context(self) -> List[dict]:
        sup = [p for p in self.paragraphs if p["is_supporting"]]
        dis = [p for p in self.paragraphs if not p["is_supporting"]]
        rng = random.Random(f"ms-dist-{self.uid}")
        rng.shuffle(dis)
        paras = sup + dis[: self.n_dist]
        rng2 = random.Random(f"ms-order-{self.uid}-{self.order_seed}")
        rng2.shuffle(paras)
        return paras

    def context_text(self) -> str:
        return "\n".join(f"[{i + 1}] {p['title']}: {p['paragraph_text']}"
                         for i, p in enumerate(self.context()))

    def distractor_titles(self) -> List[str]:
        return [p["title"] for p in self.context() if not p["is_supporting"]]

    def answer_prefix(self) -> str:
        return "Answer:"


def _render_subq(q: str, answers: List[str]) -> str:
    for i, a in enumerate(answers):
        q = q.replace(f"#{i + 1}", a)
    if ">>" in q:
        ent, rel = [s.strip() for s in q.split(">>", 1)]
        return f"What is the {rel} of {ent}?"
    q = q.strip()
    return q if q.endswith("?") else q + "?"


def load_musique(split: str = "dev") -> List[MusiqueExample]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("bdsaglam/musique", f"musique_ans_v1.0_{split}.jsonl", repo_type="dataset")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            answers = [s["answer"] for s in d["question_decomposition"]]
            hops = [(_render_subq(s["question"], answers[:i]), s["answer"])
                    for i, s in enumerate(d["question_decomposition"])]
            out.append(MusiqueExample(uid=d["id"], question=d["question"], answer=d["answer"],
                                      aliases=d.get("answer_aliases") or [],
                                      paragraphs=d["paragraphs"], gold_hops=hops))
    return out


def musique_condition(ex: MusiqueExample, cond: str) -> MusiqueExample:
    """MuSiQue interventions: number of distractor paragraphs and paragraph order."""
    if cond == "orig":
        return replace(ex, condition="orig", n_dist=4, order_seed=0)
    if cond == "dist0":
        return replace(ex, condition="dist0", n_dist=0)
    if cond == "dist8":
        return replace(ex, condition="dist8", n_dist=8)
    if cond == "order":
        return replace(ex, condition="order", order_seed=1)
    raise ValueError(cond)
