"""Hop-structured chains, the Directed Reasoning Graph (DRG) and do()-interventions.

A chain is a list of hops rendered as lines

    Hop i: <body> => <result>

DRG edge i -> j  (i < j)  iff  result_i is mentioned in body_j.

Interventions on hop j
  do_naive(j)  : mask hop j only. Estimates the *controlled direct effect* of h_j on the
                 answer, holding all downstream hops fixed.
  do_total(j)  : mask hop j and, in every DRG-descendant d of j, mask every mention of
                 result_j and of result_d (descendant values are functions of h_j, so under
                 do(h_j) they are unknown). Deterministic approximation of the *total effect*
                 without re-generating the downstream chain.
  placebo(j)   : content-preserving rewrite of hop j (true effect = 0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set

MASK = "[MASKED]"
HOP_RE = re.compile(r"^\s*Hop\s*(\d+)\s*:\s*(.*?)\s*=>\s*(.+?)\s*$")


@dataclass
class Hop:
    idx: int
    body: str
    result: str


def _clean_result(r: str) -> str:
    r = r.strip().strip(".").strip()
    r = re.sub(r"\s*\(.*?\)\s*$", "", r)  # drop trailing "(relation)" annotations
    return r.strip()


def parse_chain(text: str) -> List[Hop]:
    hops = []
    for line in text.splitlines():
        if line.strip().lower().startswith("answer"):
            break
        m = HOP_RE.match(line)
        if m:
            res = _clean_result(m.group(3))
            if res:
                hops.append(Hop(int(m.group(1)), m.group(2).strip(), res))
    return hops


def render_chain(hops: List[Hop]) -> str:
    return "".join(f"Hop {i + 1}: {h.body} => {h.result}\n" for i, h in enumerate(hops))


def _mentions(text: str, ent: str) -> bool:
    if not ent or ent == MASK or len(ent) < 2:
        return False
    return re.search(_ent_pattern(ent), text, flags=re.IGNORECASE) is not None


def _ent_pattern(ent: str) -> str:
    pre = r"\b" if ent[0].isalnum() else ""
    post = r"\b" if ent[-1].isalnum() else ""
    return pre + re.escape(ent) + post


def _mask_ents(text: str, ents: Set[str]) -> str:
    for e in sorted(ents, key=len, reverse=True):
        if e and e != MASK and len(e) >= 2:
            text = re.sub(_ent_pattern(e), MASK, text, flags=re.IGNORECASE)
    return text


# ----------------------------------------------------------------------------- DRG

def drg_edges(hops: List[Hop]) -> List[tuple]:
    return [(i, j) for j in range(len(hops)) for i in range(j)
            if _mentions(hops[j].body, hops[i].result)]


def descendants(hops: List[Hop], j: int) -> List[int]:
    children: Dict[int, List[int]] = {}
    for a, b in drg_edges(hops):
        children.setdefault(a, []).append(b)
    seen, stack = set(), [j]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return sorted(seen)


def ancestors(hops: List[Hop], j: int) -> List[int]:
    parents: Dict[int, List[int]] = {}
    for a, b in drg_edges(hops):
        parents.setdefault(b, []).append(a)
    seen, stack = set(), [j]
    while stack:
        for p in parents.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return sorted(seen)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def relevant_hops(hops: List[Hop], target: Optional[str] = None) -> List[int]:
    """Hops with a directed DRG path to the answer (ancestors-or-self of terminal hops).

    Terminal hops: hops whose result is the query target (CLUTRR); else the last hop.
    """
    if not hops:
        return []
    terms = [i for i, h in enumerate(hops) if target and norm(h.result) == norm(target)]
    if not terms:
        terms = [len(hops) - 1]
    rel = set()
    for t in terms:
        rel.add(t)
        rel.update(ancestors(hops, t))
    return sorted(rel)


# ----------------------------------------------------------------------------- do()

def do_naive(hops: List[Hop], j: int) -> List[Hop]:
    out = list(hops)
    out[j] = replace(out[j], body=MASK, result=MASK)
    return out


def do_total(hops: List[Hop], j: int) -> List[Hop]:
    desc = descendants(hops, j)
    ents = {hops[j].result} | {hops[d].result for d in desc}
    out = list(hops)
    out[j] = replace(out[j], body=MASK, result=MASK)
    for d in desc:
        out[d] = replace(out[d], body=_mask_ents(out[d].body, ents),
                         result=_mask_ents(out[d].result, ents))
    return out


CLUTRR_HOP_RE = re.compile(r"^(.+?)'s ([\w-]+) is (.+?)\.?$")


def placebo(hops: List[Hop], j: int) -> List[Hop]:
    """Content-preserving rewrite of hop j (same entities, same relation)."""
    out = list(hops)
    b = hops[j].body
    m = CLUTRR_HOP_RE.match(b)
    if m:
        nb = f"{m.group(3)} is the {m.group(2)} of {m.group(1)}."
    elif b.endswith("?"):
        nb = "To restate: " + b
    else:
        nb = "In other words, " + b
    out[j] = replace(out[j], body=nb)
    return out


def mask_marker(hops: List[Hop], j: int) -> List[Hop]:
    """Artifact control: insert the mask token into hop j WITHOUT removing any content.
    Its effect isolates the out-of-distribution cost of the [MASKED] token itself."""
    out = list(hops)
    out[j] = replace(out[j], body=f"{MASK} {hops[j].body}")
    return out
