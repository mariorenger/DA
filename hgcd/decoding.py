"""Answer-level decoding rules over a finite candidate set.

All rules consume *normalized* candidate distributions computed once per example, so any
alpha can be evaluated post hoc (cheap sweeps, no extra forward passes).

Notation (per example):
  p_cot      p(a | Q, P, y)              model's own chain y
  p_dir      p(a | Q, P)                 direct answer, no chain
  p_noctx    p(a | Q)                    no context, no chain
  p_tot[j]   p(a | Q, P, do_total(y, j))
  p_nai[j]   p(a | Q, P, do_naive(y, j))
  p_pla[j]   p(a | Q, P, placebo(y, j))
  rel        indices of DRG-relevant hops (directed path to the answer)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

EPS = 1e-12


def softmax_logp(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max()
    e = np.exp(z)
    return e / e.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log2(a + EPS) - np.log2(b + EPS))))
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def _plausible(p: np.ndarray, beta: float) -> np.ndarray:
    return p >= beta * p.max()


def _argmax_masked(s: np.ndarray, mask: np.ndarray) -> int:
    s = np.where(mask, s, -np.inf)
    return int(np.argmax(s))


def log_contrast(p_pos: np.ndarray, p_neg: np.ndarray, alpha: float, beta: float) -> int:
    s = (1 + alpha) * np.log(p_pos + EPS) - alpha * np.log(p_neg + EPS)
    return _argmax_masked(s, _plausible(p_pos, beta))


def prob_contrast(p_pos: np.ndarray, p_neg: np.ndarray, alpha: float, beta: float) -> int:
    """Probability-space contrast (1+a) p - a p_neg (Prop. 1: cancels shortcut mixtures)."""
    s = (1 + alpha) * p_pos - alpha * p_neg
    return _argmax_masked(s, _plausible(p_pos, beta))


def _mix(ps: List[np.ndarray], idx: List[int]) -> Optional[np.ndarray]:
    if not idx:
        return None
    return np.mean([ps[j] for j in idx], axis=0)


METHODS = ["direct", "cot", "cad", "adacad", "cad_chain", "hgcd", "hgcd_log", "hgcd_naive",
           "hgcd_uniform", "hgcd_placebo", "hgcd_ada"]


def decide(d: Dict, method: str, alpha: float = 1.0, beta: float = 0.1) -> int:
    """Return the index of the chosen candidate for one example record `d`."""
    p_cot, p_dir, p_noctx = d["p_cot"], d["p_dir"], d["p_noctx"]
    n_hops = len(d["p_tot"])
    rel = d["rel"]
    if method == "direct":
        return int(np.argmax(p_dir))
    if method == "cot":
        return int(np.argmax(p_cot))
    if method == "sc":  # self-consistency majority vote; ties broken by p_cot
        votes = np.bincount(d["sc_votes"], minlength=len(p_cot)).astype(float)
        return int(np.argmax(votes + 1e-3 * p_cot))
    if method == "cad":
        return log_contrast(p_dir, p_noctx, alpha, beta)
    if method == "adacad":
        return log_contrast(p_dir, p_noctx, jsd(p_dir, p_noctx), beta)
    if method == "cad_chain":
        return log_contrast(p_cot, p_dir, alpha, beta)
    # ---- HGCD family: contrast the chain against do()-interventions on its own hops
    if n_hops == 0:
        return int(np.argmax(p_cot))
    if method == "hgcd":
        neg = _mix(d["p_tot"], rel)
    elif method == "hgcd_log":
        neg = _mix(d["p_tot"], rel)
        return int(np.argmax(p_cot)) if neg is None else log_contrast(p_cot, neg, alpha, beta)
    elif method == "hgcd_naive":
        neg = _mix(d["p_nai"], rel)
    elif method == "hgcd_uniform":
        neg = _mix(d["p_tot"], list(range(n_hops)))
    elif method == "hgcd_placebo":
        neg = _mix(d["p_pla"], rel)
    elif method == "hgcd_ada":
        neg = _mix(d["p_tot"], rel)
        if neg is None:
            return int(np.argmax(p_cot))
        return prob_contrast(p_cot, neg, alpha * jsd(p_cot, neg), beta)
    else:
        raise ValueError(method)
    if neg is None:
        return int(np.argmax(p_cot))
    return prob_contrast(p_cot, neg, alpha, beta)
