"""Evaluation: accuracy, EM/F1, paired effects with bootstrap CIs, McNemar, AUC."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .decoding import decide

GREEDY_METHODS = ["direct", "cot", "sc", "cad", "adacad", "cad_chain", "calib", "poe", "hgcd", "hgcd_log",
                  "hgcd_naive", "hgcd_uniform", "hgcd_placebo", "hgcd_ada"]
NEEDS_FULL = {"hgcd_naive", "hgcd_placebo"}


# ----------------------------------------------------------------------------- answers

def _norm_ans(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: Iterable[str]) -> float:
    return float(any(_norm_ans(pred) == _norm_ans(g) for g in golds))


def f1(pred: str, golds: Iterable[str]) -> float:
    best = 0.0
    pt = _norm_ans(pred).split()
    for g in golds:
        gt = _norm_ans(g).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        p, r = ns / len(pt), ns / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


# ----------------------------------------------------------------------------- statistics

def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired binary outcomes."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(s)
    if len(set(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


# ----------------------------------------------------------------------------- decision tables

def clutrr_table(recs: List[Dict], alpha: float = 1.0, beta: float = 0.1,
                 methods: Optional[List[str]] = None) -> pd.DataFrame:
    methods = methods or GREEDY_METHODS
    rows = []
    for r in recs:
        for m in methods:
            if r["chain_src"] == "l1" and m not in ("cot", "hgcd"):
                continue
            if m in NEEDS_FULL and r.get("p_nai") is None:
                continue
            if m == "sc" and not r.get("sc_votes"):
                continue
            i = decide(r, m, alpha, beta)
            name = m if r["chain_src"] == "greedy" else f"l1+{m}"
            rows.append(dict(uid=r["uid"], k=r["k"], cond=r["cond"], method=name,
                             correct=int(i == r["gold"]), fmt_ok=int(r["fmt_ok"]),
                             n_hops=r["n_hops"], hop_fid=r.get("hop_fid", np.nan),
                             drg_connected=int(r.get("drg_connected", False))))
    return pd.DataFrame(rows)


def musique_table(recs: List[Dict], alpha: float = 1.0, beta: float = 0.1,
                  methods: Optional[List[str]] = None) -> pd.DataFrame:
    methods = methods or GREEDY_METHODS
    rows = []
    for r in recs:
        golds = [r["gold_answer"]] + list(r["aliases"])
        rows.append(dict(uid=r["uid"], k=r["k"], cond=r["cond"], method="free",
                         em=em(r["free_answer"], golds), f1=f1(r["free_answer"], golds),
                         cand_recall=max(em(c, golds) for c in r["cands"])))
        for m in methods:
            if m in NEEDS_FULL and r.get("p_nai") is None:
                continue
            if m == "sc" and not r.get("sc_votes"):
                continue
            pred = r["cands"][decide(r, m, alpha, beta)]
            rows.append(dict(uid=r["uid"], k=r["k"], cond=r["cond"], method=m,
                             em=em(pred, golds), f1=f1(pred, golds),
                             cand_recall=max(em(c, golds) for c in r["cands"])))
    df = pd.DataFrame(rows)
    df["correct"] = df["em"]
    return df


def accuracy_summary(df: pd.DataFrame, by: List[str], col: str = "correct") -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by):
        m, lo, hi = boot_ci(g[col].values)
        key = key if isinstance(key, tuple) else (key,)
        out.append(dict(zip(by, key), mean=m, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(out)


def ate_table(df: pd.DataFrame, base_cond: str = "orig", col: str = "correct") -> pd.DataFrame:
    """Paired ATE of each binary intervention per method: E[Y(1) - Y(0)] (truth = 0)."""
    out = []
    base = df[df.cond == base_cond].set_index(["uid", "method"])[col]
    for cond in sorted(set(df.cond) - {base_cond}):
        cur = df[df.cond == cond].set_index(["uid", "method"])[col]
        j = pd.concat([base.rename("y0"), cur.rename("y1")], axis=1, join="inner").reset_index()
        for m, g in j.groupby("method"):
            d = (g.y1 - g.y0).values
            mean, lo, hi = boot_ci(d)
            out.append(dict(cond=cond, method=m, ate=mean, lo=lo, hi=hi, abs_ate=abs(mean),
                            n=len(g)))
    return pd.DataFrame(out)


def paired_compare(df: pd.DataFrame, a: str, b: str, cond: str = "orig",
                   col: str = "correct") -> Dict:
    x = df[(df.cond == cond) & (df.method == a)].set_index("uid")[col]
    y = df[(df.cond == cond) & (df.method == b)].set_index("uid")[col]
    j = pd.concat([x.rename("a"), y.rename("b")], axis=1, join="inner")
    mean, lo, hi = boot_ci((j.a - j.b).values)
    return dict(a=a, b=b, diff=mean, lo=lo, hi=hi, p_mcnemar=mcnemar_p(j.a.values > 0.5,
                                                                         j.b.values > 0.5),
                n=len(j))


def estimator_table(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    aucs = []
    for ds, g in df.groupby("dataset"):
        g2 = g[g.full_correct == 1]  # PS = 1 condition (as in PNS identifiability)
        for est in ["d_total", "d_naive", "d_cf_direct", "d_cf_total", "pns_rollout", "d_placebo",
                    "d_maskmark", "drg_relevant"]:
            if est not in g or g[est].isna().all():
                continue
            aucs.append(dict(dataset=ds, estimator=est, auc_all=auc(g[est], g.label),
                             auc_ps1=auc(g2[est], g2.label), n_hops=len(g), n_ps1=len(g2)))
    df["position"] = np.where(df.label == 0, "redundant",
                              np.where(df.is_final == 1, "gold-final", "gold-nonfinal"))
    cols = [c for c in ["d_total", "d_naive", "d_cf_total", "d_cf_direct", "pns_rollout",
                         "d_placebo", "d_maskmark"] if c in df]
    pos = df.groupby(["dataset", "position"])[cols].mean().reset_index()
    return pd.DataFrame(aucs), pos


def fixed_records(recs: List[Dict]) -> List[Dict]:
    """Records whose chain is the ORIGINAL chain M(0) read under the intervened input."""
    out = []
    for r in recs:
        if r.get("fixed") is None or r.get("chain_src", "greedy") != "greedy":
            continue
        d = dict(r)
        d.update(r["fixed"])
        d["p_nai"] = d["p_pla"] = None
        d["sc_votes"] = None
        out.append(d)
    return out


def mediation_table(df: pd.DataFrame, df_fixed: pd.DataFrame, col: str = "correct",
                    base_cond: str = "orig") -> pd.DataFrame:
    """Decompose each intervention's total effect into a direct effect on the readout (NDE,
    chain held at M(0)) and an indirect effect through the regenerated chain (NIE = TE - NDE).
    Because the mediator is set exactly, Y(1, M(0)) is observed for every unit."""
    y0 = df[df.cond == base_cond].set_index(["uid", "method"])[col].rename("y0")
    out = []
    for cond in sorted(set(df_fixed.cond)):
        y1 = df[df.cond == cond].set_index(["uid", "method"])[col].rename("y1")
        yf = df_fixed[df_fixed.cond == cond].set_index(["uid", "method"])[col].rename("yf")
        j = pd.concat([y0, y1, yf], axis=1, join="inner").reset_index()
        for m, g in j.groupby("method"):
            te, te_lo, te_hi = boot_ci((g.y1 - g.y0).values)
            nde, nde_lo, nde_hi = boot_ci((g.yf - g.y0).values)
            nie, nie_lo, nie_hi = boot_ci((g.y1 - g.yf).values)
            out.append(dict(cond=cond, method=m, te=te, te_lo=te_lo, te_hi=te_hi, nde=nde,
                            nde_lo=nde_lo, nde_hi=nde_hi, nie=nie, nie_lo=nie_lo, nie_hi=nie_hi,
                            n=len(g)))
    return pd.DataFrame(out)


def form_content_table(rows: List[Dict]) -> pd.DataFrame:
    """Decompose the effect of writing a chain on p(gold):
    form    = p(gold | chain of the same shape, all hops masked) - p(gold | no chain)
    content = p(gold | real chain) - p(gold | masked chain of the same shape)."""
    df = pd.DataFrame(rows).drop_duplicates(["dataset", "uid"])
    if "p_gold_shape" not in df:
        return pd.DataFrame()
    df["form"] = df.p_gold_shape - df.p_gold_nochain
    df["content"] = df.p_full_gold - df.p_gold_shape
    out = []
    for ds, g in df.groupby("dataset"):
        f, flo, fhi = boot_ci(g.form.values)
        c, clo, chi = boot_ci(g.content.values)
        out.append(dict(dataset=ds, form=f, form_lo=flo, form_hi=fhi, content=c, content_lo=clo,
                        content_hi=chi, n=len(g)))
    return pd.DataFrame(out)
