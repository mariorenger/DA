"""End-to-end pipeline used by the Kaggle notebook and the CLI.

    python -m hgcd.pipeline --preset smoke      # CPU sanity run (minutes)
    python -m hgcd.pipeline --preset kaggle     # 1x T4, Qwen2.5-1.5B-Instruct (~2-4 h)
    python -m hgcd.pipeline --preset a100       # 1x A100, 7B model
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
import pandas as pd

from . import data as D
from . import experiments as E
from . import metrics as M
from . import report as R


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    outdir: str = "results/run"
    seed: int = 0
    batch_size: int = 8
    # CLUTRR decoding
    ks_test: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7])
    n_per_k: int = 40
    n_dev: int = 60                      # validation split (k in 2..4) for alpha selection
    conditions: List[str] = field(default_factory=lambda: ["orig", "rename", "order",
                                                           "distract", "placebo_input"])
    l1: bool = True
    l1_alpha: float = 1.0
    sc_n: int = 5                        # self-consistency samples (0 disables)
    alphas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    beta: float = 0.1
    # estimator validation (gold chains + inserted redundant hop)
    ks_est: List[int] = field(default_factory=lambda: [2, 3, 4])
    n_est_per_k: int = 30
    est_rollouts: int = 0
    # MuSiQue
    n_ms: int = 120
    ms_conditions: List[str] = field(default_factory=lambda: ["orig", "dist8", "order"])
    n_est_ms: int = 80
    run_musique: bool = True
    est_compose: bool = True             # copy-vs-compute: CLUTRR chains with the answer written in
    only_estimators: bool = False        # "reliance" study: estimator stage only
    device_map: str = ""                 # e.g. "auto" to shard a 7B model over 2 T4s


PRESETS = {
    "smoke": dict(model="Qwen/Qwen2.5-0.5B-Instruct", ks_test=[2, 3, 5], n_per_k=4, n_dev=6,
                  conditions=["orig", "rename", "distract"], l1=True, sc_n=3, ks_est=[2, 3],
                  n_est_per_k=4, est_rollouts=0, n_ms=6, ms_conditions=["orig", "dist8"],
                  n_est_ms=6, batch_size=4, alphas=[0.0, 0.5, 1.0, 2.0]),
    "kaggle": dict(),
    "kaggle_full": dict(n_per_k=80, n_dev=120, n_est_per_k=60, est_rollouts=4, n_ms=300,
                        n_est_ms=200),
    # Reliance-vs-scale study (cheap): run once per model size, estimator stage only.
    "reliance": dict(ks_est=[2, 3, 4], n_est_per_k=60, n_est_ms=150, est_rollouts=4,
                     only_estimators=True),
    "a100": dict(model="Qwen/Qwen2.5-7B-Instruct", ks_test=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                 n_per_k=100, n_dev=150, n_est_per_k=80, est_rollouts=4, n_ms=500,
                 n_est_ms=300, batch_size=16),
}


def _log(m):
    print(time.strftime("[%H:%M:%S] "), m, flush=True)


def _cached(path: str, fn):
    """Stage-level checkpoint: reuse a finished stage after a crash or session restart."""
    if os.path.exists(path):
        _log(f"reusing cached stage {os.path.basename(path)}")
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def select_alpha(recs, methods, alphas, beta, table_fn):
    rows = []
    for a in alphas:
        df = table_fn(recs, alpha=a, beta=beta, methods=methods)
        for m, g in df.groupby("method"):
            rows.append(dict(method=m, alpha=a, acc=g.correct.mean()))
    sweep = pd.DataFrame(rows)
    best = {m: float(g.sort_values(["acc", "alpha"], ascending=[False, True]).alpha.iloc[0])
            for m, g in sweep.groupby("method")}
    return best, sweep


def _table_with_alphas(recs, best, beta, table_fn):
    parts = []
    for m, a in best.items():
        parts.append(table_fn(recs, alpha=a, beta=beta, methods=[m.replace("l1+", "")]))
    df = pd.concat(parts)
    keep = set(best)
    return df[df.method.isin(keep)].drop_duplicates()


def run(cfg: Config):
    from .lm import LM
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    t0 = time.time()
    lm = LM(cfg.model, device_map=cfg.device_map or None)
    _log(f"loaded {cfg.model} on {lm.device}")
    summary = {"model": cfg.model}

    # ------------------------------------------------------------- data
    train = D.load_clutrr("train")
    test = D.load_clutrr("test")
    dev = D.load_clutrr("validation")
    rels = D.clutrr_relations(train + test)
    pool = train
    summary["clutrr_filter"] = {"train": len(train), "test": len(test), "dev": len(dev)}

    # ------------------------------------------------------------- 1) estimator validation
    est_ex = D.sample_by_k(test, cfg.ks_est, cfg.n_est_per_k, cfg.seed + 7)
    _log(f"estimator validation on {len(est_ex)} CLUTRR gold chains")
    est_rows = _cached(os.path.join(cfg.outdir, "stage_est_clutrr.pkl"),
                       lambda: E.run_estimators_clutrr(lm, est_ex, pool, rels,
                                                       rollouts=cfg.est_rollouts,
                                                       batch_size=cfg.batch_size, seed=cfg.seed))
    if cfg.est_compose:
        _log("estimator validation on CLUTRR chains + explicit answer hop (copy vs compute)")
        est_rows = est_rows + _cached(os.path.join(cfg.outdir, "stage_est_clutrr_copy.pkl"),
                                      lambda: E.run_estimators_clutrr(lm, est_ex, pool, rels,
                                                                      batch_size=cfg.batch_size,
                                                                      seed=cfg.seed, compose=True))
    ms = None
    if cfg.run_musique:
        ms = D.load_musique("dev")
        rng = np.random.default_rng(cfg.seed)
        ms_est = [ms[i] for i in rng.choice(len(ms), cfg.n_est_ms, replace=False)]
        _log(f"estimator validation on {len(ms_est)} MuSiQue gold chains")
        est_rows = est_rows + _cached(os.path.join(cfg.outdir, "stage_est_musique.pkl"),
                                      lambda: E.run_estimators_musique(lm, ms_est, seed=cfg.seed,
                                                                       rollouts=cfg.est_rollouts,
                                                                       batch_size=cfg.batch_size))
    pd.DataFrame(est_rows).to_csv(os.path.join(cfg.outdir, "estimators.csv"), index=False)
    aucs, pos = M.estimator_table(est_rows)
    R.estimator_latex(aucs, pos, os.path.join(cfg.outdir, "tab_estimators_auc.tex"),
                      os.path.join(cfg.outdir, "tab_estimators_pos.tex"))
    summary["estimator_auc"] = aucs.to_dict("records")
    summary["estimator_position"] = pos.to_dict("records")
    fc = M.form_content_table(est_rows)
    if len(fc):
        R.form_content_latex(fc, os.path.join(cfg.outdir, "tab_form_content.tex"))
        summary["form_content"] = fc.to_dict("records")
        _log("\n" + fc.to_string())
    _log("\n" + aucs.to_string())
    _log("\n" + pos.to_string())
    if cfg.only_estimators:
        summary["minutes"] = (time.time() - t0) / 60
        with open(os.path.join(cfg.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        _log(f"done (estimators only) in {summary['minutes']:.1f} min -> {cfg.outdir}")
        return summary

    # ------------------------------------------------------------- 2) alpha selection on dev
    dev_ex = D.sample_by_k(dev, [2, 3, 4], max(1, cfg.n_dev // 3), cfg.seed + 1)
    _log(f"alpha selection on {len(dev_ex)} CLUTRR validation examples")
    dev_recs = _cached(os.path.join(cfg.outdir, "stage_dev.pkl"),
                       lambda: E.run_clutrr(lm, dev_ex, pool, rels, conditions=["orig"],
                                            batch_size=cfg.batch_size, l1=False))
    best, sweep = select_alpha(dev_recs, M.GREEDY_METHODS, cfg.alphas, cfg.beta, M.clutrr_table)
    for m in M.GREEDY_METHODS:          # methods absent on dev (e.g. sc) need no alpha
        best.setdefault(m, 0.0)
    best["l1+cot"], best["l1+hgcd"] = 0.0, best.get("hgcd", 1.0)
    sweep.to_csv(os.path.join(cfg.outdir, "alpha_sweep_dev.csv"), index=False)
    summary["alpha"] = best
    _log(f"selected alphas: {best}")

    # ------------------------------------------------------------- 3) CLUTRR test
    test_ex = D.sample_by_k(test, cfg.ks_test, cfg.n_per_k, cfg.seed)
    _log(f"CLUTRR test: {len(test_ex)} examples x {len(cfg.conditions)} conditions")
    recs = _cached(os.path.join(cfg.outdir, "clutrr_records.pkl"),
                   lambda: E.run_clutrr(lm, test_ex, pool, rels, conditions=cfg.conditions,
                                        batch_size=cfg.batch_size, l1=cfg.l1,
                                        l1_alpha=cfg.l1_alpha, sc_n=cfg.sc_n))
    df = _table_with_alphas(recs, best, cfg.beta, M.clutrr_table)
    df.to_csv(os.path.join(cfg.outdir, "clutrr_decisions.csv"), index=False)
    s = R.main_table(df, os.path.join(cfg.outdir, "tab_clutrr_main.tex"))
    a = R.ate_latex(df[~df.method.str.startswith("l1+")],
                    os.path.join(cfg.outdir, "tab_clutrr_ate.tex"))
    R.figures(df, sweep, cfg.outdir, "clutrr")
    ro = R.readout_latex(df, os.path.join(cfg.outdir, "tab_clutrr_readout.tex"))
    summary["clutrr_readout"] = ro.to_dict("records")
    fx = M.fixed_records(recs)
    if fx:
        dff = _table_with_alphas(fx, {m: a for m, a in best.items() if not m.startswith("l1+")},
                                 cfg.beta, M.clutrr_table)
        med = M.mediation_table(df, dff)
        R.mediation_latex(med, os.path.join(cfg.outdir, "tab_clutrr_mediation.tex"))
        summary["clutrr_mediation"] = med.to_dict("records")
        _log("\n" + med[["cond", "method", "te", "nde", "nie"]].to_string())
    comps = [M.paired_compare(df, "hgcd", b) for b in ["cot", "sc", "cad_chain", "calib", "hgcd_placebo",
                                                        "hgcd_naive", "hgcd_uniform", "hgcd_log"]]
    if cfg.l1:
        comps.append(M.paired_compare(df, "l1+hgcd", "cot"))
    summary["clutrr_acc"] = s.to_dict("records")
    summary["clutrr_ate"] = a.to_dict("records")
    summary["clutrr_paired"] = comps
    o = df[df.cond == "orig"]
    summary["clutrr_format"] = {"fmt_ok": float(o.fmt_ok.mean()), "hop_fid": float(o.hop_fid.mean()),
                                "drg_connected": float(o.drg_connected.mean()),
                                "mean_hops": float(o.n_hops.mean())}
    _log("\n" + s.to_string())
    _log("\n" + pd.DataFrame(comps).to_string())

    # ------------------------------------------------------------- 4) MuSiQue
    if cfg.run_musique:
        rng = np.random.default_rng(cfg.seed + 3)
        ms_ex = [ms[i] for i in rng.choice(len(ms), cfg.n_ms, replace=False)]
        _log(f"MuSiQue: {len(ms_ex)} examples x {len(cfg.ms_conditions)} conditions")
        mrecs = _cached(os.path.join(cfg.outdir, "musique_records.pkl"),
                        lambda: E.run_musique(lm, ms_ex, conditions=cfg.ms_conditions,
                                              batch_size=max(1, cfg.batch_size // 2)))
        mbest = {m: best.get(m, 1.0) for m in M.GREEDY_METHODS}
        mdf = _table_with_alphas(mrecs, mbest, cfg.beta, M.musique_table)
        free = M.musique_table(mrecs, methods=["cot"])
        mdf = pd.concat([mdf, free[free.method == "free"]])
        mdf.to_csv(os.path.join(cfg.outdir, "musique_decisions.csv"), index=False)
        ms_s = R.main_table(mdf, os.path.join(cfg.outdir, "tab_musique_main.tex"), col="em",
                            by_split=False)
        ms_a = R.ate_latex(mdf, os.path.join(cfg.outdir, "tab_musique_ate.tex"), col="em")
        mfx = M.fixed_records(mrecs)
        if mfx:
            mdff = _table_with_alphas(mfx, mbest, cfg.beta, M.musique_table)
            mmed = M.mediation_table(mdf, mdff, col="em")
            R.mediation_latex(mmed, os.path.join(cfg.outdir, "tab_musique_mediation.tex"))
            summary["musique_mediation"] = mmed.to_dict("records")
            _log("\n" + mmed[["cond", "method", "te", "nde", "nie"]].to_string())
        summary["musique_em"] = ms_s.to_dict("records")
        summary["musique_ate"] = ms_a.to_dict("records")
        summary["musique_cand_recall"] = float(mdf[mdf.cond == "orig"].cand_recall.mean())
        summary["musique_paired"] = [M.paired_compare(mdf, "hgcd", b, col="em")
                                     for b in ["cot", "cad", "cad_chain", "hgcd_placebo", "hgcd_naive"]]
        _log("\n" + ms_s.to_string())

    summary["boundary_fallbacks"] = lm.n_boundary_fallback
    summary["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(cfg.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    _log(f"done in {summary['minutes']:.1f} min -> {cfg.outdir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device_map", default="")
    args = ap.parse_args()
    cfg = Config(**PRESETS[args.preset])
    if args.model:
        cfg.model = args.model
    cfg.outdir = args.outdir or f"results/{args.preset}"
    cfg.device_map = args.device_map
    run(cfg)


if __name__ == "__main__":
    main()
