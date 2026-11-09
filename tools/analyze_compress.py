"""Round 14 analysis (H-P1..H-P5, pre-registered in REVIEW.md). python tools/analyze_compress.py [results/compress]"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/compress"
B = 2000
rng = np.random.default_rng(0)


def boot(per_q: pd.Series, per_q2: pd.Series = None):
    """mean (difference) over questions with a percentile bootstrap CI."""
    x = per_q.values if per_q2 is None else (per_q - per_q2).dropna().values
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def fmt(t, scale=100):
    return f"{t[0] * scale:+.1f} [{t[1] * scale:+.1f}, {t[2] * scale:+.1f}]"


out = []
for d in sorted(glob.glob(f"{ROOT}/compress_*/")):
    name = os.path.basename(d.rstrip("/")).replace("compress_", "")
    if not os.path.exists(d + "steps.csv"):
        continue
    st = pd.read_csv(d + "steps.csv")
    pr = pd.read_csv(d + "prune.csv")
    nq = st.uid.nunique()
    pl = st[st.planted == 1]
    nonpl = st[st.planted == 0]
    r = dict(model=name, n_chains=nq, n_steps=len(st))
    # H-P1: Yu's PN on planted duplicates vs calibrated
    r["yu_planted"] = pl.yu.mean()
    r["cal_planted"] = pl.cal.mean()
    r["yu_nonplanted"] = nonpl.yu.mean()
    r["alt_found"] = st.yu.notna().mean()
    # H-P2: difficulty confound (needs keep accuracy at the planted step)
    ok = pl.dropna(subset=["yu"])
    r["rho_yu_vs_difficulty"] = spearmanr(ok.yu, 1 - ok.keep_acc)[0] if len(ok) > 5 else np.nan
    r["regen_success"] = nonpl.keep_acc.mean()          # b(j): P(correct | keep prefix) for all steps
    # pruning tables
    P = pr.copy()
    piv = lambda col, bud: P[P.budget == bud].pivot_table(index="uid", columns="method", values=col)
    for bud in ["m1", "m30", "m50"]:
        a, rem = piv("acc", bud), piv("planted_removed", bud)
        tk = piv("tok_frac", bud)
        for m in ["random", "entropy", "yu", "cal", "loo", "loo_g"]:
            r[f"{bud}_acc_{m}"] = a[m].mean()
            r[f"{bud}_tok_{m}"] = tk[m].mean()
        if bud == "m1":
            for m in ["random", "entropy", "yu", "cal", "loo", "loo_g"]:
                r[f"m1_planted_{m}"] = rem[m].mean()
            r["HP3_cal_minus_yu_planted"] = fmt(boot(rem["cal"], rem["yu"]))
        if bud == "m50":
            r["HP4_loo_g_minus_yu_acc"] = fmt(boot(a["loo_g"], a["yu"]))
            r["sec_cal_minus_yu_acc"] = fmt(boot(a["cal"], a["yu"]))
            r["sec_entropy_minus_random_acc"] = fmt(boot(a["entropy"], a["random"]))
            r["sec_loo_g_minus_cal_acc"] = fmt(boot(a["loo_g"], a["cal"]))
            r["m50_acc_none"] = a["none"].mean()
    ya = P[P.method == "yu_alpha"]
    r["yu_alpha_acc"], r["yu_alpha_tok"] = ya.acc.mean(), ya.tok_frac.mean()
    out.append(r)

df = pd.DataFrame(out)
pd.set_option("display.width", 250, "display.max_columns", 200, "display.max_colwidth", 40)
df.to_csv(f"{ROOT}/analysis_score.csv", index=False)
if len(df):
    print(df.T.to_string())

# ---- SFT (H-P5)
rows = []
for d in sorted(glob.glob(f"{ROOT}/compress_*/")):
    f = d + "sft_eval.csv"
    if not os.path.exists(f):
        continue
    e = pd.read_csv(f)
    name = os.path.basename(d.rstrip("/")).replace("compress_", "")
    full = e[e.arm == "full"].set_index("uid")
    for arm in e.arm.unique():
        a = e[e.arm == arm].set_index("uid")
        rows.append(dict(model=name, arm=arm, n=len(a), acc=a.correct.mean() * 100,
                         gen_tokens=a.gen_tokens.mean(), tok_ratio_vs_full=a.gen_tokens.mean() / full.gen_tokens.mean(),
                         acc_minus_full=fmt(boot(a.correct, full.correct)) if arm != "full" else ""))
if rows:
    s = pd.DataFrame(rows)
    s.to_csv(f"{ROOT}/analysis_sft.csv", index=False)
    print(s.round(3).to_string())
