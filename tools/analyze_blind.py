"""Round 15 analysis (H-Q1..H-Q4, pre-registered in REVIEW.md). python tools/analyze_blind.py [results/blind]"""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/blind"
B = 2000
rng = np.random.default_rng(0)


def boot_auc(st, est):
    d = st[(st.planted == 1) | (st.loadbearing == 1)]
    uids = d.uid.unique()
    by = {u: g for u, g in d.groupby("uid")}
    def auc(us):
        x = pd.concat([by[u] for u in us])
        y = (x.loadbearing == 1).astype(int)
        return roc_auc_score(y, x[est]) if y.nunique() == 2 else np.nan
    a = auc(uids)
    bs = [auc(rng.choice(uids, len(uids))) for _ in range(B)]
    return a, np.nanpercentile(bs, [2.5, 97.5])


def boot_diff(x: pd.Series, y: pd.Series):
    d = (x - y).dropna().values
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return d.mean(), *np.percentile(m, [2.5, 97.5])


def fmt(t, s=100):
    return f"{t[0] * s:+.1f} [{t[1] * s:+.1f}, {t[2] * s:+.1f}]"


rows = []
for d in sorted(glob.glob(f"{ROOT}/blind_*/")):
    if not os.path.exists(d + "steps.csv"):
        continue
    name = os.path.basename(d.rstrip("/")).replace("blind_", "")
    st, pr = pd.read_csv(d + "steps.csv"), pd.read_csv(d + "prune.csv")
    r = dict(model=name, n_chains=st.uid.nunique())
    A = {}
    for est in ["entropy", "loo", "bloo", "random"]:
        a, ci = boot_auc(st, est)
        A[est] = a
        r[f"auc_{est}"] = f"{a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    # paired AUROC difference bloo - loo (bootstrap over chains)
    d2 = st[(st.planted == 1) | (st.loadbearing == 1)]
    by = {u: g for u, g in d2.groupby("uid")}
    us = np.array(list(by))
    def dauc(sel):
        x = pd.concat([by[u] for u in sel]); y = (x.loadbearing == 1).astype(int)
        return roc_auc_score(y, x["bloo"]) - roc_auc_score(y, x["loo"]) if y.nunique() == 2 else np.nan
    bs = np.array([dauc(rng.choice(us, len(us))) for _ in range(B)])
    r["HQ1_auc_bloo_minus_loo"] = f"{dauc(us):+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]"
    P = lambda col, bud: pr[pr.budget == bud].pivot_table(index="uid", columns="method", values=col)
    a50 = P("acc_blind", "m50")
    r["HQ2_blind_readout_bloo_g_minus_loo_g_m50"] = fmt(boot_diff(a50["bloo_g"], a50["loo_g"]))
    for m in ["none", "random", "entropy", "loo", "loo_g", "bloo", "bloo_g"]:
        r[f"m50_blind_{m}"] = a50[m].mean()
        r[f"m50_plain_{m}"] = P("acc_plain", "m50")[m].mean()
    cal = pr[pr.budget == "cal"]
    for m in ["control_only", "random_m", "loo_cal", "bloo_cal"]:
        s = cal[cal.method == m]
        r[f"cal_{m}_acc_blind"], r[f"cal_{m}_acc_plain"] = s.acc_blind.mean(), s.acc_plain.mean()
        r[f"cal_{m}_real_removed"], r[f"cal_{m}_tok_frac"] = s.real_removed_frac.mean(), s.tok_frac.mean()
    rows.append(r)
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/analysis_audit.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(df.T.to_string())

out = []
for d in sorted(glob.glob(f"{ROOT}/blind_*/")):
    f = d + "sft_eval.csv"
    if not os.path.exists(f):
        continue
    e = pd.read_csv(f)
    name = os.path.basename(d.rstrip("/")).replace("blind_", "")
    full = e[e.arm == "full"].set_index("uid")
    rnd = e[e.arm == "random_m"].set_index("uid") if "random_m" in set(e.arm) else None
    loo_g = e[e.arm == "loo_g"].set_index("uid") if "loo_g" in set(e.arm) else None
    if "loo_high" in set(e.arm):                      # Round 16 order-reversal control
        lo, hi = e[e.arm == "loo_low"].set_index("uid"), e[e.arm == "loo_high"].set_index("uid")
        rd = e[e.arm == "random40"].set_index("uid")
        out.append(dict(model=name, arm="HR_loo_low_minus_loo_high", n=len(lo), acc=np.nan, gen_tokens=np.nan, tok_ratio=np.nan,
                        vs_full=fmt(boot_diff(lo.correct, hi.correct))))
        out.append(dict(model=name, arm="HR_loo_low_minus_random40", n=len(lo), acc=np.nan, gen_tokens=np.nan, tok_ratio=np.nan,
                        vs_full=fmt(boot_diff(lo.correct, rd.correct))))
    for arm in e.arm.unique():
        a = e[e.arm == arm].set_index("uid")
        o = dict(model=name, arm=arm, n=len(a), acc=100 * a.correct.mean(), gen_tokens=a.gen_tokens.mean(),
                 tok_ratio=a.gen_tokens.mean() / full.gen_tokens.mean())
        if arm != "full":
            o["vs_full"] = fmt(boot_diff(a.correct, full.correct))
        if rnd is not None and arm in ("bloo_cal", "loo_cal"):
            o["vs_random_m"] = fmt(boot_diff(a.correct, rnd.correct))
        if loo_g is not None and arm == "bloo_g":
            o["HQ4_vs_loo_g"] = fmt(boot_diff(a.correct, loo_g.correct))
        out.append(o)
if out:
    s = pd.DataFrame(out)
    s.to_csv(f"{ROOT}/analysis_sft.csv", index=False)
    print(s.round(3).to_string())
