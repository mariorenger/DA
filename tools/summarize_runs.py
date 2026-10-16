"""Build the cross-model replication table from finished run folders (no numbers typed by hand).

    python tools/summarize_runs.py results/decoding/qwen-1.5b results/decoding/smollm2-1.7b \
        --out results/decoding/tab_replication.tex
"""
import argparse
import json
import os

import pandas as pd


def stats(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    est = pd.read_csv(os.path.join(d, "estimators.csv"))
    one = est.drop_duplicates(["dataset", "uid"])
    content = (one.p_full_gold - one.p_gold_shape).groupby(one.dataset).mean()
    auc = {(r["dataset"], r["estimator"]): r["auc_all"] for r in s["estimator_auc"]}
    pos = {(r["dataset"], r["position"]): r for r in s["estimator_position"]}
    med = {(r["cond"], r["method"]): r for r in s["musique_mediation"]}
    dec = pd.read_csv(os.path.join(d, "clutrr_decisions.csv"))
    o = dec[dec.cond == "orig"].pivot_table(index="uid", columns="method", values="correct")
    return {
        "content_clutrr": content.get("clutrr"), "content_copy": content.get("clutrr_copy"),
        "content_musique": content.get("musique"),
        "copy_nonfinal_tot": pos[("clutrr_copy", "gold-nonfinal")]["d_total"],
        "copy_nonfinal_dir": pos[("clutrr_copy", "gold-nonfinal")]["d_naive"],
        "ms_nonfinal_tot": pos[("musique", "gold-nonfinal")]["d_total"],
        "ms_nonfinal_dir": pos[("musique", "gold-nonfinal")]["d_naive"],
        "auc_ms_dir": auc[("musique", "d_naive")], "auc_ms_tot": auc[("musique", "d_total")],
        "auc_copy_dir": auc[("clutrr_copy", "d_naive")], "auc_copy_tot": auc[("clutrr_copy", "d_total")],
        "cot_te": med[("dist8", "cot")]["te"], "cot_nde": med[("dist8", "cot")]["nde"],
        "cot_nie": med[("dist8", "cot")]["nie"], "dir_nde": med[("dist8", "direct")]["nde"],
        "calib_minus_cot": (o["calib"] - o["cot"]).mean(),
    }


ROWS = [
    ("Content effect of chain, CLUTRR", "content_clutrr", 100),
    ("Content effect, CLUTRR + answer hop ($T_{\\text{copy}}{=}1$)", "content_copy", 100),
    ("Content effect, MuSiQue", "content_musique", 100),
    ("CLUTRR+copy non-final hops: $\\Delta^{\\text{tot}}$ / $\\Delta^{\\text{nai}}$", ("copy_nonfinal_tot", "copy_nonfinal_dir"), 100),
    ("MuSiQue non-final hops: $\\Delta^{\\text{tot}}$ / $\\Delta^{\\text{nai}}$", ("ms_nonfinal_tot", "ms_nonfinal_dir"), 100),
    ("AUC MuSiQue: total / direct", ("auc_ms_tot", "auc_ms_dir"), 1),
    ("AUC CLUTRR+copy: total / direct", ("auc_copy_tot", "auc_copy_dir"), 1),
    ("MuSiQue \\texttt{dist8}, CoT: TE", "cot_te", 100),
    ("\\quad NDE (chain fixed) / NIE (via chain)", ("cot_nde", "cot_nie"), 100),
    ("MuSiQue \\texttt{dist8}, direct answer: NDE", "dir_nde", 100),
    ("CLUTRR accuracy: calibration $-$ CoT", "calib_minus_cot", 100),
]


def fmt(v, scale):
    return f"{v * scale:.1f}" if scale == 100 else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = {os.path.basename(r.rstrip("/")): stats(r) for r in a.runs}
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Quantity & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for label, key, scale in ROWS:
        cells = []
        for st in cols.values():
            if isinstance(key, tuple):
                cells.append(" / ".join(fmt(st[k], scale) for k in key))
            else:
                cells.append(fmt(st[key], scale))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
