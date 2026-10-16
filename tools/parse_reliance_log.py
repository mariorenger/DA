"""Parse the per-model position tables printed in a Kaggle reliance log into CSV + LaTeX.
Used for the reliance run that timed out before writing its zip (outputs are the printed means)."""
import re
import sys

import pandas as pd

ORDER = ["qwen-0.5b", "qwen-1.5b", "smollm2-1.7b", "qwen-3b", "qwen-7b"]


def parse(path):
    rows, model = [], None
    for raw in open(path, encoding="utf-8"):
        line = re.sub(r"^\s*[\d.]+s\s+\d+\s?", "", raw.rstrip("\n"))
        m = re.match(r"\[([\w.\-]+)\] exit code (\d+)", line)
        if m:
            model = m.group(1)
            continue
        m = re.match(r"\s*\d+\s+(clutrr_copy|clutrr|musique)\s+(gold-final|gold-nonfinal|redundant)\s+(.*)$", line)
        if m and model:
            vals = [float(v) if v != "NaN" else float("nan") for v in m.group(3).split()]
            keys = ["d_total", "d_naive", "d_cf_total", "d_cf_direct", "pns_goldbase", "d_placebo", "d_maskmark"]
            rows.append(dict(model=model, dataset=m.group(1), position=m.group(2), **dict(zip(keys, vals))))
    return pd.DataFrame(rows)


def latex(df, out):
    def g(model, ds, pos, col):
        r = df[(df.model == model) & (df.dataset == ds) & (df.position == pos)]
        return float("nan") if r.empty else float(r[col].iloc[0])
    models = [m for m in ORDER if m in set(df.model)]
    rows = [
        ("CLUTRR, non-final gold hop: total", "clutrr", "gold-nonfinal", "d_total"),
        ("CLUTRR, redundant hop: total", "clutrr", "redundant", "d_total"),
        ("CLUTRR, final gold hop: placebo", "clutrr", "gold-final", "d_placebo"),
        ("CLUTRR+copy, non-final: total", "clutrr_copy", "gold-nonfinal", "d_total"),
        ("CLUTRR+copy, non-final: direct", "clutrr_copy", "gold-nonfinal", "d_naive"),
        ("MuSiQue, non-final: total", "musique", "gold-nonfinal", "d_total"),
        ("MuSiQue, non-final: direct", "musique", "gold-nonfinal", "d_naive"),
        ("MuSiQue, final: total", "musique", "gold-final", "d_total"),
        ("Rollout PNS, gold baseline: necessary (non-final)", "musique", "gold-nonfinal", "pns_goldbase"),
        ("Rollout PNS, gold baseline: \\emph{redundant}", "musique", "redundant", "pns_goldbase"),
    ]
    lines = ["\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
             "Quantity ($\\Delta p(\\text{gold})$, points) & " + " & ".join(models) + " \\\\", "\\midrule"]
    for label, ds, pos, col in rows:
        cells = []
        for m in models:
            v = g(m, ds, pos, col)
            cells.append("--" if v != v else f"{100 * v:.1f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    df = parse(sys.argv[1])
    df.to_csv(sys.argv[2], index=False)
    latex(df, sys.argv[3])
