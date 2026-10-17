"""LaTeX tables for the causal-structure decoding experiments (round 9).

    python tools/round9_tables.py   ->  results/causaldec/tab_*.tex
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgcd import metrics as M  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "causaldec")
SHORT = {"qwen-1.5b": "1.5B", "qwen-3b": "3B", "qwen-7b": "7B"}


def _wide(path, metric):
    d = pd.read_csv(path)
    return d.pivot_table(index="uid", columns="method", values=metric)


def _ci(a, b):
    m, lo, hi = M.boot_ci((a - b).values)
    return f"{100 * m:+.1f}\\,{{\\scriptsize[{100 * lo:+.1f},{100 * hi:+.1f}]}}"


def clutrr_main():
    sets = [("C4_gender_clutrr", "Validation, $k{=}2$--4", ["qwen-1.5b", "qwen-3b", "qwen-7b"]),
            ("C6_fresh_short", "Unseen task, $k{=}2$--4", ["qwen-1.5b", "qwen-3b", "qwen-7b"]),
            ("C5_long_clutrr", "Unseen task, $k{=}5$--8", ["qwen-1.5b", "qwen-3b", "qwen-7b"])]
    lines = ["\\begin{tabular}{llrccccc}", "\\toprule",
             "Evaluation set & Model & $n$ & CoT & SC$_{10}$ & Extract+rules & Extract+rules, SC$_5$ "
             "& $\\Delta$ vs.\\ best baseline \\\\", "\\midrule"]
    for j, (folder, name, models) in enumerate(sets):
        if j:
            lines.append("\\midrule")
        for m in models:
            p = os.path.join(R, folder, m, "decisions.csv")
            if not os.path.exists(p):
                continue
            w = _wide(p, "correct")
            base = "sc_plus" if w.sc_plus.mean() >= w.cot_free.mean() else "cot_free"
            cells = [f"{100 * w[c].mean():.1f}" for c in ["cot_free", "sc_plus", "ext_g", "ext_g_sc"]]
            lines.append(f"{name} & {SHORT[m]} & {len(w)} & " + " & ".join(cells) + f" & {_ci(w.ext_g_sc, w[base])} \\\\")
            name = ""
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def clutrr_ablation():
    lines = ["\\begin{tabular}{llrcccc}", "\\toprule",
             "Chains & Model & $n$ & CoT & LLM on own facts & Rules on own facts & Rules $-$ LLM \\\\",
             "\\midrule"]
    for j, length in enumerate(["short", "long"]):
        if j:
            lines.append("\\midrule")
        name = "$k{=}2$--4" if length == "short" else "$k{=}5$--8"
        for m in ["qwen-1.5b", "qwen-3b", "qwen-7b"]:
            p = os.path.join(R, "C7_ablate", f"{length}-{m}", "decisions.csv")
            if not os.path.exists(p):
                continue
            w = _wide(p, "correct")
            cells = [f"{100 * w[c].mean():.1f}" for c in ["cot_free", "ext_llm", "ext_g"]]
            lines.append(f"{name} & {SHORT[m]} & {len(w)} & " + " & ".join(cells) + f" & {_ci(w.ext_g, w.ext_llm)} \\\\")
            name = ""
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def musique_selection():
    rows = [("qwen-1.5b", "M3_fix15", "M_select"), ("qwen-3b", None, "M_select")]
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Distractors & Model & CoT & SC$_5$ & Attn top-4 & Attn rerank & Self-select & Gold only \\\\",
             "\\midrule"]
    for j, cond in enumerate(["orig", "dist8"]):
        if j:
            lines.append("\\midrule")
        name = "4" if cond == "orig" else "8"
        for m, fix, base in rows:
            b = pd.read_csv(os.path.join(R, f"{base}_{cond}", m, "decisions.csv"))
            w = b[b.cond == cond].pivot_table(index="uid", columns="method", values="em")
            if fix:   # attention variants from the corrected rerun
                f = pd.read_csv(os.path.join(R, fix, m, "decisions.csv"))
                fw = f[f.cond == cond].pivot_table(index="uid", columns="method", values="em")
                w["attn_sel"], w["attn_rerank"] = fw.attn_sel, fw.attn_rerank
            else:
                s = pd.read_csv(os.path.join(R, "M2_soft_select", m, "decisions.csv"))
                sw = s[s.cond == cond].pivot_table(index="uid", columns="method", values="em")
                w["attn_rerank"] = sw.attn_rerank
            cells = [f"{100 * w[c].mean():.1f}" for c in
                     ["cot_free", "sc", "attn_sel", "attn_rerank", "self_sel", "gold_only"]]
            lines.append(f"{name} & {SHORT[m]} & " + " & ".join(cells) + " \\\\")
            name = ""
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for fn, out in [(clutrr_main, "tab_clutrr_symbolic.tex"), (clutrr_ablation, "tab_clutrr_ablation.tex"),
                    (musique_selection, "tab_musique_selection.tex")]:
        tex = fn()
        open(os.path.join(R, out), "w").write(tex + "\n")
        print(tex, "\n")
