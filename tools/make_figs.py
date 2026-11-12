"""Figures for paper 1 from results/ (CPU only). python tools/make_figs.py"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: pointer task, accuracy vs stride, per model (seed band where several seeds exist)
fam = {"Qwen2.5-1.5B": ["qwen-1.5b", "qwen-1.5b_s1", "qwen-1.5b_s2", "qwen-1.5b_s3", "qwen-1.5b_s4"],
       "Qwen2.5-3B": ["qwen-3b", "qwen-3b_s1", "qwen-3b_s2", "qwen-3b_s3", "qwen-3b_s4"],
       "Qwen2.5-7B": ["qwen-7b", "qwen-7b_s1", "qwen-7b_s2"],
       "Qwen2.5-14B (4-bit)": ["q25-14b_s4", "q25-14b_s6"],
       "Qwen3-1.7B": ["qwen3-1.7b", "qwen3-1.7b-ext"], "Qwen3-4B": ["qwen3-4b"]}
fig, ax = plt.subplots(figsize=(5.4, 3.4))
cols = plt.cm.tab10.colors
for i, (name, labs) in enumerate(fam.items()):
    per = {}
    for lab in labs:
        f = f"results/stride/stride_{lab}/stride_eval.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        for m, g in d.groupby("stride"):
            per.setdefault(m, []).append(g.correct.mean())
    xs = sorted(per)
    mean = [np.mean(per[x]) for x in xs]
    ax.plot(xs, mean, "-o", ms=3.5, color=cols[i], label=name)
    lo = [min(per[x]) for x in xs]; hi = [max(per[x]) for x in xs]
    if any(h - l > 0.02 for l, h in zip(lo, hi)):
        ax.fill_between(xs, lo, hi, color=cols[i], alpha=0.15, linewidth=0)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 3, 4, 6, 8, 12]); ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "12 (answer only)"], rotation=0, fontsize=7)
ax.set_xlabel("stride m (lookups composed per written step)"); ax.set_ylabel("mean accuracy over depths")
ax.set_title("Pointer chasing: accuracy against written stride", fontsize=9)
ax.legend(fontsize=6.5, frameon=False, loc="lower left"); ax.set_ylim(0, 1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_stride_pointer.png", dpi=200); plt.close(fig)

# ---- Fig 2: stride-4 accuracy by seed against model, and cumulative sum (no knee)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axs[0]
for i, (name, labs) in enumerate(list(fam.items())[:4]):
    vals = []
    for lab in labs:
        f = f"results/stride/stride_{lab}/stride_eval.csv"
        if os.path.exists(f):
            d = pd.read_csv(f); d = d[d.stride == 4]
            if len(d):
                vals.append(d.correct.mean())
    ax.scatter([i] * len(vals), vals, color=cols[i], s=18, alpha=0.8)
    ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color="k", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(["1.5B", "3B", "7B", "14B"]); ax.axhline(0.8, ls=":", color="gray")
ax.set_ylabel("stride-4 mean accuracy"); ax.set_title("Seed dispersion at the knee", fontsize=9); ax.set_ylim(0, 1.03)
ax = axs[1]
for i, lab in enumerate(["cumsum_qwen-1.5b", "cumsum_qwen3-1.7b", "cumsum_qwen-3b"]):
    f = f"results/stride/stride_{lab}/stride_eval.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f); m = d.groupby("stride").correct.mean()
    ax.plot([1, 2, 3, 4, 5, 6, 12], [m[s] for s in [1, 2, 3, 4, 5, 6, 12]], "-o", ms=3, color=cols[i + 4], label=lab.replace("cumsum_", ""))
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 3, 4, 6, 12]); ax.set_xticklabels(["1", "2", "3", "4", "6", "12"])
ax.set_xlabel("stride m (additions per written step)"); ax.set_title("Running sum: no knee", fontsize=9); ax.set_ylim(0, 1.03)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_seeds_and_cumsum.png", dpi=200); plt.close(fig)

# ---- Fig 3: wrong-result follow rate against model size (Qwen2.5 family) with CI
rows = []
for lab, nm in [("q25-0.5b", "0.5B"), ("q25-1.5b", "1.5B"), ("q25-3b", "3B"), ("q25-7b", "7B"), ("q25-14b-4bit", "14B"), ("q25-32b-bnb4", "32B")]:
    f = f"results/scale/scale_{lab}/copy.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f); s = d[d["mode"] == "last"].follow.values
    rng = np.random.default_rng(0); b = [rng.choice(s, len(s)).mean() for _ in range(1000)]
    rows.append((nm, s.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)))
fig, ax = plt.subplots(figsize=(4.4, 3.0))
x = range(len(rows)); y = [r[1] for r in rows]
ax.errorbar(x, y, yerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]], fmt="-o", ms=4, capsize=2, color=cols[3])
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1.03)
ax.set_ylabel("follows a wrong stated result"); ax.set_title("GSM8K, solved questions only (Qwen2.5)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_copy_scale.png", dpi=200); plt.close(fig)
print("figures written to", OUT)
