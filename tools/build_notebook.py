"""Build the self-contained Kaggle notebook from the package sources.

    python tools/build_notebook.py   ->  notebooks/hgcd_kaggle.ipynb

The notebook embeds every hgcd/*.py file via %%writefile, so it runs on Kaggle with
"Run All" without cloning this repository (Internet must be ON to download models/data).
"""

import glob
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__", "data", "chain", "prompts", "lm", "decoding", "experiments", "metrics",
           "report", "pipeline"]

INTRO = """# Do models use the hops they write? (Kaggle run-all)

**Setup on Kaggle:** *Settings -> Accelerator -> GPU T4 x2*, *Settings -> Internet -> On*, then **Run All**.

Two modes (set `MODE` in the config cell):

* `"reliance"` (round 2, main study, ~6-8 h): the **estimator benchmark across model scales**.
  Gold chains with one inserted true-but-redundant hop give known necessity labels; every
  step-importance estimator (mask direct, mask+descendants total, counterfactual replacement
  direct/total, rollout/resampling PNS, placebo and mask-marker controls) is scored by AUC.
  Qwen2.5-0.5B/1.5B/3B and SmolLM2-1.7B run in parallel on the two GPUs, then Qwen2.5-7B is
  sharded over both GPUs. Output: `tab_reliance.tex`, `reliance_vs_scale.pdf` + per-model folders.
* `"decoding"` (~4-5 h): decoding methods (CoT, SC, CAD, AdaCAD, CAD-chain, calibration, HGCD and
  ablations), binary interventions with known null effects, and the **exact mediation analysis**
  (chain held fixed vs regenerated) on CLUTRR and MuSiQue.

Every stage is checkpointed: if the session stops, run all cells again and finished stages are
reused. All numbers are produced here; nothing is hard-coded.
"""

CFG = """# ---- configuration -------------------------------------------------------------
MODE = "reliance"            # "reliance" or "decoding"
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
# reliance: (label, model, GPU queue); queues run in parallel, models in a queue sequentially
RELIANCE_JOBS = {
    "0": [("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"), ("qwen-3b", "Qwen/Qwen2.5-3B-Instruct")],
    "1": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"), ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct")],
}
RELIANCE_7B = ("qwen-7b", "Qwen/Qwen2.5-7B-Instruct")   # sharded over both GPUs afterwards; None to skip
# decoding: main model on GPU 0, second family on GPU 1
DECODING_PRESET = "kaggle"   # "kaggle" (~4-5 h) or "kaggle_full"
DECODING_JOBS = {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct")],
                 "1": [("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct")]}
"""

RUN = """import subprocess, sys, threading, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(OUT, exist_ok=True)
preset = "reliance" if MODE == "reliance" else DECODING_PRESET
jobs = RELIANCE_JOBS if MODE == "reliance" else DECODING_JOBS
ngpu = torch.cuda.device_count()
if ngpu < 2:  # single GPU: run every queue on GPU 0, one after another
    jobs = {"0": [j for q in jobs.values() for j in q]}

def run_job(label, model, gpus, extra=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    out = f"{OUT}/{MODE}_{label}"
    with open(out + ".log", "w") as log:
        rc = subprocess.call([sys.executable, "-m", "hgcd.pipeline", "--preset", preset,
                              "--model", model, "--outdir", out, *extra],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{label}] exit code {rc}; log tail:")
    print("".join(open(out + ".log").readlines()[-12:]))

def run_queue(gpu, queue):
    for label, model in queue:
        run_job(label, model, gpu)

threads = [threading.Thread(target=run_queue, args=(g, q)) for g, q in jobs.items()]
for t in threads: t.start()
for t in threads: t.join()
if MODE == "reliance" and RELIANCE_7B and ngpu >= 2:
    run_job(*RELIANCE_7B, "0,1", extra=("--device_map", "auto"))
"""

REPORT = """import json, glob, pandas as pd
from IPython.display import display
from hgcd import report as R
dirs = {os.path.basename(d).replace(MODE + "_", ""): d
        for d in sorted(glob.glob(f"{OUT}/{MODE}_*")) if os.path.isdir(d)}
order = ["qwen-0.5b", "qwen-1.5b", "qwen-3b", "qwen-7b", "smollm2-1.7b"]
dirs = {k: dirs[k] for k in order if k in dirs} | {k: v for k, v in dirs.items() if k not in order}
if MODE == "reliance":
    df = R.reliance_table(dirs, f"{OUT}/tab_reliance.tex", f"{OUT}/reliance_vs_scale.pdf")
    display(df)
    print(open(f"{OUT}/tab_reliance.tex").read())
for k, d in dirs.items():
    for f in sorted(glob.glob(f"{d}/tab_*.tex")):
        print("=" * 10, k, os.path.basename(f)); print(open(f).read())
"""


def build():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell("!pip -q install -U \"transformers>=4.44\" accelerate scikit-learn 2>/dev/null | tail -1\n"
                                  "!nvidia-smi --query-gpu=name,memory.total --format=csv || true"),
             nbf.v4.new_code_cell("import os\nos.makedirs('hgcd', exist_ok=True)\n"
                                  "os.environ.setdefault('HGCD_DATA', os.path.abspath('hgcd_data'))")]
    for m in MODULES:
        src = open(os.path.join(ROOT, "hgcd", f"{m}.py")).read()
        cells.append(nbf.v4.new_code_cell(f"%%writefile hgcd/{m}.py\n{src}"))
    cells.append(nbf.v4.new_code_cell("import os\n" + CFG))
    cells.append(nbf.v4.new_markdown_cell("## Run (both GPUs)"))
    cells.append(nbf.v4.new_code_cell(RUN))
    cells.append(nbf.v4.new_markdown_cell("## Results"))
    cells.append(nbf.v4.new_code_cell(REPORT))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "shutil.make_archive(f'{OUT}_{MODE}', 'zip', OUT)\n"
        "print('download:', f'{OUT}_{MODE}.zip')"))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = os.path.join(ROOT, "notebooks", "hgcd_kaggle.ipynb")
    nbf.write(nb, out)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
