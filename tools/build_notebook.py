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

INTRO = """# HGCD — Hop-Graph Contrastive Decoding (Kaggle run-all)

Training-free, inference-time causal intervention for multi-hop reasoning.

**Setup on Kaggle:** *Settings → Accelerator → GPU T4 x2*, *Settings → Internet → On*, then **Run All**.

What this notebook does (all numbers it prints are produced here, nothing is hard-coded):
1. **Estimator validation** — gold chains + one inserted true-but-redundant hop (true necessity 0):
   AUC of the graph-aware total-effect estimator vs. naive masking vs. placebo (CLUTRR and MuSiQue).
2. **Alpha selection** on the CLUTRR *validation* split (never on test).
3. **CLUTRR test** (ID k≤4, OOD k≥5) with binary interventions `rename / order / distract / placebo_input`
   (true effect on the answer = 0) → accuracy + paired ATE with bootstrap CIs.
4. **MuSiQue** (real text, bridge-entity chains) with `dist8 / order` interventions.
5. Writes LaTeX tables (`tab_*.tex`), figures, CSVs and `summary.json` into `/kaggle/working/results/`.

GPU 0 runs the main model; GPU 1 (optional cell) runs a second model family in parallel.
"""

CFG = """# ---- configuration -------------------------------------------------------------
# PRESET: "kaggle" (~2-4 h on one T4), "kaggle_full" (larger n, ~8-10 h), "smoke" (minutes)
PRESET = "kaggle"
MAIN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SECOND_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"   # different model family, ungated
RUN_SECOND_ON_GPU1 = True
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
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
    cells.append(nbf.v4.new_markdown_cell(
        "## Optional: second model family on GPU 1 (runs in the background)\n"
        "Log: `results/<preset>_second.log`. Check it at the end of the notebook."))
    cells.append(nbf.v4.new_code_cell(
        "import subprocess, sys, torch\n"
        "proc = None\n"
        "if RUN_SECOND_ON_GPU1 and torch.cuda.device_count() > 1:\n"
        "    os.makedirs(OUT, exist_ok=True)\n"
        "    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')\n"
        "    log = open(f'{OUT}/{PRESET}_second.log', 'w')\n"
        "    proc = subprocess.Popen([sys.executable, '-m', 'hgcd.pipeline', '--preset', PRESET,\n"
        "                             '--model', SECOND_MODEL, '--outdir', f'{OUT}/{PRESET}_second'],\n"
        "                            env=env, stdout=log, stderr=subprocess.STDOUT)\n"
        "    print('second model started, pid', proc.pid)\n"
        "else:\n"
        "    print('single GPU: skipping the second model')"))
    cells.append(nbf.v4.new_markdown_cell("## Main run (GPU 0)"))
    cells.append(nbf.v4.new_code_cell(
        "from hgcd.pipeline import Config, PRESETS, run\n"
        "cfg = Config(**PRESETS[PRESET])\n"
        "cfg.model = MAIN_MODEL if PRESET != 'smoke' else cfg.model\n"
        "cfg.outdir = f'{OUT}/{PRESET}_main'\n"
        "summary = run(cfg)"))
    cells.append(nbf.v4.new_markdown_cell("## Results"))
    cells.append(nbf.v4.new_code_cell(
        "import json, glob, pandas as pd\n"
        "from IPython.display import display, IFrame\n"
        "d = cfg.outdir\n"
        "print(json.dumps({k: summary[k] for k in ['alpha', 'clutrr_format', 'minutes'] if k in summary}, indent=1))\n"
        "for f in ['tab_estimators_auc.tex', 'tab_estimators_pos.tex', 'tab_clutrr_main.tex',\n"
        "          'tab_clutrr_ate.tex', 'tab_musique_main.tex', 'tab_musique_ate.tex']:\n"
        "    p = os.path.join(d, f)\n"
        "    if os.path.exists(p):\n"
        "        print('=' * 20, f); print(open(p).read())\n"
        "display(pd.DataFrame(summary.get('clutrr_paired', [])))\n"
        "display(pd.DataFrame(summary.get('musique_paired', [])))"))
    cells.append(nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "df = pd.read_csv(os.path.join(d, 'clutrr_decisions.csv'))\n"
        "o = df[df.cond == 'orig']\n"
        "ax = o[o.method.isin(['cot', 'cad_chain', 'hgcd_naive', 'hgcd'])].groupby(['k', 'method']).correct.mean().unstack().plot(marker='o')\n"
        "ax.set_ylabel('accuracy'); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Wait for the GPU-1 run and package everything"))
    cells.append(nbf.v4.new_code_cell(
        "if proc is not None:\n"
        "    proc.wait()\n"
        "    print(open(f'{OUT}/{PRESET}_second.log').read()[-3000:])\n"
        "import shutil\n"
        "shutil.make_archive(f'{OUT}_{PRESET}', 'zip', OUT)\n"
        "print('download:', f'{OUT}_{PRESET}.zip')"))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = os.path.join(ROOT, "notebooks", "hgcd_kaggle.ipynb")
    nbf.write(nb, out)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
