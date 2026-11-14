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
           "report", "pipeline", "causaldec", "cftrain", "egen", "cfrl", "hall", "compress", "blind", "scalecopy", "stride"]

INTRO = """# Read, not reasoned? Kaggle run-all

**Setup:** *Settings -> Accelerator -> GPU T4 x2*, *Settings -> Internet -> On*, then **Save Version -> Run All**.
Two notebooks can run at the same time (each uses its own 2 GPUs):

* **A_reliance_small** (~8 h): estimator benchmark for Qwen2.5-0.5B/1.5B/3B and SmolLM2-1.7B with the
  round-3 additions: resampled-baseline rollout, wrong-answer copy test (conditioned on chains the model
  solves), volume-matched mask control, raw log-prob effects.
* **B_7b_and_mediation** (~8 h): Qwen2.5-7B estimator benchmark (sharded over both GPUs), then the exact
  mediation analysis on MuSiQue at n=500 for Qwen2.5-1.5B and 3B.

No job starts after `MAX_START_HOURS`, so the session never hits Kaggle's 12-hour limit (which would
discard all output). Download `results_<notebook>.zip` at the end.
"""

CFG_TEMPLATE = """# ---- configuration -------------------------------------------------------------
NOTEBOOK = "{name}"
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
MAX_START_HOURS = 8.5   # never START a job after this many hours (Kaggle kills sessions at 12 h
                        # and then saves nothing); finished jobs are always kept
# PHASES run one after another; inside a phase, each GPU queue runs in parallel.
# job = (label, model, preset, extra CLI args)
PHASES = {phases}
"""

RUN = """import subprocess, sys, threading, time, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
ngpu = torch.cuda.device_count()

def run_job(label, model, preset, extra, gpus):
    if (time.time() - T0) / 3600 > MAX_START_HOURS:
        print(f"[{label}] SKIPPED: past MAX_START_HOURS"); return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    out = f"{OUT}/{preset}_{label}"
    with open(out + ".log", "w") as log:
        cmd = ([sys.executable, "-m", "hgcd.cfrl", "--model", model, "--outdir", out, *extra]
               if preset.startswith("cfrl") else
               [sys.executable, "-m", "hgcd.hall", "--model", model, "--outdir", out, *extra]
               if preset.startswith("hall") else
               [sys.executable, "-m", "hgcd.compress", "--model", model, "--outdir", out, *extra]
               if preset.startswith("compress") else
               [sys.executable, "-m", "hgcd.blind", "--model", model, "--outdir", out, *extra]
               if preset.startswith("blind") else
               [sys.executable, "-m", "hgcd.scalecopy", "--model", model, "--outdir", out, *extra]
               if preset.startswith("scale") else
               [sys.executable, "-m", "hgcd.stride", "--model", model, "--outdir", out, *extra]
               if preset.startswith("stride") else
               [sys.executable, "-m", "hgcd.egen", "--model", model, "--outdir", out, *extra]
               if preset.startswith("egen") else
               [sys.executable, "-m", "hgcd.cftrain", "--model", model, "--outdir", out, *extra]
               if preset.startswith("cftrain") else
               [sys.executable, "-m", "hgcd.causaldec", "--model", model, "--outdir", out, *extra]
               if preset.startswith("causaldec") else
               [sys.executable, "-m", "hgcd.pipeline", "--preset", preset, "--model", model,
                "--outdir", out, *extra])
        rc = subprocess.call(cmd,
                             env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{label}] exit {rc} after {(time.time() - T0) / 3600:.1f} h; log tail:")
    print("".join(open(out + ".log").readlines()[-14:]))

def run_queue(gpus, queue):
    for job in queue:
        run_job(*job, gpus)

for name, queues in PHASES.items():
    print("=" * 20, "phase", name)
    if ngpu < 2 and len(queues) > 1:  # single GPU: serialise
        queues = {"0": [j for q in queues.values() for j in q]}
    ths = [threading.Thread(target=run_queue, args=(g, q)) for g, q in queues.items()]
    for t in ths: t.start()
    for t in ths: t.join()
"""

REPORT = """import json, glob, pandas as pd
from IPython.display import display
from hgcd import report as R
dirs = {os.path.basename(d): d for d in sorted(glob.glob(f"{OUT}/*")) if os.path.isdir(d)}
rel = {k: v for k, v in dirs.items() if k.startswith("reliance")}
if rel:
    df = R.reliance_table(rel, f"{OUT}/tab_reliance.tex", f"{OUT}/reliance_vs_scale.pdf")
    display(df)
    print(open(f"{OUT}/tab_reliance.tex").read())
for k, d in dirs.items():
    if os.path.exists(f"{d}/summary.csv"):
        print("=" * 10, k); print(pd.read_csv(f"{d}/summary.csv").round(3).to_string())
    if os.path.exists(f"{d}/summary.json"):
        s = json.load(open(f"{d}/summary.json"))
        print(k, "copy_wrong:", s.get("copy_wrong"))
    for f in sorted(glob.glob(f"{d}/tab_*.tex")):
        print("=" * 10, k, os.path.basename(f)); print(open(f).read())
"""


NOTEBOOKS = {
    "A_reliance_small": """{
    "estimators": {
        "0": [("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "reliance", ()),
              ("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "reliance", ())],
        "1": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "reliance", ()),
              ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "reliance", ())],
    },
}""",
    "B_7b_and_mediation": """{
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "reliance_7b", ("--device_map", "auto"))]},
    "mediation_n500": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "mediation", ())],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "mediation", ())],
    },
}""",
    # ---- round 9: decoding with causal / structural information (hgcd/causaldec.py)
    "C_symbolic_clutrr": """{
    "small": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--max_minutes", "240"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--max_minutes", "240"))],
    },
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--device_map", "auto", "--bs", "4", "--max_minutes", "300"))]},
}""",
    "C2_extract_clutrr": """{
    "small": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--max_minutes", "240"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--max_minutes", "240"))],
    },
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--device_map", "auto", "--bs", "4", "--max_minutes", "300"))]},
}""",
    # confirmation on held-out chain lengths (k = 5, 7, 9, 10 were never looked at), with a
    # compute-matched self-consistency baseline (10 sampled chains)
    "C3_confirm_clutrr": """{
    "small": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--ks", "5", "7", "9", "10", "--seed", "1", "--sc_extra", "5",
                "--max_minutes", "300"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--ks", "5", "7", "9", "10", "--seed", "1", "--sc_extra", "5",
                "--max_minutes", "300"))],
    },
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--ks", "5", "7", "9", "10", "--seed", "1", "--sc_extra", "5",
                     "--device_map", "auto", "--bs", "4", "--max_minutes", "330"))]},
}""",
    # soft attention selection (after M_select_orig showed hard top-K filtering loses gold paragraphs)
    "M2_soft_select": """{
    "soft": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "orig", "dist8", "--sc_n", "0",
                "--ms_methods", "attn_rerank", "attn_drop2", "--max_minutes", "420"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "orig", "dist8", "--sc_n", "0",
                "--ms_methods", "attn_rerank", "attn_drop2", "--bs", "4", "--max_minutes", "420"))],
    },
}""",
    # pre-registered: gender-aware extraction on the never-used CLUTRR validation split
    "C4_gender_clutrr": """{
    "small": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--split", "validation", "--ks", "2", "3", "4", "--n_per_k", "100",
                "--seed", "2", "--sc_extra", "5", "--max_minutes", "300"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--split", "validation", "--ks", "2", "3", "4", "--n_per_k", "100",
                "--seed", "2", "--sc_extra", "5", "--max_minutes", "300"))],
    },
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--split", "validation", "--ks", "2", "3", "4", "--n_per_k", "100",
                     "--seed", "2", "--sc_extra", "5", "--device_map", "auto", "--bs", "4",
                     "--max_minutes", "330"))]},
}""",
    # pre-registered: gender-aware extraction on fresh long chains (other CLUTRR task, never used)
    "C5_long_clutrr": """{
    "small": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8",
                "--n_per_k", "50", "--seed", "5", "--sc_extra", "5", "--max_minutes", "300"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8",
                "--n_per_k", "50", "--seed", "5", "--sc_extra", "5", "--max_minutes", "300"))],
    },
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8",
                     "--n_per_k", "50", "--seed", "5", "--sc_extra", "5", "--device_map", "auto",
                     "--bs", "4", "--max_minutes", "330"))]},
}""",
    # rerun of the Qwen-1.5B attention variants after the fp16 attention bug (scores were
    # non-finite, so selection fell back to the first paragraphs) + fresh short-chain CLUTRR check
    "M3_fix15_C6": """{
    "fix": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "orig", "dist8", "--sc_n", "0",
                "--ms_methods", "attn_sel", "attn_rerank", "attn_drop2", "--max_minutes", "420"))],
        "1": [("c6-qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr",
               ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4",
                "--n_per_k", "60", "--seed", "7", "--sc_extra", "5", "--max_minutes", "300"))],
    },
}""",
    "C6_fresh_short": """{
    "small": {"0": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr",
                     ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4",
                      "--n_per_k", "60", "--seed", "7", "--sc_extra", "5", "--max_minutes", "300"))]},
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                    ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4",
                     "--n_per_k", "60", "--seed", "7", "--sc_extra", "5", "--device_map", "auto",
                     "--bs", "4", "--max_minutes", "300"))]},
}""",
    # ablation: same extracted facts, LLM composes vs rule table composes (fresh C6/C5 questions)
    "C7_ablate_small": """{
    "ablate": {
        "0": [("short-qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr", ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4", "--n_per_k", "60", "--seed", "7", "--ext_llm", "1", "--sc_extra", "0", "--max_minutes", "200")),
              ("long-qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_clutrr", ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8", "--n_per_k", "50", "--seed", "5", "--ext_llm", "1", "--sc_extra", "0", "--max_minutes", "420"))],
        "1": [("short-qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr", ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4", "--n_per_k", "60", "--seed", "7", "--ext_llm", "1", "--sc_extra", "0", "--max_minutes", "200")),
              ("long-qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_clutrr", ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8", "--n_per_k", "50", "--seed", "5", "--ext_llm", "1", "--sc_extra", "0", "--max_minutes", "420"))],
    },
}""",
    "C7_ablate_7b": """{
    "7b": {"0,1": [("short-qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                     ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "2", "3", "4", "--n_per_k", "60", "--seed", "7", "--ext_llm", "1", "--sc_extra", "0", "--device_map", "auto", "--bs", "4", "--max_minutes", "240")),
                    ("long-qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_clutrr",
                     ("--task", "clutrr", "--task_name", "gen_train23_test2to10", "--ks", "5", "6", "7", "8", "--n_per_k", "50", "--seed", "5", "--ext_llm", "1", "--sc_extra", "0", "--device_map", "auto", "--bs", "4", "--max_minutes", "470"))]},
}""",
    # round 10: structured-chain SFT (see REVIEW.md, pre-registered); one seed per notebook
    "CF1_seed0": """{
    "cf1": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "cftrain",
               ("--arms", "s_plain", "s_struct", "--seed", "0", "--max_minutes", "560"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "cftrain",
               ("--arms", "s_plain", "s_struct", "--seed", "0", "--max_minutes", "560"))],
    },
}""",
    "CF1_seed1": """{
    "cf1": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "cftrain",
               ("--arms", "s_plain", "s_struct", "--seed", "1", "--max_minutes", "560"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "cftrain",
               ("--arms", "s_plain", "s_struct", "--seed", "1", "--max_minutes", "560"))],
    },
}""",
    # round 10b: does the structured-chain recipe generalise? (see REVIEW.md, pre-registered)
    "EG1_musique": """{
    "eg1": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "egen",
               ("--dataset", "musique", "--arms", "p0", "s_ans", "s_plain", "s_struct", "--n_train", "2000",
                "--conds", "orig", "dist8", "--max_minutes", "560"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "egen",
               ("--dataset", "musique", "--arms", "s_ans", "s_plain", "s_struct", "--n_train", "1500",
                "--conds", "orig", "dist8", "--max_minutes", "560"))],
    },
}""",
    "EG1_wiki2": """{
    "eg1": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "egen",
               ("--dataset", "wiki2", "--arms", "p0", "s_ans", "s_plain", "s_struct", "--n_train", "2000",
                "--max_minutes", "560"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "egen",
               ("--dataset", "wiki2", "--arms", "p0", "s_ans", "s_plain", "s_struct", "--n_train", "1500",
                "--max_minutes", "560"))],
    },
}""",
    # round 11 (see REVIEW.md, pre-registered): causal training methods on CLUTRR, 7B, OOD type shift
    "CF3_dpo_s0": """{
    "cf3": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "cfrl",
               ("--arms", "s_ans", "s_dpo", "--seed", "0", "--max_minutes", "540"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "cfrl",
               ("--arms", "s_ans", "s_dpo", "--seed", "0", "--max_minutes", "540"))],
    },
}""",
    "CF3_dpo_s1": """{
    "cf3": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "cfrl",
               ("--arms", "s_ans", "s_dpo", "--seed", "1", "--max_minutes", "540"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "cfrl",
               ("--arms", "s_ans", "s_dpo", "--seed", "1", "--max_minutes", "540"))],
    },
}""",
    "CF3_grpo": """{
    "cf3": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "cfrl",
               ("--arms", "s_grpo", "--seed", "0", "--rl_steps", "200", "--rl_minutes", "120", "--max_minutes", "540"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "cfrl",
               ("--arms", "s_grpo", "--seed", "0", "--rl_steps", "200", "--rl_minutes", "150", "--max_minutes", "540"))],
    },
}""",
    "CF4_7b": """{
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "cfrl",
                    ("--arms", "s_plain", "s_ans", "--device_map", "auto", "--mb", "1", "--accum", "16",
                     "--bs", "4", "--seed", "0", "--max_minutes", "560"))]},
}""",
    "EG2_musique_ood": """{
    "eg1": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "egen",
               ("--dataset", "musique", "--arms", "p0", "s_ans", "s_plain", "s_struct", "--n_train", "2000",
                "--conds", "orig", "dist8", "--max_minutes", "420"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "egen",
               ("--dataset", "musique", "--arms", "s_ans", "s_plain", "s_struct", "--n_train", "1500",
                "--conds", "orig", "dist8", "--max_minutes", "420"))],
    },
    "ood": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "egen",
               ("--dataset", "wiki2", "--arms", "s_ans", "s_plain", "--n_train", "2000",
                "--train_types", "compositional", "--eval_types", "inference", "--max_minutes", "560"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "egen",
               ("--dataset", "wiki2", "--arms", "s_ans", "s_plain", "--n_train", "1500",
                "--train_types", "compositional", "--eval_types", "inference", "--max_minutes", "560"))],
    },
}""",
    # round 12 (see REVIEW.md, pre-registered): Qwen3, thinking mode, teacher distillation, hard-prompt RL
    "R12_q3_8b": """{
    "q8": {"0,1": [("qwen3-8b", "Qwen/Qwen3-8B", "cfrl",
                    ("--arms", "p_fs", "p_think", "s_plain", "s_ans", "--device_map", "auto", "--mb", "1", "--accum", "16",
                     "--bs", "8", "--think_n", "60", "--think_tokens", "1500", "--seed", "0", "--max_minutes", "560"))]},
}""",
    "R12_q3_small": """{
    "q3": {
        "0": [("qwen3-4b", "Qwen/Qwen3-4B", "cfrl",
               ("--arms", "p_fs", "p_think", "s_plain", "s_ans", "s_grpo", "--rl_hard", "1", "--rl_steps", "250",
                "--rl_minutes", "120", "--bs", "8", "--think_n", "100", "--think_tokens", "1500", "--seed", "0",
                "--max_minutes", "560"))],
        "1": [("qwen3-1.7b", "Qwen/Qwen3-1.7B", "cfrl",
               ("--arms", "p_fs", "s_plain", "s_ans", "s_grpo", "--rl_hard", "1", "--rl_steps", "250",
                "--rl_minutes", "100", "--bs", "8", "--seed", "0", "--max_minutes", "560"))],
    },
}""",
    # relaunch after the p_think NameError (covered not imported); p_think moved last so SFT results are not lost
    "R12_q3_8b_b": """{
    "q8": {"0,1": [("qwen3-8b", "Qwen/Qwen3-8B", "cfrl",
                    ("--arms", "s_plain", "s_ans", "p_think", "--device_map", "auto", "--mb", "1", "--accum", "16",
                     "--bs", "8", "--think_n", "60", "--think_tokens", "1500", "--seed", "0", "--max_minutes", "560"))]},
}""",
    "R12_q3_4b": """{
    "q4": {"0,1": [("qwen3-4b", "Qwen/Qwen3-4B", "cfrl",
                    ("--arms", "p_fs", "s_plain", "s_ans", "s_grpo", "p_think", "--rl_hard", "1", "--rl_steps", "250",
                     "--rl_minutes", "120", "--bs", "8", "--think_n", "100", "--think_tokens", "1500", "--seed", "0",
                     "--max_minutes", "560"))]},
}""",
    "R12_star": """{
    "star": {"0,1": [("qwen3-4b", "Qwen/Qwen3-4B", "cfrl",
                      ("--arms", "s_plain", "s_star", "--teacher", "Qwen/Qwen3-8B", "--teacher_device_map", "auto",
                       "--star_n", "1500", "--star_k", "2", "--bs", "8", "--seed", "0", "--max_minutes", "560"))]},
}""",
    # round 13 (see REVIEW.md, pre-registered): context-faithful answering and abstention on MuSiQue
    "R13_q25_3b": """{
    "h": {
        "0": [("qwen-3b-a", "Qwen/Qwen2.5-3B-Instruct", "hall",
               ("--arms", "p_inst", "s_plain", "s_abs", "s_abs_cf", "--bs", "6", "--seed", "0", "--max_minutes", "600"))],
        "1": [("qwen-3b-b", "Qwen/Qwen2.5-3B-Instruct", "hall",
               ("--arms", "s_dpo", "s_rl", "--bs", "6", "--seed", "0", "--max_minutes", "600"))],
    },
}""",
    "R13_q3_4b": """{
    "h": {
        "0": [("qwen3-4b-a", "Qwen/Qwen3-4B", "hall",
               ("--arms", "p_inst", "s_plain", "s_abs", "s_abs_cf", "--bs", "6", "--seed", "0", "--max_minutes", "600"))],
        "1": [("qwen3-4b-b", "Qwen/Qwen3-4B", "hall",
               ("--arms", "s_dpo", "s_rl", "--bs", "6", "--seed", "0", "--max_minutes", "600"))],
    },
}""",
    "R13_q25_3b_s1": """{
    "h": {
        "0": [("qwen-3b-s1-a", "Qwen/Qwen2.5-3B-Instruct", "hall",
               ("--arms", "s_plain", "s_abs", "s_abs_cf", "--bs", "6", "--seed", "1", "--max_minutes", "600"))],
        "1": [("qwen-3b-s1-b", "Qwen/Qwen2.5-3B-Instruct", "hall",
               ("--arms", "s_dpo", "s_rl", "--bs", "6", "--seed", "1", "--max_minutes", "600"))],
    },
}""",
    # relaunch of the Qwen3-4B part after CUDA OOM in the probe / DPO of the first launch (fixed in hall.py / cfrl.py)
    "R13_q3_4b_fix": """{
    "h": {
        "0": [("qwen3-4b-fa", "Qwen/Qwen3-4B", "hall",
               ("--arms", "s_plain", "s_abs", "--bs", "6", "--seed", "0", "--max_minutes", "600"))],
        "1": [("qwen3-4b-fb", "Qwen/Qwen3-4B", "hall",
               ("--arms", "s_dpo", "s_rl", "--bs", "6", "--seed", "0", "--rl_p", "2", "--rl_mb", "1",
                "--rl_steps", "150", "--max_minutes", "600"))],
    },
}""",
    # RL arm only, after CUDA OOM in the first launch (GRPO now applies the LM head to completion positions only)
    "R13_rl_s0": """{
    "h": {"0": [("qwen-3b-rl-s0", "Qwen/Qwen2.5-3B-Instruct", "hall",
                 ("--arms", "s_rl", "--bs", "6", "--seed", "0", "--rl_p", "2", "--rl_mb", "1", "--rl_steps", "150",
                  "--max_minutes", "600"))]},
}""",
    "R13_rl_s1": """{
    "h": {"0": [("qwen-3b-rl-s1", "Qwen/Qwen2.5-3B-Instruct", "hall",
                 ("--arms", "s_rl", "--bs", "6", "--seed", "1", "--rl_p", "2", "--rl_mb", "1", "--rl_steps", "150",
                  "--max_minutes", "600"))]},
}""",
    "M_select_orig": """{
    "orig": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "orig", "--max_minutes", "480"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "orig", "--bs", "6",
                "--max_minutes", "480"))],
    },
}""",
    "M_select_dist8": """{
    "dist8": {
        "0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "dist8", "--max_minutes", "480"))],
        "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "causaldec_musique",
               ("--task", "musique", "--n", "500", "--conditions", "dist8", "--bs", "4",
                "--max_minutes", "480"))],
    },
}""",
    "M_select_7b": """{
    "7b": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "causaldec_musique",
                    ("--task", "musique", "--n", "300", "--conditions", "orig", "dist8",
                     "--device_map", "auto", "--bs", "4", "--max_minutes", "560"))]},
}""",
    "R14_score_a": """{
    "s": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "compress",
                 ("--phase", "score", "--n", "200", "--k", "6", "--bs", "8"))],
          "1": [("qwen3-1.7b", "Qwen/Qwen3-1.7B", "compress",
                 ("--phase", "score", "--n", "200", "--k", "6", "--bs", "8", "--roll_tokens", "400"))]},
}""",
    "R14_score_b": """{
    "s": {"0": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "compress",
                 ("--phase", "score", "--n", "200", "--k", "6", "--bs", "6"))],
          "1": [("qwen3-4b", "Qwen/Qwen3-4B", "compress",
                 ("--phase", "score", "--n", "150", "--k", "6", "--bs", "4", "--roll_tokens", "400"))]},
}""",
    "R14_sft_a": """{
    "t": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "compress",
                 ("--phase", "sft", "--n_train", "800", "--k", "4", "--bs", "8"))],
          "1": [("qwen3-1.7b", "Qwen/Qwen3-1.7B", "compress",
                 ("--phase", "sft", "--n_train", "800", "--k", "4", "--bs", "8", "--roll_tokens", "400"))]},
}""",
    "R14_sft_b": """{
    "t": {"0": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "compress",
                 ("--phase", "sft", "--n_train", "800", "--k", "4", "--bs", "6", "--mb", "2", "--accum", "8"))],
          "1": [("qwen-1.5b-s1", "Qwen/Qwen2.5-1.5B-Instruct", "compress",
                 ("--phase", "sft", "--n_train", "800", "--k", "4", "--bs", "8", "--seed", "1"))]},
}""",
    "R14b_score_q25": """{
    "s": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "compress",
                 ("--phase", "score", "--n", "200", "--n_pool", "500", "--k", "6", "--bs", "8"))],
          "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "compress",
                 ("--phase", "score", "--n", "200", "--n_pool", "400", "--k", "6", "--bs", "6"))]},
}""",
    "R14b_sft_q25": """{
    "t": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "compress",
                 ("--phase", "sft", "--n_train", "400", "--k", "4", "--bs", "8"))],
          "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "compress",
                 ("--phase", "sft", "--n_train", "400", "--k", "4", "--bs", "6", "--mb", "2", "--accum", "8"))]},
}""",
    "R14c_score_q25_7b": """{
    "s": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "compress",
                   ("--phase", "score", "--n", "100", "--n_pool", "200", "--k", "4", "--n_alt", "3",
                    "--bs", "4", "--device_map", "auto"))]},
}""",
    "R14c_score_q3_8b": """{
    "s": {"0,1": [("qwen3-8b", "Qwen/Qwen3-8B", "compress",
                   ("--phase", "score", "--n", "100", "--n_pool", "200", "--k", "4", "--n_alt", "3",
                    "--bs", "4", "--roll_tokens", "400", "--device_map", "auto"))]},
}""",
    "R14c_sft_q25_7b": """{
    "t": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "compress",
                   ("--phase", "sft", "--n_train", "300", "--n_eval", "300", "--k", "3", "--n_alt", "3",
                    "--bs", "4", "--mb", "1", "--accum", "16", "--device_map", "auto",
                    "--arms", "base", "full", "random", "yu", "loo_g"))]},
}""",
    "R15_blind_a": """{
    "audit": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "blind",
                     ("--phase", "audit", "--n", "200", "--n_pool", "500", "--bs", "8")),
                    ("qwen3-1.7b", "Qwen/Qwen3-1.7B", "blind",
                     ("--phase", "audit", "--n", "200", "--n_pool", "400", "--bs", "8"))],
              "1": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "blind",
                     ("--phase", "audit", "--n", "200", "--n_pool", "400", "--bs", "6"))]},
    "sft": {"0": [("qwen-1.5b_sft", "Qwen/Qwen2.5-1.5B-Instruct", "blind",
                   ("--phase", "sft", "--n_train", "800", "--bs", "8"))],
            "1": [("qwen-3b_sft", "Qwen/Qwen2.5-3B-Instruct", "blind",
                   ("--phase", "sft", "--n_train", "800", "--bs", "6", "--mb", "2", "--accum", "8"))]},
}""",
    "R15_blind_b": """{
    "sft": {"0,1": [("qwen3-1.7b_sft", "Qwen/Qwen3-1.7B", "blind",
                     ("--phase", "sft", "--n_train", "800", "--bs", "8", "--mb", "2", "--accum", "8",
                      "--device_map", "auto"))]},
}""",
    "R15_blind_7b": """{
    "audit": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "blind",
                       ("--phase", "audit", "--n", "100", "--n_pool", "200", "--bs", "4",
                        "--device_map", "auto"))]},
}""",
    "R16_order_a": """{
    "o": {"0": [("qwen-1.5b_order", "Qwen/Qwen2.5-1.5B-Instruct", "blind",
                 ("--phase", "order", "--n_train", "800", "--bs", "8"))],
          "1": [("qwen3-1.7b_order", "Qwen/Qwen3-1.7B", "blind",
                 ("--phase", "order", "--n_train", "800", "--bs", "8"))]},
}""",
    "R16_order_b": """{
    "o": {"0": [("qwen-3b_order", "Qwen/Qwen2.5-3B-Instruct", "blind",
                 ("--phase", "order", "--n_train", "800", "--bs", "6", "--mb", "2", "--accum", "8"))]},
}""",
    "R16_order_7b": """{
    "o": {"0,1": [("qwen-7b_order", "Qwen/Qwen2.5-7B-Instruct", "blind",
                   ("--phase", "order", "--n_train", "300", "--n_eval", "300", "--bs", "4", "--mb", "1",
                    "--accum", "16", "--device_map", "auto"))]},
}""",
    "R17_compact_a": """{
    "c": {"0": [("qwen-1.5b_compact", "Qwen/Qwen2.5-1.5B-Instruct", "blind",
                 ("--phase", "compact", "--n_train", "800", "--bs", "8"))],
          "1": [("qwen3-1.7b_compact", "Qwen/Qwen3-1.7B", "blind",
                 ("--phase", "compact", "--n_train", "800", "--bs", "8"))]},
}""",
    "R17_compact_b": """{
    "c": {"0": [("qwen-3b_compact", "Qwen/Qwen2.5-3B-Instruct", "blind",
                 ("--phase", "compact", "--n_train", "800", "--bs", "6", "--mb", "2", "--accum", "8"))]},
}""",
    "R17_compact_7b": """{
    "c": {"0,1": [("qwen-7b_compact", "Qwen/Qwen2.5-7B-Instruct", "blind",
                   ("--phase", "compact", "--n_train", "300", "--n_eval", "300", "--bs", "4", "--mb", "1",
                    "--accum", "16", "--device_map", "auto"))]},
}""",
    "R18_scale_small": """{
    "s": {"0": [("q25-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "scale", ("--n", "200", "--n_pool", "800", "--bs", "8")),
                ("q25-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "scale", ("--n", "200", "--n_pool", "500", "--bs", "8")),
                ("q25-3b", "Qwen/Qwen2.5-3B-Instruct", "scale", ("--n", "200", "--n_pool", "400", "--bs", "6"))],
          "1": [("q3-1.7b", "Qwen/Qwen3-1.7B", "scale", ("--n", "200", "--n_pool", "400", "--bs", "8")),
                ("q3-4b", "Qwen/Qwen3-4B", "scale", ("--n", "200", "--n_pool", "300", "--bs", "6"))]},
}""",
    "R18_scale_big": """{
    "s": {"0,1": [("q25-7b", "Qwen/Qwen2.5-7B-Instruct", "scale",
                   ("--n", "200", "--n_pool", "300", "--bs", "4", "--device_map", "auto")),
                  ("q25-14b-4bit", "Qwen/Qwen2.5-14B-Instruct", "scale",
                   ("--n", "200", "--n_pool", "300", "--bs", "4", "--load_in_4bit")),
                  ("q25-32b-4bit", "Qwen/Qwen2.5-32B-Instruct", "scale",
                   ("--n", "150", "--n_pool", "220", "--bs", "4", "--load_in_4bit"))]},
}""",
    "R18_hall_q25_7b": """{
    "h": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "hall",
                   ("--arms", "p_inst", "s_plain", "s_abs", "s_abs_cf", "--n_train", "1000", "--bs", "4",
                    "--mb", "1", "--accum", "16", "--device_map", "auto", "--probe", "0", "--seed", "0",
                    "--max_minutes", "520"))]},
}""",
    "R18_hall_q3_8b": """{
    "h": {"0,1": [("qwen3-8b", "Qwen/Qwen3-8B", "hall",
                   ("--arms", "p_inst", "s_plain", "s_abs", "s_abs_cf", "--n_train", "1000", "--bs", "4",
                    "--mb", "1", "--accum", "16", "--device_map", "auto", "--probe", "0", "--seed", "0",
                    "--max_minutes", "520"))]},
}""",
    "R19_compact2_a": """{
    "c": {"0": [("qwen-1.5b_compact2", "Qwen/Qwen2.5-1.5B-Instruct", "blind",
                 ("--phase", "compact2", "--n_train", "800", "--bs", "8"))],
          "1": [("qwen3-1.7b_compact2", "Qwen/Qwen3-1.7B", "blind",
                 ("--phase", "compact2", "--n_train", "800", "--bs", "8"))]},
}""",
    "R19_compact2_b": """{
    "c": {"0": [("qwen-3b_compact2", "Qwen/Qwen2.5-3B-Instruct", "blind",
                 ("--phase", "compact2", "--n_train", "800", "--bs", "6", "--mb", "2", "--accum", "8"))]},
}""",
    "R18b_scale_32b": """{
    "s": {"0,1": [("q25-32b-bnb4", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "scale",
                   ("--n", "150", "--n_pool", "220", "--bs", "4", "--load_in_4bit"))]},
}""",
    "R20_stride_a": """{
    "t": {"0": [("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "stride", ("--bs", "16"))],
          "1": [("qwen3-1.7b", "Qwen/Qwen3-1.7B", "stride", ("--bs", "16"))]},
}""",
    "R20_stride_b": """{
    "t": {"0": [("qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "stride", ("--bs", "12", "--mb", "4", "--accum", "4"))]},
}""",
    "R20_stride_7b": """{
    "t": {"0,1": [("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "stride",
                   ("--bs", "8", "--mb", "2", "--accum", "8", "--n_train", "2000", "--device_map", "auto"))]},
}""",
    "R21_stride_ext": """{
    "t": {"0": [("qwen3-1.7b-ext", "Qwen/Qwen3-1.7B", "stride", ("--bs", "16", "--strides", "5", "6", "8"))],
          "1": [("qwen3-4b", "Qwen/Qwen3-4B", "stride", ("--bs", "12", "--mb", "4", "--accum", "4",
                                                         "--strides", "1", "4", "6", "8", "12"))]},
}""",
    "R22_seeds": """{
    "t": {"0": [("qwen-1.5b_s1", "Qwen/Qwen2.5-1.5B-Instruct", "stride", ("--bs", "16", "--seed", "1", "--strides", "3", "4")),
                ("qwen-1.5b_s2", "Qwen/Qwen2.5-1.5B-Instruct", "stride", ("--bs", "16", "--seed", "2", "--strides", "3", "4"))],
          "1": [("qwen-3b_s1", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--seed", "1", "--strides", "3", "4")),
                ("qwen-3b_s2", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--seed", "2", "--strides", "3", "4"))]},
}""",
    "R22_sum_a": """{
    "t": {"0": [("sum_qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "stride",
                 ("--bs", "16", "--task", "sum", "--strides", "1", "2", "3", "4", "6", "12"))],
          "1": [("sum_qwen3-1.7b", "Qwen/Qwen3-1.7B", "stride",
                 ("--bs", "16", "--task", "sum", "--strides", "1", "2", "3", "4", "6", "12"))]},
}""",
    "R22_sum_b": """{
    "t": {"0": [("sum_qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--task", "sum", "--strides", "1", "2", "3", "4", "6", "12"))]},
}""",
    "R23_cumsum_a": """{
    "t": {"0": [("cumsum_qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "stride",
                 ("--bs", "16", "--task", "cumsum", "--strides", "1", "2", "3", "4", "5", "6", "12"))],
          "1": [("cumsum_qwen3-1.7b", "Qwen/Qwen3-1.7B", "stride",
                 ("--bs", "16", "--task", "cumsum", "--strides", "1", "2", "3", "4", "5", "6", "12"))]},
}""",
    "R23_cumsum_b": """{
    "t": {"0": [("cumsum_qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--task", "cumsum", "--strides", "1", "2", "3", "4", "5", "6", "12"))]},
}""",
    "R24_seeds_a": """{
    "t": {"0": [("qwen-3b_s3", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--seed", "3", "--strides", "4")),
                ("qwen-3b_s4", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--seed", "4", "--strides", "4"))],
          "1": [("qwen-1.5b_s3", "Qwen/Qwen2.5-1.5B-Instruct", "stride", ("--bs", "16", "--seed", "3", "--strides", "4")),
                ("qwen-1.5b_s4", "Qwen/Qwen2.5-1.5B-Instruct", "stride", ("--bs", "16", "--seed", "4", "--strides", "4"))]},
}""",
    "R24_seeds_7b": """{
    "t": {"0,1": [("qwen-7b_s1", "Qwen/Qwen2.5-7B-Instruct", "stride",
                   ("--bs", "8", "--mb", "2", "--accum", "8", "--n_train", "2000", "--device_map", "auto", "--seed", "1", "--strides", "4")),
                  ("qwen-7b_s2", "Qwen/Qwen2.5-7B-Instruct", "stride",
                   ("--bs", "8", "--mb", "2", "--accum", "8", "--n_train", "2000", "--device_map", "auto", "--seed", "2", "--strides", "4"))]},
}""",
    "R25_14b_s4": """{
    "t": {"0,1": [("q25-14b_s4", "Qwen/Qwen2.5-14B-Instruct", "stride",
                   ("--bs", "6", "--mb", "1", "--accum", "16", "--n_train", "2000", "--device_map", "auto",
                    "--load_in_4bit", "--no_merge", "--strides", "4"))]},
}""",
    "R25_14b_s6": """{
    "t": {"0,1": [("q25-14b_s6", "Qwen/Qwen2.5-14B-Instruct", "stride",
                   ("--bs", "6", "--mb", "1", "--accum", "16", "--n_train", "2000", "--device_map", "auto",
                    "--load_in_4bit", "--no_merge", "--strides", "6"))]},
}""",
    "R26_swap_a": """{
    "t": {"0": [("swap_qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "stride",
                 ("--bs", "16", "--task", "swap", "--strides", "1", "2", "3", "4", "12"))],
          "1": [("swap_qwen3-1.7b", "Qwen/Qwen3-1.7B", "stride",
                 ("--bs", "16", "--task", "swap", "--strides", "1", "2", "3", "4", "12"))]},
}""",
    "R26_swap_b": """{
    "t": {"0": [("swap_qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--task", "swap", "--strides", "1", "2", "3", "4", "12"))]},
}""",
    "R27_14b_s5": """{
    "t": {"0,1": [("q25-14b_s5e3", "Qwen/Qwen2.5-14B-Instruct", "stride",
                   ("--bs", "6", "--mb", "1", "--accum", "16", "--n_train", "2000", "--epochs", "3", "--device_map", "auto",
                    "--load_in_4bit", "--no_merge", "--strides", "5"))]},
}""",
    "R27_14b_s6e3": """{
    "t": {"0,1": [("q25-14b_s6e3", "Qwen/Qwen2.5-14B-Instruct", "stride",
                   ("--bs", "6", "--mb", "1", "--accum", "16", "--n_train", "2000", "--epochs", "3", "--device_map", "auto",
                    "--load_in_4bit", "--no_merge", "--strides", "6"))]},
}""",
    "R28_swap2_a": """{
    "t": {"0": [("swap2_qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "stride",
                 ("--bs", "16", "--task", "swap", "--n_train", "9000", "--epochs", "3", "--max_new", "128", "--strides", "1", "2", "3"))],
          "1": [("swap2_qwen3-1.7b", "Qwen/Qwen3-1.7B", "stride",
                 ("--bs", "16", "--task", "swap", "--n_train", "9000", "--epochs", "3", "--max_new", "128", "--strides", "1", "2", "3"))]},
}""",
    "R28_swap2_b": """{
    "t": {"0": [("swap2_qwen-3b", "Qwen/Qwen2.5-3B-Instruct", "stride",
                 ("--bs", "12", "--mb", "4", "--accum", "4", "--task", "swap", "--n_train", "9000", "--epochs", "3", "--max_new", "128",
                  "--strides", "1", "2", "3"))]},
}""",
}


def build():
    for name, phases in NOTEBOOKS.items():
        build_one(name, phases)


def build_one(name, phases):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell("!pip -q uninstall -y torchao 2>/dev/null | tail -1\n"
                                  "!pip -q install -U \"transformers>=4.44\" accelerate scikit-learn peft bitsandbytes 2>/dev/null | tail -1\n"
                                  "!nvidia-smi --query-gpu=name,memory.total --format=csv || true"),
             nbf.v4.new_code_cell("import os\nos.makedirs('hgcd', exist_ok=True)\n"
                                  "os.environ.setdefault('HGCD_DATA', os.path.abspath('hgcd_data'))")]
    for m in MODULES:
        src = open(os.path.join(ROOT, "hgcd", f"{m}.py")).read()
        cells.append(nbf.v4.new_code_cell(f"%%writefile hgcd/{m}.py\n{src}"))
    cells.append(nbf.v4.new_code_cell("import os\n" + CFG_TEMPLATE.format(name=name, phases=phases)))
    cells.append(nbf.v4.new_markdown_cell("## Run (both GPUs)"))
    cells.append(nbf.v4.new_code_cell(RUN))
    cells.append(nbf.v4.new_markdown_cell("## Results"))
    cells.append(nbf.v4.new_code_cell(REPORT))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "shutil.make_archive(f'{OUT}_{NOTEBOOK}', 'zip', OUT)\n"
        "print('download:', f'{OUT}_{NOTEBOOK}.zip')"))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = os.path.join(ROOT, "notebooks", f"run_{name}.ipynb")
    nbf.write(nb, out)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
