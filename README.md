# HGCD — Hop-Graph Contrastive Decoding

Training-free, inference-time causal intervention for multi-hop reasoning: the model is contrasted
against **interventions on its own reasoning graph** (graph-aware `do()` on each answer-relevant hop),
with an evaluation protocol whose interventions have **known ground-truth effects**.

| Path | What |
|---|---|
| `paper/main.tex` | LaTeX paper (tables are `\input` from `results/kaggle_main/`) |
| `notebooks/hgcd_kaggle.ipynb` | Self-contained Kaggle notebook (Run All) |
| `hgcd/` | Package: data + interventions, chain/DRG, LM scoring, decoding rules, experiments, metrics, report |
| `tests/` | Unit tests for the logic (parsing, DRG, do(), decoding rules, Prop. 1 check) |
| `tools/build_notebook.py` | Regenerates the notebook from `hgcd/*.py` |
| `REVIEW.md` | Adversarial self-review and the revisions it triggered |
| `results/pilot_cpu/` | CPU smoke run (Qwen2.5-0.5B, tiny n) — sanity check only, not evidence |

## Run on Kaggle (2× T4, free)

1. Upload `notebooks/hgcd_kaggle.ipynb` → *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**.
2. Pick `PRESET` in the config cell: `kaggle` (~2–4 h), `kaggle_full` (~8–10 h), `smoke` (minutes).
3. **Run All**. GPU 0 runs Qwen2.5-1.5B-Instruct; GPU 1 runs SmolLM2-1.7B-Instruct (second model family).
4. Download `results_<preset>.zip`, unzip so that `results/kaggle_main/` contains `tab_*.tex`,
   then compile the paper (`cd paper && latexmk -pdf main.tex` or `tectonic main.tex`).

## Run on one A100 (1 day)

```bash
pip install -U "transformers>=4.44" accelerate scikit-learn pandas matplotlib
python -m hgcd.pipeline --preset a100 --outdir results/a100_main          # Qwen2.5-7B-Instruct
```

## Local checks

```bash
python -m pytest -q tests
python -m hgcd.pipeline --preset smoke --outdir results/smoke             # CPU, ~30-60 min
python tools/build_notebook.py
```

## Pipeline stages (each writes CSV + LaTeX)

1. **Estimator validation**: gold chains + one inserted true-but-redundant hop → AUC of graph-aware
   total-effect masking vs naive masking vs placebo (vs rollout PNS when `est_rollouts>0`).
2. **α selection** on the CLUTRR validation split (never on test).
3. **CLUTRR test**: ID k≤4 / OOD k≥5; interventions `rename / order / distract / placebo_input`
   (true effect 0) → accuracy and paired ATE with bootstrap CIs; L1 chains; self-consistency.
4. **MuSiQue**: real text, bridge chains; `dist8 / order` interventions; EM/F1, candidate recall.
