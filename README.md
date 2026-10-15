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

## Hướng dẫn nhanh (tiếng Việt)

1. **Kaggle**: tải `notebooks/hgcd_kaggle.ipynb` lên Kaggle → *Settings*: **GPU T4 x2**, **Internet: On** → **Run All**.
   - Lần đầu để `PRESET = "kaggle"` (khoảng 2–4 giờ) để kiểm tra trên GPU và đo thời gian chạy.
   - Số liệu đưa vào paper: dùng `PRESET = "kaggle_full"` (khoảng 8–10 giờ, nằm trong giới hạn 12 giờ của Kaggle).
   - GPU 0 chạy Qwen2.5-1.5B, GPU 1 chạy SmolLM2-1.7B (họ mô hình thứ hai).
   - Nếu phiên bị ngắt, chạy lại notebook: các bước đã xong được tự đọc lại từ checkpoint (`stage_*.pkl`, `*_records.pkl`).
2. **A100 (1 ngày)**: `python -m hgcd.pipeline --preset a100 --outdir results/a100_main` (Qwen2.5-7B).
3. **Đưa kết quả vào paper**: giải nén `results_<preset>.zip` sao cho có thư mục `results/kaggle_main/` chứa các file `tab_*.tex`,
   rồi biên dịch `paper/main.tex`. Bảng nào chưa có kết quả sẽ hiện ô "Pending".
4. **Điểm dừng (go/no-go)** theo giả thuyết H1: AUC(Δ^total) ≥ 0.75 và Δ^placebo, Δ^marker ≈ 0.
   Nếu không đạt thì không thuê A100; bài khi đó chuyển thành bài phân tích.
