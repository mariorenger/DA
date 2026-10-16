#!/usr/bin/env bash
# One rented A100 (80 GB), ~1 day: larger models for the scale question (reviewer W2).
set -euo pipefail
pip install -q -U "transformers>=4.44" accelerate scikit-learn pandas matplotlib
for m in "qwen-14b Qwen/Qwen2.5-14B-Instruct" "olmo2-7b allenai/OLMo-2-1124-7B-Instruct" "qwen-7b Qwen/Qwen2.5-7B-Instruct"; do
  set -- $m
  python -m hgcd.pipeline --preset reliance --model "$2" --outdir "results/a100/reliance_$1" --device_map auto
done
python -m hgcd.pipeline --preset mediation --model Qwen/Qwen2.5-7B-Instruct --outdir results/a100/mediation_qwen-7b
python -m hgcd.pipeline --preset mediation --model Qwen/Qwen2.5-14B-Instruct --outdir results/a100/mediation_qwen-14b --device_map auto
