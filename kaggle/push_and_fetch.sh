#!/usr/bin/env bash
# Push the notebook to Kaggle as a private GPU kernel, wait for it, download the results.
# Needs KAGGLE_API_TOKEN in the environment (never commit it). "NvidiaTeslaT4" = the 2x T4 machine.
set -euo pipefail
NB="${1:-A_reliance_small}"   # A_reliance_small | B_7b_and_mediation
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN (Kaggle settings -> API -> Generate New Token)}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-$(kaggle kernels list --mine --page-size 1 2>/dev/null | awk 'NR==3{split($1,a,"/"); print a[1]}')}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
python3 "$ROOT/tools/build_notebook.py" >/dev/null
cp "$ROOT/notebooks/run_$NB.ipynb" "$WORK/hgcd_kaggle.ipynb"
SLUG="hgcd-$(echo "$NB" | tr "_A-Z" "-a-z")"
cat > "$WORK/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USERNAME/$SLUG", "title": "$SLUG", "code_file": "hgcd_kaggle.ipynb",
 "language": "python", "kernel_type": "notebook", "is_private": true,
 "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
 "kernel_sources": []}
JSON
kaggle kernels push -p "$WORK"
echo "pushed $KAGGLE_USERNAME/$SLUG; polling status"
while true; do
  s=$(kaggle kernels status "$KAGGLE_USERNAME/$SLUG" 2>&1 | tail -1); echo "$(date +%T) $s"
  case "$s" in *complete*|*error*|*cancel*) break;; esac
  sleep 300
done
mkdir -p "$ROOT/results/kaggle_out_$NB"
kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$ROOT/results/kaggle_out_$NB"
echo "downloaded to results/kaggle_out_$NB"
