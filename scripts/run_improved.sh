#!/bin/bash
# Run improved experiments: Focal Loss + FER2013 pre-training
# Prerequisites:
#   1. notebook 40 (FER2013 pre-training) must be completed first
#   2. data/benchmark/fer2013_prepared/ must exist
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "  STEP 1: Pre-train ResNet18 on FER2013"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/40_pretrain_fer2013.ipynb" \
    --output "40_pretrain_fer2013_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "=========================================="
echo "  STEP 2: Improved Experiments"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/41_frontonly_improved.ipynb" \
    --output "41_frontonly_improved_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "=========================================="
echo "  IMPROVED EXPERIMENTS DONE!"
echo "=========================================="
