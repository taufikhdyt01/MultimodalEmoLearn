#!/bin/bash
# Run front-only transfer learning experiments (notebooks 26-31)
# Prerequisite: notebooks 18-25 (from scratch) must be completed first
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  FRONT-ONLY TRANSFER LEARNING"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

for nb in 26_cnn_tl_frontonly_7class 27_late_fusion_tl_frontonly_7class 28_intermediate_tl_frontonly_7class \
          29_cnn_tl_frontonly_4class 30_late_fusion_tl_frontonly_4class 31_intermediate_tl_frontonly_4class; do
    echo ""
    echo ">> Running $nb ..."
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
    echo ">> Done: $nb"
done

echo ""
echo "=========================================="
echo "  ALL TRANSFER LEARNING FRONT-ONLY DONE!"
echo "=========================================="
