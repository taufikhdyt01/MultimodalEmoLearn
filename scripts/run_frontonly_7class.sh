#!/bin/bash
# Run front-only 7-class from scratch experiments (notebooks 18-21)
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  FRONT-ONLY 7-CLASS FROM SCRATCH"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

for nb in 18_cnn_frontonly_7class 19_fcnn_frontonly_7class 20_late_fusion_frontonly_7class 21_intermediate_frontonly_7class; do
    echo ""
    echo ">> Running $nb ..."
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
    echo ">> Done: $nb"
done

echo ""
echo "=========================================="
echo "  ALL 7-CLASS FRONT-ONLY DONE!"
echo "=========================================="