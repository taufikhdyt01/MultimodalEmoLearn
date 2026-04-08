#!/bin/bash
# Run front-only 4-class from scratch experiments (notebooks 22-25)
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  FRONT-ONLY 4-CLASS FROM SCRATCH"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

for nb in 22_cnn_frontonly_4class 23_fcnn_frontonly_4class 24_late_fusion_frontonly_4class 25_intermediate_frontonly_4class; do
    echo ""
    echo ">> Running $nb ..."
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
    echo ">> Done: $nb"
done

echo ""
echo "=========================================="
echo "  ALL 4-CLASS FRONT-ONLY DONE!"
echo "=========================================="
