#!/bin/bash
# Run conf60 experiments (confidence >= 60% filtered dataset)
# Prerequisites:
#   1. data/dataset_frontonly_conf60/ must exist
#   2. Run prepare_conf60_all.py first for augmented + 4-class
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

# Step 0: Generate derived datasets if needed
if [ ! -d "data/dataset_frontonly_conf60_4class" ]; then
    echo "Generating conf60 augmented + 4-class datasets..."
    python scripts/prepare_conf60_all.py
fi

echo "=========================================="
echo "  CONF60: FROM SCRATCH 7-CLASS"
echo "=========================================="
for nb in 43_cnn_conf60_7class 44_fcnn_conf60_7class 45_late_fusion_conf60_7class 46_intermediate_conf60_7class; do
    echo ">> $nb"
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
done

echo "=========================================="
echo "  CONF60: FROM SCRATCH 4-CLASS"
echo "=========================================="
for nb in 47_cnn_conf60_4class 48_fcnn_conf60_4class 49_late_fusion_conf60_4class 50_intermediate_conf60_4class; do
    echo ">> $nb"
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
done

echo "=========================================="
echo "  CONF60: TRANSFER LEARNING"
echo "=========================================="
for nb in 51_cnn_tl_conf60_7class 52_late_fusion_tl_conf60_7class 53_intermediate_tl_conf60_7class \
          54_cnn_tl_conf60_4class 55_late_fusion_tl_conf60_4class 56_intermediate_tl_conf60_4class; do
    echo ">> $nb"
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=7200
done

echo "=========================================="
echo "  CONF60: COMPARISON"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/57_comparison_conf60.ipynb" \
    --output "57_comparison_conf60_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=3600

echo ""
echo "=========================================="
echo "  ALL CONF60 EXPERIMENTS DONE!"
echo "=========================================="
