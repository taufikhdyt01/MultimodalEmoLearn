#!/bin/bash
# Run undersampling + conf60 experiments
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "  UNDERSAMPLING + CONF60"
echo "=========================================="

# Generate undersampled conf60 dataset if not exists
if [ ! -d "data/dataset_frontonly_conf60_under_660_4class" ]; then
    echo "Generating conf60 under_660 dataset..."
    python scripts/prepare_undersampled_conf60.py
fi

jupyter nbconvert --to notebook --execute "notebooks/58_undersampled_conf60.ipynb" \
    --output "58_undersampled_conf60_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=7200

echo ""
echo "=========================================="
echo "  DONE!"
echo "=========================================="
