#!/bin/bash
# Run undersampling experiments
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "  UNDERSAMPLING EXPERIMENTS"
echo "=========================================="

# Generate undersampled datasets (if not exists)
if [ ! -d "data/dataset_frontonly_under_660_4class" ]; then
    echo "Generating undersampled datasets..."
    python scripts/prepare_undersampled.py
fi

jupyter nbconvert --to notebook --execute "notebooks/42_frontonly_undersampled.ipynb" \
    --output "42_frontonly_undersampled_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "=========================================="
echo "  UNDERSAMPLING DONE!"
echo "=========================================="
