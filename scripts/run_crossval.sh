#!/bin/bash
# Run 5-Fold Cross-Validation via notebook 34
# Prerequisite: user_ids_*.npy files must exist in data/dataset_frontonly/
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  5-FOLD CROSS-VALIDATION (FRONT-ONLY)"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

if [ ! -f "data/dataset_frontonly/user_ids_all.npy" ]; then
    echo "ERROR: user_ids_all.npy not found!"
    echo "Upload user_ids_*.npy files to data/dataset_frontonly/ first."
    exit 1
fi

echo ">> Running notebook 34 (5-Fold CV)..."
jupyter nbconvert --to notebook --execute "notebooks/34_crossval_frontonly.ipynb" \
    --output "34_crossval_frontonly_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "=========================================="
echo "  5-FOLD CV DONE!"
echo "  Results: models/frontonly/crossval/"
echo "  Notebook: $OUTDIR/34_crossval_frontonly_executed.ipynb"
echo "=========================================="
