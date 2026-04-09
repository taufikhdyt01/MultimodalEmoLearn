#!/bin/bash
# Run LOSO cross-validation via notebook 33
# Prerequisite: user_ids_*.npy files must exist in data/dataset_frontonly/
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  LOSO CROSS-VALIDATION (FRONT-ONLY)"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

# Check prerequisite
if [ ! -f "data/dataset_frontonly/user_ids_all.npy" ]; then
    echo "ERROR: user_ids_all.npy not found!"
    echo "Upload user_ids_*.npy files to data/dataset_frontonly/ first."
    exit 1
fi

echo ">> Running notebook 33 (LOSO)..."
jupyter nbconvert --to notebook --execute "notebooks/33_loso_frontonly.ipynb" \
    --output "33_loso_frontonly_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=36000

echo ""
echo "=========================================="
echo "  LOSO DONE!"
echo "  Results: models/frontonly/loso/"
echo "  Notebook: $OUTDIR/33_loso_frontonly_executed.ipynb"
echo "=========================================="
