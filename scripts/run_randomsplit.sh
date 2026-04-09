#!/bin/bash
# Run Random Split evaluation via notebook 35
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

echo "=========================================="
echo "  RANDOM SPLIT (FRONT-ONLY)"
echo "=========================================="

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo ">> Running notebook 35 (Random Split)..."
jupyter nbconvert --to notebook --execute "notebooks/35_randomsplit_frontonly.ipynb" \
    --output "35_randomsplit_frontonly_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=7200

echo ""
echo "=========================================="
echo "  RANDOM SPLIT DONE!"
echo "  Results: models/frontonly/randomsplit/"
echo "  Notebook: $OUTDIR/35_randomsplit_frontonly_executed.ipynb"
echo "=========================================="
