#!/bin/bash
# Run benchmark LOSO (JAFFE) and 10-Fold CV (CK+)
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "  BENCHMARK: JAFFE LOSO (10 folds)"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/38_benchmark_jaffe_loso.ipynb" \
    --output "38_benchmark_jaffe_loso_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=36000

echo ""
echo "=========================================="
echo "  BENCHMARK: CK+ 10-Fold CV"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/39_benchmark_ckplus_cv10.ipynb" \
    --output "39_benchmark_ckplus_cv10_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=36000

echo ""
echo "=========================================="
echo "  BENCHMARK CV DONE!"
echo "=========================================="
