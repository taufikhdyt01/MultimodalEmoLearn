#!/bin/bash
# Run benchmark experiments (JAFFE + CK+)
set -e
cd "$(dirname "$0")/.."

conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "  BENCHMARK: JAFFE (LOSO, 10 folds)"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/36_benchmark_jaffe.ipynb" \
    --output "36_benchmark_jaffe_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=36000

echo ""
echo "=========================================="
echo "  BENCHMARK: CK+ (10-Fold CV)"
echo "=========================================="
jupyter nbconvert --to notebook --execute "notebooks/37_benchmark_ckplus.ipynb" \
    --output "37_benchmark_ckplus_executed.ipynb" --output-dir "$OUTDIR" \
    --ExecutePreprocessor.timeout=36000

echo ""
echo "=========================================="
echo "  BENCHMARK DONE!"
echo "  Results: models/benchmark/"
echo "=========================================="
