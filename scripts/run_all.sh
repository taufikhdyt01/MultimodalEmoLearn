#!/bin/bash
# Run all training notebooks sequentially
# Usage: bash scripts/run_all.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  MultimodalEmoLearn - Run All Experiments"
echo "============================================"
echo ""

# Check environment
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

NOTEBOOKS=(
    "notebooks/01_train_cnn.ipynb"
    "notebooks/02_train_fcnn.ipynb"
    "notebooks/03_late_fusion.ipynb"
    "notebooks/04_intermediate_fusion.ipynb"
    "notebooks/05_comparison.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Running: $nb"
    echo "  Started: $(date)"
    echo "========================================"

    jupyter nbconvert --to notebook --execute "$nb" \
        --output "$(basename $nb .ipynb)_executed.ipynb" \
        --output-dir "notebooks/results/" \
        --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=python3

    echo "  Finished: $(date)"
done

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETED!"
echo "  Results: notebooks/results/"
echo "  Models: models/"
echo "============================================"
