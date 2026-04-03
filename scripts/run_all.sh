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

# Generate 4-class dataset from 7-class (if not exists)
if [ ! -d "data/dataset_4class" ]; then
    echo "Generating 4-class dataset..."
    python src/preprocessing/prepare_dataset_4class.py
fi

mkdir -p notebooks/results

NOTEBOOKS=(
    # 7-class experiments
    "notebooks/01_train_cnn.ipynb"
    "notebooks/02_train_fcnn.ipynb"
    "notebooks/03_late_fusion.ipynb"
    "notebooks/04_intermediate_fusion.ipynb"
    "notebooks/05_comparison.ipynb"
    # 4-class experiments
    "notebooks/06_train_cnn_4class.ipynb"
    "notebooks/07_train_fcnn_4class.ipynb"
    "notebooks/08_late_fusion_4class.ipynb"
    "notebooks/09_intermediate_fusion_4class.ipynb"
    "notebooks/10_comparison_4class.ipynb"
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
