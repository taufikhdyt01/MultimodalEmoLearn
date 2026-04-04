#!/bin/bash
# Run only Transfer Learning notebooks (11-17)
# Prerequisite: notebooks 01-10 already completed (need FCNN models for fusion)
# Usage: bash scripts/run_transfer.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  MultimodalEmoLearn - Transfer Learning"
echo "============================================"
echo ""

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Generate 4-class dataset if not exists
if [ ! -d "data/dataset_4class" ]; then
    echo "Generating 4-class dataset..."
    python src/preprocessing/prepare_dataset_4class.py
fi

mkdir -p notebooks/results

NOTEBOOKS=(
    # 7-class transfer learning
    "notebooks/11_train_cnn_transfer.ipynb"
    "notebooks/12_late_fusion_transfer.ipynb"
    "notebooks/13_intermediate_fusion_transfer.ipynb"
    # 4-class transfer learning
    "notebooks/14_train_cnn_transfer_4class.ipynb"
    "notebooks/15_late_fusion_transfer_4class.ipynb"
    "notebooks/16_intermediate_fusion_transfer_4class.ipynb"
    # Final comparison (all experiments)
    "notebooks/17_comparison_all.ipynb"
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
echo "  TRANSFER LEARNING EXPERIMENTS COMPLETED!"
echo "  Results: notebooks/results/"
echo "  Models: models/cnn_transfer/ & models/4class/cnn_transfer/"
echo "============================================"
