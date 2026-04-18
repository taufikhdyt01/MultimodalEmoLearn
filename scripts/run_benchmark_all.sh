#!/bin/bash
# Runner for all benchmark notebooks (Skema 1 + Skema 2)
# Prerequisites:
#   - data/benchmark/rafdb_7class & rafdb_4class generated (scripts/prepare_rafdb.py)
#   - data/benchmark/kdef_7class & kdef_4class generated (scripts/prepare_kdef.py)
#   - data/benchmark/ckplus_* & jaffe_* already generated (scripts/prepare_benchmark.py)
#   - data/dataset_frontonly_conf60/ for Primer

set -e
cd "$(dirname "$0")/.."
conda activate emotrain 2>/dev/null || true

OUTDIR="notebooks/results"
mkdir -p "$OUTDIR"

run_nb() {
    local nb="$1"
    echo ""
    echo "=========================================="
    echo "  RUN: $nb"
    echo "=========================================="
    jupyter nbconvert --to notebook --execute "notebooks/$nb" \
        --output "${nb%.ipynb}_executed.ipynb" --output-dir "$OUTDIR" \
        --ExecutePreprocessor.timeout=43200
}

# Skema 1: self train-test
run_nb "60_benchmark_rafdb.ipynb"
run_nb "61_benchmark_kdef.ipynb"
run_nb "62_benchmark_primer.ipynb"

# CK+/JAFFE already executed previously — rerun for micro F1 if desired:
# run_nb "36_benchmark_jaffe.ipynb"
# run_nb "37_benchmark_ckplus.ipynb"

# Skema 2: cross-dataset → Primer (panjang! 4-8 jam)
run_nb "63_crossdataset_to_primer.ipynb"

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "=========================================="
