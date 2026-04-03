"""
Prepare 4-Class Dataset
========================
Remap 7 emosi → 4 kelas:
  neutral(0), happy(1), sad(2), negative(3)
  negative = angry + fearful + disgusted + surprised

Menggunakan dataset 7-kelas yang sudah ada, hanya remap label.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

# ============== CONFIG ==============
DATASET_7_DIR = Path("data/dataset")
DATASET_7_AUG_DIR = Path("data/dataset_augmented")
OUTPUT_DIR = Path("data/dataset_4class")
OUTPUT_AUG_DIR = Path("data/dataset_4class_augmented")

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]

# Mapping: 7-class index → 4-class index
# 0:neutral→0, 1:happy→1, 2:sad→2, 3:angry→3, 4:fearful→3, 5:disgusted→3, 6:surprised→3
REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}
# ====================================


def remap_labels(y):
    """Remap array label dari 7-kelas ke 4-kelas."""
    return np.array([REMAP[label] for label in y], dtype=np.int32)


def process_dataset(src_dir, dst_dir):
    """Copy images/landmarks, remap labels, update weights."""
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\n  Source: {src_dir}")
    print(f"  Output: {dst_dir}")

    for split in ["train", "val", "test"]:
        # Copy images & landmarks (unchanged)
        for dtype in ["images", "landmarks"]:
            src = src_dir / f"X_{split}_{dtype}.npy"
            dst = dst_dir / f"X_{split}_{dtype}.npy"
            if src.exists() and not dst.exists():
                print(f"  Linking {src.name}...")
                import shutil
                shutil.copy2(src, dst)

        # Remap labels
        y_src = src_dir / f"y_{split}.npy"
        y_dst = dst_dir / f"y_{split}.npy"
        if y_src.exists():
            y = np.load(y_src)
            y_new = remap_labels(y)
            np.save(y_dst, y_new)

            counts_old = Counter(y.tolist())
            counts_new = Counter(y_new.tolist())
            print(f"  {split}: {len(y)} samples")
            print(f"    7-class: {dict(sorted(counts_old.items()))}")
            print(f"    4-class: {dict(sorted(counts_new.items()))}")

    # Class weights for 4-class
    y_train = np.load(dst_dir / "y_train.npy")
    counts = Counter(y_train.tolist())
    beta = 0.999
    weights = []
    for i in range(4):
        eff = (1 - beta ** counts[i]) / (1 - beta)
        weights.append(1.0 / eff)
    min_w = min(weights)
    weights = [round(w / min_w, 4) for w in weights]

    weight_info = {
        "emotions": EMOTIONS_4,
        "num_classes": 4,
        "remap": {"angry": "negative", "fearful": "negative", "disgusted": "negative", "surprised": "negative"},
        "train_counts": {EMOTIONS_4[i]: counts[i] for i in range(4)},
        "weights_array": weights,
        "effective_number_weights": {EMOTIONS_4[i]: weights[i] for i in range(4)},
        "method": "Class-Balanced Loss (Cui et al., 2019), beta=0.999",
    }
    with open(dst_dir / "class_weights.json", "w") as f:
        json.dump(weight_info, f, indent=2)

    # Label map
    with open(dst_dir / "label_map.json", "w") as f:
        json.dump({str(i): e for i, e in enumerate(EMOTIONS_4)}, f, indent=2)

    # Dataset info
    info = {
        "num_classes": 4,
        "emotions": EMOTIONS_4,
        "remap_from_7class": REMAP,
        "train_samples": len(y_train),
        "train_distribution": {EMOTIONS_4[i]: counts[i] for i in range(4)},
        "class_weights": weights,
    }
    with open(dst_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Class weights (4-class):")
    for i, emo in enumerate(EMOTIONS_4):
        print(f"    {emo:>10s}: count={counts[i]:>5d}, weight={weights[i]:.4f}")


def main():
    print("=" * 60)
    print("PREPARE 4-CLASS DATASET")
    print("  neutral, happy, sad, negative(angry+fearful+disgusted+surprised)")
    print("=" * 60)

    print("\n[1/2] Original dataset...")
    process_dataset(DATASET_7_DIR, OUTPUT_DIR)

    print("\n[2/2] Augmented dataset...")
    process_dataset(DATASET_7_AUG_DIR, OUTPUT_AUG_DIR)

    print(f"\n{'=' * 60}")
    print("SELESAI!")
    print(f"  4-class original:  {OUTPUT_DIR}")
    print(f"  4-class augmented: {OUTPUT_AUG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
