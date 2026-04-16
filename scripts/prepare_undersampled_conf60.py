"""
Prepare Undersampled Conf60 Dataset
====================================
Undersample neutral on conf60 dataset (confidence >= 60%).
Combining 2 best strategies so far.

Usage:
    python scripts/prepare_undersampled_conf60.py
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUTPUT_BASE = PROJECT_ROOT / "data"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]
REMAP_4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}
SEED = 42


def undersample_train(y_train, target_neutral, seed=42):
    rng = np.random.RandomState(seed)
    neutral_idx = np.where(y_train == 0)[0]
    other_idx = np.where(y_train != 0)[0]
    selected_neutral = rng.choice(neutral_idx, size=min(target_neutral, len(neutral_idx)), replace=False)
    selected = np.concatenate([selected_neutral, other_idx])
    rng.shuffle(selected)
    return selected


def create_undersampled(target_neutral, suffix):
    out_7c = OUTPUT_BASE / f"dataset_frontonly_conf60_under{suffix}"
    out_4c = OUTPUT_BASE / f"dataset_frontonly_conf60_under{suffix}_4class"

    print(f"\n{'='*60}")
    print(f"  UNDERSAMPLE CONF60: neutral -> {target_neutral} (suffix: {suffix})")
    print(f"{'='*60}")

    X_train_img = np.load(SRC_DIR / "X_train_images.npy")
    X_train_lm = np.load(SRC_DIR / "X_train_landmarks.npy")
    y_train = np.load(SRC_DIR / "y_train.npy")

    print(f"\n  Original train (conf60): {len(y_train)} samples")
    counts = Counter(y_train.tolist())
    for i, emo in enumerate(EMOTIONS_7):
        print(f"    {emo:>12s}: {counts.get(i, 0)}")

    selected = undersample_train(y_train, target_neutral, SEED)
    X_us_img = X_train_img[selected]
    X_us_lm = X_train_lm[selected]
    y_us = y_train[selected]

    print(f"\n  Undersampled train: {len(y_us)} samples")
    counts_us = Counter(y_us.tolist())
    for i, emo in enumerate(EMOTIONS_7):
        print(f"    {emo:>12s}: {counts_us.get(i, 0)}")

    os.makedirs(out_7c, exist_ok=True)
    np.save(out_7c / "X_train_images.npy", X_us_img)
    np.save(out_7c / "X_train_landmarks.npy", X_us_lm)
    np.save(out_7c / "y_train.npy", y_us)

    for split in ["val", "test"]:
        for s in ["images", "landmarks"]:
            np.save(out_7c / f"X_{split}_{s}.npy", np.load(SRC_DIR / f"X_{split}_{s}.npy"))
        np.save(out_7c / f"y_{split}.npy", np.load(SRC_DIR / f"y_{split}.npy"))

    info = {
        "base": "dataset_frontonly_conf60",
        "strategy": f"conf60_undersample_neutral_{target_neutral}",
        "train_samples": len(y_us),
        "distribution_7class": {EMOTIONS_7[k]: int(v) for k, v in sorted(counts_us.items())},
    }
    json.dump(info, open(out_7c / "dataset_info.json", "w"), indent=2)
    json.dump({emo: i for i, emo in enumerate(EMOTIONS_7)}, open(out_7c / "label_map.json", "w"), indent=2)
    print(f"  Saved 7-class: {out_7c}")

    # 4-class remap
    os.makedirs(out_4c, exist_ok=True)
    for split in ["train", "val", "test"]:
        y = np.load(out_7c / f"y_{split}.npy")
        y4 = np.array([REMAP_4[int(c)] for c in y], dtype=np.int64)
        np.save(out_4c / f"y_{split}.npy", y4)
        for s in ["images", "landmarks"]:
            np.save(out_4c / f"X_{split}_{s}.npy", np.load(out_7c / f"X_{split}_{s}.npy"))

    y4_train = np.load(out_4c / "y_train.npy")
    counts_4 = Counter(y4_train.tolist())
    print(f"\n  4-class train: {len(y4_train)} samples")
    for i, emo in enumerate(EMOTIONS_4):
        print(f"    {emo:>12s}: {counts_4.get(i, 0)}")

    info_4 = {
        "base": "dataset_frontonly_conf60",
        "strategy": f"conf60_undersample_neutral_{target_neutral}_4class",
        "train_samples": len(y4_train),
        "distribution_4class": {EMOTIONS_4[k]: int(v) for k, v in sorted(counts_4.items())},
    }
    json.dump(info_4, open(out_4c / "dataset_info.json", "w"), indent=2)
    json.dump({emo: i for i, emo in enumerate(EMOTIONS_4)}, open(out_4c / "label_map.json", "w"), indent=2)
    print(f"  Saved 4-class: {out_4c}")


if __name__ == "__main__":
    # Focus on under_660 only (sweet spot from previous experiments)
    create_undersampled(660, "_660")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
