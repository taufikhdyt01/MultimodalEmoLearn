"""
Prepare all conf60 dataset variants (same as prepare_frontonly_all.py but for conf60).
Prerequisite: data/dataset_frontonly_conf60/ must exist.

Generates:
  data/dataset_frontonly_conf60_augmented/
  data/dataset_frontonly_conf60_4class/
  data/dataset_frontonly_conf60_4class_augmented/

Usage:
    python scripts/prepare_conf60_all.py
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
SRC_DATASET = BASE / "data" / "dataset_frontonly_conf60"
AUG_DATASET = BASE / "data" / "dataset_frontonly_conf60_augmented"
FOURCLASS_DIR = BASE / "data" / "dataset_frontonly_conf60_4class"
FOURCLASS_AUG_DIR = BASE / "data" / "dataset_frontonly_conf60_4class_augmented"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]
REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}

IMG_SIZE = 224
TARGET_MIN = 150


def augment_image(img):
    import cv2
    augmented = []
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    for angle in [-15, 15]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    for factor in [0.8, 1.2]:
        augmented.append(np.clip(img * factor, 0, 1).astype(np.float32))
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), 10, 1.0)
    augmented.append(cv2.warpAffine(flipped, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT))
    return augmented


def augment_landmark(lm):
    augmented = []
    coords = lm.reshape(68, 2)
    flipped = coords.copy(); flipped[:, 0] = 1.0 - flipped[:, 0]
    augmented.append(flipped.flatten())
    for angle_deg in [-15, 15]:
        angle = np.radians(angle_deg)
        rot = coords.copy(); rot -= 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        new_x = rot[:, 0]*cos_a - rot[:, 1]*sin_a
        new_y = rot[:, 0]*sin_a + rot[:, 1]*cos_a
        rot[:, 0] = new_x + 0.5; rot[:, 1] = new_y + 0.5
        augmented.append(rot.flatten())
    augmented.append(lm.copy())
    augmented.append(lm.copy())
    flipped_rot = flipped.copy(); flipped_rot -= 0.5
    angle = np.radians(10); cos_a, sin_a = np.cos(angle), np.sin(angle)
    new_x = flipped_rot[:, 0]*cos_a - flipped_rot[:, 1]*sin_a
    new_y = flipped_rot[:, 0]*sin_a + flipped_rot[:, 1]*cos_a
    flipped_rot[:, 0] = new_x + 0.5; flipped_rot[:, 1] = new_y + 0.5
    augmented.append(flipped_rot.flatten())
    return augmented


def do_augmentation():
    print("\n" + "="*60)
    print("STEP 1: AUGMENT MINORITY CLASSES (conf60)")
    print("="*60)
    os.makedirs(AUG_DATASET, exist_ok=True)
    y_train = np.load(SRC_DATASET / "y_train.npy")
    X_train_img = np.load(SRC_DATASET / "X_train_images.npy")
    X_train_lm = np.load(SRC_DATASET / "X_train_landmarks.npy")
    counts = Counter(y_train.tolist())
    print(f"  Original train: {len(y_train)} samples")
    for i in sorted(counts): print(f"    {EMOTIONS_7[i]:>10s}: {counts[i]}")

    aug_images, aug_lm, aug_labels = list(X_train_img), list(X_train_lm), list(y_train)
    for cls_idx in range(7):
        cls_count = counts.get(cls_idx, 0)
        if cls_count >= TARGET_MIN or cls_count == 0: continue
        needed = TARGET_MIN - cls_count
        cls_indices = np.where(y_train == cls_idx)[0]
        print(f"  Augmenting {EMOTIONS_7[cls_idx]}: {cls_count} -> {TARGET_MIN} (+{needed})")
        added = 0
        while added < needed:
            for idx in cls_indices:
                if added >= needed: break
                for a_img, a_lm in zip(augment_image(X_train_img[idx]), augment_landmark(X_train_lm[idx])):
                    if added >= needed: break
                    aug_images.append(a_img); aug_lm.append(a_lm); aug_labels.append(cls_idx); added += 1

    np.save(AUG_DATASET / "X_train_images.npy", np.array(aug_images, dtype=np.float32))
    np.save(AUG_DATASET / "X_train_landmarks.npy", np.array(aug_lm, dtype=np.float32))
    np.save(AUG_DATASET / "y_train.npy", np.array(aug_labels, dtype=np.int64))
    print(f"  Augmented train: {len(aug_labels)} samples")

    for split in ["val", "test"]:
        for s in ["images", "landmarks"]:
            np.save(AUG_DATASET / f"X_{split}_{s}.npy", np.load(SRC_DATASET / f"X_{split}_{s}.npy"))
        np.save(AUG_DATASET / f"y_{split}.npy", np.load(SRC_DATASET / f"y_{split}.npy"))

    import shutil
    for f in ["label_map.json", "dataset_info.json"]:
        if (SRC_DATASET / f).exists(): shutil.copy2(SRC_DATASET / f, AUG_DATASET / f)
    print("  Done!")


def remap_to_4class(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        y = np.load(src_dir / f"y_{split}.npy")
        y4 = np.array([REMAP[int(c)] for c in y], dtype=np.int64)
        for s in ["images", "landmarks"]:
            np.save(out_dir / f"X_{split}_{s}.npy", np.load(src_dir / f"X_{split}_{s}.npy"))
        np.save(out_dir / f"y_{split}.npy", y4)
        print(f"    {split}: {len(y4)} -> {dict(sorted(Counter(y4.tolist()).items()))}")
    json.dump({emo: i for i, emo in enumerate(EMOTIONS_4)}, open(out_dir / "label_map.json", "w"), indent=2)


def do_4class():
    print("\n" + "="*60)
    print("STEP 2: 4-CLASS REMAP (conf60)")
    print("="*60)
    print("\n  Original 7-class -> 4-class:")
    remap_to_4class(SRC_DATASET, FOURCLASS_DIR)
    print("\n  Augmented 7-class -> 4-class:")
    remap_to_4class(AUG_DATASET, FOURCLASS_AUG_DIR)
    print("  Done!")


if __name__ == "__main__":
    do_augmentation()
    do_4class()
    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"  7-class:     {SRC_DATASET}")
    print(f"  7-class aug: {AUG_DATASET}")
    print(f"  4-class:     {FOURCLASS_DIR}")
    print(f"  4-class aug: {FOURCLASS_AUG_DIR}")
    print("="*60)
