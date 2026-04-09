"""
Prepare all front-only dataset variants:
1. 7-class front-only (already done)
2. 7-class front-only augmented
3. 4-class front-only
4. 4-class front-only augmented

Temporarily patches the hardcoded paths in augment and 4class scripts.
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
SRC_DATASET = BASE / "data" / "dataset_frontonly"
AUG_DATASET = BASE / "data" / "dataset_frontonly_augmented"
FOURCLASS_DIR = BASE / "data" / "dataset_frontonly_4class"
FOURCLASS_AUG_DIR = BASE / "data" / "dataset_frontonly_4class_augmented"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]
REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}

IMG_SIZE = 224
TARGET_MIN = 150


# ── STEP 1: AUGMENT ──────────────────────────────────
def augment_image(img):
    """Generate augmented versions of an image."""
    import cv2
    augmented = []

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # Small rotations
    for angle in [-15, 15]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # Brightness
    for factor in [0.8, 1.2]:
        bright = np.clip(img * factor, 0, 1).astype(np.float32)
        augmented.append(bright)

    # Flip + rotation
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), 10, 1.0)
    augmented.append(cv2.warpAffine(flipped, M, (IMG_SIZE, IMG_SIZE),
                                     borderMode=cv2.BORDER_REFLECT))

    return augmented


def augment_landmark(lm):
    """Generate augmented versions of landmark (136-dim)."""
    augmented = []
    coords = lm.reshape(68, 2)

    # Flip: mirror x
    flipped = coords.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    augmented.append(flipped.flatten())

    # Rotations
    for angle_deg in [-15, 15]:
        angle = np.radians(angle_deg)
        cx, cy = 0.5, 0.5
        rot = coords.copy()
        rot[:, 0] -= cx
        rot[:, 1] -= cy
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        new_x = rot[:, 0] * cos_a - rot[:, 1] * sin_a
        new_y = rot[:, 0] * sin_a + rot[:, 1] * cos_a
        rot[:, 0] = new_x + cx
        rot[:, 1] = new_y + cy
        augmented.append(rot.flatten())

    # Brightness doesn't change landmarks
    augmented.append(lm.copy())
    augmented.append(lm.copy())

    # Flip + rotation
    flipped_rot = flipped.copy()
    flipped_rot[:, 0] -= 0.5
    flipped_rot[:, 1] -= 0.5
    angle = np.radians(10)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    new_x = flipped_rot[:, 0] * cos_a - flipped_rot[:, 1] * sin_a
    new_y = flipped_rot[:, 0] * sin_a + flipped_rot[:, 1] * cos_a
    flipped_rot[:, 0] = new_x + 0.5
    flipped_rot[:, 1] = new_y + 0.5
    augmented.append(flipped_rot.flatten())

    return augmented


def do_augmentation():
    """Augment minority classes in front-only dataset."""
    print("\n" + "="*60)
    print("STEP 1: AUGMENT MINORITY CLASSES (front-only)")
    print("="*60)

    os.makedirs(AUG_DATASET, exist_ok=True)

    y_train = np.load(SRC_DATASET / "y_train.npy")
    X_train_img = np.load(SRC_DATASET / "X_train_images.npy")
    X_train_lm = np.load(SRC_DATASET / "X_train_landmarks.npy")

    counts = Counter(y_train.tolist())
    print(f"  Original train: {len(y_train)} samples")
    for emo_idx in sorted(counts):
        print(f"    {EMOTIONS_7[emo_idx]:>10s}: {counts[emo_idx]}")

    # Find minority classes
    aug_images = list(X_train_img)
    aug_lm = list(X_train_lm)
    aug_labels = list(y_train)

    for cls_idx in range(7):
        cls_count = counts.get(cls_idx, 0)
        if cls_count >= TARGET_MIN or cls_count == 0:
            continue

        needed = TARGET_MIN - cls_count
        cls_indices = np.where(y_train == cls_idx)[0]
        print(f"  Augmenting {EMOTIONS_7[cls_idx]}: {cls_count} -> {TARGET_MIN} (+{needed})")

        added = 0
        while added < needed:
            for idx in cls_indices:
                if added >= needed:
                    break
                augs_img = augment_image(X_train_img[idx])
                augs_lm = augment_landmark(X_train_lm[idx])
                for a_img, a_lm in zip(augs_img, augs_lm):
                    if added >= needed:
                        break
                    aug_images.append(a_img)
                    aug_lm.append(a_lm)
                    aug_labels.append(cls_idx)
                    added += 1

    X_aug_img = np.array(aug_images, dtype=np.float32)
    X_aug_lm = np.array(aug_lm, dtype=np.float32)
    y_aug = np.array(aug_labels, dtype=np.int64)

    print(f"  Augmented train: {len(y_aug)} samples")

    np.save(AUG_DATASET / "X_train_images.npy", X_aug_img)
    np.save(AUG_DATASET / "X_train_landmarks.npy", X_aug_lm)
    np.save(AUG_DATASET / "y_train.npy", y_aug)

    # Copy val/test unchanged
    for split in ["val", "test"]:
        for suffix in ["images", "landmarks"]:
            src = SRC_DATASET / f"X_{split}_{suffix}.npy"
            if src.exists():
                np.save(AUG_DATASET / f"X_{split}_{suffix}.npy",
                        np.load(src))
        src = SRC_DATASET / f"y_{split}.npy"
        if src.exists():
            np.save(AUG_DATASET / f"y_{split}.npy", np.load(src))

    # Copy metadata
    for f in ["label_map.json", "dataset_info.json"]:
        src = SRC_DATASET / f
        if src.exists():
            import shutil
            shutil.copy2(src, AUG_DATASET / f)

    print("  Done!")


# ── STEP 2: 4-CLASS REMAP ────────────────────────────
def remap_to_4class(src_dir, out_dir, name):
    """Remap 7-class labels to 4-class."""
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        y = np.load(src_dir / f"y_{split}.npy")
        y4 = np.array([REMAP[int(c)] for c in y], dtype=np.int64)

        # Copy images and landmarks
        for suffix in ["images", "landmarks"]:
            src = src_dir / f"X_{split}_{suffix}.npy"
            if src.exists():
                np.save(out_dir / f"X_{split}_{suffix}.npy", np.load(src))
        np.save(out_dir / f"y_{split}.npy", y4)

        counts = Counter(y4.tolist())
        print(f"    {split}: {len(y4)} -> {dict(counts)}")

    # Label map
    label_map = {emo: i for i, emo in enumerate(EMOTIONS_4)}
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Class weights
    y_train = np.load(out_dir / "y_train.npy")
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    weights = {EMOTIONS_4[k]: round(max_count / v, 4) for k, v in sorted(counts.items())}
    print(f"    Weights: {weights}")


def do_4class():
    """Create 4-class versions."""
    print("\n" + "="*60)
    print("STEP 2: 4-CLASS REMAP (front-only)")
    print("="*60)

    print(f"\n  Original 7-class -> 4-class:")
    remap_to_4class(SRC_DATASET, FOURCLASS_DIR, "frontonly")

    print(f"\n  Augmented 7-class -> 4-class:")
    remap_to_4class(AUG_DATASET, FOURCLASS_AUG_DIR, "frontonly_aug")

    print("  Done!")


if __name__ == "__main__":
    do_augmentation()
    do_4class()
    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"  7-class:         {SRC_DATASET}")
    print(f"  7-class aug:     {AUG_DATASET}")
    print(f"  4-class:         {FOURCLASS_DIR}")
    print(f"  4-class aug:     {FOURCLASS_AUG_DIR}")
    print("="*60)
