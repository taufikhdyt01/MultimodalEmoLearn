"""
Augmentasi Data untuk Kelas Minoritas
=======================================
Augmentasi image dan landmark untuk kelas dengan sample < threshold.
Menyimpan dataset baru (augmented) terpisah dari dataset original.

Teknik augmentasi:
- Horizontal flip
- Rotasi kecil (-15 s/d +15 derajat)
- Brightness adjustment (+/- 20%)
- Kombinasi flip + rotasi

Usage:
    python src/preprocessing/augment_minority.py
    python src/preprocessing/augment_minority.py --target-min 200
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import Counter

# ============== KONFIGURASI ==============
DATASET_DIR = Path("data/dataset")
OUTPUT_DIR = Path("data/dataset_augmented")
IMG_SIZE = 224
TARGET_MIN = 150  # Target minimum sample per kelas
RANDOM_SEED = 42
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
# ==========================================


def augment_image(img, technique, rng):
    """Apply satu teknik augmentasi ke image (224x224x3, float32 0-1)."""
    h, w = img.shape[:2]

    if technique == "hflip":
        aug_img = np.fliplr(img).copy()

    elif technique == "rotate_pos":
        angle = rng.uniform(5, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    elif technique == "rotate_neg":
        angle = rng.uniform(-15, -5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    elif technique == "bright_up":
        factor = rng.uniform(1.05, 1.20)
        aug_img = np.clip(img * factor, 0, 1).astype(np.float32)

    elif technique == "bright_down":
        factor = rng.uniform(0.80, 0.95)
        aug_img = np.clip(img * factor, 0, 1).astype(np.float32)

    elif technique == "hflip_rotate_pos":
        aug_img = np.fliplr(img).copy()
        angle = rng.uniform(5, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    elif technique == "hflip_rotate_neg":
        aug_img = np.fliplr(img).copy()
        angle = rng.uniform(-15, -5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    elif technique == "hflip_bright_up":
        aug_img = np.fliplr(img).copy()
        factor = rng.uniform(1.05, 1.20)
        aug_img = np.clip(aug_img * factor, 0, 1).astype(np.float32)

    elif technique == "hflip_bright_down":
        aug_img = np.fliplr(img).copy()
        factor = rng.uniform(0.80, 0.95)
        aug_img = np.clip(aug_img * factor, 0, 1).astype(np.float32)

    else:
        aug_img = img.copy()

    return aug_img


def augment_landmark(landmark, technique, rng):
    """Apply augmentasi ke landmark (136,) float32.
    Landmark normalized [0,1] relative to crop box.
    """
    lm = landmark.copy().reshape(68, 2)

    if "hflip" in technique:
        lm[:, 0] = 1.0 - lm[:, 0]

    if "rotate_pos" in technique:
        angle = rng.uniform(5, 15) * np.pi / 180
        cx, cy = 0.5, 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = lm[:, 0] - cx
        y = lm[:, 1] - cy
        lm[:, 0] = x * cos_a - y * sin_a + cx
        lm[:, 1] = x * sin_a + y * cos_a + cy

    elif "rotate_neg" in technique:
        angle = rng.uniform(-15, -5) * np.pi / 180
        cx, cy = 0.5, 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = lm[:, 0] - cx
        y = lm[:, 1] - cy
        lm[:, 0] = x * cos_a - y * sin_a + cx
        lm[:, 1] = x * sin_a + y * cos_a + cy

    # Brightness tidak mempengaruhi landmark
    return lm.flatten().astype(np.float32)


TECHNIQUES = [
    "hflip",
    "rotate_pos",
    "rotate_neg",
    "bright_up",
    "bright_down",
    "hflip_rotate_pos",
    "hflip_rotate_neg",
    "hflip_bright_up",
    "hflip_bright_down",
]


def main():
    parser = argparse.ArgumentParser(description="Augment Minority Classes")
    parser.add_argument("--target-min", type=int, default=TARGET_MIN,
                        help=f"Target minimum sample per kelas (default: {TARGET_MIN})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("AUGMENTASI KELAS MINORITAS")
    print(f"  Target minimum per kelas: {args.target_min}")
    print("=" * 60)

    # 1. Load original training data
    print("\n[1/4] Loading training data...")
    X_train_img = np.load(DATASET_DIR / "X_train_images.npy")
    X_train_lm = np.load(DATASET_DIR / "X_train_landmarks.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")

    counts = Counter(y_train.tolist())
    print(f"  Original: {len(y_train)} samples")
    for i, emo in enumerate(EMOTIONS):
        print(f"    {emo:>10s}: {counts.get(i, 0)}")

    # 2. Identifikasi kelas yang perlu augmentasi
    print(f"\n[2/4] Identifikasi kelas minoritas (< {args.target_min})...")
    classes_to_augment = {}
    for i, emo in enumerate(EMOTIONS):
        count = counts.get(i, 0)
        if count < args.target_min:
            needed = args.target_min - count
            classes_to_augment[i] = needed
            print(f"    {emo:>10s}: {count} -> butuh +{needed} augmentasi")

    if not classes_to_augment:
        print("  Semua kelas sudah >= target. Tidak perlu augmentasi.")
        return

    # 3. Generate augmented samples
    print(f"\n[3/4] Generating augmented samples...")
    aug_images = []
    aug_landmarks = []
    aug_labels = []

    for class_idx, n_needed in classes_to_augment.items():
        # Ambil semua sample dari kelas ini
        mask = y_train == class_idx
        class_images = X_train_img[mask]
        class_landmarks = X_train_lm[mask]
        n_original = len(class_images)

        print(f"    {EMOTIONS[class_idx]}: generating {n_needed} from {n_original} originals...")

        generated = 0
        while generated < n_needed:
            # Pilih random sample dari kelas ini
            idx = rng.randint(0, n_original)
            # Pilih random teknik
            tech = TECHNIQUES[rng.randint(0, len(TECHNIQUES))]

            aug_img = augment_image(class_images[idx], tech, rng)
            aug_lm = augment_landmark(class_landmarks[idx], tech, rng)

            aug_images.append(aug_img)
            aug_landmarks.append(aug_lm)
            aug_labels.append(class_idx)
            generated += 1

    aug_images = np.array(aug_images, dtype=np.float32)
    aug_landmarks = np.array(aug_landmarks, dtype=np.float32)
    aug_labels = np.array(aug_labels, dtype=np.int32)

    print(f"    Total augmented: {len(aug_labels)}")

    # 4. Gabung dan simpan
    print(f"\n[4/4] Menggabung dan menyimpan...")
    X_train_img_aug = np.concatenate([X_train_img, aug_images])
    X_train_lm_aug = np.concatenate([X_train_lm, aug_landmarks])
    y_train_aug = np.concatenate([y_train, aug_labels])

    # Shuffle
    indices = rng.permutation(len(y_train_aug))
    X_train_img_aug = X_train_img_aug[indices]
    X_train_lm_aug = X_train_lm_aug[indices]
    y_train_aug = y_train_aug[indices]

    # Simpan augmented train data
    np.save(OUTPUT_DIR / "X_train_images.npy", X_train_img_aug)
    np.save(OUTPUT_DIR / "X_train_landmarks.npy", X_train_lm_aug)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_aug)

    # Copy val dan test tanpa perubahan (TIDAK boleh augmentasi val/test)
    for fname in ["X_val_images.npy", "X_val_landmarks.npy", "y_val.npy",
                   "X_test_images.npy", "X_test_landmarks.npy", "y_test.npy",
                   "label_map.json", "dataset_info.json"]:
        src = DATASET_DIR / fname
        dst = OUTPUT_DIR / fname
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    # Update class weights untuk augmented data
    new_counts = Counter(y_train_aug.tolist())
    beta = 0.999
    weights = []
    for i in range(7):
        eff = (1 - beta ** new_counts[i]) / (1 - beta)
        weights.append(1.0 / eff)
    min_w = min(weights)
    weights = [round(w / min_w, 4) for w in weights]

    weight_info = {
        "emotions": EMOTIONS,
        "train_counts_original": {EMOTIONS[i]: counts.get(i, 0) for i in range(7)},
        "train_counts_augmented": {EMOTIONS[i]: new_counts[i] for i in range(7)},
        "augmented_added": {EMOTIONS[i]: classes_to_augment.get(i, 0) for i in range(7)},
        "effective_number_weights": {EMOTIONS[i]: weights[i] for i in range(7)},
        "weights_array": weights,
        "method": "Class-Balanced Loss (Cui et al., 2019), beta=0.999",
        "augmentation_techniques": TECHNIQUES,
        "target_min_per_class": args.target_min,
    }
    with open(OUTPUT_DIR / "class_weights.json", "w") as f:
        json.dump(weight_info, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SELESAI!")
    print(f"  Original train: {len(y_train)} samples")
    print(f"  Augmented train: {len(y_train_aug)} samples (+{len(aug_labels)})")
    print(f"")
    print(f"  Distribusi setelah augmentasi:")
    for i, emo in enumerate(EMOTIONS):
        orig = counts.get(i, 0)
        aug = new_counts[i]
        added = aug - orig
        w = weights[i]
        print(f"    {emo:>10s}: {orig:>5d} -> {aug:>5d} (+{added:>3d})  weight={w:.4f}")
    print(f"")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Val/Test: TIDAK diaugmentasi (copied as-is)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
