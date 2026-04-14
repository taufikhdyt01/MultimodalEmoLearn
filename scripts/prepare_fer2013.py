"""
Prepare FER2013 Dataset for Pre-Training
==========================================
Download dari Kaggle, convert ke numpy arrays.

FER2013: 35,887 grayscale 48x48 images, 7 emotions.
Digunakan untuk pre-train ResNet18 sebelum fine-tune ke dataset sendiri.

Prerequisite:
    Download fer2013.csv dari https://www.kaggle.com/datasets/msambare/fer2013
    Atau dataset folder format dari https://www.kaggle.com/datasets/msambare/fer2013
    Taruh di data/benchmark/fer2013/

Usage:
    python scripts/prepare_fer2013.py
"""
import os
import csv
import json
import numpy as np
import cv2
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FER2013_DIR = PROJECT_ROOT / "data" / "benchmark" / "fer2013"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark" / "fer2013_prepared"

IMG_SIZE = 224
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
# FER2013 label order: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
# Remap to our order: neutral=0, happy=1, sad=2, angry=3, fearful=4, disgusted=5, surprised=6
FER_TO_OUR = {0: 3, 1: 5, 2: 4, 3: 1, 4: 2, 5: 6, 6: 0}
OUR_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def load_from_csv(csv_path):
    """Load FER2013 from CSV format (pixels column)."""
    samples = {"train": [], "val": [], "test": []}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = int(row["emotion"])
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            usage = row.get("Usage", "Training")

            if usage == "Training":
                samples["train"].append((pixels, emotion))
            elif usage == "PublicTest":
                samples["val"].append((pixels, emotion))
            else:
                samples["test"].append((pixels, emotion))

    return samples


def load_from_folders(base_dir):
    """Load FER2013 from folder format (train/test with subfolders per emotion)."""
    emotion_map = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "sad": 4, "surprise": 5, "neutral": 6,
    }

    samples = {"train": [], "val": [], "test": []}

    for split_name, split_dir in [("train", "train"), ("test", "test")]:
        split_path = base_dir / split_dir
        if not split_path.exists():
            continue
        for emo_folder in sorted(split_path.iterdir()):
            if not emo_folder.is_dir():
                continue
            emo_name = emo_folder.name.lower()
            if emo_name not in emotion_map:
                continue
            emo_idx = emotion_map[emo_name]
            for img_file in sorted(emo_folder.glob("*.png")) + sorted(emo_folder.glob("*.jpg")):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples[split_name].append((img, emo_idx))

    # Use 10% of train as val if no separate val
    if not samples["val"] and samples["train"]:
        rng = np.random.RandomState(42)
        rng.shuffle(samples["train"])
        n_val = int(len(samples["train"]) * 0.1)
        samples["val"] = samples["train"][:n_val]
        samples["train"] = samples["train"][n_val:]

    return samples


def process_and_save(samples, output_dir):
    """Resize to 224x224, convert grayscale to RGB, save as numpy arrays.
    Uses memory-mapped files to avoid OOM on large datasets."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_samples in samples.items():
        if not split_samples:
            continue

        n = len(split_samples)
        img_path = output_dir / f"X_{split_name}_images.npy"
        lbl_path = output_dir / f"y_{split_name}.npy"

        # Create memory-mapped array to avoid OOM
        images = np.lib.format.open_memmap(
            str(img_path), mode='w+', dtype=np.float32,
            shape=(n, IMG_SIZE, IMG_SIZE, 3))
        labels = np.zeros(n, dtype=np.int64)

        for i, (pixels, emo) in enumerate(split_samples):
            img = cv2.resize(pixels, (IMG_SIZE, IMG_SIZE))
            img_rgb = np.stack([img, img, img], axis=-1)
            images[i] = img_rgb.astype(np.float32) / 255.0
            labels[i] = FER_TO_OUR[emo]

            if (i + 1) % 5000 == 0 or (i + 1) == n:
                print(f"    {split_name}: {i + 1}/{n}")

        del images  # flush memmap
        np.save(lbl_path, labels)

        counts = Counter(labels.tolist())
        print(f"  {split_name}: {n} samples")
        for idx in sorted(counts.keys()):
            print(f"    {OUR_EMOTIONS[idx]:>10s}: {counts[idx]}")

    # Save metadata
    info = {
        "dataset": "FER2013",
        "num_classes": 7,
        "emotions": OUR_EMOTIONS,
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "original_size": 48,
        "note": "Grayscale -> RGB (3-channel repeat), resized 48->224",
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    label_map = {emo: i for i, emo in enumerate(OUR_EMOTIONS)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)


def main():
    print("=" * 60)
    print("PREPARE FER2013 FOR PRE-TRAINING")
    print("=" * 60)

    # Try CSV format first
    csv_path = FER2013_DIR / "fer2013.csv"
    if csv_path.exists():
        print(f"\nLoading from CSV: {csv_path}")
        samples = load_from_csv(csv_path)
    else:
        # Try folder format
        print(f"\nLoading from folders: {FER2013_DIR}")
        samples = load_from_folders(FER2013_DIR)

    total = sum(len(v) for v in samples.values())
    if total == 0:
        print("ERROR: No data found!")
        print(f"  Expected CSV at: {csv_path}")
        print(f"  Or folders at: {FER2013_DIR}/train/, {FER2013_DIR}/test/")
        return

    print(f"\nTotal: {total} samples")
    for k, v in samples.items():
        print(f"  {k}: {len(v)}")

    print(f"\nProcessing and saving to {OUTPUT_DIR}...")
    process_and_save(samples, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
