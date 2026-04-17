"""
Prepare RAF-DB Dataset (from Kaggle shuvoalok/raf-db-dataset)
==============================================================
Downloads RAF-DB (15,339 images = full basic-emotion public release) from Kaggle.
Structure: DATASET/train/{1-7}/*.jpg + DATASET/test/{1-7}/*.jpg

RAF-DB label mapping (from original paper):
  1 = Surprise, 2 = Fear, 3 = Disgust, 4 = Happiness,
  5 = Sadness,  6 = Anger, 7 = Neutral

Generates:
  data/benchmark/rafdb_7class/   (train/test split from official)
  data/benchmark/rafdb_4class/
Each contains:
  X_train_images.npy, X_train_landmarks.npy, y_train.npy
  X_test_images.npy,  X_test_landmarks.npy,  y_test.npy

Usage:
    python scripts/prepare_rafdb.py
"""
import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RAW_DIR = BENCHMARK_DIR / "rafdb_raw"
IMG_SIZE = 224
KAGGLE_SLUG = "shuvoalok/raf-db-dataset"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]

# RAF-DB folder (1-7) -> our 7-class index
#   1=Surprise -> surprised(6), 2=Fear -> fearful(4), 3=Disgust -> disgusted(5),
#   4=Happy -> happy(1), 5=Sad -> sad(2), 6=Angry -> angry(3), 7=Neutral -> neutral(0)
RAFDB_TO_7 = {1: 6, 2: 4, 3: 5, 4: 1, 5: 2, 6: 3, 7: 0}
REMAP_4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}

LANDMARKS_68_MAP = [
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91, 78,
    82, 13, 312, 308, 317, 14, 87,
]


def init_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = str(PROJECT_ROOT / "tools" / "face_landmarker_v2_with_blendshapes.task")
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(landmarker, image_rgb):
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    face_lm = result.face_landmarks[0]
    coords = []
    for idx in LANDMARKS_68_MAP:
        if idx < len(face_lm):
            coords.append(face_lm[idx].x)
            coords.append(face_lm[idx].y)
        else:
            coords.extend([0.0, 0.0])
    return np.array(coords, dtype=np.float32)


def load_and_resize(path, target_size=224):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (target_size, target_size))


def download_rafdb():
    """Download + unzip RAF-DB from Kaggle."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_root = RAW_DIR / "DATASET"

    if dataset_root.exists() and any(dataset_root.iterdir()):
        print(f"  RAF-DB already extracted at: {dataset_root}")
        return dataset_root

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"  Downloading {KAGGLE_SLUG} ...")
    api.dataset_download_files(KAGGLE_SLUG, path=str(RAW_DIR), unzip=True, quiet=False)

    # Check for nested zip
    zips = list(RAW_DIR.glob("*.zip"))
    for z in zips:
        print(f"  Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()

    if not dataset_root.exists():
        raise RuntimeError(f"Expected {dataset_root} after extraction. "
                           f"Contents: {list(RAW_DIR.iterdir())}")
    return dataset_root


def collect_split(split_dir):
    """Return list of (path, rafdb_folder_label) for one split."""
    samples = []
    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            cls = int(folder.name)
        except ValueError:
            continue
        if cls not in RAFDB_TO_7:
            continue
        for img_file in sorted(folder.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((img_file, cls))
    return samples


def build_arrays(samples, landmarker):
    n = len(samples)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    landmarks = np.zeros((n, 136), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    skipped_load = skipped_face = 0

    for i, (path, cls) in enumerate(samples):
        img = load_and_resize(path)
        if img is None:
            valid[i] = False
            skipped_load += 1
            continue
        lm = extract_landmarks(landmarker, img)
        if lm is None:
            valid[i] = False
            skipped_face += 1
            continue
        images[i] = img.astype(np.float32) / 255.0
        landmarks[i] = lm
        labels[i] = RAFDB_TO_7[cls]
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    {i + 1}/{n}  (skipped load={skipped_load}, no-face={skipped_face})")

    return images[valid], landmarks[valid], labels[valid]


def save_split(out_dir, images_tr, lm_tr, y_tr, images_te, lm_te, y_te, emotions):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train_images.npy", images_tr)
    np.save(out_dir / "X_train_landmarks.npy", lm_tr)
    np.save(out_dir / "y_train.npy", y_tr)
    np.save(out_dir / "X_test_images.npy", images_te)
    np.save(out_dir / "X_test_landmarks.npy", lm_te)
    np.save(out_dir / "y_test.npy", y_te)

    counts_tr = Counter(y_tr.tolist())
    counts_te = Counter(y_te.tolist())
    info = {
        "dataset": "rafdb",
        "source": f"kaggle:{KAGGLE_SLUG}",
        "num_classes": len(emotions),
        "emotions": emotions,
        "train_samples": int(len(y_tr)),
        "test_samples": int(len(y_te)),
        "train_distribution": {emotions[k]: int(v) for k, v in sorted(counts_tr.items())},
        "test_distribution": {emotions[k]: int(v) for k, v in sorted(counts_te.items())},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump({emo: i for i, emo in enumerate(emotions)}, f, indent=2)

    print(f"\n  Saved: {out_dir}")
    print(f"  Train: {len(y_tr)}  Test: {len(y_te)}")
    for emo in emotions:
        idx = emotions.index(emo)
        print(f"    {emo:>10s}: train={counts_tr.get(idx, 0):>5d}  test={counts_te.get(idx, 0):>5d}")


def main():
    print(f"\n{'='*60}")
    print("  PREPARE RAF-DB")
    print(f"{'='*60}")

    dataset_root = download_rafdb()
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise RuntimeError(f"Expected train/ and test/ under {dataset_root}. "
                           f"Got: {list(dataset_root.iterdir())}")

    samples_tr = collect_split(train_dir)
    samples_te = collect_split(test_dir)
    print(f"  Train samples collected: {len(samples_tr)}")
    print(f"  Test  samples collected: {len(samples_te)}")

    landmarker = init_landmarker()

    print(f"\n  Building TRAIN arrays ...")
    img_tr, lm_tr, y_tr = build_arrays(samples_tr, landmarker)
    print(f"\n  Building TEST arrays ...")
    img_te, lm_te, y_te = build_arrays(samples_te, landmarker)

    # 7-class
    save_split(BENCHMARK_DIR / "rafdb_7class",
               img_tr, lm_tr, y_tr, img_te, lm_te, y_te, EMOTIONS_7)

    # 4-class remap
    y_tr_4 = np.array([REMAP_4[int(v)] for v in y_tr], dtype=np.int64)
    y_te_4 = np.array([REMAP_4[int(v)] for v in y_te], dtype=np.int64)
    save_split(BENCHMARK_DIR / "rafdb_4class",
               img_tr, lm_tr, y_tr_4, img_te, lm_te, y_te_4, EMOTIONS_4)

    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == "__main__":
    main()
