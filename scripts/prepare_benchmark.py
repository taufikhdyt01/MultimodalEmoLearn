"""
Prepare Benchmark Datasets (JAFFE & CK+)
==========================================
Preprocessing: resize 224x224, extract 68 landmarks, save numpy arrays.

Generates:
  data/benchmark/jaffe_7class/    → 7-class (213 samples)
  data/benchmark/jaffe_4class/    → 4-class (213 samples)
  data/benchmark/ckplus_7class/   → 7-class, drop contempt (636 samples)
  data/benchmark/ckplus_4class/   → 4-class, contempt → negative (654 samples)

Usage:
    python scripts/prepare_benchmark.py
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
IMG_SIZE = 224

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]

# Map folder names to standard emotion names
FOLDER_TO_EMOTION = {
    "Anger": "angry",
    "Disgust": "disgusted",
    "Fear": "fearful",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprised": "surprised",
    "Contempt": "contempt",  # CK+ only
}

EMOTION_TO_7CLASS = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fearful": 4, "disgusted": 5, "surprised": 6,
}

REMAP_4CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}
# Contempt → negative (index 3 in 4-class)
CONTEMPT_4CLASS = 3


def init_mediapipe():
    """Initialize MediaPipe FaceLandmarker (tasks API)."""
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


# MediaPipe 478 → 68 landmark mapping (from src/utils/face_crop_landmark.py)
LANDMARKS_68_MAP = [
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91, 78,
    82, 13, 312, 308, 317, 14, 87,
]


def extract_landmarks(landmarker, image_rgb):
    """Extract 68 landmarks from face image, return (136,) array or None."""
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    face_lm = result.face_landmarks[0]
    coords = []
    for idx in LANDMARKS_68_MAP:
        if idx < len(face_lm):
            coords.append(face_lm[idx].x)  # normalized 0-1
            coords.append(face_lm[idx].y)
        else:
            coords.extend([0.0, 0.0])
    return np.array(coords, dtype=np.float32)


def load_and_resize(path, target_size=224):
    """Load image (TIFF/PNG/JPG), convert to RGB, resize to target_size."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Handle grayscale (JAFFE is grayscale)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (target_size, target_size))
    return img


def collect_samples(dataset_dir, dataset_name, include_contempt=False):
    """Collect all samples from folder-based dataset."""
    samples = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        emotion = FOLDER_TO_EMOTION.get(folder.name)
        if emotion is None:
            print(f"  SKIP unknown folder: {folder.name}")
            continue
        if emotion == "contempt" and not include_contempt:
            print(f"  SKIP contempt ({len(list(folder.glob('*')))} files)")
            continue

        for img_file in sorted(folder.glob("*")):
            if img_file.suffix.lower() not in (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"):
                continue
            # Extract subject ID from filename
            if dataset_name == "jaffe":
                subject = img_file.stem[:2]  # e.g. "KA" from "KA.HA1.29.tiff"
            else:  # ck+
                subject = img_file.stem.split("_")[0]  # e.g. "S010" from "S010_006_00000015.png"

            samples.append({
                "path": str(img_file),
                "emotion": emotion,
                "subject": subject,
            })

    return samples


def build_arrays(samples, face_mesh, num_classes=7):
    """Build numpy arrays from samples."""
    n = len(samples)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    landmarks = np.zeros((n, 136), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    subjects = []
    valid_mask = np.ones(n, dtype=bool)

    for i, s in enumerate(samples):
        img = load_and_resize(s["path"])
        if img is None:
            print(f"    SKIP (load failed): {s['path']}")
            valid_mask[i] = False
            continue

        lm = extract_landmarks(face_mesh, img)
        if lm is None:
            print(f"    SKIP (no face): {s['path']}")
            valid_mask[i] = False
            continue

        images[i] = img.astype(np.float32) / 255.0
        landmarks[i] = lm

        emo = s["emotion"]
        if num_classes == 7:
            labels[i] = EMOTION_TO_7CLASS.get(emo, -1)
        else:  # 4-class
            if emo == "contempt":
                labels[i] = CONTEMPT_4CLASS
            elif emo in EMOTION_TO_7CLASS:
                labels[i] = REMAP_4CLASS[EMOTION_TO_7CLASS[emo]]

        subjects.append(s["subject"])

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    {i + 1}/{n}")

    # Filter valid
    images = images[valid_mask]
    landmarks = landmarks[valid_mask]
    labels = labels[valid_mask]
    subjects = [s for s, v in zip(subjects, valid_mask) if v]

    return images, landmarks, labels, np.array(subjects)


def save_dataset(output_dir, images, landmarks, labels, subjects, emotions_list, dataset_name):
    """Save dataset as numpy arrays + metadata."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir / "X_images.npy", images)
    np.save(output_dir / "X_landmarks.npy", landmarks)
    np.save(output_dir / "y_labels.npy", labels)
    np.save(output_dir / "subjects.npy", subjects)

    # Label map
    label_map = {emo: i for i, emo in enumerate(emotions_list)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Info
    counts = Counter(labels.tolist())
    info = {
        "dataset": dataset_name,
        "total_samples": len(labels),
        "num_classes": len(emotions_list),
        "emotions": emotions_list,
        "num_subjects": len(set(subjects)),
        "subjects": sorted(set(subjects.tolist())),
        "distribution": {emotions_list[k]: int(v) for k, v in sorted(counts.items())},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"  Total: {len(labels)} samples, {len(set(subjects))} subjects")
    for emo in emotions_list:
        idx = label_map[emo]
        print(f"    {emo:>10s}: {counts.get(idx, 0)}")


def main():
    face_mesh = init_mediapipe()

    datasets = [
        ("jaffe", BENCHMARK_DIR / "jaffe", False),
        ("ckplus", BENCHMARK_DIR / "ck+", False),     # 7-class (no contempt)
        ("ckplus", BENCHMARK_DIR / "ck+", True),       # with contempt (for 4-class)
    ]

    for dataset_name, dataset_dir, include_contempt in datasets:
        suffix = "_with_contempt" if include_contempt else ""
        print(f"\n{'='*60}")
        print(f"  {dataset_name.upper()}{suffix}")
        print(f"{'='*60}")

        samples = collect_samples(dataset_dir, dataset_name, include_contempt)
        print(f"  Collected: {len(samples)} samples")

        if not include_contempt:
            # 7-class
            print(f"\n  Building 7-class arrays...")
            images, landmarks, labels, subjects = build_arrays(samples, face_mesh, 7)
            save_dataset(
                BENCHMARK_DIR / f"{dataset_name}_7class",
                images, landmarks, labels, subjects, EMOTIONS_7, dataset_name)

            # 4-class (remap from 7-class, no contempt)
            print(f"\n  Building 4-class arrays (remap)...")
            labels_4 = np.array([REMAP_4CLASS[int(l)] for l in labels], dtype=np.int64)
            save_dataset(
                BENCHMARK_DIR / f"{dataset_name}_4class",
                images, landmarks, labels_4, subjects, EMOTIONS_4, dataset_name)
        else:
            # 4-class with contempt → negative
            print(f"\n  Building 4-class arrays (with contempt -> negative)...")
            images, landmarks, labels, subjects = build_arrays(samples, face_mesh, 4)
            save_dataset(
                BENCHMARK_DIR / f"{dataset_name}_4class_contempt",
                images, landmarks, labels, subjects, EMOTIONS_4, f"{dataset_name}+contempt")

    # landmarker doesn't need explicit close
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
