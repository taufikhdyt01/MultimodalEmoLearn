"""
Prepare KDEF Dataset (from official KDEF_and_AKDEF.zip at kdef.se)
====================================================================
Assumes user has placed KDEF_and_AKDEF.zip at project root.

KDEF filename pattern: [Series][Gender][SubjectID][Emotion][Angle].JPG
  Series:  A or B (each subject photographed twice)
  Gender:  F or M
  Subject: 01-35 per gender -> 70 unique subjects total
  Emotion: AF=afraid(fearful), AN=angry, DI=disgust, HA=happy,
           NE=neutral,         SA=sad,   SU=surprise
  Angle:   FL=full-left, HL=half-left, S=straight, HR=half-right, FR=full-right

Total: 70 subjects * 2 series * 7 emotions * 5 angles = 4900 images.

Generates (subject-wise 80/10/10 split; 70 subjects -> 56/7/7):
  data/benchmark/kdef_7class/
  data/benchmark/kdef_4class/
Each contains:
  X_train_images.npy, X_train_landmarks.npy, y_train.npy, subjects_train.npy
  X_val_images.npy,   X_val_landmarks.npy,   y_val.npy,   subjects_val.npy
  X_test_images.npy,  X_test_landmarks.npy,  y_test.npy,  subjects_test.npy

Usage:
    python scripts/prepare_kdef.py
    python scripts/prepare_kdef.py --angle S        # straight only (980 imgs)
"""
import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ZIP_PATH = PROJECT_ROOT / "KDEF_and_AKDEF.zip"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
IMG_SIZE = 224
SEED = 42

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]

KDEF_EMOTION_TO_7 = {
    "NE": 0,  # neutral
    "HA": 1,  # happy
    "SA": 2,  # sad
    "AN": 3,  # angry
    "AF": 4,  # fearful (afraid)
    "DI": 5,  # disgusted
    "SU": 6,  # surprised
}
REMAP_4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}
ALL_ANGLES = {"FL", "HL", "S", "HR", "FR"}

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


def decode_img_bytes(raw_bytes, target_size=224):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (target_size, target_size))


def parse_kdef_name(filename):
    """Parse filename like 'AF01ANS.JPG' or 'BM35SAFL.JPG'.

    Returns (subject_id, emotion_code, angle) or None if invalid.
    Subject ID = first 4 chars (e.g. 'AF01', 'BM35').
    """
    stem = Path(filename).stem.upper()
    if len(stem) < 7:
        return None
    # Series+Gender+ID = first 4 chars (AF01, BM35 etc)
    subject_id = stem[:4]
    emotion = stem[4:6]
    angle = stem[6:]  # remaining: S, FL, FR, HL, HR
    if emotion not in KDEF_EMOTION_TO_7:
        return None
    if angle not in ALL_ANGLES:
        return None
    return subject_id, emotion, angle


def collect_samples(zip_path, allowed_angles=None):
    """Return list of (zip_arcname, subject, subject_gender_id, emotion_code, angle)."""
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("KDEF_and_AKDEF/KDEF/"):
                continue
            if not name.lower().endswith(".jpg"):
                continue
            parsed = parse_kdef_name(name.split("/")[-1])
            if parsed is None:
                continue
            subject_id, emotion, angle = parsed
            if allowed_angles and angle not in allowed_angles:
                continue
            # Subject for LOSO/split = gender+id, ignoring series (A/B = same person twice)
            gender_id = subject_id[1:]  # 'F01' or 'M35' (without series letter)
            samples.append((name, subject_id, gender_id, emotion, angle))
    return samples


def subject_split(unique_subjects, seed=SEED, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.RandomState(seed)
    subs = sorted(unique_subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_va]), set(subs[n_tr + n_va:])


def build_arrays(samples, zip_path, landmarker):
    n = len(samples)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    landmarks = np.zeros((n, 136), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    subjects = []
    valid = np.ones(n, dtype=bool)
    skip_load = skip_face = 0

    with zipfile.ZipFile(zip_path) as zf:
        for i, (arcname, _series_sub, gender_id, emotion, _angle) in enumerate(samples):
            raw = zf.read(arcname)
            img = decode_img_bytes(raw)
            if img is None:
                valid[i] = False
                skip_load += 1
                continue
            lm = extract_landmarks(landmarker, img)
            if lm is None:
                valid[i] = False
                skip_face += 1
                continue
            images[i] = img.astype(np.float32) / 255.0
            landmarks[i] = lm
            labels[i] = KDEF_EMOTION_TO_7[emotion]
            subjects.append(gender_id)

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"    {i + 1}/{n}  (skipped load={skip_load}, no-face={skip_face})")

    images = images[valid]
    landmarks = landmarks[valid]
    labels = labels[valid]
    subjects = np.array(subjects)
    return images, landmarks, labels, subjects


def save_split(out_dir, split_data, emotions, source_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (imgs, lms, ys, subs) in split_data.items():
        np.save(out_dir / f"X_{split_name}_images.npy", imgs)
        np.save(out_dir / f"X_{split_name}_landmarks.npy", lms)
        np.save(out_dir / f"y_{split_name}.npy", ys)
        np.save(out_dir / f"subjects_{split_name}.npy", subs)

    info = {
        **source_info,
        "num_classes": len(emotions),
        "emotions": emotions,
        "splits": {},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
    }
    for split_name, (imgs, lms, ys, subs) in split_data.items():
        counts = Counter(ys.tolist())
        info["splits"][split_name] = {
            "samples": int(len(ys)),
            "subjects": int(len(set(subs.tolist()))),
            "distribution": {emotions[k]: int(v) for k, v in sorted(counts.items())},
        }

    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump({emo: i for i, emo in enumerate(emotions)}, f, indent=2)

    print(f"\n  Saved: {out_dir}")
    for split_name, d in info["splits"].items():
        print(f"    {split_name:>5s}: {d['samples']:>5d} samples, {d['subjects']} subjects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", default="all",
                    choices=["all", "S", "front3"],
                    help="all=5 angles (4900), S=straight only (980), front3=S+HL+HR (2940)")
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Expected {ZIP_PATH}. Download from https://kdef.se/ first.")

    if args.angle == "all":
        allowed = ALL_ANGLES
        suffix = ""
    elif args.angle == "S":
        allowed = {"S"}
        suffix = "_frontonly"
    else:  # front3
        allowed = {"S", "HL", "HR"}
        suffix = "_front3"

    print(f"\n{'='*60}")
    print(f"  PREPARE KDEF  (angles={sorted(allowed)})")
    print(f"{'='*60}")

    samples = collect_samples(ZIP_PATH, allowed)
    print(f"  Collected: {len(samples)} images")

    gender_ids = sorted({s[2] for s in samples})
    print(f"  Unique subjects (gender+id): {len(gender_ids)}")

    train_subs, val_subs, test_subs = subject_split(gender_ids)
    print(f"  Split  -> train:{len(train_subs)}  val:{len(val_subs)}  test:{len(test_subs)}")

    samples_by_split = {"train": [], "val": [], "test": []}
    for s in samples:
        gid = s[2]
        if gid in train_subs:
            samples_by_split["train"].append(s)
        elif gid in val_subs:
            samples_by_split["val"].append(s)
        else:
            samples_by_split["test"].append(s)

    for name, lst in samples_by_split.items():
        print(f"    {name:>5s}: {len(lst)} samples")

    landmarker = init_landmarker()
    built = {}
    for split_name in ["train", "val", "test"]:
        print(f"\n  Building {split_name} arrays ...")
        imgs, lms, ys, subs = build_arrays(samples_by_split[split_name], ZIP_PATH, landmarker)
        built[split_name] = (imgs, lms, ys, subs)

    # 7-class
    src = {"dataset": "kdef",
           "source": "official:kdef.se (KDEF_and_AKDEF.zip)",
           "angle_filter": sorted(allowed)}
    save_split(BENCHMARK_DIR / f"kdef_7class{suffix}", built, EMOTIONS_7, src)

    # 4-class remap
    built_4 = {}
    for split_name, (imgs, lms, ys, subs) in built.items():
        ys4 = np.array([REMAP_4[int(v)] for v in ys], dtype=np.int64)
        built_4[split_name] = (imgs, lms, ys4, subs)
    save_split(BENCHMARK_DIR / f"kdef_4class{suffix}", built_4, EMOTIONS_4, src)

    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == "__main__":
    main()
