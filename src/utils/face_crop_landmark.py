"""
Face Crop & Landmark Extraction using MediaPipe
=================================================
Deteksi wajah, crop, resize ke 224x224, dan ekstrak 68 landmark dari frame.
Mendukung data lama (20 Sample) dan data baru (17 User, front+side).

Usage:
    python src/utils/face_crop_landmark.py
    python src/utils/face_crop_landmark.py --workers 6
    python src/utils/face_crop_landmark.py --source new --users 200 201
    python src/utils/face_crop_landmark.py --source old --samples 1 2 3
    python src/utils/face_crop_landmark.py --dry-run
"""

import os
import csv
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============== KONFIGURASI ==============
FACE_SIZE = 224                    # Output face size (224x224)
FACE_PADDING = 0.3                 # 30% padding di sekitar wajah
MIN_DETECTION_CONFIDENCE = 0.3     # Threshold deteksi (rendah supaya side view terdeteksi)

# Path ke model MediaPipe
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
MODEL_PATH = str(_PROJECT_ROOT / "tools" / "face_landmarker_v2_with_blendshapes.task")

# MediaPipe 478 -> 68 landmark mapping (standar dlib 68-point)
LANDMARK_68_INDICES = [
    # Jaw (17 points)
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    # Right eyebrow (5 points)
    70, 63, 105, 66, 107,
    # Left eyebrow (5 points)
    336, 296, 334, 293, 300,
    # Nose (9 points)
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    # Right eye (6 points)
    33, 160, 158, 133, 153, 144,
    # Left eye (6 points)
    362, 385, 387, 263, 373, 380,
    # Outer lip (12 points)
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91,
    # Inner lip (8 points)
    78, 82, 13, 312, 308, 317, 14, 87,
]
assert len(LANDMARK_68_INDICES) == 68, f"Expected 68 landmarks, got {len(LANDMARK_68_INDICES)}"

# Path data
OLD_DATA_DIR = Path("data/processed")
NEW_DATA_DIR = Path("data/processed_new")
OUTPUT_DIR = Path("data/final")
# ==========================================


def create_landmarker():
    """Buat FaceLandmarker instance (harus dipanggil per-process)."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_batch(task_batch):
    """Proses batch frames dalam satu worker (reuse landmarker)."""
    landmarker = create_landmarker()
    results = []

    for frame_path, output_face_path, output_landmark_path in task_batch:
        result = _process_one(landmarker, frame_path, output_face_path, output_landmark_path)
        results.append(result)

    landmarker.close()
    return results


def _process_one(landmarker, frame_path, output_face_path, output_landmark_path):
    """Proses satu frame: detect face, crop, resize, extract landmarks."""
    if os.path.exists(output_face_path) and os.path.exists(output_landmark_path):
        return frame_path, "skipped", None, output_landmark_path

    img = cv2.imread(str(frame_path))
    if img is None:
        return frame_path, "error: cannot read image", None, output_landmark_path

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection = landmarker.detect(mp_image)

    if not detection.face_landmarks:
        return frame_path, "no_face", None, output_landmark_path

    face_lms = detection.face_landmarks[0]

    # --- Extract 68 landmarks (normalized 0-1) ---
    landmarks_68 = []
    for idx in LANDMARK_68_INDICES:
        lm = face_lms[idx]
        landmarks_68.append((lm.x, lm.y))

    # --- Hitung bounding box wajah dari semua landmark ---
    all_x = [lm.x * w for lm in face_lms]
    all_y = [lm.y * h for lm in face_lms]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Tambah padding
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_w = face_w * FACE_PADDING
    pad_h = face_h * FACE_PADDING

    x_min = max(0, int(x_min - pad_w))
    y_min = max(0, int(y_min - pad_h))
    x_max = min(w, int(x_max + pad_w))
    y_max = min(h, int(y_max + pad_h))

    # --- Crop dan resize ---
    face_crop = img[y_min:y_max, x_min:x_max]
    if face_crop.size == 0:
        return frame_path, "error: empty crop", None, output_landmark_path

    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

    # --- Simpan cropped face ---
    os.makedirs(os.path.dirname(output_face_path), exist_ok=True)
    cv2.imwrite(str(output_face_path), face_resized)

    # --- Normalize landmark relative to crop box ---
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    landmarks_normalized = []
    for (lx, ly) in landmarks_68:
        px = lx * w
        py = ly * h
        nx = (px - x_min) / crop_w if crop_w > 0 else 0
        ny = (py - y_min) / crop_h if crop_h > 0 else 0
        landmarks_normalized.append((round(nx, 6), round(ny, 6)))

    # --- Simpan landmark CSV ---
    os.makedirs(os.path.dirname(output_landmark_path), exist_ok=True)
    with open(output_landmark_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "x", "y"])
        for i, (x, y) in enumerate(landmarks_normalized):
            writer.writerow([i, x, y])

    return frame_path, "ok", landmarks_normalized, output_landmark_path


def get_sample_to_userid_map():
    """Baca mapping Sample -> user_id dari cleaned_data.xlsx."""
    import openpyxl
    mapping = {}
    if not OLD_DATA_DIR.exists():
        return mapping

    for sample_dir in sorted(OLD_DATA_DIR.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("Sample"):
            continue
        for xlsx in sample_dir.rglob("cleaned_data.xlsx"):
            wb = openpyxl.load_workbook(xlsx, read_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, max_row=2, values_only=True):
                uid = str(row[1])  # kolom user_id
                mapping[sample_dir.name] = uid
            wb.close()
            break
    return mapping


def discover_old_frames():
    """Temukan semua frame dari data lama (20 Sample), output pakai user_id."""
    tasks = []
    if not OLD_DATA_DIR.exists():
        return tasks

    sample_uid_map = get_sample_to_userid_map()
    print(f"  Mapping Sample -> user_id: {sample_uid_map}")

    for sample_dir in sorted(OLD_DATA_DIR.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("Sample"):
            continue

        uid = sample_uid_map.get(sample_dir.name)
        if not uid:
            print(f"  [WARN] Tidak ditemukan user_id untuk {sample_dir.name}, skip")
            continue

        # Kumpulkan semua frame dari semua sub-task (tanpa sub-folder tantangan)
        for frames_dir in sample_dir.rglob("cleaned_frames"):
            for frame_file in sorted(frames_dir.glob("*.jpg")):
                stem = frame_file.stem
                out_face = OUTPUT_DIR / "old" / uid / "front" / "faces" / f"{stem}.jpg"
                out_lm = OUTPUT_DIR / "old" / uid / "front" / "landmarks" / f"{stem}.csv"
                tasks.append((str(frame_file), str(out_face), str(out_lm)))

    return tasks


def discover_new_frames():
    """Temukan semua frame dari data baru (17 User, front+side)."""
    tasks = []
    if not NEW_DATA_DIR.exists():
        return tasks

    for user_dir in sorted(NEW_DATA_DIR.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.isdigit():
            continue
        uid = user_dir.name

        for angle in ["front", "side"]:
            angle_dir = user_dir / angle
            if not angle_dir.exists():
                continue

            for frame_file in sorted(angle_dir.glob("*.jpg")):
                stem = frame_file.stem
                out_face = OUTPUT_DIR / "new" / uid / angle / "faces" / f"{stem}.jpg"
                out_lm = OUTPUT_DIR / "new" / uid / angle / "landmarks" / f"{stem}.csv"
                tasks.append((str(frame_file), str(out_face), str(out_lm)))

    return tasks


def chunk_list(lst, n):
    """Bagi list menjadi n chunks yang kurang-lebih sama besar."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Face Crop & Landmark Extraction")
    parser.add_argument("--workers", type=int, default=4,
                        help="Jumlah parallel workers (default: 4)")
    parser.add_argument("--source", choices=["all", "old", "new"], default="all",
                        help="Sumber data: all, old (20 Sample), new (17 User)")
    parser.add_argument("--users", nargs="*", type=str, default=None,
                        help="Filter user tertentu untuk data new")
    parser.add_argument("--samples", nargs="*", type=str, default=None,
                        help="Filter sample tertentu untuk data old (angka saja)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya tampilkan rencana")
    args = parser.parse_args()

    print("=" * 60)
    print("FACE CROP & LANDMARK EXTRACTION (MediaPipe)")
    print("=" * 60)

    # 1. Discover frames
    all_tasks = []

    if args.source in ("all", "old"):
        print(f"\n[1] Scanning data lama ({OLD_DATA_DIR})...")
        old_tasks = discover_old_frames()
        if args.samples:
            filter_names = {f"Sample {s}" for s in args.samples}
            old_tasks = [t for t in old_tasks if any(fn in t[0] for fn in filter_names)]
        print(f"  Ditemukan {len(old_tasks)} frames dari data lama")
        all_tasks.extend(old_tasks)

    if args.source in ("all", "new"):
        print(f"\n[2] Scanning data baru ({NEW_DATA_DIR})...")
        new_tasks = discover_new_frames()
        if args.users:
            new_tasks = [t for t in new_tasks
                         if any(f"/{u}/" in t[0].replace("\\", "/") for u in args.users)]
        print(f"  Ditemukan {len(new_tasks)} frames dari data baru")
        all_tasks.extend(new_tasks)

    print(f"\n  Total frames: {len(all_tasks)}")

    if args.dry_run:
        print("\n[DRY RUN]")
        for t in all_tasks[:5]:
            print(f"  {t[0]}")
            print(f"    -> face: {t[1]}")
            print(f"    -> lm:   {t[2]}")
        if len(all_tasks) > 5:
            print(f"  ... dan {len(all_tasks) - 5} lainnya")
        return

    if not all_tasks:
        print("Tidak ada frame untuk diproses.")
        return

    # 2. Process frames in batches (reuse landmarker per worker)
    num_workers = min(args.workers, len(all_tasks))
    batches = chunk_list(all_tasks, num_workers)

    print(f"\n[3] Memproses {len(all_tasks)} frames dengan {num_workers} workers...")

    completed = 0
    skipped = 0
    no_face = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            batch_results = future.result()
            for frame_path, status, landmarks, lm_path in batch_results:
                if status == "ok":
                    completed += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "no_face":
                    no_face += 1
                else:
                    errors += 1

            total_done = completed + skipped + no_face + errors
            pct = total_done / len(all_tasks) * 100
            print(f"  Progress: {total_done}/{len(all_tasks)} ({pct:.0f}%) "
                  f"- OK: {completed}, Skip: {skipped}, NoFace: {no_face}, Err: {errors}")

    # 3. Summary
    print(f"\n{'=' * 60}")
    print(f"SELESAI!")
    print(f"  Faces cropped: {completed}")
    print(f"  Skipped (sudah ada): {skipped}")
    print(f"  No face detected: {no_face}")
    print(f"  Errors: {errors}")
    print(f"  Detection rate: {completed}/{completed + no_face} "
          f"({completed / max(1, completed + no_face) * 100:.1f}%)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    if no_face > 0:
        print(f"\n[INFO] {no_face} frame tanpa wajah terdeteksi.")
        print("  Ini normal untuk frame yang blur, terlalu jauh, atau tertutup.")


if __name__ == "__main__":
    main()
