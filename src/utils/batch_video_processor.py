"""
Batch Video Processor for MultimodalEmoLearn
=============================================
Ekstraksi frame dari video berdasarkan timestamp emosi di CSV.
Mendukung multi-user, multi-angle (front/side), dan parallel processing.

Usage:
    python src/utils/batch_video_processor.py
    python src/utils/batch_video_processor.py --workers 8
    python src/utils/batch_video_processor.py --users 200 201 202
    python src/utils/batch_video_processor.py --dry-run
"""

import subprocess
import os
import csv
import json
import argparse
import sys
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ============== KONFIGURASI ==============
DATA_DIR = Path("data/new")
EMOTIONS_CSV = DATA_DIR / "emotions.csv"
OUTPUT_BASE = Path("data/processed_new")
ANGLES = ["front", "side"]  # sudut kamera yang akan diproses
FRAME_QUALITY = 2  # kualitas JPEG (2 = tinggi, 31 = rendah)

# Path ke ffmpeg/ffprobe (gunakan binary lokal jika ada)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # src/utils -> src -> project root
_TOOLS_DIR = _PROJECT_ROOT / "tools"
FFMPEG = str(_TOOLS_DIR / "ffmpeg.exe") if (_TOOLS_DIR / "ffmpeg.exe").exists() else "ffmpeg"
FFPROBE = str(_TOOLS_DIR / "ffprobe.exe") if (_TOOLS_DIR / "ffprobe.exe").exists() else "ffprobe"
# ==========================================


def parse_video_start_time(video_filename):
    """Ekstrak waktu mulai dari nama file video.
    Contoh: '2025-11-27 11-16-05.mp4' -> datetime(2025, 11, 27, 11, 16, 5)
    """
    stem = Path(video_filename).stem  # '2025-11-27 11-16-05'
    return datetime.strptime(stem, "%Y-%m-%d %H-%M-%S")


def get_video_duration(video_path):
    """Dapatkan durasi video dalam detik menggunakan ffprobe."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        print(f"  [WARN] Gagal baca durasi: {video_path}")
        return None


def find_video_for_timestamp(videos_info, emotion_dt):
    """Cari video yang mencakup timestamp emosi tertentu.
    videos_info: list of (video_path, start_datetime, end_datetime)
    """
    for vpath, vstart, vend in videos_info:
        if vstart <= emotion_dt <= vend:
            return vpath, vstart
    return None, None


def extract_single_frame(args):
    """Ekstrak satu frame dari video pada posisi seek tertentu.
    Menggunakan input seeking (-ss sebelum -i) untuk kecepatan maksimal.
    """
    video_path, seek_seconds, output_path = args

    if os.path.exists(output_path):
        return output_path, "skipped"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        FFMPEG,
        "-ss", f"{seek_seconds:.3f}",  # Input seeking (CEPAT)
        "-i", str(video_path),
        "-frames:v", "1",              # Hanya 1 frame
        "-q:v", str(FRAME_QUALITY),
        "-y",                           # Overwrite
        "-loglevel", "error",
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return output_path, "ok"
    else:
        return output_path, f"error: {result.stderr.decode().strip()}"


def load_emotions(csv_path, target_user_ids=None):
    """Load emotions dari CSV, grouped by user_id.
    Returns: dict { user_id: [ {timestamp, emotions...}, ... ] }
    """
    user_emotions = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            uid = row["user_id"].strip('"')
            if target_user_ids and uid not in target_user_ids:
                continue

            timestamp_str = row["timestamp"].strip('"')
            try:
                ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            emotion_data = {
                "id": row["id"].strip('"'),
                "timestamp": ts,
                "timestamp_str": timestamp_str,
                "neutral": row.get("neutral", "").strip('"'),
                "happy": row.get("happy", "").strip('"'),
                "sad": row.get("sad", "").strip('"'),
                "angry": row.get("angry", "").strip('"'),
                "fearful": row.get("fearful", "").strip('"'),
                "disgusted": row.get("disgusted", "").strip('"'),
                "surprised": row.get("surprised", "").strip('"'),
            }
            user_emotions[uid].append(emotion_data)

    # Sort by timestamp
    for uid in user_emotions:
        user_emotions[uid].sort(key=lambda x: x["timestamp"])

    return user_emotions


def discover_videos(user_dir, angle):
    """Temukan semua video untuk sudut tertentu dan hitung durasi.
    Returns: list of (video_path, start_datetime, end_datetime)
    """
    # Handle case-insensitive folder names (Screen vs screen)
    angle_dir = None
    for name in os.listdir(user_dir):
        if name.lower() == angle.lower():
            angle_dir = user_dir / name
            break

    if not angle_dir or not angle_dir.exists():
        return []

    videos = []
    for f in sorted(angle_dir.iterdir()):
        if f.suffix.lower() in (".mp4", ".mkv", ".avi", ".webm"):
            try:
                start_dt = parse_video_start_time(f.name)
            except ValueError:
                continue
            duration = get_video_duration(f)
            if duration is None:
                continue
            from datetime import timedelta
            end_dt = start_dt + timedelta(seconds=duration)
            videos.append((f, start_dt, end_dt))

    # Sort by start time
    videos.sort(key=lambda x: x[1])
    return videos


def process_user(user_id, emotions, data_dir, output_base, angles, dry_run=False):
    """Proses semua video untuk satu user. Returns list of extraction tasks."""
    user_dir = data_dir / str(user_id)
    if not user_dir.exists():
        print(f"  [SKIP] Folder user {user_id} tidak ditemukan: {user_dir}")
        return []

    tasks = []

    for angle in angles:
        videos_info = discover_videos(user_dir, angle)
        if not videos_info:
            print(f"  [SKIP] User {user_id}/{angle}: tidak ada video")
            continue

        matched = 0
        for emo in emotions:
            emo_dt = emo["timestamp"]
            video_path, video_start = find_video_for_timestamp(videos_info, emo_dt)

            if video_path is None:
                continue

            seek_seconds = (emo_dt - video_start).total_seconds()
            ts_str = emo_dt.strftime("%Y%m%d_%H%M%S")
            out_dir = output_base / str(user_id) / angle
            out_path = out_dir / f"frame_{ts_str}_emo{emo['id']}.jpg"

            tasks.append((str(video_path), seek_seconds, str(out_path)))
            matched += 1

        print(f"  User {user_id}/{angle}: {matched}/{len(emotions)} timestamps matched ke video")

    return tasks


def save_emotion_labels(user_emotions, output_base):
    """Simpan label emosi per user sebagai CSV untuk training nanti."""
    for uid, emotions in user_emotions.items():
        out_dir = output_base / str(uid)
        os.makedirs(out_dir, exist_ok=True)
        label_path = out_dir / "labels.csv"

        with open(label_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "emotion_id", "timestamp", "neutral", "happy", "sad",
                "angry", "fearful", "disgusted", "surprised"
            ])
            for emo in emotions:
                writer.writerow([
                    emo["id"], emo["timestamp_str"],
                    emo["neutral"], emo["happy"], emo["sad"],
                    emo["angry"], emo["fearful"], emo["disgusted"], emo["surprised"]
                ])


def main():
    parser = argparse.ArgumentParser(description="Batch Video Frame Extractor")
    parser.add_argument("--workers", type=int, default=4,
                        help="Jumlah parallel workers (default: 4)")
    parser.add_argument("--users", nargs="*", type=str, default=None,
                        help="Proses user tertentu saja (misal: --users 200 201)")
    parser.add_argument("--angles", nargs="*", type=str, default=ANGLES,
                        help="Sudut kamera (default: front side)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya tampilkan rencana tanpa ekstraksi")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help=f"Direktori data (default: {DATA_DIR})")
    parser.add_argument("--csv", type=str, default=str(EMOTIONS_CSV),
                        help=f"Path ke emotions CSV (default: {EMOTIONS_CSV})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_BASE),
                        help=f"Direktori output (default: {OUTPUT_BASE})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv)
    output_base = Path(args.output)

    print("=" * 60)
    print("BATCH VIDEO FRAME EXTRACTOR")
    print("=" * 60)

    # 1. Load emotions
    print(f"\n[1/4] Loading emotions dari {csv_path}...")
    target_uids = set(args.users) if args.users else None

    # Jika tidak ada user spesifik, deteksi dari folder
    if target_uids is None:
        target_uids = set()
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                target_uids.add(d.name)
        print(f"  Ditemukan {len(target_uids)} user folders: {sorted(target_uids, key=int)}")

    user_emotions = load_emotions(csv_path, target_uids)
    total_emotions = sum(len(v) for v in user_emotions.values())
    print(f"  Loaded {total_emotions} emotion records untuk {len(user_emotions)} users")

    # 2. Buat extraction tasks
    print(f"\n[2/4] Mapping timestamps ke video...")
    all_tasks = []
    for uid in sorted(user_emotions.keys(), key=int):
        tasks = process_user(
            uid, user_emotions[uid], data_dir, output_base, args.angles, args.dry_run
        )
        all_tasks.extend(tasks)

    print(f"\n  Total frame yang akan diekstrak: {len(all_tasks)}")

    if args.dry_run:
        print("\n[DRY RUN] Tidak ada frame yang diekstrak.")
        for task in all_tasks[:10]:
            print(f"  {task[0]} @ {task[1]:.1f}s -> {task[2]}")
        if len(all_tasks) > 10:
            print(f"  ... dan {len(all_tasks) - 10} lainnya")
        return

    if not all_tasks:
        print("\nTidak ada frame untuk diekstrak. Periksa mapping user_id dan video.")
        return

    # 3. Simpan labels
    print(f"\n[3/4] Menyimpan emotion labels...")
    save_emotion_labels(user_emotions, output_base)
    print(f"  Labels disimpan di {output_base}/{{user_id}}/labels.csv")

    # 4. Ekstrak frame secara parallel
    print(f"\n[4/4] Mengekstrak {len(all_tasks)} frames dengan {args.workers} workers...")

    completed = 0
    errors = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_single_frame, t): t for t in all_tasks}

        for future in as_completed(futures):
            path, status = future.result()
            if status == "ok":
                completed += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  [ERROR] {path}: {status}")

            total_done = completed + skipped + errors
            if total_done % 50 == 0 or total_done == len(all_tasks):
                pct = total_done / len(all_tasks) * 100
                print(f"  Progress: {total_done}/{len(all_tasks)} ({pct:.0f}%) "
                      f"- OK: {completed}, Skip: {skipped}, Error: {errors}")

    print(f"\n{'=' * 60}")
    print(f"SELESAI!")
    print(f"  Extracted: {completed}")
    print(f"  Skipped (sudah ada): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_base}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
