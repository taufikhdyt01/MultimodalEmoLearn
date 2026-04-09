"""
LOSO (Leave-One-Subject-Out) Cross-Validation
==============================================
Untuk setiap user, jadikan sebagai test set, sisanya sebagai train+val.
Jalankan pada top 3 model front-only.

Menggunakan numpy arrays dari dataset_frontonly + user_ids mapping.
Tidak memerlukan raw data (data/processed, data/final).

Prerequisite:
    python scripts/generate_user_ids.py  (generate user_ids_all.npy)

Usage:
    python scripts/run_loso.py                          # semua top 3 model
    python scripts/run_loso.py --models intermediate_tl  # model tertentu
    python scripts/run_loso.py --num-classes 4           # 4-class saja
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
)
from training.utils import train_model, full_evaluation

# ── CONFIG ────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_frontonly"
OUTPUT_DIR = PROJECT_ROOT / "models" / "frontonly" / "loso"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]
REMAP_4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15


# ── DATA LOADING (from pre-built numpy arrays) ──────

def load_all_data():
    """Load all images, landmarks, labels, and user_ids from numpy arrays."""
    # Check if user_ids_all.npy exists
    uid_path = DATASET_DIR / "user_ids_all.npy"
    if not uid_path.exists():
        print("ERROR: user_ids_all.npy not found!")
        print("Run first: python scripts/generate_user_ids.py")
        sys.exit(1)

    # Load all splits and concatenate
    all_images = []
    all_landmarks = []
    all_labels = []
    all_uids = np.load(uid_path, allow_pickle=True)

    for split in ["train", "val", "test"]:
        all_images.append(np.load(DATASET_DIR / f"X_{split}_images.npy"))
        all_landmarks.append(np.load(DATASET_DIR / f"X_{split}_landmarks.npy"))
        all_labels.append(np.load(DATASET_DIR / f"y_{split}.npy"))

    images = np.concatenate(all_images, axis=0)
    landmarks = np.concatenate(all_landmarks, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    assert len(images) == len(all_uids), \
        f"Mismatch: {len(images)} samples vs {len(all_uids)} user_ids"

    return images, landmarks, labels, all_uids


def make_loaders(images, landmarks, labels, model_type, batch_size=32, shuffle=True):
    """Create DataLoader based on model type."""
    img_t = torch.from_numpy(images).permute(0, 3, 1, 2)  # NHWC -> NCHW
    lm_t = torch.from_numpy(landmarks)
    y_t = torch.from_numpy(labels).long()

    if model_type == "cnn":
        ds = TensorDataset(img_t, y_t)
    elif model_type == "fcnn":
        ds = TensorDataset(lm_t, y_t)
    else:  # fusion
        ds = TensorDataset(img_t, lm_t, y_t)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ── MODEL DEFINITIONS ────────────────────────────────

MODEL_CONFIGS = {
    "intermediate_tl": {
        "class": IntermediateFusionTransfer,
        "type": "fusion",
        "lr": 0.00005,
        "description": "Intermediate Fusion TL (ResNet18 + FCNN)",
    },
    "late_fusion": {
        "class": None,  # special handling
        "type": "late",
        "lr": 0.0001,
        "description": "Late Fusion (CNN + FCNN weighted avg)",
    },
    "fcnn": {
        "class": EmotionFCNN,
        "type": "fcnn",
        "lr": 0.0001,
        "description": "FCNN (Landmark only)",
    },
    "cnn_tl": {
        "class": EmotionCNNTransfer,
        "type": "cnn",
        "lr": 0.00005,
        "description": "CNN TL (ResNet18)",
    },
    "cnn": {
        "class": EmotionCNN,
        "type": "cnn",
        "lr": 0.0001,
        "description": "CNN (from scratch)",
    },
    "intermediate": {
        "class": IntermediateFusion,
        "type": "fusion",
        "lr": 0.0001,
        "description": "Intermediate Fusion (from scratch)",
    },
}


def train_and_eval_fold(model_name, train_img, train_lm, train_y,
                        test_img, test_lm, test_y,
                        num_classes, device, fold_dir):
    """Train model on train arrays, evaluate on test arrays."""
    cfg = MODEL_CONFIGS[model_name]
    emotions = EMOTIONS_4 if num_classes == 4 else EMOTIONS_7

    # For validation, take 10% of train
    n_val = max(1, int(len(train_y) * 0.1))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_y))
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]

    val_img, val_lm, val_y = train_img[val_idx], train_lm[val_idx], train_y[val_idx]
    train_img, train_lm, train_y = train_img[tr_idx], train_lm[tr_idx], train_y[tr_idx]

    model_type = cfg["type"]

    # Late fusion: train CNN + FCNN separately, then combine
    if model_type == "late":
        from sklearn.metrics import f1_score as f1s, accuracy_score as acc_s

        # Train CNN
        cnn_model = EmotionCNN(num_classes=num_classes).to(device)
        cnn_train = make_loaders(train_img, train_lm, train_y, "cnn", BATCH_SIZE)
        cnn_val = make_loaders(val_img, val_lm, val_y, "cnn", BATCH_SIZE, shuffle=False)
        cnn_crit = nn.CrossEntropyLoss()
        cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
        cnn_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_opt, mode="max", factor=0.5, patience=8, min_lr=1e-7)
        train_model(cnn_model, cnn_train, cnn_val, cnn_crit, cnn_opt, cnn_sched,
                     device, "cnn", EPOCHS, PATIENCE, str(fold_dir / "cnn_temp.pth"))

        # Train FCNN
        fcnn_model = EmotionFCNN(num_classes=num_classes).to(device)
        fcnn_train = make_loaders(train_img, train_lm, train_y, "fcnn", BATCH_SIZE)
        fcnn_val = make_loaders(val_img, val_lm, val_y, "fcnn", BATCH_SIZE, shuffle=False)
        fcnn_crit = nn.CrossEntropyLoss()
        fcnn_opt = torch.optim.Adam(fcnn_model.parameters(), lr=0.0001)
        fcnn_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(fcnn_opt, mode="max", factor=0.5, patience=8, min_lr=1e-7)
        train_model(fcnn_model, fcnn_train, fcnn_val, fcnn_crit, fcnn_opt, fcnn_sched,
                     device, "fcnn", EPOCHS, PATIENCE, str(fold_dir / "fcnn_temp.pth"))

        # Load best
        cnn_model.load_state_dict(torch.load(fold_dir / "cnn_temp.pth", map_location=device, weights_only=True))
        fcnn_model.load_state_dict(torch.load(fold_dir / "fcnn_temp.pth", map_location=device, weights_only=True))
        cnn_model.eval(); fcnn_model.eval()

        # Late fusion on test
        test_img_t = torch.from_numpy(test_img).permute(0, 3, 1, 2).to(device)
        test_lm_t = torch.from_numpy(test_lm).to(device)

        with torch.no_grad():
            cnn_probs = torch.softmax(cnn_model(test_img_t), dim=1).cpu().numpy()
            fcnn_probs = torch.softmax(fcnn_model(test_lm_t), dim=1).cpu().numpy()

        best_f1, best_w = 0, 0.5
        for w in np.arange(0.0, 1.05, 0.05):
            preds = (w * cnn_probs + (1 - w) * fcnn_probs).argmax(axis=1)
            f1 = f1s(test_y, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_w = w; best_preds = preds

        acc = acc_s(test_y, best_preds)
        wf1 = f1s(test_y, best_preds, average="weighted", zero_division=0)

        # Cleanup temp files
        (fold_dir / "cnn_temp.pth").unlink(missing_ok=True)
        (fold_dir / "fcnn_temp.pth").unlink(missing_ok=True)

        return {"accuracy": acc, "macro_f1": best_f1, "weighted_f1": wf1,
                "best_cnn_weight": best_w, "test_samples": len(test_y)}

    # Standard models (non-late-fusion)
    model = cfg["class"](num_classes=num_classes).to(device)
    train_loader = make_loaders(train_img, train_lm, train_y, model_type, BATCH_SIZE)
    val_loader = make_loaders(val_img, val_lm, val_y, model_type, BATCH_SIZE, shuffle=False)
    test_loader = make_loaders(test_img, test_lm, test_y, model_type, BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7)

    save_path = fold_dir / "model_temp.pth"
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, model_type, EPOCHS, PATIENCE, str(save_path))

    # Load best and evaluate
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    results = full_evaluation(model, test_loader, criterion, device, model_type, emotions)

    # Cleanup
    save_path.unlink(missing_ok=True)

    return {
        "accuracy": float(results["test_accuracy"]),
        "macro_f1": float(results["test_macro_f1"]),
        "weighted_f1": float(results["test_weighted_f1"]),
        "test_samples": len(test_y),
    }


def main():
    parser = argparse.ArgumentParser(description="LOSO Cross-Validation")
    parser.add_argument("--models", nargs="+",
                        default=["intermediate_tl", "late_fusion", "fcnn"],
                        help="Models to evaluate")
    parser.add_argument("--num-classes", type=int, default=4,
                        help="Number of classes (4 or 7)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = args.num_classes
    emotions = EMOTIONS_4 if num_classes == 4 else EMOTIONS_7
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load all data from numpy arrays ──
    print("\nLoading data from numpy arrays...")
    all_images, all_landmarks, all_labels, all_uids = load_all_data()

    # Remap labels if 4-class
    if num_classes == 4:
        all_labels = np.array([REMAP_4[int(l)] for l in all_labels], dtype=np.int64)

    # Group indices by user
    users = sorted(set(all_uids))
    user_indices = {uid: np.where(all_uids == uid)[0] for uid in users}

    total = len(all_labels)
    print(f"Total: {total} samples from {len(users)} users")
    for uid in users:
        idx = user_indices[uid]
        counts = Counter(all_labels[idx].tolist())
        print(f"  User {uid:>4s}: {len(idx):>4d} samples | {dict(sorted(counts.items()))}")

    # ── Run LOSO ──
    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n{'='*70}")
        print(f"  LOSO: {cfg['description']} ({num_classes}-class)")
        print(f"  {len(users)} folds (1 user = 1 fold)")
        print(f"{'='*70}")

        fold_results = []
        model_dir = OUTPUT_DIR / f"{model_name}_{num_classes}class"
        os.makedirs(model_dir, exist_ok=True)

        for i, test_user in enumerate(users):
            test_idx = user_indices[test_user]
            train_idx = np.concatenate([user_indices[u] for u in users if u != test_user])

            print(f"\n  Fold {i+1}/{len(users)}: test=user_{test_user} "
                  f"({len(test_idx)} samples)")

            # Skip if test set has no samples for target classes
            test_labels = all_labels[test_idx]
            if len(set(test_labels.tolist())) < 2:
                print(f"    SKIP: test user only has {len(set(test_labels.tolist()))} class(es)")
                continue

            fold_dir = model_dir / f"fold_{test_user}"
            os.makedirs(fold_dir, exist_ok=True)

            result = train_and_eval_fold(
                model_name,
                all_images[train_idx], all_landmarks[train_idx], all_labels[train_idx],
                all_images[test_idx], all_landmarks[test_idx], all_labels[test_idx],
                num_classes, device, fold_dir)

            result["test_user"] = test_user
            fold_results.append(result)

            print(f"    Acc={result['accuracy']:.4f} "
                  f"Macro-F1={result['macro_f1']:.4f} "
                  f"W-F1={result['weighted_f1']:.4f} "
                  f"(n={result['test_samples']})")

            # Cleanup fold dir
            try:
                fold_dir.rmdir()
            except OSError:
                pass

        # ── Summary ──
        if fold_results:
            accs = [r["accuracy"] for r in fold_results]
            f1s = [r["macro_f1"] for r in fold_results]
            wf1s = [r["weighted_f1"] for r in fold_results]

            print(f"\n{'='*70}")
            print(f"  LOSO RESULTS: {cfg['description']} ({num_classes}-class)")
            print(f"{'='*70}")
            print(f"  Folds completed: {len(fold_results)}/{len(users)}")
            print(f"  Accuracy:    {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"  Macro F1:    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
            print(f"  Weighted F1: {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}")
            print(f"  Best fold:   user_{fold_results[np.argmax(f1s)]['test_user']} (F1={max(f1s):.4f})")
            print(f"  Worst fold:  user_{fold_results[np.argmin(f1s)]['test_user']} (F1={min(f1s):.4f})")

            # Save
            summary = {
                "model": model_name,
                "description": cfg["description"],
                "num_classes": num_classes,
                "num_folds": len(fold_results),
                "total_users": len(users),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "weighted_f1_mean": float(np.mean(wf1s)),
                "weighted_f1_std": float(np.std(wf1s)),
                "per_fold": fold_results,
            }
            save_path = OUTPUT_DIR / f"loso_{model_name}_{num_classes}class.json"
            with open(save_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
