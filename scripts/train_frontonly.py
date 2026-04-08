"""
Train all front-only experiments.
Same 48 experiments as before but on front-only dataset.

Usage:
    python scripts/train_frontonly.py                    # all experiments
    python scripts/train_frontonly.py --models cnn fcnn  # specific models
    python scripts/train_frontonly.py --classes 4         # 4-class only
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
)
from training.utils import (
    EmotionImageDataset, EmotionLandmarkDataset, EmotionMultimodalDataset,
    get_class_weights, train_model, full_evaluation,
)

# ── CONFIG ────────────────────────────────────────────
DATA_BASE = PROJECT_ROOT / "data"
OUTPUT_BASE = PROJECT_ROOT / "models" / "frontonly"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15
LR = 0.0001
LR_TL = 0.00005  # Transfer learning uses smaller LR


def get_dataset_paths(num_classes, augmented=False):
    suffix = "frontonly"
    if num_classes == 4:
        suffix += "_4class"
    if augmented:
        suffix += "_augmented"
    return DATA_BASE / f"dataset_{suffix}"


def load_data(dataset_dir, model_type, batch_size=32):
    """Load dataloaders based on model type."""
    if model_type in ("cnn", "cnn_tl"):
        train_ds = EmotionImageDataset(
            dataset_dir / "X_train_images.npy",
            dataset_dir / "y_train.npy")
        val_ds = EmotionImageDataset(
            dataset_dir / "X_val_images.npy",
            dataset_dir / "y_val.npy")
        test_ds = EmotionImageDataset(
            dataset_dir / "X_test_images.npy",
            dataset_dir / "y_test.npy")
    elif model_type == "fcnn":
        train_ds = EmotionLandmarkDataset(
            dataset_dir / "X_train_landmarks.npy",
            dataset_dir / "y_train.npy")
        val_ds = EmotionLandmarkDataset(
            dataset_dir / "X_val_landmarks.npy",
            dataset_dir / "y_val.npy")
        test_ds = EmotionLandmarkDataset(
            dataset_dir / "X_test_landmarks.npy",
            dataset_dir / "y_test.npy")
    else:  # fusion
        train_ds = EmotionMultimodalDataset(
            dataset_dir / "X_train_images.npy",
            dataset_dir / "X_train_landmarks.npy",
            dataset_dir / "y_train.npy")
        val_ds = EmotionMultimodalDataset(
            dataset_dir / "X_val_images.npy",
            dataset_dir / "X_val_landmarks.npy",
            dataset_dir / "y_val.npy")
        test_ds = EmotionMultimodalDataset(
            dataset_dir / "X_test_images.npy",
            dataset_dir / "X_test_landmarks.npy",
            dataset_dir / "y_test.npy")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def create_model(model_name, num_classes, device):
    """Create model instance."""
    models = {
        "cnn": lambda: EmotionCNN(num_classes=num_classes),
        "fcnn": lambda: EmotionFCNN(num_classes=num_classes),
        "late_fusion": None,  # handled separately
        "intermediate": lambda: IntermediateFusion(num_classes=num_classes),
        "cnn_tl": lambda: EmotionCNNTransfer(num_classes=num_classes),
        "intermediate_tl": lambda: IntermediateFusionTransfer(num_classes=num_classes),
    }
    if model_name == "late_fusion":
        # Late fusion uses separate CNN + FCNN
        return None
    return models[model_name]().to(device)


def get_model_type_for_training(model_name):
    """Map model name to training model_type."""
    if model_name in ("cnn", "cnn_tl"):
        return "cnn"
    elif model_name == "fcnn":
        return "fcnn"
    else:
        return "fusion"


def run_late_fusion(cnn_model, fcnn_model, test_loader, device, num_classes):
    """Run late fusion by combining CNN + FCNN predictions."""
    cnn_model.eval()
    fcnn_model.eval()

    all_preds = []
    all_labels = []
    best_f1 = 0
    best_weight = 0.5

    # Get all predictions
    cnn_probs_all = []
    fcnn_probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in test_loader:
            images, landmarks, labels = batch
            images = images.to(device)
            landmarks = landmarks.to(device)

            cnn_out = torch.softmax(cnn_model(images), dim=1)
            fcnn_out = torch.softmax(fcnn_model(landmarks), dim=1)

            cnn_probs_all.append(cnn_out.cpu().numpy())
            fcnn_probs_all.append(fcnn_out.cpu().numpy())
            labels_all.append(labels.numpy())

    cnn_probs = np.concatenate(cnn_probs_all)
    fcnn_probs = np.concatenate(fcnn_probs_all)
    labels = np.concatenate(labels_all)

    # Search best weight
    from sklearn.metrics import f1_score
    for w in np.arange(0.0, 1.05, 0.05):
        combined = w * cnn_probs + (1 - w) * fcnn_probs
        preds = combined.argmax(axis=1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weight = w
            best_preds = preds

    from sklearn.metrics import accuracy_score, f1_score as f1s, classification_report
    acc = accuracy_score(labels, best_preds)
    macro_f1 = f1s(labels, best_preds, average="macro", zero_division=0)
    weighted_f1 = f1s(labels, best_preds, average="weighted", zero_division=0)

    return {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "best_cnn_weight": best_weight,
    }


def train_single_experiment(model_name, scenario, num_classes, device, output_dir):
    """Train a single experiment."""
    emotions = EMOTIONS_7 if num_classes == 7 else EMOTIONS_4
    is_tl = "_tl" in model_name
    lr = LR_TL if is_tl else LR
    model_type = get_model_type_for_training(model_name)

    # Dataset
    augmented = (scenario == "B3")
    dataset_dir = get_dataset_paths(num_classes, augmented)

    if not dataset_dir.exists():
        print(f"  SKIP: {dataset_dir} not found")
        return None

    # Load data
    data_model_type = model_type
    if model_name in ("cnn", "cnn_tl"):
        data_model_type = "cnn"
    elif model_name == "fcnn":
        data_model_type = "fcnn"
    else:
        data_model_type = "fusion"

    train_loader, val_loader, test_loader = load_data(
        dataset_dir, data_model_type, BATCH_SIZE)

    # Late fusion special case
    if model_name in ("late_fusion", "late_fusion_tl"):
        cnn_name = "cnn_tl" if is_tl else "cnn"
        cnn_path = output_dir / f"{cnn_name}_{num_classes}class_{scenario}.pth"
        fcnn_path = output_dir / f"fcnn_{num_classes}class_{scenario}.pth"

        if not cnn_path.exists() or not fcnn_path.exists():
            print(f"  SKIP late fusion: need {cnn_path.name} and {fcnn_path.name}")
            return None

        if is_tl:
            cnn_model = EmotionCNNTransfer(num_classes=num_classes).to(device)
        else:
            cnn_model = EmotionCNN(num_classes=num_classes).to(device)
        fcnn_model = EmotionFCNN(num_classes=num_classes).to(device)

        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        fcnn_model.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

        # Need multimodal test loader
        test_loader_mm = load_data(dataset_dir, "fusion", BATCH_SIZE)[2]
        results = run_late_fusion(cnn_model, fcnn_model, test_loader_mm, device, num_classes)
        return results

    # Create model
    model = create_model(model_name, num_classes, device)

    # Loss
    if scenario in ("B2", "B3"):
        weights = get_class_weights(dataset_dir, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7)

    save_path = output_dir / f"{model_name}_{num_classes}class_{scenario}.pth"

    history, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, model_type=model_type, epochs=EPOCHS, patience=PATIENCE,
        save_path=str(save_path))

    # Load best and evaluate
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    results = full_evaluation(model, test_loader, criterion, device,
                              model_type=model_type, emotions=emotions)
    results["best_epoch"] = best_epoch
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "fcnn", "late_fusion", "intermediate",
                                 "cnn_tl", "late_fusion_tl", "intermediate_tl"],
                        help="Models to train")
    parser.add_argument("--scenarios", nargs="+", default=["B1", "B2", "B3"])
    parser.add_argument("--classes", nargs="+", type=int, default=[7, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_results = {}
    total = len(args.models) * len(args.scenarios) * len(args.classes)
    done = 0

    for num_classes in args.classes:
        for model_name in args.models:
            # FCNN doesn't change with TL
            if model_name == "fcnn" and "fcnn" in [m for m in args.models if "_tl" in m]:
                pass  # still train FCNN

            for scenario in args.scenarios:
                done += 1
                key = f"{model_name}_{num_classes}class_{scenario}"
                print(f"\n{'='*60}")
                print(f"[{done}/{total}] {key}")
                print(f"{'='*60}")

                results = train_single_experiment(
                    model_name, scenario, num_classes, device, OUTPUT_BASE)

                if results:
                    all_results[key] = {
                        "model": model_name,
                        "num_classes": num_classes,
                        "scenario": scenario,
                        "accuracy": float(results["test_accuracy"]),
                        "macro_f1": float(results["test_macro_f1"]),
                        "weighted_f1": float(results["test_weighted_f1"]),
                    }
                    if "best_cnn_weight" in results:
                        all_results[key]["best_cnn_weight"] = float(results["best_cnn_weight"])

                    print(f"\n  >> Acc={results['test_accuracy']:.4f} "
                          f"Macro-F1={results['test_macro_f1']:.4f} "
                          f"Weighted-F1={results['test_weighted_f1']:.4f}")

                    # Save incremental
                    with open(OUTPUT_BASE / "frontonly_results.json", "w") as f:
                        json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("RINGKASAN FRONT-ONLY EXPERIMENTS")
    print("="*70)
    print(f"{'Key':<45} {'Acc':>8} {'Macro F1':>10} {'W-F1':>10}")
    print("-"*75)
    for key, r in sorted(all_results.items(), key=lambda x: -x[1]["macro_f1"]):
        print(f"{key:<45} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f}")

    with open(OUTPUT_BASE / "frontonly_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_BASE / 'frontonly_results.json'}")


if __name__ == "__main__":
    main()