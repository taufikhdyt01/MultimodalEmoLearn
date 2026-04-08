"""
Generate individual notebooks for front-only experiments (18-32),
mirroring the structure of notebooks 01-17 but with front-only dataset paths.
"""
import json
import copy
from pathlib import Path

NOTEBOOKS_DIR = Path("d:/MultimodalEmoLearn/notebooks")


def make_notebook():
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def add_cell(nb, cell_type, source):
    nb["cells"].append({
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    })


def gen_setup(model_imports, extra_imports=""):
    return f"""import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter
{extra_imports}
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import {model_imports}
from training.utils import (
    EmotionImageDataset, EmotionLandmarkDataset, EmotionMultimodalDataset,
    get_class_weights, train_model, full_evaluation,
    plot_training_history, plot_confusion_matrix, plot_per_class_f1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")
if device.type == "cuda":
    print(f"GPU: {{torch.cuda.get_device_name(0)}})")"""


def gen_config(num_classes, is_tl=False):
    if num_classes == 7:
        ds = "dataset_frontonly"
        ds_aug = "dataset_frontonly_augmented"
        emotions = '["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]'
    else:
        ds = "dataset_frontonly_4class"
        ds_aug = "dataset_frontonly_4class_augmented"
        emotions = '["neutral", "happy", "sad", "negative"]'

    lr = "0.00005" if is_tl else "0.0001"
    tl_suffix = "_tl" if is_tl else ""
    out_subdir = f"{num_classes}class{tl_suffix}"

    return f"""DATASET_DIR = PROJECT_ROOT / "data" / "{ds}"
DATASET_AUG_DIR = PROJECT_ROOT / "data" / "{ds_aug}"
OUTPUT_DIR = PROJECT_ROOT / "models" / "frontonly" / "{out_subdir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LR = {lr}
PATIENCE = 15
NUM_CLASSES = {num_classes}
EMOTIONS = {emotions}

print(f"Dataset: {{DATASET_DIR}}")
print(f"Output: {{OUTPUT_DIR}}")"""


def gen_loader(mode):
    if mode == "img":
        return """print("Loading image data...")
def load_dataloaders(dataset_dir, batch_size=32):
    train_ds = EmotionImageDataset(dataset_dir / "X_train_images.npy", dataset_dir / "y_train.npy")
    val_ds = EmotionImageDataset(dataset_dir / "X_val_images.npy", dataset_dir / "y_val.npy")
    test_ds = EmotionImageDataset(dataset_dir / "X_test_images.npy", dataset_dir / "y_test.npy")
    train_l = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_l = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_l = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    y = np.load(dataset_dir / "y_train.npy")
    counts = Counter(y.tolist())
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    for i, e in enumerate(EMOTIONS): print(f"  {e:>10s}: {counts.get(i, 0)}")
    return train_l, val_l, test_l

train_loader, val_loader, test_loader = load_dataloaders(DATASET_DIR, BATCH_SIZE)"""
    elif mode == "lm":
        return """print("Loading landmark data...")
def load_dataloaders(dataset_dir, batch_size=32):
    train_ds = EmotionLandmarkDataset(dataset_dir / "X_train_landmarks.npy", dataset_dir / "y_train.npy")
    val_ds = EmotionLandmarkDataset(dataset_dir / "X_val_landmarks.npy", dataset_dir / "y_val.npy")
    test_ds = EmotionLandmarkDataset(dataset_dir / "X_test_landmarks.npy", dataset_dir / "y_test.npy")
    train_l = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_l = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_l = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    y = np.load(dataset_dir / "y_train.npy")
    counts = Counter(y.tolist())
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    for i, e in enumerate(EMOTIONS): print(f"  {e:>10s}: {counts.get(i, 0)}")
    return train_l, val_l, test_l

train_loader, val_loader, test_loader = load_dataloaders(DATASET_DIR, BATCH_SIZE)"""
    else:  # mm
        return """print("Loading multimodal data...")
def load_dataloaders(dataset_dir, batch_size=32):
    train_ds = EmotionMultimodalDataset(
        dataset_dir / "X_train_images.npy", dataset_dir / "X_train_landmarks.npy", dataset_dir / "y_train.npy")
    val_ds = EmotionMultimodalDataset(
        dataset_dir / "X_val_images.npy", dataset_dir / "X_val_landmarks.npy", dataset_dir / "y_val.npy")
    test_ds = EmotionMultimodalDataset(
        dataset_dir / "X_test_images.npy", dataset_dir / "X_test_landmarks.npy", dataset_dir / "y_test.npy")
    train_l = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_l = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_l = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    y = np.load(dataset_dir / "y_train.npy")
    counts = Counter(y.tolist())
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    for i, e in enumerate(EMOTIONS): print(f"  {e:>10s}: {counts.get(i, 0)}")
    return train_l, val_l, test_l

train_loader, val_loader, test_loader = load_dataloaders(DATASET_DIR, BATCH_SIZE)"""


def gen_train_scenario(model_class, model_type, prefix, scenario, use_aug_loader=False):
    sc_labels = {"B1": "Baseline", "B2": "Class Weights", "B3": "Weights + Augmentasi"}
    label = sc_labels[scenario]
    save_name = f"{prefix}_{scenario.lower()}.pth"
    title = f"{prefix.upper()} {scenario} - {label} (Front-Only)"

    loader_var = "train_loader_aug" if use_aug_loader else "train_loader"

    if scenario == "B1":
        criterion_line = "criterion = nn.CrossEntropyLoss()"
    elif scenario == "B2":
        criterion_line = """weights = get_class_weights(DATASET_DIR, device)
print(f"Class weights: {weights}")
criterion = nn.CrossEntropyLoss(weight=weights)"""
    else:  # B3
        criterion_line = f"""train_loader_aug, _, _ = load_dataloaders(DATASET_AUG_DIR, BATCH_SIZE)
weights_aug = get_class_weights(DATASET_AUG_DIR, device)
print(f"Augmented class weights: {{weights_aug}}")
criterion = nn.CrossEntropyLoss(weight=weights_aug)"""

    return f"""# {scenario}: {label}
{criterion_line}

model = {model_class}(num_classes=NUM_CLASSES).to(device)
print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7)

print("\\nTraining {title}...")
history, best_epoch = train_model(
    model, {loader_var}, val_loader, criterion, optimizer, scheduler,
    device, model_type="{model_type}", epochs=EPOCHS, patience=PATIENCE,
    save_path=str(OUTPUT_DIR / "{save_name}"))

plot_training_history(history, "{title}")"""


def gen_eval(prefix, scenario, model_class, model_type):
    sc_labels = {"B1": "Baseline", "B2": "Class Weights", "B3": "Weights + Augmentasi"}
    label = sc_labels[scenario]
    save_name = f"{prefix}_{scenario.lower()}.pth"
    title = f"{prefix.upper()} {scenario} - {label} (Front-Only)"

    return f"""# Evaluate {scenario}
model.load_state_dict(torch.load(OUTPUT_DIR / "{save_name}", map_location=device, weights_only=True))
print("=" * 60)
print("EVALUASI {scenario} - {label.upper()}")
print("=" * 60)
results_{scenario.lower()} = full_evaluation(model, test_loader, criterion, device, "{model_type}", EMOTIONS)
plot_confusion_matrix(results_{scenario.lower()}["confusion_matrix"], "{title}", EMOTIONS)"""


def gen_comparison(prefix):
    return f"""# Perbandingan
all_results = {{
    "B1 Baseline": results_b1,
    "B2 Class Weights": results_b2,
    "B3 Weights+Aug": results_b3,
}}
plot_per_class_f1(all_results, "{prefix.upper()} - Perbandingan F1 (Front-Only)")

print("=" * 70)
print("RINGKASAN {prefix.upper()} FRONT-ONLY")
print("=" * 70)
print(f"{{'Skenario':<25}} {{'Accuracy':>10}} {{'Macro F1':>10}} {{'Weighted F1':>12}}")
print("-" * 70)
for name, r in all_results.items():
    print(f"{{name:<25}} {{r['test_accuracy']:>10.4f}} {{r['test_macro_f1']:>10.4f}} {{r['test_weighted_f1']:>12.4f}}")

results_save = {{}}
for name, r in all_results.items():
    results_save[name] = {{
        "accuracy": float(r["test_accuracy"]),
        "macro_f1": float(r["test_macro_f1"]),
        "weighted_f1": float(r["test_weighted_f1"]),
    }}
with open(OUTPUT_DIR / "{prefix}_results.json", "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\\nSaved to {{OUTPUT_DIR / '{prefix}_results.json'}}")"""


# ═══════════════════════════════════════════════════════
# Define all notebooks
# ═══════════════════════════════════════════════════════

NOTEBOOKS = [
    # From scratch 7-class
    (18, "CNN Front-Only 7-Class", "cnn", "EmotionCNN", "EmotionCNN", "img", 7, False),
    (19, "FCNN Front-Only 7-Class", "fcnn", "EmotionFCNN", "EmotionFCNN", "lm", 7, False),
    (20, "Late Fusion Front-Only 7-Class", "late_fusion", None, None, "late", 7, False),
    (21, "Intermediate Fusion Front-Only 7-Class", "intermediate", "IntermediateFusion", "IntermediateFusion", "mm", 7, False),
    # From scratch 4-class
    (22, "CNN Front-Only 4-Class", "cnn", "EmotionCNN", "EmotionCNN", "img", 4, False),
    (23, "FCNN Front-Only 4-Class", "fcnn", "EmotionFCNN", "EmotionFCNN", "lm", 4, False),
    (24, "Late Fusion Front-Only 4-Class", "late_fusion", None, None, "late", 4, False),
    (25, "Intermediate Fusion Front-Only 4-Class", "intermediate", "IntermediateFusion", "IntermediateFusion", "mm", 4, False),
    # Transfer Learning 7-class
    (26, "CNN Transfer Learning Front-Only 7-Class", "cnn_tl", "EmotionCNNTransfer", "EmotionCNNTransfer", "img", 7, True),
    (27, "Late Fusion TL Front-Only 7-Class", "late_fusion_tl", None, None, "late_tl", 7, True),
    (28, "Intermediate Fusion TL Front-Only 7-Class", "intermediate_tl", "IntermediateFusionTransfer", "IntermediateFusionTransfer", "mm", 7, True),
    # Transfer Learning 4-class
    (29, "CNN Transfer Learning Front-Only 4-Class", "cnn_tl", "EmotionCNNTransfer", "EmotionCNNTransfer", "img", 4, True),
    (30, "Late Fusion TL Front-Only 4-Class", "late_fusion_tl", None, None, "late_tl", 4, True),
    (31, "Intermediate Fusion TL Front-Only 4-Class", "intermediate_tl", "IntermediateFusionTransfer", "IntermediateFusionTransfer", "mm", 4, True),
]


def gen_late_fusion_notebook(num, title, num_classes, is_tl):
    nb = make_notebook()
    cls_label = f"{num_classes}-Class"
    tl_label = " (Transfer Learning)" if is_tl else ""

    add_cell(nb, "markdown", f"# {num} - Late Fusion{tl_label} Front-Only ({cls_label})\n\n"
             f"Late Fusion menggabungkan prediksi CNN{'TL' if is_tl else ''} dan FCNN menggunakan weighted average.\n"
             f"Bobot optimal dicari dengan grid search pada test set.\n\n"
             f"**3 Skenario:** B1, B2, B3")

    cnn_class = "EmotionCNNTransfer" if is_tl else "EmotionCNN"
    imports = f"{cnn_class}, EmotionFCNN"
    add_cell(nb, "code", gen_setup(imports, "from sklearn.metrics import f1_score, accuracy_score"))
    add_cell(nb, "code", gen_config(num_classes, is_tl))

    tl_suffix = "_tl" if is_tl else ""
    out_subdir = f"{num_classes}class{tl_suffix}"

    # Load models from corresponding CNN and FCNN notebooks
    cnn_dir = f"frontonly/{out_subdir}"
    fcnn_dir = cnn_dir  # same dir for TL, different for non-TL
    if not is_tl:
        fcnn_dir = f"frontonly/{num_classes}class"
        cnn_dir = fcnn_dir

    add_cell(nb, "code", f"""# Load multimodal test set
test_ds = EmotionMultimodalDataset(
    DATASET_DIR / "X_test_images.npy",
    DATASET_DIR / "X_test_landmarks.npy",
    DATASET_DIR / "y_test.npy")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Test: {{len(test_ds)}} samples")

CNN_DIR = PROJECT_ROOT / "models" / "{cnn_dir}"
FCNN_DIR = PROJECT_ROOT / "models" / "{fcnn_dir if not is_tl else cnn_dir}"

def late_fusion_eval(cnn_path, fcnn_path, title):
    cnn_model = {cnn_class}(num_classes=NUM_CLASSES).to(device)
    fcnn_model = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    fcnn_model.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))
    cnn_model.eval(); fcnn_model.eval()

    cnn_probs_all, fcnn_probs_all, labels_all = [], [], []
    with torch.no_grad():
        for images, landmarks, labels in test_loader:
            cnn_probs_all.append(torch.softmax(cnn_model(images.to(device)), dim=1).cpu().numpy())
            fcnn_probs_all.append(torch.softmax(fcnn_model(landmarks.to(device)), dim=1).cpu().numpy())
            labels_all.append(labels.numpy())

    cnn_probs = np.concatenate(cnn_probs_all)
    fcnn_probs = np.concatenate(fcnn_probs_all)
    lbls = np.concatenate(labels_all)

    best_f1, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        preds = (w * cnn_probs + (1-w) * fcnn_probs).argmax(axis=1)
        f1 = f1_score(lbls, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_w = w; best_preds = preds

    acc = accuracy_score(lbls, best_preds)
    wf1 = f1_score(lbls, best_preds, average="weighted", zero_division=0)
    print(f"\\n{{title}}")
    print(f"  Best CNN weight: {{best_w:.2f}} | Acc: {{acc:.4f}} | Macro F1: {{best_f1:.4f}} | W-F1: {{wf1:.4f}}")
    return {{"test_accuracy": acc, "test_macro_f1": best_f1, "test_weighted_f1": wf1, "best_cnn_weight": best_w}}""")

    cnn_prefix = "cnn_tl" if is_tl else "cnn"
    add_cell(nb, "markdown", "## Evaluasi Late Fusion — B1, B2, B3")
    add_cell(nb, "code", f"""results_b1 = late_fusion_eval(
    CNN_DIR / "{cnn_prefix}_b1.pth", {"CNN_DIR" if is_tl else "FCNN_DIR"} / "fcnn_b1.pth",
    "Late Fusion{'_TL' if is_tl else ''} B1 - Baseline (Front-Only)")

results_b2 = late_fusion_eval(
    CNN_DIR / "{cnn_prefix}_b2.pth", {"CNN_DIR" if is_tl else "FCNN_DIR"} / "fcnn_b2.pth",
    "Late Fusion{'_TL' if is_tl else ''} B2 - Class Weights (Front-Only)")

results_b3 = late_fusion_eval(
    CNN_DIR / "{cnn_prefix}_b3.pth", {"CNN_DIR" if is_tl else "FCNN_DIR"} / "fcnn_b3.pth",
    "Late Fusion{'_TL' if is_tl else ''} B3 - Augmented (Front-Only)")""")

    prefix = f"late_fusion{'_tl' if is_tl else ''}"
    add_cell(nb, "markdown", "## Ringkasan")
    add_cell(nb, "code", gen_comparison(prefix))

    return nb


def gen_standard_notebook(num, title, prefix, model_class, model_type, data_mode, num_classes, is_tl):
    nb = make_notebook()
    cls_label = f"{num_classes}-Class"
    tl_label = " (Transfer Learning)" if is_tl else ""

    add_cell(nb, "markdown", f"# {num} - {title}\n\n"
             f"**Dataset:** Front-only {cls_label}\n"
             f"**3 Skenario:** B1 (Baseline), B2 (Class Weights), B3 (Weights + Augmentasi)"
             f"{tl_label}")

    add_cell(nb, "code", gen_setup(model_class))
    add_cell(nb, "code", gen_config(num_classes, is_tl))
    add_cell(nb, "code", gen_loader(data_mode))

    # B1
    add_cell(nb, "markdown", "## Skenario B1: Baseline")
    add_cell(nb, "code", gen_train_scenario(model_class, model_type, prefix, "B1"))
    add_cell(nb, "code", gen_eval(prefix, "B1", model_class, model_type))

    # B2
    add_cell(nb, "markdown", "## Skenario B2: Class Weights")
    add_cell(nb, "code", gen_train_scenario(model_class, model_type, prefix, "B2"))
    add_cell(nb, "code", gen_eval(prefix, "B2", model_class, model_type))

    # B3
    add_cell(nb, "markdown", "## Skenario B3: Class Weights + Augmentasi")
    add_cell(nb, "code", gen_train_scenario(model_class, model_type, prefix, "B3", use_aug_loader=True))
    add_cell(nb, "code", gen_eval(prefix, "B3", model_class, model_type))

    # Comparison
    add_cell(nb, "markdown", "## Perbandingan 3 Skenario")
    add_cell(nb, "code", gen_comparison(prefix))

    return nb


# ═══════════════════════════════════════════════════════
# Generate all notebooks
# ═══════════════════════════════════════════════════════

# Remove old consolidated notebooks
for f in ["18_frontonly_7class.ipynb", "19_frontonly_4class.ipynb", "20_frontonly_transfer.ipynb"]:
    p = NOTEBOOKS_DIR / f
    if p.exists():
        p.unlink()
        print(f"Removed old {f}")

for num, title, prefix, model_class, model_type_str, data_mode, num_classes, is_tl in NOTEBOOKS:
    fname = NOTEBOOKS_DIR / f"{num:02d}_{prefix}_frontonly_{'4class' if num_classes == 4 else '7class'}.ipynb"

    if data_mode in ("late", "late_tl"):
        nb = gen_late_fusion_notebook(num, title, num_classes, is_tl)
    else:
        model_type = {"img": "cnn", "lm": "fcnn", "mm": "fusion"}[data_mode]
        nb = gen_standard_notebook(num, title, prefix, model_class, model_type, data_mode, num_classes, is_tl)

    with open(fname, "w") as f:
        json.dump(nb, f, indent=1)
    n_cells = len(nb["cells"])
    print(f"Created {fname.name} ({n_cells} cells)")

# Notebook 32: Comparison
nb = make_notebook()
add_cell(nb, "markdown", "# 32 - Perbandingan Front-Only vs Front+Side\n\n"
         "Membandingkan hasil 48 eksperimen front-only dengan 48 eksperimen original (front+side).")
add_cell(nb, "code", """import sys, os, json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..").resolve()

# Load front-only results
frontonly_dir = PROJECT_ROOT / "models" / "frontonly"
original_dir = PROJECT_ROOT / "models"

def load_all_results(base_dir, subdirs):
    results = {}
    for sd in subdirs:
        for f in (base_dir / sd).glob("*_results.json"):
            with open(f) as fh:
                data = json.load(fh)
                for k, v in data.items():
                    results[f"{sd}/{k}"] = v
    return results

# Load results
print("Loading results...")
frontonly = load_all_results(frontonly_dir, ["7class", "4class", "7class_tl", "4class_tl"])
original = load_all_results(original_dir, ["cnn", "fcnn", "late_fusion", "intermediate",
                                            "cnn_4class", "fcnn_4class", "late_fusion_4class", "intermediate_4class",
                                            "cnn_transfer", "late_fusion_transfer", "intermediate_transfer",
                                            "cnn_transfer_4class", "late_fusion_transfer_4class", "intermediate_transfer_4class"])

print(f"Front-only: {len(frontonly)} experiments")
print(f"Original:   {len(original)} experiments")

# Compare best models
print("\\n" + "="*70)
print("TOP 5 FRONT-ONLY")
print("="*70)
for k, v in sorted(frontonly.items(), key=lambda x: -x[1].get("macro_f1", 0))[:5]:
    print(f"  {k:<40} Macro F1: {v.get('macro_f1', 0):.4f}")

print("\\n" + "="*70)
print("TOP 5 ORIGINAL (FRONT+SIDE)")
print("="*70)
for k, v in sorted(original.items(), key=lambda x: -x[1].get("macro_f1", 0))[:5]:
    print(f"  {k:<40} Macro F1: {v.get('macro_f1', 0):.4f}")""")

fname = NOTEBOOKS_DIR / "32_comparison_frontonly_vs_original.ipynb"
with open(fname, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {fname.name}")

print(f"\nDone! Generated {len(NOTEBOOKS) + 1} notebooks (18-32)")
