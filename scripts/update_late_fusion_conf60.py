"""
Update Late Fusion conf60 notebooks (45, 49, 52, 55) to run B1/B2/B3.
Uses pre-trained CNN and FCNN from corresponding B1/B2/B3 files.
"""
import json
from pathlib import Path

NB_DIR = Path("d:/MultimodalEmoLearn/notebooks")

NOTEBOOKS = [
    (45, "late_fusion_conf60_7class", 7, False),
    (49, "late_fusion_conf60_4class", 4, False),
    (52, "late_fusion_tl_conf60_7class", 7, True),
    (55, "late_fusion_tl_conf60_4class", 4, True),
]


def gen_setup(num_classes, is_tl):
    cnn_class = "EmotionCNNTransfer" if is_tl else "EmotionCNN"
    if num_classes == 7:
        ds = "dataset_frontonly_conf60"
        emotions = '["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]'
    else:
        ds = "dataset_frontonly_conf60_4class"
        emotions = '["neutral", "happy", "sad", "negative"]'

    tl_suffix = "_tl" if is_tl else ""
    out_subdir = f"{num_classes}class{tl_suffix}"
    prefix = "late_fusion_tl" if is_tl else "late_fusion"
    cnn_prefix = "cnn_tl" if is_tl else "cnn"

    return f"""import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import {cnn_class}, EmotionFCNN
from training.utils import EmotionMultimodalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")
if device.type == "cuda":
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")

DATASET_DIR = PROJECT_ROOT / "data" / "{ds}"
OUTPUT_DIR = PROJECT_ROOT / "models" / "frontonly_conf60" / "{out_subdir}"
CNN_DIR = OUTPUT_DIR  # CNN and FCNN saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_CLASSES = {num_classes}
EMOTIONS = {emotions}
CNN_PREFIX = "{cnn_prefix}"
PREFIX = "{prefix}"

# Load test multimodal
test_ds = EmotionMultimodalDataset(
    DATASET_DIR / "X_test_images.npy",
    DATASET_DIR / "X_test_landmarks.npy",
    DATASET_DIR / "y_test.npy")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Output: {{OUTPUT_DIR}}")
print(f"Test samples: {{len(test_ds)}}")"""


def gen_late_fusion_scenario(cnn_class, prefix, cnn_prefix):
    return f"""def evaluate_late_fusion(scenario):
    cnn_path = CNN_DIR / f"{{CNN_PREFIX}}_{{scenario}}.pth"
    fcnn_path = CNN_DIR / f"fcnn_{{scenario}}.pth"

    if not cnn_path.exists():
        print(f"SKIP {{scenario}}: {{cnn_path.name}} not found")
        return None
    if not fcnn_path.exists():
        print(f"SKIP {{scenario}}: {{fcnn_path.name}} not found")
        return None

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
        if f1 > best_f1: best_f1 = f1; best_w = w; best_preds = preds

    acc = accuracy_score(lbls, best_preds)
    wf1 = f1_score(lbls, best_preds, average="weighted", zero_division=0)
    print(f"  {{scenario.upper()}}: w={{best_w:.2f}} Acc={{acc:.4f}} Macro-F1={{best_f1:.4f}} W-F1={{wf1:.4f}}")
    return {{"accuracy": float(acc), "macro_f1": float(best_f1), "weighted_f1": float(wf1), "best_cnn_weight": float(best_w)}}

# Run B1, B2, B3
all_results = {{}}
print("\\nRunning Late Fusion B1, B2, B3...")
for sc_key, sc_name in [("b1", "B1 Baseline"), ("b2", "B2 Class Weights"), ("b3", "B3 Weights+Aug")]:
    r = evaluate_late_fusion(sc_key)
    if r: all_results[sc_name] = r

# Save
with open(OUTPUT_DIR / f"{{PREFIX}}_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\\nSaved: {{OUTPUT_DIR / (PREFIX + '_results.json')}}")

print("\\n" + "=" * 60)
print(f"RINGKASAN {{PREFIX.upper()}} (conf60)")
print("=" * 60)
for name, r in all_results.items():
    print(f"  {{name:<25}} Acc={{r['accuracy']:.4f}} F1={{r['macro_f1']:.4f}}")"""


for num, prefix, num_classes, is_tl in NOTEBOOKS:
    cnn_class = "EmotionCNNTransfer" if is_tl else "EmotionCNN"
    cnn_prefix = "cnn_tl" if is_tl else "cnn"
    lf_prefix = "late_fusion_tl" if is_tl else "late_fusion"

    cls_label = f"{num_classes}-Class"
    tl_label = " (Transfer Learning)" if is_tl else ""

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source":
             f"# {num} - Late Fusion{tl_label} conf60 {cls_label}\n\n"
             f"Late Fusion: weighted average CNN + FCNN\n"
             f"**Dataset:** Front-only conf60 {cls_label}\n"
             f"**Skenario:** B1 (Baseline), B2 (Class Weights), B3 (Weights + Augmentasi)"},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
             "source": gen_setup(num_classes, is_tl)},
            {"cell_type": "markdown", "metadata": {}, "source": "## Late Fusion B1, B2, B3"},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
             "source": gen_late_fusion_scenario(cnn_class, lf_prefix, cnn_prefix)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

    fname = NB_DIR / f"{num:02d}_{prefix}.ipynb"
    with open(fname, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {fname.name}")

print("\nDone! Late Fusion notebooks now run B1/B2/B3")
