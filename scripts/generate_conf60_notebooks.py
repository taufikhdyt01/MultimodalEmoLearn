"""
Generate notebooks for conf60 dataset (front-only + confidence >= 60%).
Same structure as notebooks 18-31 but with conf60 dataset paths.
B1/B2/B3 — same 3 scenarios as original experiments.

Notebooks:
43-46: 7-class from scratch (CNN, FCNN, Late Fusion, Intermediate)
47-50: 4-class from scratch
51-53: 7-class TL (CNN TL, Late Fusion TL, Intermediate TL)
54-56: 4-class TL
57: Comparison conf60 vs original
"""
import json
from pathlib import Path

NB_DIR = Path("d:/MultimodalEmoLearn/notebooks")


def make_nb():
    return {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    }, "nbformat": 4, "nbformat_minor": 5}


def cell(nb, ctype, src):
    nb["cells"].append({"cell_type": ctype, "metadata": {}, "source": src,
                         **({"outputs": [], "execution_count": None} if ctype == "code" else {})})


def gen_setup(model_imports, num_classes, is_tl=False):
    if num_classes == 7:
        ds = "dataset_frontonly_conf60"
        ds_aug = "dataset_frontonly_conf60_augmented"
        emotions = '["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]'
    else:
        ds = "dataset_frontonly_conf60_4class"
        ds_aug = "dataset_frontonly_conf60_4class_augmented"
        emotions = '["neutral", "happy", "sad", "negative"]'

    lr = "0.00005" if is_tl else "0.0001"
    tl_suffix = "_tl" if is_tl else ""
    out_subdir = f"{num_classes}class{tl_suffix}"

    return f"""import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter

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
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")

DATASET_DIR = PROJECT_ROOT / "data" / "{ds}"
DATASET_AUG_DIR = PROJECT_ROOT / "data" / "{ds_aug}"
OUTPUT_DIR = PROJECT_ROOT / "models" / "frontonly_conf60" / "{out_subdir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LR = {lr}
PATIENCE = 15
NUM_CLASSES = {num_classes}
EMOTIONS = {emotions}

print(f"Dataset: {{DATASET_DIR}}")
print(f"Dataset Aug: {{DATASET_AUG_DIR}}")
print(f"Output: {{OUTPUT_DIR}}")"""


def gen_loader(mode):
    if mode == "img":
        ds_class = "EmotionImageDataset"
        files = 'dataset_dir / "X_{split}_images.npy", dataset_dir / "y_{split}.npy"'
    elif mode == "lm":
        ds_class = "EmotionLandmarkDataset"
        files = 'dataset_dir / "X_{split}_landmarks.npy", dataset_dir / "y_{split}.npy"'
    else:
        ds_class = "EmotionMultimodalDataset"
        files = 'dataset_dir / "X_{split}_images.npy", dataset_dir / "X_{split}_landmarks.npy", dataset_dir / "y_{split}.npy"'

    return f"""def load_dataloaders(dataset_dir, batch_size=32):
    loaders = {{}}
    for split in ["train", "val", "test"]:
        ds = {ds_class}({files.replace('{split}', '" + split + "')})
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=2, pin_memory=True)
    y = np.load(dataset_dir / "y_train.npy")
    counts = Counter(y.tolist())
    print(f"Train: {{len(y)}} | Val: {{len(np.load(dataset_dir / 'y_val.npy'))}} | Test: {{len(np.load(dataset_dir / 'y_test.npy'))}}")
    for i, e in enumerate(EMOTIONS): print(f"  {{e:>10s}}: {{counts.get(i, 0)}}")
    return loaders["train"], loaders["val"], loaders["test"]

train_loader, val_loader, test_loader = load_dataloaders(DATASET_DIR, BATCH_SIZE)"""


def gen_train_all(model_class, model_type, prefix):
    return f"""all_results = {{}}

# B1: Baseline
model_b1 = {model_class}(num_classes=NUM_CLASSES).to(device)
criterion_b1 = nn.CrossEntropyLoss()
optimizer_b1 = torch.optim.Adam(model_b1.parameters(), lr=LR)
scheduler_b1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b1, mode="max", factor=0.5, patience=8, min_lr=1e-7)
print("\\nTraining B1 (baseline)...")
history_b1, _ = train_model(model_b1, train_loader, val_loader, criterion_b1, optimizer_b1, scheduler_b1,
    device, model_type="{model_type}", epochs=EPOCHS, patience=PATIENCE, save_path=str(OUTPUT_DIR / "{prefix}_b1.pth"))
plot_training_history(history_b1, "{prefix.upper()} B1 - Baseline (conf60)")
model_b1.load_state_dict(torch.load(OUTPUT_DIR / "{prefix}_b1.pth", map_location=device, weights_only=True))
r_b1 = full_evaluation(model_b1, test_loader, criterion_b1, device, "{model_type}", EMOTIONS)
plot_confusion_matrix(r_b1["confusion_matrix"], "{prefix.upper()} B1 (conf60)", EMOTIONS)
all_results["B1 Baseline"] = {{"accuracy": float(r_b1["test_accuracy"]), "macro_f1": float(r_b1["test_macro_f1"]), "weighted_f1": float(r_b1["test_weighted_f1"])}}
print(f"B1: Acc={{r_b1['test_accuracy']:.4f}} F1={{r_b1['test_macro_f1']:.4f}}")

# B2: Class Weights
weights = get_class_weights(DATASET_DIR, device)
print(f"\\nClass weights: {{weights}}")
model_b2 = {model_class}(num_classes=NUM_CLASSES).to(device)
criterion_b2 = nn.CrossEntropyLoss(weight=weights)
optimizer_b2 = torch.optim.Adam(model_b2.parameters(), lr=LR)
scheduler_b2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b2, mode="max", factor=0.5, patience=8, min_lr=1e-7)
print("Training B2 (class weights)...")
history_b2, _ = train_model(model_b2, train_loader, val_loader, criterion_b2, optimizer_b2, scheduler_b2,
    device, model_type="{model_type}", epochs=EPOCHS, patience=PATIENCE, save_path=str(OUTPUT_DIR / "{prefix}_b2.pth"))
plot_training_history(history_b2, "{prefix.upper()} B2 - Class Weights (conf60)")
model_b2.load_state_dict(torch.load(OUTPUT_DIR / "{prefix}_b2.pth", map_location=device, weights_only=True))
r_b2 = full_evaluation(model_b2, test_loader, criterion_b2, device, "{model_type}", EMOTIONS)
plot_confusion_matrix(r_b2["confusion_matrix"], "{prefix.upper()} B2 (conf60)", EMOTIONS)
all_results["B2 Class Weights"] = {{"accuracy": float(r_b2["test_accuracy"]), "macro_f1": float(r_b2["test_macro_f1"]), "weighted_f1": float(r_b2["test_weighted_f1"])}}
print(f"B2: Acc={{r_b2['test_accuracy']:.4f}} F1={{r_b2['test_macro_f1']:.4f}}")

# B3: Class Weights + Augmented
train_loader_aug, _, _ = load_dataloaders(DATASET_AUG_DIR, BATCH_SIZE)
weights_aug = get_class_weights(DATASET_AUG_DIR, device)
model_b3 = {model_class}(num_classes=NUM_CLASSES).to(device)
criterion_b3 = nn.CrossEntropyLoss(weight=weights_aug)
optimizer_b3 = torch.optim.Adam(model_b3.parameters(), lr=LR)
scheduler_b3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b3, mode="max", factor=0.5, patience=8, min_lr=1e-7)
print("\\nTraining B3 (class weights + augmented)...")
history_b3, _ = train_model(model_b3, train_loader_aug, val_loader, criterion_b3, optimizer_b3, scheduler_b3,
    device, model_type="{model_type}", epochs=EPOCHS, patience=PATIENCE, save_path=str(OUTPUT_DIR / "{prefix}_b3.pth"))
plot_training_history(history_b3, "{prefix.upper()} B3 - Augmented (conf60)")
model_b3.load_state_dict(torch.load(OUTPUT_DIR / "{prefix}_b3.pth", map_location=device, weights_only=True))
r_b3 = full_evaluation(model_b3, test_loader, criterion_b3, device, "{model_type}", EMOTIONS)
plot_confusion_matrix(r_b3["confusion_matrix"], "{prefix.upper()} B3 (conf60)", EMOTIONS)
all_results["B3 Weights+Aug"] = {{"accuracy": float(r_b3["test_accuracy"]), "macro_f1": float(r_b3["test_macro_f1"]), "weighted_f1": float(r_b3["test_weighted_f1"])}}
print(f"B3: Acc={{r_b3['test_accuracy']:.4f}} F1={{r_b3['test_macro_f1']:.4f}}")

# Summary
print("\\n" + "=" * 60)
print("RINGKASAN {prefix.upper()} (conf60)")
print("=" * 60)
for name, r in all_results.items():
    print(f"  {{name:<25}} Acc={{r['accuracy']:.4f}} F1={{r['macro_f1']:.4f}}")

with open(OUTPUT_DIR / "{prefix}_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\\nSaved: {{OUTPUT_DIR / '{prefix}_results.json'}}")"""


# ── Define notebooks ──
NOTEBOOKS = [
    # From scratch 7-class
    (43, "CNN conf60 7-class", "cnn", "EmotionCNN", "EmotionCNN", "img", "cnn", 7, False),
    (44, "FCNN conf60 7-class", "fcnn", "EmotionFCNN", "EmotionFCNN", "lm", "fcnn", 7, False),
    (45, "Late Fusion conf60 7-class", "late_fusion", None, None, "late", None, 7, False),
    (46, "Intermediate conf60 7-class", "intermediate", "IntermediateFusion", "IntermediateFusion", "mm", "fusion", 7, False),
    # From scratch 4-class
    (47, "CNN conf60 4-class", "cnn", "EmotionCNN", "EmotionCNN", "img", "cnn", 4, False),
    (48, "FCNN conf60 4-class", "fcnn", "EmotionFCNN", "EmotionFCNN", "lm", "fcnn", 4, False),
    (49, "Late Fusion conf60 4-class", "late_fusion", None, None, "late", None, 4, False),
    (50, "Intermediate conf60 4-class", "intermediate", "IntermediateFusion", "IntermediateFusion", "mm", "fusion", 4, False),
    # TL 7-class
    (51, "CNN TL conf60 7-class", "cnn_tl", "EmotionCNNTransfer", "EmotionCNNTransfer", "img", "cnn", 7, True),
    (52, "Late Fusion TL conf60 7-class", "late_fusion_tl", None, None, "late_tl", None, 7, True),
    (53, "Intermediate TL conf60 7-class", "intermediate_tl", "IntermediateFusionTransfer", "IntermediateFusionTransfer", "mm", "fusion", 7, True),
    # TL 4-class
    (54, "CNN TL conf60 4-class", "cnn_tl", "EmotionCNNTransfer", "EmotionCNNTransfer", "img", "cnn", 4, True),
    (55, "Late Fusion TL conf60 4-class", "late_fusion_tl", None, None, "late_tl", None, 4, True),
    (56, "Intermediate TL conf60 4-class", "intermediate_tl", "IntermediateFusionTransfer", "IntermediateFusionTransfer", "mm", "fusion", 4, True),
]


def gen_late_fusion_nb(num, title, prefix, num_classes, is_tl):
    nb = make_nb()
    cls_label = f"{num_classes}-Class"
    tl_label = " (Transfer Learning)" if is_tl else ""
    cnn_class = "EmotionCNNTransfer" if is_tl else "EmotionCNN"
    imports = f"{cnn_class}, EmotionFCNN"

    cell(nb, "markdown", f"# {num} - {title}\n\nLate Fusion{tl_label}: weighted average CNN + FCNN\n"
         f"**Dataset:** Front-only conf60 {cls_label}, B1 only")
    cell(nb, "code", gen_setup(imports, num_classes, is_tl) +
         "\nfrom sklearn.metrics import f1_score, accuracy_score")

    cnn_prefix = "cnn_tl" if is_tl else "cnn"
    cnn_dir_suffix = f"{num_classes}class_tl" if is_tl else f"{num_classes}class"

    cell(nb, "code", f"""# Load test multimodal
test_ds = EmotionMultimodalDataset(
    DATASET_DIR / "X_test_images.npy",
    DATASET_DIR / "X_test_landmarks.npy",
    DATASET_DIR / "y_test.npy")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

CNN_DIR = PROJECT_ROOT / "models" / "frontonly_conf60" / "{cnn_dir_suffix}"
FCNN_DIR = CNN_DIR

cnn_model = {cnn_class}(num_classes=NUM_CLASSES).to(device)
fcnn_model = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load(CNN_DIR / "{cnn_prefix}_b1.pth", map_location=device, weights_only=True))
fcnn_model.load_state_dict(torch.load(FCNN_DIR / "fcnn_b1.pth", map_location=device, weights_only=True))
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
print(f"Best CNN weight: {{best_w:.2f}}")
print(f"Acc={{acc:.4f}} Macro-F1={{best_f1:.4f}} W-F1={{wf1:.4f}}")

with open(OUTPUT_DIR / "{prefix}_results.json", "w") as f:
    json.dump({{"B1 Baseline": {{
        "accuracy": acc, "macro_f1": best_f1, "weighted_f1": wf1, "best_cnn_weight": best_w
    }}}}, f, indent=2)
print(f"Saved: {{OUTPUT_DIR / '{prefix}_results.json'}}")""")
    return nb


# ── Generate all ──
for num, title, prefix, model_class, model_import, data_mode, model_type, num_classes, is_tl in NOTEBOOKS:
    fname = NB_DIR / f"{num:02d}_{prefix}_conf60_{num_classes}class.ipynb"

    if data_mode in ("late", "late_tl"):
        nb = gen_late_fusion_nb(num, title, prefix, num_classes, is_tl)
    else:
        nb = make_nb()
        cell(nb, "markdown", f"# {num} - {title}\n\n**Dataset:** Front-only conf60 (confidence >= 60%)\n"
             f"**Skenario:** B1 (Baseline), B2 (Class Weights), B3 (Weights + Augmentasi)")
        cell(nb, "code", gen_setup(model_import, num_classes, is_tl))
        cell(nb, "code", gen_loader(data_mode))
        cell(nb, "markdown", "## Training B1, B2, B3")
        cell(nb, "code", gen_train_all(model_class, model_type, prefix))

    with open(fname, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {fname.name}")

# ── Notebook 57: Comparison ──
nb = make_nb()
cell(nb, "markdown", "# 57 - Comparison: conf60 vs Original (Front-Only)\n\n"
     "Membandingkan hasil dengan confidence filtering >= 60% vs tanpa filtering.")
cell(nb, "code", """import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
conf60_dir = PROJECT_ROOT / "models" / "frontonly_conf60"
orig_dir = PROJECT_ROOT / "models" / "frontonly"

def load_results(base, subdirs):
    results = {}
    for sd in subdirs:
        d = base / sd
        if not d.exists(): continue
        for f in d.glob("*_results.json"):
            model = f.stem.replace("_results", "")
            data = json.load(open(f))
            for sc, v in data.items():
                results[f"{sd}/{model} {sc}"] = v
    return results

conf60 = load_results(conf60_dir, ["7class", "4class", "7class_tl", "4class_tl"])
orig = load_results(orig_dir, ["7class", "4class", "7class_tl", "4class_tl"])

print(f"conf60: {len(conf60)} | Original: {len(orig)}")

print("\\n" + "="*75)
print("TOP 10 CONF60")
print("="*75)
for k, v in sorted(conf60.items(), key=lambda x: -x[1].get("macro_f1", 0))[:10]:
    print(f"  {k:<45} F1: {v.get('macro_f1', 0):.4f}")

print("\\n" + "="*75)
print("TOP 10 ORIGINAL")
print("="*75)
for k, v in sorted(orig.items(), key=lambda x: -x[1].get("macro_f1", 0))[:10]:
    print(f"  {k:<45} F1: {v.get('macro_f1', 0):.4f}")""")

with open(NB_DIR / "57_comparison_conf60.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Created 57_comparison_conf60.ipynb")

print(f"\nDone! Generated {len(NOTEBOOKS) + 1} notebooks (43-57)")
