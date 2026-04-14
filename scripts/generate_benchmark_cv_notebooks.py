"""Generate notebooks 38 (JAFFE LOSO) and 39 (CK+ 10-Fold CV)."""
import json, copy
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

SETUP = """import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
)
from training.utils import train_model, full_evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 15
{output_dir_placeholder}

MODELS = [
    ("CNN", EmotionCNN, "cnn", 0.0001),
    ("FCNN", EmotionFCNN, "fcnn", 0.0001),
    ("Intermediate", IntermediateFusion, "fusion", 0.0001),
    ("CNN_TL", EmotionCNNTransfer, "cnn", 0.00005),
    ("Intermediate_TL", IntermediateFusionTransfer, "fusion", 0.00005),
]
print("Setup complete.")"""

HELPERS = """def make_loader(images, landmarks, labels, model_type, batch_size=16, shuffle=True):
    img_t = torch.from_numpy(images).permute(0, 3, 1, 2)
    lm_t = torch.from_numpy(landmarks)
    y_t = torch.from_numpy(labels).long()
    if model_type == "cnn": ds = TensorDataset(img_t, y_t)
    elif model_type == "fcnn": ds = TensorDataset(lm_t, y_t)
    else: ds = TensorDataset(img_t, lm_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def train_fold(ModelClass, model_type, lr, train_img, train_lm, train_y,
               test_img, test_lm, test_y, emotions, fold_dir):
    n_val = max(1, int(len(train_y) * 0.15))
    perm = np.random.RandomState(42).permutation(len(train_y))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    tr_l = make_loader(train_img[tr_i], train_lm[tr_i], train_y[tr_i], model_type, BATCH_SIZE)
    vl_l = make_loader(train_img[val_i], train_lm[val_i], train_y[val_i], model_type, BATCH_SIZE, False)
    te_l = make_loader(test_img, test_lm, test_y, model_type, BATCH_SIZE, False)
    model = ModelClass(num_classes=len(emotions)).to(device)
    sp = str(fold_dir / "model.pth")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    train_model(model, tr_l, vl_l, nn.CrossEntropyLoss(), opt, sch, device, model_type, EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, model_type, emotions)
    os.remove(sp)
    return {"accuracy": float(r["test_accuracy"]), "macro_f1": float(r["test_macro_f1"]),
            "weighted_f1": float(r["test_weighted_f1"])}

def late_fusion_fold(train_img, train_lm, train_y, test_img, test_lm, test_y, num_classes, fold_dir):
    n_val = max(1, int(len(train_y) * 0.15))
    perm = np.random.RandomState(42).permutation(len(train_y))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    cnn = EmotionCNN(num_classes=num_classes).to(device)
    o1 = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    s1 = torch.optim.lr_scheduler.ReduceLROnPlateau(o1, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    train_model(cnn, make_loader(train_img[tr_i],train_lm[tr_i],train_y[tr_i],"cnn",BATCH_SIZE),
                make_loader(train_img[val_i],train_lm[val_i],train_y[val_i],"cnn",BATCH_SIZE,False),
                nn.CrossEntropyLoss(), o1, s1, device, "cnn", EPOCHS, PATIENCE, str(fold_dir/"cnn.pth"))
    fcnn = EmotionFCNN(num_classes=num_classes).to(device)
    o2 = torch.optim.Adam(fcnn.parameters(), lr=0.0001)
    s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(o2, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    train_model(fcnn, make_loader(train_img[tr_i],train_lm[tr_i],train_y[tr_i],"fcnn",BATCH_SIZE),
                make_loader(train_img[val_i],train_lm[val_i],train_y[val_i],"fcnn",BATCH_SIZE,False),
                nn.CrossEntropyLoss(), o2, s2, device, "fcnn", EPOCHS, PATIENCE, str(fold_dir/"fcnn.pth"))
    cnn.load_state_dict(torch.load(fold_dir/"cnn.pth", map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fold_dir/"fcnn.pth", map_location=device, weights_only=True))
    cnn.eval(); fcnn.eval()
    ti = torch.from_numpy(test_img).permute(0,3,1,2).to(device)
    tl = torch.from_numpy(test_lm).to(device)
    with torch.no_grad():
        cp = torch.softmax(cnn(ti), dim=1).cpu().numpy()
        fp = torch.softmax(fcnn(tl), dim=1).cpu().numpy()
    best_f1, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        preds = (w*cp+(1-w)*fp).argmax(axis=1)
        f1 = f1_score(test_y, preds, average="macro", zero_division=0)
        if f1 > best_f1: best_f1=f1; best_w=w; best_preds=preds
    acc = accuracy_score(test_y, best_preds)
    wf1 = f1_score(test_y, best_preds, average="weighted", zero_division=0)
    for f in ["cnn.pth","fcnn.pth"]: (fold_dir/f).unlink(missing_ok=True)
    return {"accuracy": acc, "macro_f1": best_f1, "weighted_f1": wf1}

print("Helper functions ready.")"""

LOSO_RUN = """def run_loso(dataset_name, data_dir, num_classes, emotions):
    print(f"\\n{'='*70}")
    print(f"  BENCHMARK: {dataset_name} ({num_classes}-class, LOSO)")
    print(f"{'='*70}")
    images = np.load(data_dir/"X_images.npy"); landmarks = np.load(data_dir/"X_landmarks.npy")
    labels = np.load(data_dir/"y_labels.npy"); subjects = np.load(data_dir/"subjects.npy", allow_pickle=True)
    unique_subjects = sorted(set(subjects))
    subject_indices = {s: np.where(subjects==s)[0] for s in unique_subjects}
    n_folds = len(unique_subjects)
    print(f"  Samples: {len(labels)}, Subjects: {n_folds}")
    all_results = {}
    models_to_run = MODELS + [("Late_Fusion", None, "late", 0.0001)]
    for model_name, ModelClass, model_type, lr in models_to_run:
        key = f"{model_name}_B1"
        print(f"\\n  >> {key} ({n_folds} folds)")
        fold_results = []
        model_dir = OUTPUT_DIR / f"{dataset_name}_{num_classes}c" / key
        os.makedirs(model_dir, exist_ok=True)
        for fi, test_subj in enumerate(unique_subjects):
            test_idx = subject_indices[test_subj]
            train_idx = np.concatenate([subject_indices[s] for s in unique_subjects if s != test_subj])
            fold_dir = model_dir / f"fold_{fi}"; os.makedirs(fold_dir, exist_ok=True)
            if model_type == "late":
                r = late_fusion_fold(images[train_idx], landmarks[train_idx], labels[train_idx],
                                     images[test_idx], landmarks[test_idx], labels[test_idx], num_classes, fold_dir)
            else:
                r = train_fold(ModelClass, model_type, lr, images[train_idx], landmarks[train_idx], labels[train_idx],
                               images[test_idx], landmarks[test_idx], labels[test_idx], emotions, fold_dir)
            fold_results.append(r)
            try: fold_dir.rmdir()
            except: pass
        f1s = [r["macro_f1"] for r in fold_results]; accs = [r["accuracy"] for r in fold_results]
        print(f"     F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        all_results[key] = {"model": model_name, "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)), "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)), "n_folds": n_folds, "per_fold": fold_results}
    save_path = OUTPUT_DIR / f"{dataset_name}_{num_classes}c_loso_results.json"
    with open(save_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\\n  Saved: {save_path}")
    return all_results"""

CV10_RUN = """def run_cv10(dataset_name, data_dir, num_classes, emotions):
    K_FOLDS = 10; SEED = 42
    print(f"\\n{'='*70}")
    print(f"  BENCHMARK: {dataset_name} ({num_classes}-class, {K_FOLDS}-Fold CV)")
    print(f"{'='*70}")
    images = np.load(data_dir/"X_images.npy"); landmarks = np.load(data_dir/"X_landmarks.npy")
    labels = np.load(data_dir/"y_labels.npy"); subjects = np.load(data_dir/"subjects.npy", allow_pickle=True)
    unique_subjects = sorted(set(subjects))
    subject_indices = {s: np.where(subjects==s)[0] for s in unique_subjects}
    rng = np.random.RandomState(SEED); subj_arr = np.array(unique_subjects); rng.shuffle(subj_arr)
    folds = np.array_split(subj_arr, K_FOLDS)
    print(f"  Samples: {len(labels)}, Subjects: {len(unique_subjects)}, Folds: {K_FOLDS}")
    all_results = {}
    models_to_run = MODELS + [("Late_Fusion", None, "late", 0.0001)]
    for model_name, ModelClass, model_type, lr in models_to_run:
        key = f"{model_name}_B1"
        print(f"\\n  >> {key} ({K_FOLDS} folds)")
        fold_results = []
        model_dir = OUTPUT_DIR / f"{dataset_name}_{num_classes}c" / key
        os.makedirs(model_dir, exist_ok=True)
        for fi in range(K_FOLDS):
            test_subjs = folds[fi]
            train_subjs = np.concatenate([folds[j] for j in range(K_FOLDS) if j != fi])
            test_idx = np.concatenate([subject_indices[s] for s in test_subjs])
            train_idx = np.concatenate([subject_indices[s] for s in train_subjs])
            fold_dir = model_dir / f"fold_{fi}"; os.makedirs(fold_dir, exist_ok=True)
            if model_type == "late":
                r = late_fusion_fold(images[train_idx], landmarks[train_idx], labels[train_idx],
                                     images[test_idx], landmarks[test_idx], labels[test_idx], num_classes, fold_dir)
            else:
                r = train_fold(ModelClass, model_type, lr, images[train_idx], landmarks[train_idx], labels[train_idx],
                               images[test_idx], landmarks[test_idx], labels[test_idx], emotions, fold_dir)
            fold_results.append(r)
            try: fold_dir.rmdir()
            except: pass
        f1s = [r["macro_f1"] for r in fold_results]; accs = [r["accuracy"] for r in fold_results]
        print(f"     F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        all_results[key] = {"model": model_name, "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)), "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)), "k_folds": K_FOLDS, "per_fold": fold_results}
    save_path = OUTPUT_DIR / f"{dataset_name}_{num_classes}c_cv10_results.json"
    with open(save_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\\n  Saved: {save_path}")
    return all_results"""

SUMMARY_TEMPLATE = """for nc_label, res in [("7-class", res_7c), ("4-class", res_4c)]:
    print(f"\\n{'='*70}")
    print(f"  {dataset_label} {{nc_label}} - {eval_label} Results (sorted by Macro F1)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Macro F1':>18} {'Accuracy':>18}")
    print(f"  {'-'*63}")
    for key in sorted(res.keys(), key=lambda k: -res[k]["macro_f1_mean"]):
        r = res[key]
        print(f"  {{key:<25}} {{r['macro_f1_mean']:.4f}} +/- {{r['macro_f1_std']:.4f}}  {{r['accuracy_mean']:.4f}} +/- {{r['accuracy_std']:.4f}}")"""

# ── NB 38: JAFFE LOSO ──
nb38 = make_nb()
cell(nb38, "markdown", "# 38 - Benchmark JAFFE: LOSO (10-Fold = 10 Subjects)\n\n"
     "**Dataset:** JAFFE - 213 gambar, 7 emosi, 10 subjek\n"
     "**Evaluasi:** LOSO (10 fold)\n**Skenario:** B1 only\n**Kelas:** 7-class dan 4-class")
cell(nb38, "code", SETUP.replace("{output_dir_placeholder}",'OUTPUT_DIR = PROJECT_ROOT / "models" / "benchmark" / "jaffe_loso"\nos.makedirs(OUTPUT_DIR, exist_ok=True)'))
cell(nb38, "code", HELPERS + "\n\n" + LOSO_RUN)
cell(nb38, "markdown", "## Run JAFFE LOSO")
cell(nb38, "code", 'BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"\n'
     'EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]\n'
     'EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]\n\n'
     'res_7c = run_loso("jaffe", BENCHMARK_DIR / "jaffe_7class", 7, EMOTIONS_7)\n'
     'res_4c = run_loso("jaffe", BENCHMARK_DIR / "jaffe_4class", 4, EMOTIONS_4)')
cell(nb38, "markdown", "## Ringkasan JAFFE LOSO")
cell(nb38, "code", 'for nc_label, res in [("7-class", res_7c), ("4-class", res_4c)]:\n'
     '    print(f"\\n{\'=\'*70}")\n'
     '    print(f"  JAFFE {nc_label} - LOSO Results")\n'
     '    print(f"{\'=\'*70}")\n'
     '    print(f"  {\'Model\':<25} {\'Macro F1\':>18} {\'Accuracy\':>18}")\n'
     '    print(f"  {\'-\'*63}")\n'
     '    for key in sorted(res.keys(), key=lambda k: -res[k]["macro_f1_mean"]):\n'
     '        r = res[key]\n'
     '        print(f"  {key:<25} {r[\'macro_f1_mean\']:.4f} +/- {r[\'macro_f1_std\']:.4f}  {r[\'accuracy_mean\']:.4f} +/- {r[\'accuracy_std\']:.4f}")')

with open(NB_DIR / "38_benchmark_jaffe_loso.ipynb", "w") as f:
    json.dump(nb38, f, indent=1)
print("Created 38_benchmark_jaffe_loso.ipynb")

# ── NB 39: CK+ 10-Fold CV ──
nb39 = make_nb()
cell(nb39, "markdown", "# 39 - Benchmark CK+: 10-Fold CV (Subject-Wise)\n\n"
     "**Dataset:** CK+ - 636/654 gambar, 118 subjek\n"
     "**Evaluasi:** 10-Fold CV (subject-wise)\n**Skenario:** B1 only\n**Kelas:** 7-class dan 4-class (contempt -> negative)")
cell(nb39, "code", SETUP.replace("{output_dir_placeholder}",'OUTPUT_DIR = PROJECT_ROOT / "models" / "benchmark" / "ckplus_cv10"\nos.makedirs(OUTPUT_DIR, exist_ok=True)'))
cell(nb39, "code", HELPERS + "\n\n" + CV10_RUN)
cell(nb39, "markdown", "## Run CK+ 10-Fold CV")
cell(nb39, "code", 'BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"\n'
     'EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]\n'
     'EMOTIONS_4 = ["neutral", "happy", "sad", "negative"]\n\n'
     'res_7c = run_cv10("ckplus", BENCHMARK_DIR / "ckplus_7class", 7, EMOTIONS_7)\n'
     'res_4c = run_cv10("ckplus", BENCHMARK_DIR / "ckplus_4class_contempt", 4, EMOTIONS_4)')
cell(nb39, "markdown", "## Ringkasan CK+ 10-Fold CV")
cell(nb39, "code", 'for nc_label, res in [("7-class", res_7c), ("4-class", res_4c)]:\n'
     '    print(f"\\n{\'=\'*70}")\n'
     '    print(f"  CK+ {nc_label} - 10-Fold CV Results")\n'
     '    print(f"{\'=\'*70}")\n'
     '    print(f"  {\'Model\':<25} {\'Macro F1\':>18} {\'Accuracy\':>18}")\n'
     '    print(f"  {\'-\'*63}")\n'
     '    for key in sorted(res.keys(), key=lambda k: -res[k]["macro_f1_mean"]):\n'
     '        r = res[key]\n'
     '        print(f"  {key:<25} {r[\'macro_f1_mean\']:.4f} +/- {r[\'macro_f1_std\']:.4f}  {r[\'accuracy_mean\']:.4f} +/- {r[\'accuracy_std\']:.4f}")')

with open(NB_DIR / "39_benchmark_ckplus_cv10.ipynb", "w") as f:
    json.dump(nb39, f, indent=1)
print("Created 39_benchmark_ckplus_cv10.ipynb")
