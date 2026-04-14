"""
Training Utilities
===================
Shared functions for training, evaluation, and visualization.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)


# ============== LOSS FUNCTIONS ==============

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for imbalanced classification.

    Reduces loss for well-classified samples, focuses on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default=2.0). Higher = more focus on hard samples.
        alpha: class weights (tensor). If None, no class weighting.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============== DATASET ==============

class EmotionImageDataset(Dataset):
    """Dataset untuk CNN (images only)."""

    def __init__(self, images_npy, labels_npy):
        self.images = np.load(images_npy)  # (N, 224, 224, 3) float32
        self.labels = np.load(labels_npy)  # (N,) int

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # HWC -> CHW for PyTorch
        img = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)
        label = torch.LongTensor([self.labels[idx]])[0]
        return img, label


class EmotionLandmarkDataset(Dataset):
    """Dataset untuk FCNN (landmarks only)."""

    def __init__(self, landmarks_npy, labels_npy):
        self.landmarks = np.load(landmarks_npy)  # (N, 136) float32
        self.labels = np.load(labels_npy)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lm = torch.FloatTensor(self.landmarks[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return lm, label


class EmotionMultimodalDataset(Dataset):
    """Dataset untuk Fusion (images + landmarks)."""

    def __init__(self, images_npy, landmarks_npy, labels_npy):
        self.images = np.load(images_npy)
        self.landmarks = np.load(landmarks_npy)
        self.labels = np.load(labels_npy)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)
        lm = torch.FloatTensor(self.landmarks[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return img, lm, label


# ============== TRAINING ==============

def get_class_weights(dataset_dir, device="cpu"):
    """Load class weights dari JSON."""
    path = Path(dataset_dir) / "class_weights.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        weights = torch.FloatTensor(data["weights_array"]).to(device)
        return weights
    return None


def train_one_epoch(model, loader, criterion, optimizer, device, model_type="cnn"):
    """Train satu epoch. model_type: 'cnn', 'fcnn', atau 'fusion'."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        if model_type == "fusion":
            images, landmarks, labels = batch
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(images, landmarks)
        elif model_type == "cnn":
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        else:  # fcnn
            landmarks, labels = batch
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(landmarks)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_type="cnn"):
    """Evaluate model. Returns loss, accuracy, all predictions, all labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        if model_type == "fusion":
            images, landmarks, labels = batch
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(images, landmarks)
        elif model_type == "cnn":
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        else:
            landmarks, labels = batch
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(landmarks)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, model_type="cnn", epochs=50, patience=15,
                save_path="best_model.pth"):
    """Full training loop with early stopping."""
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    best_epoch = 0

    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Val F1':>8} {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, model_type
        )
        val_loss, val_acc, val_preds, val_labels, _ = evaluate(
            model, val_loader, criterion, device, model_type
        )
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(f"{epoch:>6d} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f} {val_f1:>8.4f} {current_lr:>10.6f}"
              f"  ({elapsed:.1f}s)")

        # Save best model (by macro F1 for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
            break

    print(f"\nBest: epoch {best_epoch}, val_acc={best_val_acc:.4f}, val_f1={best_val_f1:.4f}")
    print(f"Model saved: {save_path}")

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history, best_epoch


# ============== EVALUATION ==============

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def full_evaluation(model, test_loader, criterion, device, model_type="cnn", emotions=None):
    """Complete evaluation with classification report and confusion matrix."""
    if emotions is None:
        emotions = EMOTIONS

    test_loss, test_acc, preds, labels, probs = evaluate(
        model, test_loader, criterion, device, model_type
    )

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print()
    print("Classification Report:")
    class_labels = list(range(len(emotions)))
    print(classification_report(labels, preds, labels=class_labels, target_names=emotions, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=class_labels)

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
        "confusion_matrix": cm,
    }
    return results


# ============== VISUALIZATION ==============

def plot_training_history(history, title="Training History"):
    """Plot training & validation loss/accuracy curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix", emotions=None):
    """Plot confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if emotions is None:
        emotions = EMOTIONS

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=emotions, yticklabels=emotions, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig


def plot_per_class_f1(results_dict, title="Per-Class F1 Score Comparison", emotions=None):
    """Plot F1 score per emosi untuk beberapa model."""
    import matplotlib.pyplot as plt

    if emotions is None:
        emotions = EMOTIONS

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(emotions))
    width = 0.8 / len(results_dict)

    for i, (name, results) in enumerate(results_dict.items()):
        class_labels = list(range(len(emotions)))
        report = classification_report(
            results["labels"], results["predictions"],
            labels=class_labels, target_names=emotions, output_dict=True, zero_division=0
        )
        f1_scores = [report[emo]["f1-score"] for emo in emotions]
        ax.bar(x + i * width, f1_scores, width, label=name, alpha=0.8)

    ax.set_xticks(x + width * (len(results_dict) - 1) / 2)
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    return fig
