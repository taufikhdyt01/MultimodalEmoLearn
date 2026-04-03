"""
Model Architectures for Emotion Recognition
=============================================
CNN, FCNN, Intermediate Fusion (Feature-Level Fusion) - PyTorch implementation.
"""

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    """CNN untuk fitur penampilan (citra wajah 224x224x3).

    4 Conv blocks (32->64->128->256) + 2 FC layers (512->256).
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 256-dim feature vector (untuk fusion)."""
        x = self.features(x)
        return self.classifier(x)


class EmotionFCNN(nn.Module):
    """FCNN untuk fitur geometrik (68 landmarks = 136 fitur).

    5 Dense layers (256->512->512->256->128).
    """

    def __init__(self, input_dim=136, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 128-dim feature vector (untuk fusion)."""
        return self.features(x)


class IntermediateFusion(nn.Module):
    """Intermediate Fusion (Feature-Level Fusion): CNN + FCNN digabung di level fitur.

    Referensi: Boulahia et al. (2021) - "Early, intermediate and late fusion strategies
    for robust deep learning-based multimodal action recognition."

    Image stream (CNN) -> 256-dim feature
    Landmark stream (FCNN) -> 128-dim feature
    Concatenate -> 384-dim -> Dense(512->256) -> 7 classes
    """

    def __init__(self, num_classes=7, landmark_dim=136):
        super().__init__()

        # Image stream (CNN - 3 blocks, lighter than standalone CNN)
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        # Landmark stream
        self.landmark_features = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        # Post-fusion: 256 (image) + 128 (landmark) = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, landmark):
        img_feat = self.image_features(image)
        img_feat = self.image_fc(img_feat)
        lm_feat = self.landmark_features(landmark)
        fused = torch.cat([img_feat, lm_feat], dim=1)
        return self.fusion_head(fused)