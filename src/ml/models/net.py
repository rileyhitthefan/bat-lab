"""
Unified BatClassifier with mel spectrogram, location, and numeric features.
Matches the architecture used in training notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Classification accuracy (0-1)."""
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()


def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for training."""
    return F.cross_entropy(logits, y)


class BatClassifier(nn.Module):
    """Unified model with mel spec, location, and numeric features."""

    def __init__(
        self,
        n_classes: int,
        n_locations: int,
        num_feat_dim: int,
        loc_embed_dim: int = 16,
        num_feat_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        # CNN for mel spectrogram (single channel input)
        self.conv = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # During "force overfit" mode in the notebook this was set to 0.0
            nn.Dropout2d(0.0),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.0),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.0),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Keep output size stable so the classifier input is easy
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Location embedding
        self.loc_embed = nn.Embedding(
            num_embeddings=max(1, n_locations),
            embedding_dim=loc_embed_dim,
        )

        # Numeric feature encoder
        self.num_fc = nn.Sequential(
            nn.Linear(num_feat_dim, num_feat_hidden),
            nn.BatchNorm1d(num_feat_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Combined classifier
        combined_dim = 256 * 4 * 4 + loc_embed_dim + num_feat_hidden

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        loc_ids: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, 1, n_mels, T]
        z = self.conv(x).flatten(1)
        le = self.loc_embed(loc_ids)
        nf = self.num_fc(num_feats)

        combined = torch.cat([z, le, nf], dim=1)
        logits = self.fc(combined)
        logits = logits / self.temperature.clamp_min(0.5)
        return logits