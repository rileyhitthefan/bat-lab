"""
This file defines the neural network architecture used for bat species classification.

The model is multimodal and combines:
- CNN features from the full spectrogram
- CNN features from the single-call spectrogram
- a learned embedding for recording location
- a small neural network applied to numeric features

All of these representations are concatenated and passed through fully connected
layers to produce class logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(1)

class BatClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_locations: int,
        num_feat_dim: int,
        loc_embed_dim: int = 16,
        num_feat_hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MelEncoder()
        self.loc_embed = nn.Embedding(max(1, n_locations), loc_embed_dim)
        self.num_fc = nn.Sequential(
            nn.Linear(num_feat_dim, num_feat_hidden),
            nn.BatchNorm1d(num_feat_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        spec_dim = 128 * 4 * 4
        combined_dim = (2 * spec_dim) + loc_embed_dim + num_feat_hidden

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

    def forward(self, x_full, x_call, loc_ids, num_feats):
        z_full = self.encoder(x_full)
        z_call = self.encoder(x_call)
        le = self.loc_embed(loc_ids)
        nf = self.num_fc(num_feats)
        combined = torch.cat([z_full, z_call, le, nf], dim=1)
        logits = self.fc(combined)
        logits = logits / self.temperature.clamp_min(0.5)
        return logits

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()

def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)
