import torch
import torch.nn as nn

class SmallAudioCNN(nn.Module):
    def __init__(self, n_classes:int, n_locations:int, loc_embed_dim:int=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.loc_embed = nn.Embedding(num_embeddings=max(1,n_locations), embedding_dim=loc_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(64*4*4 + loc_embed_dim, 256), nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1))  # for calibration
    def forward(self, x, loc_ids):
        z = self.conv(x).flatten(1)             # [B, 1024]
        le = self.loc_embed(loc_ids)            # [B, loc_embed_dim]
        logits = self.fc(torch.cat([z, le], dim=1)) / self.temperature.clamp_min(0.5)
        return logits