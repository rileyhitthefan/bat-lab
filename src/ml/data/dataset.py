"""PyTorch Dataset for bat call classification (mel + numeric features)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .spectrograms import cache_mel_full, cache_mel_call, cache_mel
from .features import cache_numeric_features


def _ctx_to_cfg_dict(ctx: "DatasetContext") -> dict:
    """Build a dict for cache_mel_full / cache_numeric_features from DatasetContext."""
    c = ctx.cfg
    return {
        "cache_dir": ctx.cache_dir,
        "numfeat_cache_dir": ctx.numfeat_cache_dir,
        "sr": c.sr,
        "mono": c.mono,
        "duration_s": c.duration_s,
        "n_fft": c.n_fft,
        "hop_length": c.hop_length,
        "n_mels": c.n_mels,
        "fmin": c.fmin,
        "fmax": c.fmax,
    }


@dataclass
class DatasetContext:
    """Shared context for BatDataset: paths, config, label/loc mappings, feature normalization."""
    root_dir: str
    cache_dir: str
    numfeat_cache_dir: str
    cfg: object  # Config from ml.config
    label_to_idx: Dict[str, int]
    loc_to_idx: Dict[str, int]
    feat_mean: np.ndarray
    feat_std: np.ndarray


class BatDataset(Dataset):
    """Dataset that returns (x, label_idx, loc_idx, numeric_feats, filepath) per sample.
    
    Uses single mel spectrogram (matches training notebook architecture).
    """

    def __init__(self, df: pd.DataFrame, ctx: DatasetContext):
        self.df = df.reset_index(drop=True)
        self.ctx = ctx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        cfg_dict = _ctx_to_cfg_dict(self.ctx)

        # Mel spectrogram (single channel, matches training)
        mel_path = cache_mel(row.filepath, self.ctx.root_dir, cfg_dict)
        mel = np.load(mel_path)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]

        # Labels
        y = torch.tensor(self.ctx.label_to_idx[row.label], dtype=torch.long)
        loc = torch.tensor(self.ctx.loc_to_idx.get(row.location, 0), dtype=torch.long)

        # Numeric features (normalized)
        num_path = cache_numeric_features(row.filepath, self.ctx.root_dir, cfg_dict)
        num_feats = np.load(num_path)
        num_feats = (num_feats - self.ctx.feat_mean) / (self.ctx.feat_std + 1e-8)
        num_feats = torch.tensor(num_feats, dtype=torch.float32)

        return x, y, loc, num_feats, row.filepath
