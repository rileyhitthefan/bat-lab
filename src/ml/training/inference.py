"""
This file contains all logic related to loading a trained model and running inference.

It reconstructs the model using saved metadata, generates the same features used
during training, and applies confidence thresholds to determine whether a prediction
should be trusted or marked as unknown.

This separation ensures inference behavior matches training exactly.
"""
from __future__ import annotations
import os
import sys
from dataclasses import asdict
from typing import Union

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import librosa

from ..models.net import BatClassifier
from ..data.spectrograms import cache_mel_full, cache_mel_call
from ..data.audio import extract_single_call_window
from ..data.features import compute_numeric_features_from_y


def _cfg_to_dict(cfg) -> dict:
    """Normalize Config or dict to a dict for cache/feature code (sr, cache_dir, n_fft, etc.)."""
    if isinstance(cfg, dict):
        return dict(cfg)
    return asdict(cfg)


def _cfg_attr(cfg, key: str, default=None):
    """Get attribute or key from Config or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def load_thresholds(path: str, default_min_conf: float) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            th = yaml.safe_load(f) or {}
    else:
        th = {}
    th["_default_min_conf"] = th.get("_default_min_conf", default_min_conf)
    return th

def get_min_conf(thresholds: dict, species: str, location: str, default_min_conf: float) -> float:
    per = thresholds.get("per", {})
    if species in per and location in per[species]:
        return float(per[species][location])
    return float(thresholds.get("_default_min_conf", default_min_conf))

def load_best(model_path: str, device: str):
    """Load checkpoint saved with BatClassifier (MLModel2 / scripts/train.py)."""
    # Checkpoints saved from notebooks/scripts reference BatClassifier from __main__.
    # When Streamlit runs, __main__ is app.py, so register our class so unpickling works.
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "BatClassifier"):
        setattr(main, "BatClassifier", BatClassifier)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Handle (model, meta) tuple (common in MLModel2 notebook)
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 2:
        model = ckpt[0].to(device)
        meta = ckpt[1]
    elif isinstance(ckpt, dict):
        if "meta" not in ckpt:
            raise ValueError(
                "Checkpoint dict must contain 'meta'. Train with: python scripts/train.py --data-root <path>"
            )
        meta = ckpt["meta"]
        if "model_state" in ckpt:
            model = BatClassifier(
                n_classes=len(meta["species"]),
                n_locations=len(meta["locations"]),
                num_feat_dim=meta["numeric_feat_dim"],
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
        elif "model" in ckpt:
            model = ckpt["model"].to(device)
        else:
            raise ValueError("Checkpoint must contain 'model_state' or 'model'.")
    elif isinstance(ckpt, torch.nn.Module):
        raise ValueError(
            "Checkpoint file contains only the model (no metadata). "
            "The app needs 'meta' (label_to_idx, feat_mean, locations, etc.). "
            "Re-save from your notebook/script as: "
            "torch.save({'model_state': model.state_dict(), 'meta': meta}, path) "
            "or train with: python scripts/train.py --data-root <path>"
        )
    else:
        raise ValueError(
            f"Checkpoint format not supported (got {type(ckpt).__name__}). "
            "Use a dict with 'model_state' and 'meta', or a (model, meta) tuple. "
            "Train with: python scripts/train.py --data-root <path>"
        )

    model.eval()
    return model, meta

@torch.no_grad()
def predict_file(
    wav_path: str,
    location: str,
    model_path: str,
    thresholds_yaml: str,
    root_dir: str,
    cfg: Union[dict, object],
    device: str,
):
    """
    Run inference on a single wav file using the trained model.
    cfg can be a Config instance or a dict (e.g. with cache_dir/numfeat_cache_dir overrides for temp dirs).
    """
    model, meta = load_best(model_path, device)
    default_min_conf = _cfg_attr(cfg, "default_min_conf", 0.5)
    thresholds = load_thresholds(thresholds_yaml, default_min_conf)
    cfg_dict = _cfg_to_dict(cfg)

    loc_to_idx = meta["loc_to_idx"]
    if location not in loc_to_idx:
        location = meta["locations"][0]
    loc_idx = torch.tensor([loc_to_idx[location]], dtype=torch.long)

    # Spectrograms (same pipeline as MLModel2 / training)
    mel_full = np.load(cache_mel_full(wav_path, root_dir, cfg_dict))
    mel_call = np.load(cache_mel_call(wav_path, root_dir, cfg_dict))
    x_full = torch.tensor(mel_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x_call = torch.tensor(mel_call, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Numeric features (normalized like training)
    sr = _cfg_attr(cfg, "sr")
    mono = _cfg_attr(cfg, "mono")
    fmin = _cfg_attr(cfg, "fmin")
    fmax = _cfg_attr(cfg, "fmax")
    y_full, sr = librosa.load(wav_path, sr=sr, mono=mono)
    y_call, _ = extract_single_call_window(y_full, sr, min_freq=fmin, max_freq=fmax)
    num_feats = compute_numeric_features_from_y(y_call, sr, cfg_dict)

    feat_mean = np.array(meta["feat_mean"], dtype=np.float32)
    feat_std  = np.array(meta["feat_std"], dtype=np.float32)
    num_feats = (num_feats - feat_mean) / (feat_std + 1e-8)
    num_feats = torch.tensor(num_feats, dtype=torch.float32).unsqueeze(0)

    logits = model(x_full.to(device), x_call.to(device), loc_idx.to(device), num_feats.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top2_idx = np.argsort(probs)[-2:][::-1]

    idx_to_label = {v: k for k, v in meta["label_to_idx"].items()}
    top_label = idx_to_label[top_idx]
    p = float(probs[top_idx])
    top2 = [(idx_to_label[i], float(probs[i])) for i in top2_idx]

    min_conf = get_min_conf(thresholds, top_label, location, default_min_conf)
    return {
        "label": top_label,
        "prob": p,
        "location": location,
        "is_unknown": p < min_conf,
        "min_conf": min_conf,
        "top2": top2,
    }
