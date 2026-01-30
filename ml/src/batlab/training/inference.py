"""
This file contains all logic related to loading a trained model and running inference.

It reconstructs the model using saved metadata, generates the same features used
during training, and applies confidence thresholds to determine whether a prediction
should be trusted or marked as unknown.

This separation ensures inference behavior matches training exactly.
"""
from __future__ import annotations
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from ..models.net import BatClassifier
from ..data.spectrograms import cache_mel_full, cache_mel_call
from ..data.audio import extract_single_call_window
from ..data.features import compute_numeric_features_from_y

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
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt["meta"]
    model = BatClassifier(
        n_classes=len(meta["species"]),
        n_locations=len(meta["locations"]),
        num_feat_dim=meta["numeric_feat_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta

@torch.no_grad()
def predict_file(wav_path: str, location: str, model_path: str, thresholds_yaml: str, root_dir: str, cfg, device: str):
    model, meta = load_best(model_path, device)
    thresholds = load_thresholds(thresholds_yaml, cfg.default_min_conf)

    loc_to_idx = meta["loc_to_idx"]
    if location not in loc_to_idx:
        location = meta["locations"][0]
    loc_idx = torch.tensor([loc_to_idx[location]], dtype=torch.long)

    # spectrograms
    mel_full = np.load(cache_mel_full(wav_path, root_dir, cfg.cache_dir, cfg))
    mel_call = np.load(cache_mel_call(wav_path, root_dir, cfg.cache_dir, cfg))
    x_full = torch.tensor(mel_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x_call = torch.tensor(mel_call, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # numeric features
    y_full, sr = librosa.load(wav_path, sr=cfg.sr, mono=cfg.mono)
    y_call, _ = extract_single_call_window(y_full, sr, min_freq=cfg.fmin, max_freq=cfg.fmax)
    num_feats = compute_numeric_features_from_y(y_call, sr, cfg)

    feat_mean = np.array(meta["feat_mean"], dtype=np.float32)
    feat_std  = np.array(meta["feat_std"], dtype=np.float32)
    num_feats = (num_feats - feat_mean) / (feat_std + 1e-8)
    num_feats = torch.tensor(num_feats, dtype=torch.float32).unsqueeze(0)

    logits = model(x_full.to(device), x_call.to(device), loc_idx.to(device), num_feats.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))

    idx_to_label = {v: k for k, v in meta["label_to_idx"].items()}
    top_label = idx_to_label[top_idx]
    p = float(probs[top_idx])

    min_conf = get_min_conf(thresholds, top_label, location, cfg.default_min_conf)
    return {"label": top_label, "prob": p, "location": location, "is_unknown": p < min_conf, "min_conf": min_conf}
