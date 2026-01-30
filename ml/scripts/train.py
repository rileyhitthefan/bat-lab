"""
This file is the main entry point for training the BatLab machine learning model.
This is the file I run from the terminal when I want to train a new model.

Instead of having all of the machine learning code in one large script, this file
acts more like a controller that connects all of the different pieces of the project
together.

High-level flow of this file:
- Load configuration values (sample rate, mel settings, training hyperparameters)
- Scan the bat_calls directory and build a manifest of all .wav files
- Filter and split the data into train / validation / test sets
- Compute normalization statistics for numeric features
- Create PyTorch datasets and dataloaders
- Initialize the BatClassifier model
- Train the model using early stopping
- Calibrate model confidence using temperature scaling
- Save the best model checkpoint and metadata

This file intentionally contains very little machine learning logic itself.
Almost all of the real work is done inside the batlab package so the code is
cleaner, modular, and easier to understand.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from batlab.config import Config, ensure_dirs
from batlab.utils.logging_utils import get_logger
from batlab.utils.seed import set_seed
from batlab.utils.device import get_device
from batlab.data.manifest import build_manifest_from_dir, load_manifest
from batlab.data.features import compute_feature_stats, NUMERIC_FEAT_DIM
from batlab.data.dataset import BatDataset, DatasetContext
from batlab.models.net import BatClassifier
from batlab.training.trainer import train_model
from batlab.training.calibration import calibrate_temperature

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to your 'Bat Calls' root folder")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default=None, help="Optional: existing manifest CSV")
    return ap.parse_args()

def main():
    args = parse_args()
    logger = get_logger()
    project_root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(project_root / args.config)
    ensure_dirs(cfg, project_root)

    device = get_device()
    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {args.data_root}")

    manifest_csv = args.manifest or str(project_root / cfg.manifest_csv)
    if args.manifest is None:
        logger.info("Building manifest...")
        df = build_manifest_from_dir(args.data_root, manifest_csv)
    else:
        logger.info("Loading manifest...")
        df = load_manifest(manifest_csv)

    # filter labels with >=3
    label_counts = df["label"].value_counts()
    keep = label_counts[label_counts >= 3].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    logger.info(f"Samples after filtering: {len(df)} | classes: {df['label'].nunique()}")

    species_names = sorted(df["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(species_names)}
    location_names = sorted(df["location"].unique())
    loc_to_idx = {name: i for i, name in enumerate(location_names)}

    train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=cfg.seed)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["label"], random_state=cfg.seed)
    logger.info(f"Split: train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    feat_stats_path = project_root / cfg.model_dir / "feature_stats.json"
    feat_mean, feat_std = compute_feature_stats(train_df, args.data_root, str(project_root / cfg.numfeat_cache_dir), cfg, str(feat_stats_path))
    logger.info(f"Saved feature stats: {feat_stats_path}")

    ctx = DatasetContext(
        root_dir=args.data_root,
        cache_dir=str(project_root / cfg.cache_dir),
        numfeat_cache_dir=str(project_root / cfg.numfeat_cache_dir),
        cfg=cfg,
        label_to_idx=label_to_idx,
        loc_to_idx=loc_to_idx,
        feat_mean=np.array(feat_mean, dtype=np.float32),
        feat_std=np.array(feat_std, dtype=np.float32),
    )

    train_ds = BatDataset(train_df, ctx)
    val_ds = BatDataset(val_df, ctx)
    test_ds = BatDataset(test_df, ctx)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True),
    }

    meta = {
        "species": species_names,
        "label_to_idx": label_to_idx,
        "locations": location_names,
        "loc_to_idx": loc_to_idx,
        "numeric_feat_dim": NUMERIC_FEAT_DIM,
        "feat_mean": np.array(feat_mean).tolist(),
        "feat_std": np.array(feat_std).tolist(),
    }

    model = BatClassifier(
        n_classes=len(species_names),
        n_locations=len(location_names),
        num_feat_dim=NUMERIC_FEAT_DIM,
    ).to(device)

    result = train_model(model, loaders, device, cfg.lr, cfg.epochs, str(project_root / cfg.model_dir), meta)
    logger.info(f"Best checkpoint: {result.best_path}")

    # temp calibration
    ckpt = torch.load(result.best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    T = calibrate_temperature(model, loaders["val"], device)
    logger.info(f"Calibrated temperature: {T:.3f}")
    torch.save({"model_state": model.state_dict(), "meta": meta}, result.best_path)

if __name__ == "__main__":
    main()
