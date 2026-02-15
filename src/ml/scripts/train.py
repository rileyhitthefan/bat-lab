"""Train the BatLab model. Requires ml.data.manifest and ml.data.dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..config import Config, ensure_dirs
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..utils.device import get_device
from ..models.net import BatClassifier
from ..training.trainer import train_model
from ..training.calibration import calibrate_temperature
from ..data.manifest import build_manifest_from_dir, load_manifest
from ..data.features import compute_feature_stats, NUMERIC_FEAT_DIM
from ..data.dataset import BatDataset, DatasetContext


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to your 'Bat Calls' root folder")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default=None, help="Optional: existing manifest CSV")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = get_logger()
    project_root = Path(__file__).resolve().parents[3]
    cfg = Config.from_yaml(project_root / args.config)
    ensure_dirs(cfg, project_root)

    device = get_device()
    set_seed(cfg.seed)
    logger.info("Device: %s", device)
    logger.info("Data root: %s", args.data_root)

    manifest_csv = args.manifest or str(project_root / cfg.manifest_csv)
    if args.manifest is None:
        logger.info("Building manifest...")
        df = build_manifest_from_dir(args.data_root, manifest_csv)
    else:
        logger.info("Loading manifest...")
        df = load_manifest(manifest_csv)

    label_counts = df["label"].value_counts()
    keep = label_counts[label_counts >= 3].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    logger.info("Samples after filtering: %d | classes: %d", len(df), df["label"].nunique())

    species_names = sorted(df["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(species_names)}
    location_names = sorted(df["location"].unique())
    loc_to_idx = {name: i for i, name in enumerate(location_names)}

    train_df, test_val_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=cfg.seed
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, stratify=test_val_df["label"], random_state=cfg.seed
    )
    logger.info("Split: train %d | val %d | test %d", len(train_df), len(val_df), len(test_df))

    feat_stats_path = project_root / cfg.model_dir / "feature_stats.json"
    feat_mean, feat_std = compute_feature_stats(
        train_df,
        args.data_root,
        str(project_root / cfg.numfeat_cache_dir),
        cfg,
        str(feat_stats_path),
    )
    logger.info("Saved feature stats: %s", feat_stats_path)

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
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
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

    result = train_model(
        model,
        loaders,
        device,
        cfg.lr,
        cfg.epochs,
        str(project_root / cfg.model_dir),
        meta,
    )
    logger.info("Best checkpoint: %s", result.best_path)

    ckpt = torch.load(result.best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    T = calibrate_temperature(model, loaders["val"], device)
    logger.info("Calibrated temperature: %.3f", T)
    torch.save({"model_state": model.state_dict(), "meta": meta}, result.best_path)


if __name__ == "__main__":
    main()
