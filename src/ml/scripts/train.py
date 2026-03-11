"""Train bat call classification model from command line."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.ml.config import Config, ensure_dirs
from src.ml.data.dataset import BatDataset, DatasetContext
from src.ml.data.features import NUMERIC_FEAT_DIM, compute_feature_stats
from src.ml.data.manifest import build_manifest_from_dir, load_manifest
from src.ml.models.net import BatClassifier
from src.ml.training.calibration import calibrate_temperature
from src.ml.training.inference import load_best
from src.ml.training.trainer import train_model
from src.ml.utils.device import get_device


def parse_args():
    ap = argparse.ArgumentParser(description="Train bat call classification model")
    ap.add_argument("--data-root", required=True, help="Root directory containing bat call .wav files")
    ap.add_argument("--manifest", help="Path to manifest CSV (will build if not provided)")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    ap.add_argument("--min-samples", type=int, default=3, help="Minimum samples per class to keep")
    ap.add_argument("--test-size", type=float, default=0.3, help="Test+val split size")
    ap.add_argument("--val-size", type=float, default=0.5, help="Val fraction of test+val")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (uses config default if not set)")
    return ap.parse_args()


def build_loaders(
    df: pd.DataFrame,
    root_dir: str,
    cfg: Config,
    project_root: Path,
    min_samples: int = 3,
    test_size: float = 0.3,
    val_size: float = 0.5,
    seed: int = None,
):
    """Build train/val/test loaders with feature normalization."""
    if seed is None:
        seed = cfg.seed

    # Filter classes with insufficient samples
    label_counts = df["label"].value_counts()
    # At minimum, stratified splitting requires at least 2 samples per class.
    effective_min = max(min_samples, 2)
    labels_to_keep = label_counts[label_counts >= effective_min].index
    df_filtered = df[df["label"].isin(labels_to_keep)].copy()

    print(f"\n=== DATA FILTERING ===")
    print(f"Original samples: {len(df)}")
    print(f"Filtered samples: {len(df_filtered)}")
    print(f"Classes kept (>= {effective_min} samples): {len(labels_to_keep)}")
    print(f"Classes removed: {len(df) - len(df_filtered)}")

    if len(df_filtered) == 0:
        raise ValueError("No samples remaining after filtering!")

    # Create mappings
    species_names = sorted(df_filtered["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(species_names)}

    location_names = sorted(df_filtered["location"].unique())
    loc_to_idx = {name: i for i, name in enumerate(location_names)}

    # Split data (stratified by label)
    train_df, test_val_df = train_test_split(
        df_filtered,
        test_size=test_size,
        stratify=df_filtered["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=val_size,
        stratify=test_val_df["label"],
        random_state=seed,
    )

    print(f"\n=== DATA SPLIT ===")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Compute feature normalization stats on training data only
    print(f"\n=== COMPUTING FEATURE STATS ===")
    cache_dir = project_root / cfg.cache_dir
    numfeat_cache_dir = project_root / cfg.numfeat_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    numfeat_cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = {
        "cache_dir": str(cache_dir),
        "numfeat_cache_dir": str(numfeat_cache_dir),
        "sr": cfg.sr,
        "mono": cfg.mono,
        "duration_s": cfg.duration_s,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "n_mels": cfg.n_mels,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
    }

    stats_path = str(project_root / cfg.model_dir / "feature_stats.json")
    feat_mean, feat_std = compute_feature_stats(
        train_df, root_dir, str(numfeat_cache_dir), cfg_dict, stats_path
    )
    print(f"Feature stats saved to {stats_path}")
    print(f"Mean shape: {feat_mean.shape}, Std shape: {feat_std.shape}")

    # Create dataset contexts
    ctx = DatasetContext(
        root_dir=root_dir,
        cache_dir=str(cache_dir),
        numfeat_cache_dir=str(numfeat_cache_dir),
        cfg=cfg,
        label_to_idx=label_to_idx,
        loc_to_idx=loc_to_idx,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    # Create datasets
    train_ds = BatDataset(train_df, ctx)
    val_ds = BatDataset(val_df, ctx)
    test_ds = BatDataset(test_df, ctx)

    # On Windows, num_workers > 0 often causes OSError (invalid handle) when spawning.
    num_workers = 0 if sys.platform == "win32" else cfg.num_workers

    # Create data loaders
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # Build metadata
    meta = {
        "species": species_names,
        "label_to_idx": label_to_idx,
        "locations": location_names,
        "loc_to_idx": loc_to_idx,
        "numeric_feat_dim": NUMERIC_FEAT_DIM,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
    }

    return loaders, meta


def evaluate_model(model, loader, device: str) -> float:
    """Evaluate model accuracy on a dataset."""
    from src.ml.models.net import accuracy

    model.eval()
    total_acc, total_n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x, y, loc, num_feats, _ = batch
                x = x.to(device)
            elif len(batch) == 6:
                x_full, x_call, y, loc, num_feats, _ = batch
                x = x_full.to(device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            y, loc, num_feats = y.to(device), loc.to(device), num_feats.to(device)
            logits = model(x, loc, num_feats)
            bs = x.size(0)
            total_acc += accuracy(logits, y) * bs
            total_n += bs
    return total_acc / total_n if total_n > 0 else 0.0


def train_from_manifest_df(
    df: pd.DataFrame,
    data_root: str = "",
    config: str = "configs/default.yaml",
    min_samples: int = 3,
    test_size: float = 0.3,
    val_size: float = 0.5,
    seed: int | None = None,
) -> dict:
    """
    Run the full training pipeline given an in-memory manifest DataFrame.

    Args:
        df: DataFrame with columns ["filepath", "label", "location"].
        data_root: Root directory for audio files (used only for cache key construction).
        config: Relative path to the config YAML from project root.
        min_samples: Minimum samples per class to keep.
        test_size: Test+val split size.
        val_size: Fraction of test+val set used for validation.
        seed: Optional random seed (falls back to config.seed if None).

    Returns:
        Dictionary with keys: best_path, test_acc, temperature, history.
    """
    project_root = Path(__file__).resolve().parents[3]

    # Load config and ensure directories
    config_path = project_root / config
    cfg = Config.from_yaml(config_path)
    ensure_dirs(cfg, project_root)

    # Set random seed
    seed_val = seed if seed is not None else cfg.seed
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # Persist manifest so other parts of the app (e.g. species list) can reuse it
    manifest_path = project_root / cfg.manifest_csv
    df.to_csv(manifest_path, index=False)

    # Build data loaders
    loaders, meta = build_loaders(
        df,
        data_root,
        cfg,
        project_root,
        min_samples=min_samples,
        test_size=test_size,
        val_size=val_size,
        seed=seed_val,
    )

    # Create model
    device = get_device()
    print(f"\n=== MODEL ===")
    print(f"Device: {device}")
    print(f"Classes: {len(meta['species'])}")
    print(f"Locations: {len(meta['locations'])}")
    print(f"Numeric features: {meta['numeric_feat_dim']}")

    model = BatClassifier(
        n_classes=len(meta["species"]),
        n_locations=len(meta["locations"]),
        num_feat_dim=meta["numeric_feat_dim"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    print(f"\n=== TRAINING ===")
    model_dir = project_root / cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    result = train_model(
        model,
        loaders,
        device,
        lr=cfg.lr,
        epochs=cfg.epochs,
        model_dir=str(model_dir),
        meta=meta,
    )

    print(f"\nBest checkpoint: {result.best_path}")
    print(f"Training history: {len(result.history)} epochs")

    # Visualize training history and save plot next to the model checkpoint.
    try:
        import matplotlib.pyplot as plt

        history = result.history
        if history:
            epochs = [h["epoch"] for h in history]
            train_loss = [h["train"]["loss"] for h in history]
            val_loss = [h["val"]["loss"] for h in history]
            train_acc = [h["train"]["acc"] for h in history]
            val_acc = [h["val"]["acc"] for h in history]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax_loss = axes[0]
            ax_loss.plot(epochs, train_loss, label="Train loss")
            ax_loss.plot(epochs, val_loss, label="Val loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Loss")
            ax_loss.legend()

            ax_acc = axes[1]
            ax_acc.plot(epochs, train_acc, label="Train acc")
            ax_acc.plot(epochs, val_acc, label="Val acc")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title("Accuracy")
            ax_acc.legend()

            plt.tight_layout()
            plot_path = model_dir / "training_history.png"
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Saved training history plot to {plot_path}")
        else:
            print("No training history to plot.")
    except Exception as e:
        print(f"Could not visualize training history: {e}")

    # Evaluate on test set
    print(f"\n=== EVALUATION (TEST SET) ===")
    best_model, _meta = load_best(result.best_path, device)
    test_acc = evaluate_model(best_model, loaders["test"], device)
    print(f"Test accuracy: {test_acc:.4f}")

    # Temperature scaling calibration
    print(f"\n=== CALIBRATION (TEMPERATURE SCALING) ===")
    temperature = calibrate_temperature(best_model, loaders["val"], device)
    calibrated_model = best_model
    print(f"Optimal temperature: {temperature:.4f}")

    # Save calibrated model checkpoint (overwrite best path)
    print(f"\n=== SAVING CALIBRATED MODEL ===")
    torch.save(
        {
            "model_state": calibrated_model.state_dict(),
            "meta": _meta,
            "temperature": temperature,
        },
        result.best_path,
    )
    print(f"Saved calibrated model with temperature={temperature:.4f} to {result.best_path}")

    return {
        "best_path": str(result.best_path),
        "test_acc": float(test_acc),
        "temperature": float(temperature),
        "history": result.history,
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    # Load config
    config_path = project_root / args.config
    cfg = Config.from_yaml(config_path)
    ensure_dirs(cfg, project_root)

    # Build or load manifest
    if args.manifest:
        manifest_path = args.manifest
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        print(f"Loading manifest from {manifest_path}...")
        df = load_manifest(manifest_path)
    else:
        manifest_path = project_root / cfg.manifest_csv
        print(f"Building manifest from {args.data_root}...")
        df = build_manifest_from_dir(args.data_root, str(manifest_path))
        print(f"Manifest saved to {manifest_path}")

    print(f"\n=== MANIFEST SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Species: {df['label'].nunique()} ({sorted(df['label'].unique())})")
    print(f"Locations: {df['location'].nunique()} ({sorted(df['location'].unique())})")

    # Delegate to shared training routine
    train_from_manifest_df(
        df=df,
        data_root=args.data_root,
        config=args.config,
        min_samples=args.min_samples,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

