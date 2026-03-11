"""Run inference on a single audio file."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.config import Config, ensure_dirs
from src.ml.utils.device import get_device
from src.ml.training.inference import predict_file


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to .wav file")
    ap.add_argument("--location", default="unknown", help="Location for threshold lookup")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    ap.add_argument("--data-root", required=True, help="Root dir for cache paths")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = Config.from_yaml(project_root / args.config)
    ensure_dirs(cfg, project_root)
    device = get_device()
    out = predict_file(
        wav_path=args.wav,
        location=args.location,
        model_path=args.ckpt,
        thresholds_yaml=str(project_root / args.thresholds),
        root_dir=args.data_root,
        cfg=cfg,
        device=device,
    )
    print(out)


if __name__ == "__main__":
    main()

