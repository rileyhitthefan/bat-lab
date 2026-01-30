"""
This file is used to run inference on a single audio file after a model has already
been trained.

I run this script when I want to give the model one .wav file and get back a predicted
bat species, along with a confidence score and an "unknown" decision if the confidence
is too low.

This script:
- Loads the same configuration used during training
- Loads a trained model checkpoint
- Generates the same features used during training (mel spectrograms + numeric features)
- Runs the model forward pass
- Applies confidence thresholds to determine whether the prediction should be trusted

Like train.py, this file mostly connects pieces together rather than implementing
machine learning logic directly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from batlab.config import Config, ensure_dirs
from batlab.utils.device import get_device
from batlab.training.inference import predict_file

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--location", default="unknown")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    return ap.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
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
