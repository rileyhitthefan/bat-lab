"""
This file handles configuration for the entire project.

All hyperparameters and audio processing settings are stored in a YAML file
(configs/default.yaml) and loaded into a Config object here.

The goal of this file is to avoid hard-coding values like sample rate, number of mel
bins, learning rate, or number of epochs directly into the code. This makes the project
easier to modify, debug, and reproduce.

This file also ensures that important directories (model checkpoints and cache folders)
exist before training starts.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass(frozen=True)
class Config:
    sr: int
    mono: bool
    duration_s: float
    hop_length: int
    n_fft: int
    n_mels: int
    fmin: int
    fmax: int

    batch_size: int
    num_workers: int
    lr: float
    epochs: int

    model_dir: str
    cache_dir: str
    numfeat_cache_dir: str
    manifest_csv: str
    thresholds_yaml: str
    default_min_conf: float
    seed: int

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        p = Path(path)
        data: Dict[str, Any] = yaml.safe_load(p.read_text())
        # clamp fmax to Nyquist-1000 like your notebook
        data["fmax"] = min(int(data["fmax"]), int(data["sr"]//2 - 1000))
        return Config(**data)

def ensure_dirs(cfg: Config, project_root: Path) -> None:
    for rel in [cfg.model_dir, cfg.cache_dir, cfg.numfeat_cache_dir]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)
