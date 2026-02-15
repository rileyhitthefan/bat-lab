"""Manifest building and loading for BatLab training data."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def infer_species_and_location(path: str, root_dir: str) -> tuple[str, str]:
    """Extract species and location from folder structure. Expects root/species/location/file.wav."""
    path = path.replace("\\", "/")
    root_dir = str(root_dir).replace("\\", "/")

    if path.startswith(root_dir):
        rel_path = path[len(root_dir) :].lstrip("/")
    else:
        rel_path = path

    parts = [p for p in rel_path.split("/") if p]

    if len(parts) >= 3:
        species = parts[0]
        location = parts[1]
    elif len(parts) >= 2:
        species = parts[0]
        location = "unknown"
    else:
        species = "unknown"
        location = "unknown"

    return species, location


def build_manifest_from_dir(root_dir: str, out_csv: str) -> pd.DataFrame:
    """Scan root_dir for .wav files and build manifest CSV with filepath, label, location."""
    rows = []
    root_dir = str(root_dir)
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                full = os.path.join(dirpath, fn)
                label, location = infer_species_and_location(full, root_dir)
                rows.append({"filepath": full, "label": label, "location": location})

    df = pd.DataFrame(rows).sort_values("filepath")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_manifest(csv_path: str) -> pd.DataFrame:
    """Load manifest CSV. Must have columns filepath, label, location."""
    df = pd.read_csv(csv_path)
    required = {"filepath", "label", "location"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must have columns {required}; got {list(df.columns)}")
    return df
