"""Manifest building and loading for BatLab training data.

Matches MLModel2.ipynb: species from first folder in path, location from filename rules.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Recorder ID used for filename-based location (MLModel2.ipynb)
SPECIAL_RECORDER = "24F3190166BC0811"


def infer_species_and_location(path: str, root_dir: str) -> tuple[str, str]:
    """
    Extract species from folder structure and location from filename rules.
    Matches MLModel2.ipynb (Manifest helpers).
    - Species: first path component under root (e.g. root/NYCTHE/... -> NYCTHE).
    - Location: from filename: TCUBAT-AS -> ASTP; SPECIAL_RECORDER + RHICAP/NYCTHE -> CAVE,
      else CSWY; otherwise UNKNOWN.
    """
    path = path.replace("\\", "/")
    root_dir = str(root_dir).replace("\\", "/")

    if path.startswith(root_dir):
        rel_path = path[len(root_dir):].lstrip("/")
    else:
        rel_path = path

    parts = [p for p in rel_path.split("/") if p]

    # Species: first folder
    if len(parts) >= 1:
        species = parts[0]
    else:
        species = "unknown"

    # Location: from filename (not folder)
    fname = os.path.basename(path)
    if "TCUBAT-AS" in fname:
        location = "ASTP"
    elif SPECIAL_RECORDER in fname:
        if species in {"RHICAP", "NYCTHE"}:
            location = "CAVE"
        else:
            location = "CSWY"
    else:
        location = "UNKNOWN"

    return species, location


def build_manifest_from_dir(root_dir: str, out_csv: str, verbose: bool = True) -> pd.DataFrame:
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

    if verbose:
        print("\n=== MANIFEST DIAGNOSTICS ===")
        print(f"Total files found: {len(df)}")
        print("\nSpecies distribution:")
        print(df["label"].value_counts())
        print("\nLocation distribution:")
        print(df["location"].value_counts())
        print("\nSample paths:")
        for i in range(min(3, len(df))):
            print(f"  {df.iloc[i]['filepath']}")
            print(f"  -> Species: {df.iloc[i]['label']}, Location: {df.iloc[i]['location']}")
        print(f"\nWrote {out_csv} with {len(df)} rows")

    return df


def load_manifest(csv_path: str) -> pd.DataFrame:
    """Load manifest CSV. Must have columns filepath, label, location."""
    df = pd.read_csv(csv_path)
    required = {"filepath", "label", "location"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must have columns {required}; got {list(df.columns)}")
    return df
