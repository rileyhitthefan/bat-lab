from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.ml.config import Config, ensure_dirs
from src.ml.scripts.train import build_loaders
from src.ml.training.inference import load_best
from src.ml.training.calibration import calibrate_temperature
from src.ml.training.trainer import train_model
from src.db.connection import get_call_library_data


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3] / "model_checkpoints" / "local"
MODELS_DIR = BASE_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------

@dataclass
class TrainingExample:
    detector_id: str
    species_code: str
    file_path: str


@dataclass
class SubsetTrainingJob:
    model_name: str
    created_at: str
    selected_detectors: List[str]
    selected_species_by_detector: Dict[str, List[str]]
    num_examples: int
    base_model_path: str
    output_model_path: str
    metadata_path: str


# -----------------------------------------------------------------------------
# UI SELECTION PARSING
# -----------------------------------------------------------------------------

def normalize_train_selections(train_selections: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Convert Streamlit session_state['train_selections'] into:
        {
            "DET-001": ["MYLU", "EPFU"],
            "DET-002": ["MYLU"]
        }

    Ignores detectors that were not selected or have no species.
    """
    normalized: Dict[str, List[str]] = {}

    for detector_id, info in train_selections.items():
        if not info.get("selected", False):
            continue

        species_list = info.get("species", [])
        if species_list:
            normalized[detector_id] = species_list

    return normalized


# -----------------------------------------------------------------------------
# DATABASE ACCESS
# -----------------------------------------------------------------------------

def fetch_subset_training_examples(conn, detector_species_map: Dict[str, List[str]]) -> List[TrainingExample]:
    """
    Pull just the training examples needed for the custom model, using the
    same Call_Library materialisation logic as src.db.connection.get_call_library_data.

    Ignores the `conn` argument (kept for backward compatibility) and builds
    examples from the Call_Library blobs materialised to disk.
    """
    call_df = get_call_library_data()
    if call_df is None or getattr(call_df, "empty", True):
        return []

    det_allow = set(detector_species_map.keys())
    examples: List[TrainingExample] = []
    for row in call_df.itertuples(index=False):
        det = getattr(row, "location")
        sp = getattr(row, "bat")
        fp = getattr(row, "file")
        if det not in det_allow:
            continue
        allowed_species = detector_species_map.get(det) or []
        if sp not in allowed_species:
            continue
        examples.append(
            TrainingExample(
                detector_id=str(det),
                species_code=str(sp),
                file_path=str(fp),
            )
        )

    return examples


# -----------------------------------------------------------------------------
# DATASET BUILDING
# -----------------------------------------------------------------------------

def build_label_map(examples: List[TrainingExample]) -> Dict[str, int]:
    """
    Build a contiguous label map for the subset species only.
    Example:
        {"MYLU": 0, "EPFU": 1, "LABO": 2}
    """
    species = sorted({ex.species_code for ex in examples})
    return {sp: idx for idx, sp in enumerate(species)}


def validate_examples(examples: List[TrainingExample]) -> Tuple[bool, str]:
    if not examples:
        return False, "No matching training files were found for the selected detectors/species."

    missing = [ex.file_path for ex in examples if not os.path.exists(ex.file_path)]
    if missing:
        preview = "\n".join(missing[:10])
        return False, (
            "Some DB records point to files that do not exist on disk.\n"
            f"First missing files:\n{preview}"
        )

    return True, "OK"


# -----------------------------------------------------------------------------
# TRAINING STUB
# -----------------------------------------------------------------------------

def fine_tune_subset_model(
    examples: List[TrainingExample],
    label_map: Dict[str, int],
    base_model_path: str,
    output_model_path: str,
) -> Dict[str, Any]:
    """
    Train a subset model using the existing BatLab training pipeline.

    This mirrors `scripts/train.py::train_from_manifest_df` but runs on an
    in-memory list of `TrainingExample` for a subset of detectors/species.
    """
    project_root = Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------------
    # 1) Build manifest DataFrame from examples
    # ------------------------------------------------------------------
    records = []
    for ex in examples:
        records.append(
            {
                "filepath": ex.file_path,
                "label": ex.species_code,
                "location": ex.detector_id,
            }
        )

    df = pd.DataFrame(records, columns=["filepath", "label", "location"])
    if df.empty:
        raise ValueError("No training examples provided to fine_tune_subset_model.")

    # ------------------------------------------------------------------
    # 2) Load config and ensure directories
    # ------------------------------------------------------------------
    config_path = project_root / "configs" / "default.yaml"
    cfg = Config.from_yaml(config_path)
    ensure_dirs(cfg, project_root)

    # ------------------------------------------------------------------
    # 3) Set random seed (reuse cfg.seed)
    # ------------------------------------------------------------------
    seed_val = cfg.seed
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # ------------------------------------------------------------------
    # 4) Build loaders/metadata for this subset
    # ------------------------------------------------------------------
    data_root = ""  # filepaths in df are already absolute
    loaders, meta = build_loaders(
        df,
        data_root,
        cfg,
        project_root,
        min_samples=3,
        test_size=0.3,
        val_size=0.5,
        seed=seed_val,
    )

    # ------------------------------------------------------------------
    # 5) Load base model checkpoint and adapt classifier for subset
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, _base_meta = load_best(base_model_path, device)

    # Resize final classifier layer to the subset number of classes, keeping
    # prior layers (feature extractor) intact for transfer learning.
    if not hasattr(base_model, "fc") or not isinstance(base_model.fc, nn.Sequential):
        raise ValueError("Base model does not have expected 'fc' Sequential head.")

    last_layer = base_model.fc[-1]
    if not isinstance(last_layer, nn.Linear):
        raise ValueError("Last layer of base model 'fc' head is not nn.Linear.")

    in_features = last_layer.in_features
    num_subset_classes = len(meta["species"])
    base_model.fc[-1] = nn.Linear(in_features, num_subset_classes, bias=True).to(device)

    # Optionally freeze earlier layers to train only classifier head.
    for name, param in base_model.named_parameters():
        # Simple heuristic: only layers inside "fc" are trainable
        param.requires_grad = name.startswith("fc.")

    model = base_model

    # Train and get best checkpoint under MODELS_DIR (local subset directory)
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(
        model=model,
        loaders=loaders,
        device=device,
        lr=cfg.lr,
        epochs=cfg.epochs,
        model_dir=str(model_dir),
        meta=meta,
    )

    # ------------------------------------------------------------------
    # 6) Evaluate + calibrate (temperature scaling), then save calibrated
    #    subset checkpoint to output_model_path (model_checkpoints/local/...).
    # ------------------------------------------------------------------
    best_model, best_meta = load_best(result.best_path, device)

    from src.ml.scripts.train import evaluate_model  # avoid circular import at top

    test_acc = float(evaluate_model(best_model, loaders["test"], device))
    temperature = float(calibrate_temperature(best_model, loaders["val"], device))

    checkpoint = {
        "model_state": best_model.state_dict(),
        "meta": best_meta,
        "temperature": temperature,
    }
    output_model_path = str(output_model_path)
    torch.save(checkpoint, output_model_path)

    return {
        "base_model_path": base_model_path,
        "output_model_path": output_model_path,
        "num_examples": len(examples),
        "num_classes": len(label_map),
        "classes": label_map,
        "test_acc": test_acc,
        "temperature": temperature,
        "history": result.history,
    }

# -----------------------------------------------------------------------------
# MODEL REGISTRY / METADATA
# -----------------------------------------------------------------------------

def make_model_name(detector_species_map: Dict[str, List[str]]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    det_count = len(detector_species_map)
    species_count = len({sp for sps in detector_species_map.values() for sp in sps})
    return f"subset_model_{det_count}det_{species_count}sp_{timestamp}"


def save_training_metadata(
    model_name: str,
    detector_species_map: Dict[str, List[str]],
    examples: List[TrainingExample],
    base_model_path: str,
    output_model_path: str,
) -> str:
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    payload = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "selected_detectors": list(detector_species_map.keys()),
        "selected_species_by_detector": detector_species_map,
        "num_examples": len(examples),
        "base_model_path": base_model_path,
        "output_model_path": str(output_model_path),
        "files": [asdict(ex) for ex in examples],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(metadata_path)


# -----------------------------------------------------------------------------
# MAIN ENTRYPOINT FOR UI
# -----------------------------------------------------------------------------

def create_subset_model_from_ui_selection(
    train_selections: Dict[str, Dict[str, Any]],
    base_model_path: str,
    conn=None,
    call_library_df=None,
) -> SubsetTrainingJob:
    """
    Main function your Streamlit UI should call.

    Data sources: pass `call_library_df`
        with columns `file`, `bat`, `location` (as returned by
        `src.db.connection.get_call_library_data()`), and you may pass `conn=None`
    """
    detector_species_map = normalize_train_selections(train_selections)

    if not detector_species_map:
        raise ValueError("No detectors/species were selected for training.")

    if call_library_df is not None:
        required_cols = {"file", "bat", "location"}
        if getattr(call_library_df, "empty", True):
            raise ValueError("No training calls found in the database.")
        missing_cols = required_cols.difference(set(getattr(call_library_df, "columns", [])))
        if missing_cols:
            raise ValueError(
                f"Call library data is missing required columns: {sorted(missing_cols)}"
            )

        det_allow = set(detector_species_map.keys())
        examples: List[TrainingExample] = []
        for row in call_library_df.itertuples(index=False):
            det = getattr(row, "location")
            sp = getattr(row, "bat")
            fp = getattr(row, "file")
            if det not in det_allow:
                continue
            allowed_species = detector_species_map.get(det) or []
            if sp not in allowed_species:
                continue
            examples.append(
                TrainingExample(
                    detector_id=str(det),
                    species_code=str(sp),
                    file_path=str(fp),
                )
            )
    else:
        if conn is None:
            raise ValueError("No training data source provided (expected call_library_df or conn).")
        examples = fetch_subset_training_examples(conn, detector_species_map)

    ok, message = validate_examples(examples)
    if not ok:
        raise ValueError(message)

    label_map = build_label_map(examples)
    model_name = make_model_name(detector_species_map)

    output_model_path = MODELS_DIR / f"{model_name}.pt" 

    fine_tune_subset_model(
        examples=examples,
        label_map=label_map,
        base_model_path=base_model_path,
        output_model_path=str(output_model_path),
    )

    metadata_path = save_training_metadata(
        model_name=model_name,
        detector_species_map=detector_species_map,
        examples=examples,
        base_model_path=base_model_path,
        output_model_path=str(output_model_path),
    )

    return SubsetTrainingJob(
        model_name=model_name,
        created_at=datetime.now().isoformat(),
        selected_detectors=list(detector_species_map.keys()),
        selected_species_by_detector=detector_species_map,
        num_examples=len(examples),
        base_model_path=base_model_path,
        output_model_path=str(output_model_path),
        metadata_path=metadata_path,
    )