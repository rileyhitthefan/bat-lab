from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]   # ml/
MODELS_DIR = BASE_DIR / "models" / "custom"
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
    Pull just the training examples needed for the custom model.

    ASSUMED TABLE SHAPE:
        training_files(
            detector_id VARCHAR,
            species_code VARCHAR,
            file_path VARCHAR,
            is_verified BOOLEAN
        )

    Adjust the SQL if your schema uses different table/column names.
    """
    rows: List[TrainingExample] = []
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT detector_id, species_code, file_path
        FROM training_files
        WHERE detector_id = %s
          AND species_code IN ({placeholders})
          AND is_verified = 1
    """

    for detector_id, species_codes in detector_species_map.items():
        if not species_codes:
            continue

        placeholders = ",".join(["%s"] * len(species_codes))
        final_query = query.format(placeholders=placeholders)
        params = [detector_id, *species_codes]

        cursor.execute(final_query, params)
        result = cursor.fetchall()

        for row in result:
            rows.append(
                TrainingExample(
                    detector_id=row["detector_id"],
                    species_code=row["species_code"],
                    file_path=row["file_path"],
                )
            )

    cursor.close()
    return rows


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
    Replace this function body with your actual PyTorch training code.

    Intended flow:
      1. Load your base model checkpoint.
      2. Replace / resize final classifier layer to len(label_map).
      3. Freeze early layers if using transfer learning.
      4. Train only upper layers (or partially unfreeze later).
      5. Save subset checkpoint separately.

    For now this creates a placeholder JSON artifact so the pipeline works end-to-end.
    """
    artifact = {
        "status": "placeholder_saved",
        "base_model_path": base_model_path,
        "output_model_path": output_model_path,
        "num_examples": len(examples),
        "num_classes": len(label_map),
        "classes": label_map,
    }

    # Temporary placeholder output so your UI/db integration can be tested first
    with open(output_model_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact


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
    conn,
    train_selections: Dict[str, Dict[str, Any]],
    base_model_path: str,
) -> SubsetTrainingJob:
    """
    Main function your Streamlit UI should call.
    """
    detector_species_map = normalize_train_selections(train_selections)

    if not detector_species_map:
        raise ValueError("No detectors/species were selected for training.")

    examples = fetch_subset_training_examples(conn, detector_species_map)

    ok, message = validate_examples(examples)
    if not ok:
        raise ValueError(message)

    label_map = build_label_map(examples)
    model_name = make_model_name(detector_species_map)

    output_model_path = MODELS_DIR / f"{model_name}.json"   # change to .pt when real training is added

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