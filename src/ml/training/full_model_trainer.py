from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import shutil

from src.ml.config import Config


@dataclass
class FullModelTrainingJob:
    model_name: str
    num_examples: int
    output_model_path: str
    metadata_path: str


def retrain_full_model_from_ui(conn=None, call_library_df=None, model_name: str | None = None):
    """
    Placeholder full-model retraining hook.

    Replace the middle section with your real training pipeline.
    Right now this just copies the current base model into a timestamped file
    so the UI flow works end-to-end.
    """
    # File is at src/ml/training/, so repo root is 3 levels up.
    project_root = Path(__file__).resolve().parents[3]
    cfg = Config.from_yaml(project_root / "configs" / "default.yaml")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (model_name or f"full_model_{timestamp}").strip().replace(" ", "_")

    output_dir = project_root / cfg.model_dir / "full_retrained" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = project_root / cfg.model_dir / "colab" / "best_model.pt"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found at {base_model_path}")

    output_model_path = output_dir / f"{safe_name}.pt"
    metadata_path = output_dir / f"{safe_name}_metadata.json"

    # ------------------------------------------------------------------
    # TODO: replace this copy step with real full-model training
    # ------------------------------------------------------------------
    shutil.copy2(base_model_path, output_model_path)

    num_examples = 0
    if call_library_df is not None:
        try:
            num_examples = len(call_library_df)
        except Exception:
            num_examples = 0

    payload = {
        "model_name": safe_name,
        "created_at": timestamp,
        "num_examples": num_examples,
        "type": "full_model_retrain",
        "output_model_path": str(output_model_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return FullModelTrainingJob(
        model_name=safe_name,
        num_examples=num_examples,
        output_model_path=str(output_model_path),
        metadata_path=str(metadata_path),
    )