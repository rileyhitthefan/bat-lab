"""
App-facing classification: takes Streamlit uploads, builds a temporary dataset,
runs ML inference (same pipeline as MLModel2 / training), returns (known_df, unknown_df).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, ensure_dirs
from .utils.device import get_device
from .training.inference import predict_file


def _empty_results():
    return (
        pd.DataFrame(columns=["Filename", "Species Prediction", "Confidence Level"]),
        pd.DataFrame(columns=["Filename"]),
    )


def _build_upload_cfg_dict(cfg: Config, tmpdir: Path) -> dict:
    """
    Build a config dict for inference when processing uploads.
    Uses cache dirs inside the temp directory so project cache is not polluted.
    """
    cache_dir = tmpdir / "cache_mels"
    numfeat_cache_dir = tmpdir / "cache_numfeats"
    cache_dir.mkdir(exist_ok=True)
    numfeat_cache_dir.mkdir(exist_ok=True)
    d = {
        "sr": cfg.sr,
        "mono": cfg.mono,
        "duration_s": cfg.duration_s,
        "hop_length": cfg.hop_length,
        "n_fft": cfg.n_fft,
        "n_mels": cfg.n_mels,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
        "cache_dir": str(cache_dir),
        "numfeat_cache_dir": str(numfeat_cache_dir),
        "default_min_conf": cfg.default_min_conf,
    }
    return d


def classify_uploaded_files(
    uploaded_files: list[Any],
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify uploaded audio files using the trained ML model (same pipeline as MLModel2).

    Builds a temporary dataset: writes uploads to a temp dir, computes mel spectrograms
    and numeric features (cached in temp), runs model inference, applies confidence
    thresholds. Returns identified species and unknown results.

    Args:
        uploaded_files: Streamlit file uploader list (file-like objects with .name).
        model_path: Path to .pt checkpoint. Default: from config model_dir/best_model.pt
        config_path: Path to config YAML. Default: configs/default.yaml
        thresholds_path: Path to thresholds YAML. Default: from config thresholds_yaml

    Returns:
        (known_df, unknown_df) with columns Filename, Species Prediction, Confidence Level.
    """
    if not uploaded_files:
        return _empty_results()

    project_root = Path(__file__).resolve().parents[2]
    config_path = config_path or project_root / "configs" / "default.yaml"
    cfg = Config.from_yaml(config_path)
    ensure_dirs(cfg, project_root)

    # Default paths from config (same as training output: scripts/train.py → best_model.pt)
    model_path = model_path or (project_root / cfg.model_dir / "best_model2.pt")
    thresholds_path = thresholds_path or (project_root / cfg.thresholds_yaml)

    if not Path(model_path).exists():
        # No trained model: return all uploads as unknown (no placeholder data)
        print(f"[BatLab] Model not found at {model_path}; all {len(uploaded_files)} file(s) marked unknown.")
        unknown = [{"Filename": f.name} for f in uploaded_files]
        return (
            pd.DataFrame(columns=["Filename", "Species Prediction", "Confidence Level"]),
            pd.DataFrame(unknown),
        )

    print(f"[BatLab] Classifying {len(uploaded_files)} file(s) with model {model_path}")
    device = get_device()
    location = "unknown"  # model will use first known location if "unknown" not in meta

    known_results = []
    unknown_results = []

    with tempfile.TemporaryDirectory(prefix="batlab_upload_") as tmpdir:
        root_dir = str(Path(tmpdir).resolve())
        tmpdir_path = Path(tmpdir)
        upload_cfg = _build_upload_cfg_dict(cfg, tmpdir_path)

        # Build upload "dataset": one wav path per file, then run inference (same as MLModel2)
        for file in uploaded_files:
            filename = file.name
            if not filename.lower().endswith(".wav"):
                print(f"[BatLab] {filename}: skipped (not .wav)")
                unknown_results.append({"Filename": filename})
                continue

            wav_path = tmpdir_path / filename
            wav_path.write_bytes(file.getvalue())
            abs_path = str(wav_path.resolve())

            try:
                out = predict_file(
                    wav_path=abs_path,
                    location=location,
                    model_path=str(model_path),
                    thresholds_yaml=str(thresholds_path),
                    root_dir=root_dir,
                    cfg=upload_cfg,
                    device=device,
                )
            except Exception as e:
                print(f"[BatLab] {filename}: ERROR - {e}")
                unknown_results.append({"Filename": filename})
                continue

            prob_pct = f"{out['prob'] * 100:.2f}%"
            top2 = out.get("top2", [])
            top2_str = " | ".join(f"{l}={p*100:.1f}%" for l, p in top2) if top2 else ""
            if out["is_unknown"]:
                print(f"[BatLab] {filename}: unknown (top prob={out['prob']:.3f} < min_conf={out['min_conf']:.3f})")
                print(f"[BatLab]   top2: {top2_str}")
                unknown_results.append({"Filename": filename})
            else:
                print(f"[BatLab] {filename}: {out['label']} @ {prob_pct}")
                if top2_str:
                    print(f"[BatLab]   top2: {top2_str}")
                known_results.append({
                    "Filename": filename,
                    "Species Prediction": out["label"],
                    "Confidence Level": prob_pct,
                })

    # Debug summary to terminal
    known_df = pd.DataFrame(known_results)
    unknown_df = pd.DataFrame(unknown_results)
    print("[BatLab] --- classification summary ---")
    print(f"[BatLab] identified: {len(known_df)} | unknown: {len(unknown_df)}")
    if not known_df.empty:
        print(known_df.to_string(index=False))
    if not unknown_df.empty:
        print("Unknown:", list(unknown_df["Filename"]))
    print("[BatLab] ------------------------------")

    return known_df, unknown_df
