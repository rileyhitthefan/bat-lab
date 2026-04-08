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

EXPORT_COLUMNS = [
    "IN FILE",
    "DATE",
    "TIME",
    "AUTO ID*",
    "MATCHING",
    "Fc",
    "Dur",
    "Fmax",
    "Fmin",
    "SC",
]


def _empty_results():
    return (
        pd.DataFrame(columns=["Filename", "Species Prediction", "Confidence Level"]),
        pd.DataFrame(columns=["Filename"]),
        pd.DataFrame(columns=EXPORT_COLUMNS),
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


def _build_colab_upload_cfg_dict(tmpdir: Path) -> dict:
    """
    Build a config dict that matches the MLModel2 Colab notebook settings.

    This is used when running inference with the Colab-trained checkpoint so
    that the feature extraction pipeline (sampling rate, hop length, mel
    parameters, frequency range, and default confidence threshold) matches the
    environment where that model was originally trained and evaluated.
    """
    cache_dir = tmpdir / "cache_mels"
    numfeat_cache_dir = tmpdir / "cache_numfeats"
    cache_dir.mkdir(exist_ok=True)
    numfeat_cache_dir.mkdir(exist_ok=True)
    return {
        # From MLModel2 CONFIG in the notebook
        "sr": 48000,
        "mono": True,
        "duration_s": 1.0,
        "hop_length": 128,
        "n_fft": 2048,
        "n_mels": 96,
        "fmin": 2000,
        "fmax": 90000,
        # Per-call cache dirs inside the temp directory
        "cache_dir": str(cache_dir),
        "numfeat_cache_dir": str(numfeat_cache_dir),
        # Notebook default minimum confidence
        "default_min_conf": 0.70,
    }


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

    # Choose a sensible default checkpoint:
    # - Prefer the local-trained model (model_checkpoints/best_model.pt) if present
    # - Otherwise fall back to the colab-trained model (model_checkpoints/colab/best_model.pt)
    if model_path is None:
        colab_default = project_root / cfg.model_dir / "colab" / "best_model.pt"
        local_default = project_root / cfg.model_dir / "best_model.pt"
        if local_default.exists():
            model_path = local_default
        else:
            model_path = colab_default
    else:
        model_path = Path(model_path)
        
    # Heuristic: treat any checkpoint under a "colab" subdirectory as a Colab-trained model.
    is_colab_model = "colab" in {p.lower() for p in model_path.parts}

    if thresholds_path is None:
        if is_colab_model:
            # Use thresholds next to the Colab checkpoint if present; fall back to config otherwise.
            candidate = model_path.with_name("thresholds.yaml")
            thresholds_path = candidate if candidate.exists() else (project_root / cfg.thresholds_yaml)
        else:
            thresholds_path = project_root / cfg.thresholds_yaml
    else:
        thresholds_path = Path(thresholds_path)

    if not model_path.exists():
        # No trained model: return all uploads as unknown (no placeholder data)
        print(f"[BatLab] Model not found at {model_path}; all {len(uploaded_files)} file(s) marked unknown.")
        unknown = [{"Filename": f.name} for f in uploaded_files]
        export_rows = [
            {
                "IN FILE": f.name,
                "DATE": "",
                "TIME": "",
                "AUTO ID*": "UNKNOWN",
                "MATCHING": "",
                "Fc": "",
                "Dur": "",
                "Fmax": "",
                "Fmin": "",
                "SC": "",
            }
            for f in uploaded_files
        ]
        return (
            pd.DataFrame(columns=["Filename", "Species Prediction", "Confidence Level"]),
            pd.DataFrame(unknown),
            pd.DataFrame(export_rows, columns=EXPORT_COLUMNS),
        )

    print(f"[BatLab] Classifying {len(uploaded_files)} file(s) with model {model_path}")
    device = get_device()
    location = "unknown"  # model will use first known location if "unknown" not in meta

    known_results = []
    unknown_results = []
    export_results = []

    with tempfile.TemporaryDirectory(prefix="batlab_upload_") as tmpdir:
        root_dir = str(Path(tmpdir).resolve())
        tmpdir_path = Path(tmpdir)

        # Match feature pipeline to the origin of the checkpoint.
        if is_colab_model:
            upload_cfg = _build_colab_upload_cfg_dict(tmpdir_path)
        else:
            upload_cfg = _build_upload_cfg_dict(cfg, tmpdir_path)

        # Build upload "dataset": one wav path per file, then run inference (same as MLModel2)
        for file in uploaded_files:
            filename = file.name
            if not filename.lower().endswith(".wav"):
                print(f"[BatLab] {filename}: skipped (not .wav)")
                unknown_results.append({"Filename": filename})
                export_results.append({
                    "IN FILE": filename,
                    "DATE": "",
                    "TIME": "",
                    "AUTO ID*": "UNKNOWN",
                    "MATCHING": "",
                    "Fc": "",
                    "Dur": "",
                    "Fmax": "",
                    "Fmin": "",
                    "SC": "",
                })
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
                # Debug: log raw prediction dict (includes label, prob, top2, etc.)
                print(f"[BatLab] Raw prediction for {filename}: {out}")
            except Exception as e:
                print(f"[BatLab] {filename}: ERROR - {e}")
                unknown_results.append({"Filename": filename})
                export_results.append({
                    "IN FILE": filename,
                    "DATE": "",
                    "TIME": "",
                    "AUTO ID*": "UNKNOWN",
                    "MATCHING": "",
                    "Fc": "",
                    "Dur": "",
                    "Fmax": "",
                    "Fmin": "",
                    "SC": "",
                })
                continue

            prob_pct = f"{out['prob'] * 100:.2f}%"
            top2 = out.get("top2", [])
            top2_str = " | ".join(f"{l}={p*100:.1f}%" for l, p in top2) if top2 else ""
            export_results.append({
                "IN FILE": out["file_name"],
                "DATE": out["date"],
                "TIME": out["time"],
                "AUTO ID*": out["auto_id"],
                "MATCHING": out["matching"],
                "Fc": round(out["Fc"], 3) if pd.notna(out["Fc"]) else "",
                "Dur": round(out["Dur"], 3) if pd.notna(out["Dur"]) else "",
                "Fmax": round(out["Fmax"], 3) if pd.notna(out["Fmax"]) else "",
                "Fmin": round(out["Fmin"], 3) if pd.notna(out["Fmin"]) else "",
                "SC": round(out["Sc"], 6) if pd.notna(out["Sc"]) else "",
            })
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
