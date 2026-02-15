import json
import os
from typing import Union

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .spectrograms import load_wav_fixed
from .audio import extract_single_call_window

NUMERIC_FEAT_NAMES = [
    "duration_sec",
    "rms_mean",
    "rms_std",
    "hf_centroid_hz",
    "hf_bandwidth_hz",
    "hf_peak_freq_hz",
    "zcr_mean",
    "zcr_std",
    # NEW pulse-timing features:
    "pulse_count",
    "pulse_rate_hz",
    "mean_pulse_width_ms",
    "mean_ipi_ms",
    "duty_cycle",
]
NUMERIC_FEAT_DIM = len(NUMERIC_FEAT_NAMES)


def compute_pulse_timing_features(y: np.ndarray, sr: int, cfg: dict):
    """
    Compute pulse-count and timing features from call audio.
    Returns (pulse_count, pulse_rate_hz, mean_pulse_width_ms, mean_ipi_ms, duty_cycle).
    """
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    duration_sec = len(y) / float(sr)
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = np.sqrt(np.mean(S ** 2, axis=0) + 1e-8)

    med = np.median(rms)
    peak_thr = med + 0.2 * (np.max(rms) - med)
    above = rms >= peak_thr
    edges = np.diff(np.concatenate([[0], above.astype(np.float32), [0]]))
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    pulse_count = float(len(run_starts))
    pulse_rate_hz = pulse_count / duration_sec

    if len(run_starts) > 0 and len(run_ends) > 0:
        n = min(len(run_starts), len(run_ends))
        widths_frames = run_ends[:n] - run_starts[:n]
        widths_ms = (widths_frames * hop / sr) * 1000.0
        mean_pulse_width_ms = float(np.mean(widths_ms))
        if n > 1:
            gaps_frames = run_starts[1:n] - run_ends[: n - 1]
            gaps_ms = (gaps_frames * hop / sr) * 1000.0
            mean_ipi_ms = float(np.mean(gaps_ms))
        else:
            mean_ipi_ms = 0.0
    else:
        mean_pulse_width_ms = 0.0
        mean_ipi_ms = 0.0

    duty_cycle = float(np.sum(above) / len(above)) if len(above) > 0 else 0.0

    return (
        pulse_count,
        pulse_rate_hz,
        mean_pulse_width_ms,
        mean_ipi_ms,
        duty_cycle,
    )


def compute_numeric_features_from_y(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Compute numeric descriptors of the call."""
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    fmin = cfg["fmin"]
    fmax = cfg["fmax"]

    duration_sec = len(y) / float(sr)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if band_mask.sum() == 0:
        S_band = S
        freqs_band = freqs
    else:
        S_band = S[band_mask, :]
        freqs_band = freqs[band_mask]

    power = S_band ** 2 + 1e-8
    mean_spec = power.mean(axis=1)
    spec_sum = mean_spec.sum()

    hf_centroid = float((freqs_band * mean_spec).sum() / spec_sum)
    hf_peak_freq = float(freqs_band[int(np.argmax(mean_spec))])
    hf_bandwidth = float(
        np.sqrt(((freqs_band - hf_centroid) ** 2 * mean_spec).sum() / spec_sum)
    )

    rms = np.sqrt(np.mean(S_band**2, axis=0))
    rms_mean = float(rms.mean())
    rms_std = float(rms.std() + 1e-8)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    zcr_mean = float(zcr.mean())
    zcr_std = float(zcr.std() + 1e-8)

    pulse_count, pulse_rate_hz, mean_width_ms, mean_ipi_ms, duty_cycle = compute_pulse_timing_features(y, sr, cfg)

    feats = np.array(
        [duration_sec, rms_mean, rms_std, hf_centroid, hf_bandwidth,
          hf_peak_freq, zcr_mean, zcr_std,
          pulse_count, pulse_rate_hz, mean_width_ms, mean_ipi_ms, duty_cycle],
        dtype=np.float32,
    )
    return feats


def cache_numeric_features(path: str, root_dir_param: str, cfg: dict) -> str:
    """Compute and cache numeric features for a wav file. Uses call window (matches inference)."""
    rel_path = path.replace(root_dir_param, "").replace("/", "_").replace("\\", "_")
    key = rel_path.replace(".wav", f"_{cfg['sr']}sr_numfeat.npy")
    out_path = os.path.join(cfg["numfeat_cache_dir"], key)

    if os.path.exists(out_path):
        return out_path

    os.makedirs(cfg["numfeat_cache_dir"], exist_ok=True)
    y_full, sr = librosa.load(path, sr=cfg["sr"], mono=cfg["mono"])
    y_call, _ = extract_single_call_window(
        y_full, sr, min_freq=cfg["fmin"], max_freq=cfg["fmax"],
        n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
    )
    feats = compute_numeric_features_from_y(y_call, sr, cfg)
    np.save(out_path, feats.astype(np.float32))
    return out_path


def _cfg_to_dict(cfg: Union[dict, object], numfeat_cache_dir: str) -> dict:
    """Build a dict for cache_numeric_features from Config or dict."""
    if isinstance(cfg, dict):
        return {**cfg, "numfeat_cache_dir": numfeat_cache_dir}
    return {
        "numfeat_cache_dir": numfeat_cache_dir,
        "sr": cfg.sr,
        "mono": cfg.mono,
        "duration_s": cfg.duration_s,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
    }


def compute_feature_stats(
    df: pd.DataFrame,
    root_dir: str,
    numfeat_cache_dir: str,
    cfg: Union[dict, object],
    stats_path: str,
):
    """Compute mean and std for numeric features across training data; save to stats_path. Returns (mean, std)."""
    cfg_dict = _cfg_to_dict(cfg, numfeat_cache_dir)
    all_feats = []

    for _, row in df.iterrows():
        feat_path = cache_numeric_features(row["filepath"], root_dir, cfg_dict)
        feats = np.load(feat_path)
        all_feats.append(feats)

    all_feats = np.stack(all_feats, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_feats)

    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}, f, indent=2)

    return scaler.mean_, scaler.scale_
