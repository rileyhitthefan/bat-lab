import json
import os
import re
from pathlib import Path
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

def parse_date_time_from_filename(filename: str) -> tuple[str, str]:
    """
    Extract date and time from filenames like:
    24F3190166BC0811_20250530_181400T.WAV
    00152_TADAEG_TCUBAT-AS_20250626_033753_000.wav

    Returns:
        (date_str, time_str)
        date_str -> YYYY-MM-DD
        time_str -> HH:MM:SS
    """
    name = Path(filename).name

    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return "", ""

    date_raw, time_raw = m.group(1), m.group(2)
    date_str = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    time_str = f"{time_raw[0:2]}:{time_raw[2:4]}:{time_raw[4:6]}"
    return date_str, time_str


def compute_call_measurements(y: np.ndarray, sr: int, cfg: dict) -> dict:
    """
    Compute export measurements for one extracted call window.

    Returns:
        Fc   -> dominant / brightest frequency in kHz
        Dur  -> duration in ms
        Fmin -> minimum active frequency in kHz
        Fmax -> maximum active frequency in kHz
        Sc   -> approximate slope in kHz/ms
    """
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    fmin_cfg = cfg["fmin"]
    fmax_cfg = cfg["fmax"]

    if y is None or len(y) == 0:
        return {
            "Fc": np.nan,
            "Dur": np.nan,
            "Fmin": np.nan,
            "Fmax": np.nan,
            "Sc": np.nan,
        }

    duration_ms = (len(y) / float(sr)) * 1000.0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times_ms = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=sr, hop_length=hop, n_fft=n_fft
    ) * 1000.0

    band_mask = (freqs >= fmin_cfg) & (freqs <= fmax_cfg)
    if np.any(band_mask):
        S = S[band_mask, :]
        freqs = freqs[band_mask]

    if S.size == 0:
        return {
            "Fc": np.nan,
            "Dur": duration_ms,
            "Fmin": np.nan,
            "Fmax": np.nan,
            "Sc": np.nan,
        }

    power = S ** 2
    mean_spec = power.mean(axis=1)

    fc_hz = float(freqs[int(np.argmax(mean_spec))])

    global_peak = float(np.max(power))
    thr = max(global_peak * 0.20, 1e-12)
    active = power >= thr

    active_rows = np.where(active.any(axis=1))[0]
    if len(active_rows) > 0:
        fmin_hz = float(freqs[active_rows[0]])
        fmax_hz = float(freqs[active_rows[-1]])
    else:
        fmin_hz = np.nan
        fmax_hz = np.nan

    ridge_freqs = []
    ridge_times = []
    for t in range(power.shape[1]):
        col = power[:, t]
        if np.max(col) >= thr:
            ridge_idx = int(np.argmax(col))
            ridge_freqs.append(freqs[ridge_idx] / 1000.0)
            ridge_times.append(times_ms[t])

    if len(ridge_times) >= 2:
        try:
            slope, _ = np.polyfit(ridge_times, ridge_freqs, 1)
            sc_val = float(slope)
        except Exception:
            sc_val = np.nan
    else:
        sc_val = np.nan

    return {
        "Fc": fc_hz / 1000.0,
        "Dur": duration_ms,
        "Fmin": fmin_hz / 1000.0 if not np.isnan(fmin_hz) else np.nan,
        "Fmax": fmax_hz / 1000.0 if not np.isnan(fmax_hz) else np.nan,
        "Sc": sc_val,
    }
