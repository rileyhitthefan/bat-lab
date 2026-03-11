import os
import librosa
import numpy as np
from .audio import extract_single_call_window

def load_wav_fixed(path, sr, mono, duration_s):
    y, s = librosa.load(path, sr=sr, mono=mono)
    target_len = int(sr*duration_s)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len-len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y, sr

def wav_to_mel(y, sr, n_fft, hop_length, n_mels, fmin, fmax):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)
    return S_norm.astype(np.float32)

def cache_mel_full(path: str, root_dir_param: str, cfg: dict):
    rel_path = path.replace(root_dir_param, "").replace("/", "_").replace("\\", "_")
    key = rel_path.replace(".wav", f"_FULL_{cfg['sr']}sr_{cfg['n_mels']}mels.npy")
    out_path = os.path.join(cfg["cache_dir"], key)
    if os.path.exists(out_path):
        return out_path
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    y, sr = load_wav_fixed(path, cfg["sr"], cfg["mono"], cfg["duration_s"])
    mel = wav_to_mel(y, sr, cfg["n_fft"], cfg["hop_length"], cfg["n_mels"], cfg["fmin"], cfg["fmax"])
    target_frames = int(np.ceil(cfg["sr"] * cfg["duration_s"] / cfg["hop_length"]))
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode='constant', constant_values=0.0)
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    np.save(out_path, mel)
    return out_path

def cache_mel_call(path: str, root_dir_param: str, cfg: dict):
    rel_path = path.replace(root_dir_param, "").replace("/", "_").replace("\\", "_")
    key = rel_path.replace(".wav", f"_CALL_{cfg['sr']}sr_{cfg['n_mels']}mels.npy")
    out_path = os.path.join(cfg["cache_dir"], key)
    if os.path.exists(out_path):
        return out_path
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    y_full, sr = librosa.load(path, sr=cfg["sr"], mono=cfg["mono"])
    y_call, sr = extract_single_call_window(y_full, sr, min_freq=cfg["fmin"], max_freq=cfg["fmax"])
    mel = wav_to_mel(y_call, sr, cfg["n_fft"], cfg["hop_length"], cfg["n_mels"], cfg["fmin"], cfg["fmax"])
    target_frames = int(np.ceil(cfg["sr"] * cfg["duration_s"] / cfg["hop_length"]))
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode='constant', constant_values=0.0)
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    np.save(out_path, mel)
    return out_path

def cache_mel(path: str, root_dir_param: str, cfg: dict):
    """Cache mel spectrogram (matches training notebook)."""
    rel_path = path.replace(root_dir_param, "").replace("/", "_").replace("\\", "_")
    key = rel_path.replace(".wav", f"_{cfg['sr']}sr_{cfg['n_mels']}mels.npy")
    out_path = os.path.join(cfg["cache_dir"], key)
    if os.path.exists(out_path):
        return out_path
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    y, sr = load_wav_fixed(path, cfg["sr"], cfg["mono"], cfg["duration_s"])
    mel = wav_to_mel(y, sr, cfg["n_fft"], cfg["hop_length"], cfg["n_mels"], cfg["fmin"], cfg["fmax"])
    np.save(out_path, mel)
    return out_path
