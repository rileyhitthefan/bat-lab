import librosa
import numpy as np
from scipy.signal import medfilt

def extract_single_call_window(
    y,
    sr,
    min_freq=20000,
    max_freq=90000,
    context_ms=3.0,
    n_fft=2048,
    hop_length=128,
):
    """
    Returns (y_call, sr) cropped around the strongest detected bat call.
    """

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.effects.preemphasis(y, coef=0.97)

    # STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(band):
        return y, sr  # fallback

    band_energy = S[band, :].max(axis=0)
    band_energy = medfilt(band_energy, kernel_size=7)

    peak = band_energy.max()
    med = np.median(band_energy)
    thr = med + 0.35 * (peak - med)

    mask = band_energy >= thr
    if not np.any(mask):
        center = int(np.argmax(band_energy))
        start_f = max(0, center - 5)
        end_f = min(S.shape[1] - 1, center + 5)
    else:
        idx = np.where(mask)[0]
        start_f, end_f = idx[0], idx[-1]

    pad = int((context_ms / 1000) * sr / hop_length)
    start_f = max(0, start_f - pad)
    end_f = min(S.shape[1] - 1, end_f + pad)

    start_samp = int(start_f * hop_length)
    end_samp = min(len(y), int((end_f + 1) * hop_length))

    return y[start_samp:end_samp], sr
