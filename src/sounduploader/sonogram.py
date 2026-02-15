from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

def gen_mag_spectrogram_fixed(x: np.ndarray, fs: float, ms: float, overlap_perc: float):
    """Compute the magnitude spectrogram with proper normalization."""
    x = x.astype(np.float64)
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)
    freqs, times, Sxx = signal.spectrogram(
        x, fs=fs, window='hann', nperseg=nfft, noverlap=noverlap, scaling='density'
    )
    mag_spec = np.flipud(Sxx)
    return mag_spec, nfft


def gen_spectrogram_fixed(
    audio_samples: np.ndarray,
    sampling_rate: int,
    fft_win_length: float,
    fft_overlap: float,
    crop_spec: bool = True,
    max_freq: int = 256,
    min_freq: int = 0,
) -> np.ndarray:
    """Generate a log-scaled magnitude spectrogram."""
    spec, _ = gen_mag_spectrogram_fixed(audio_samples, sampling_rate, fft_win_length, fft_overlap)
    spec_log = np.log10(spec + 1e-10)
    if crop_spec:
        spec_log = spec_log[min_freq:max_freq, :]
    return spec_log


def gen_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap):
    """Alias for file_upload compatibility."""
    return gen_spectrogram_fixed(audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec=True, max_freq=256, min_freq=0)


def denoise_aggressive(spec_noisy: np.ndarray, percentile: float = 25) -> np.ndarray:
    noise_floor = np.percentile(spec_noisy, percentile, axis=1, keepdims=True)
    spec_denoise = spec_noisy - noise_floor
    return np.clip(spec_denoise, 0, None)


def process_spectrogram_fixed(
    spec: np.ndarray,
    denoise_spec: bool = True,
    smooth_spec: bool = True,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """Denoise, smooth, and enhance spectrogram."""
    if denoise_spec:
        spec = denoise_aggressive(spec, percentile=20)
    if smooth_spec:
        spec = filters.gaussian(spec, sigma=0.5)
    if enhance_contrast:
        from skimage import morphology
        spec = filters.median(spec, footprint=morphology.rectangle(3, 1))
    return spec


process_spectrogram = process_spectrogram_fixed  # alias for file_upload


def create_sonogram(filepath: str, include_axes: bool = True, mode: str = 'bat', hp_cutoff: int = 2000) -> str:
    rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if hp_cutoff > 0:
        sos = signal.butter(4, hp_cutoff, 'hp', fs=rate, output='sos')
        data = signal.sosfilt(sos, data)
    if mode == 'bats':
        fft_win, overlap, max_freq, min_freq = 0.005, 0.75, 1000, 100
    else:
        fft_win, overlap, max_freq, min_freq = 0.02, 0.5, 256, 0
    spec = gen_spectrogram_fixed(data, rate, fft_win, overlap, max_freq=max_freq, min_freq=min_freq)
    spec = process_spectrogram_fixed(spec)
    p25, p95 = np.percentile(spec, 25), np.percentile(spec, 95)
    spec_normalized = np.clip((spec - p25) / (p95 - p25 + 1e-10), 0, 1)
    sonogram_path = filepath.replace(".wav", f"_sonogram_{mode}.png")
    if include_axes:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(spec_normalized, aspect="auto", origin="lower", cmap='viridis')
        plt.colorbar(im, ax=ax, label='Normalized Intensity')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Frequency (bins)')
        ax.set_title(f'Spectrogram ({mode} mode)')
        plt.tight_layout()
    else:
        plt.figure()
        plt.imshow(spec_normalized, aspect="auto", origin="lower", cmap='viridis')
        plt.axis("off")
    plt.savefig(sonogram_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()
    return sonogram_path
