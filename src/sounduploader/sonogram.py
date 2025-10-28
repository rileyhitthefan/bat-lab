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
    step = nfft - noverlap
    
    freqs, times, Sxx = signal.spectrogram(
        x, 
        fs=fs,
        window='hann',
        nperseg=nfft,
        noverlap=noverlap,
        scaling='density'
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
    spec, x_win_len = gen_mag_spectrogram_fixed(audio_samples, sampling_rate, fft_win_length, fft_overlap)
    
    # Apply log scaling
    spec_log = np.log10(spec + 1e-10)
    
    if crop_spec:
        spec_log = spec_log[min_freq:max_freq, :]
    
    return spec_log


def denoise_aggressive(spec_noisy: np.ndarray, percentile: float = 25) -> np.ndarray:
    """
    Aggressive denoising by subtracting noise floor estimated from low percentile.
    """
    # Estimate noise floor from each frequency band
    noise_floor = np.percentile(spec_noisy, percentile, axis=1, keepdims=True)
    spec_denoise = spec_noisy - noise_floor
    spec_denoise = np.clip(spec_denoise, 0, None)
    return spec_denoise


def process_spectrogram_fixed(
    spec: np.ndarray,
    denoise_spec: bool = True,
    smooth_spec: bool = True,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """Denoise, smooth, and enhance spectrogram."""
    if denoise_spec:
        # Aggressive noise floor subtraction
        spec = denoise_aggressive(spec, percentile=20)
    
    if smooth_spec:
        # Light Gaussian smoothing
        spec = filters.gaussian(spec, sigma=0.5)
    
    if enhance_contrast:
        # Morphological operations to enhance isolated peaks (bat calls)
        from skimage import morphology
        # Enhance vertical structures (frequency continuity)
        spec = filters.median(spec, footprint=morphology.rectangle(3, 1))
    
    return spec


def create_sonogram(filepath: str, include_axes: bool = True, mode: str = 'bat', hp_cutoff: int = 2000) -> str:
    """
    Create and save a spectrogram optimized for bat detection.
    
    Parameters
    ----------
    filepath : str
        Path to .wav file
    include_axes : bool
        If True, include frequency/time axes and colorbar
    mode : str
        'standard' - full spectrogram (0-250 freq bins)
        'bats' - optimized for bat echolocation (100-1000 freq bins, 5ms window)
    hp_cutoff : int
        High-pass filter cutoff in Hz (default 2000). Set to 0 to disable.
    """
    rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # High-pass filter to remove low-frequency noise
    if hp_cutoff > 0:
        sos = signal.butter(4, hp_cutoff, 'hp', fs=rate, output='sos')
        data = signal.sosfilt(sos, data)

    # Adjust parameters based on mode
    if mode == 'bats':
        fft_win = 0.005
        overlap = 0.75
        max_freq = 1000
        min_freq = 100
    else:
        fft_win = 0.02
        overlap = 0.5
        max_freq = 256
        min_freq = 0

    # Generate spectrogram
    spec = gen_spectrogram_fixed(data, rate, fft_win, overlap, max_freq=max_freq, min_freq=min_freq)
    spec = process_spectrogram_fixed(spec)

    # Percentile-based normalization
    p25 = np.percentile(spec, 25)
    p95 = np.percentile(spec, 95)
    spec_normalized = np.clip((spec - p25) / (p95 - p25 + 1e-10), 0, 1)

    # Save
    sonogram_path = filepath.replace(".wav", f"_sonogram_{mode}.png")
    
    if include_axes:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(spec_normalized, aspect="auto", origin="lower", cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Intensity')
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