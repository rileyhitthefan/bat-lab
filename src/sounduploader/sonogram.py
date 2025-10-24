from skimage import filters
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom


def denoise(spec_noisy: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Denoise a spectrogram by subtracting the mean energy per frequency band.

    Parameters
    ----------
    spec_noisy : np.ndarray
        Noisy spectrogram.
    mask : np.ndarray | None
        Boolean mask for selecting relevant time steps (optional).

    Returns
    -------
    np.ndarray
        Denoised spectrogram with negative values clipped to zero.
    """
    if mask is None:
        me = np.mean(spec_noisy, 1)
        spec_denoise = spec_noisy - me[:, np.newaxis]
    else:
        mask_inv = np.invert(mask)
        spec_denoise = spec_noisy.copy()
        if np.sum(mask) > 0:
            me = np.mean(spec_denoise[:, mask], 1)
            spec_denoise[:, mask] -= me[:, np.newaxis]
        if np.sum(mask_inv) > 0:
            me_inv = np.mean(spec_denoise[:, mask_inv], 1)
            spec_denoise[:, mask_inv] -= me_inv[:, np.newaxis]
    spec_denoise.clip(min=0, out=spec_denoise)
    return spec_denoise


def gen_mag_spectrogram(x: np.ndarray, fs: float, ms: float, overlap_perc: float) -> tuple[np.ndarray, int]:
    """
    Compute the magnitude spectrogram.

    Parameters
    ----------
    x : np.ndarray
        Audio samples.
    fs : float
        Sampling rate (Hz).
    ms : float
        Window length in seconds.
    overlap_perc : float
        Fraction of overlap between windows (0–1).

    Returns
    -------
    tuple[np.ndarray, int]
        Magnitude spectrogram (freq × time) and number of FFT windows.
    """
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1] - noverlap) // step)
    strides = (x.strides[0], step * x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins
    complex_spec = np.fft.rfft(x_wins_han, axis=0)
    mag_spec = (np.conjugate(complex_spec) * complex_spec).real
    spec = np.flipud(mag_spec[1:, :])
    return spec, x_wins.shape[0]


def gen_spectrogram(
    audio_samples: np.ndarray,
    sampling_rate: int,
    fft_win_length: float,
    fft_overlap: float,
    crop_spec: bool = True,
    max_freq: int = 256,
    min_freq: int = 0,
) -> np.ndarray:
    """
    Generate a log-scaled magnitude spectrogram.

    Parameters
    ----------
    audio_samples : np.ndarray
        Raw audio waveform.
    sampling_rate : int
        Sampling rate (Hz).
    fft_win_length : float
        FFT window size (s).
    fft_overlap : float
        Overlap percentage between windows (0–1).
    crop_spec : bool, default=True
        Whether to apply a band-pass crop.
    max_freq : int, default=256
        Max frequency band index.
    min_freq : int, default=0
        Min frequency band index.

    Returns
    -------
    np.ndarray
        Log-scaled magnitude spectrogram.
    """
    spec, x_win_len = gen_mag_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap)
    if crop_spec:
        freq = abs(np.fft.rfftfreq(x_win_len) * sampling_rate)
        freq = np.flip(freq)
        spec = spec[-max_freq:-min_freq, :]
        req_height = max_freq - min_freq
        if spec.shape[0] < req_height:
            pad = np.zeros((req_height - spec.shape[0], spec.shape[1]))
            spec = np.vstack((pad, spec))
    log_scaling = 2.0 * (1.0 / sampling_rate) * (
        1.0 / (np.abs(np.hanning(int(fft_win_length * sampling_rate))) ** 2).sum()
    )
    return np.log(1.0 + log_scaling * spec)


def process_spectrogram(
    spec: np.ndarray,
    denoise_spec: bool = True,
    mean_log_mag: float = 0.5,
    smooth_spec: bool = True,
) -> np.ndarray:
    """
    Optionally denoise and smooth a log-magnitude spectrogram.

    Parameters
    ----------
    spec : np.ndarray
        Input spectrogram.
    denoise_spec : bool, default=True
        Apply mean-based denoising.
    mean_log_mag : float, default=0.5
        Threshold for silence masking.
    smooth_spec : bool, default=True
        Apply Gaussian smoothing.

    Returns
    -------
    np.ndarray
        Cleaned and smoothed spectrogram.
    """
    if denoise_spec:
        mask = spec.mean(0) > mean_log_mag
        spec = denoise(spec, mask)
    if smooth_spec:
        spec = filters.gaussian(spec, 1.0)
    return spec
