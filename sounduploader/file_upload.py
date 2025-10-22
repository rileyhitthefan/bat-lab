import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from .sonogram import gen_spectrogram, process_spectrogram


def is_wav(filename: str) -> bool:
    """Check if the uploaded file has a .wav extension."""
    return filename.lower().endswith(".wav")


def save_file(file, upload_dir: str = "sounduploader/static/uploads") -> str:
    """
    Save uploaded file to the temporary upload directory.

    Parameters
    ----------
    file : werkzeug.FileStorage
        File uploaded from the Flask form.
    upload_dir : str, default='sounduploader/static/uploads'
        Directory to store temporary files.

    Returns
    -------
    str
        Path to saved file.
    """
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, file.filename)
    file.save(path)
    return path


def create_sonogram(filepath: str) -> str:
    """
    Create and save a spectrogram (sonogram) image from a .wav file.

    Parameters
    ----------
    filepath : str
        Path to the .wav file.

    Returns
    -------
    str
        Path to generated sonogram image (.png).
    """
    rate, data = wavfile.read(filepath)
    if data.ndim > 1:  # Convert stereo → mono
        data = np.mean(data, axis=1)

    # Generate and process spectrogram
    spec = gen_spectrogram(data, rate, 0.02, 0.5)
    spec = process_spectrogram(spec)

    # Save sonogram
    sonogram_path = filepath.replace(".wav", "_sonogram.png")
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.axis("off")
    plt.savefig(sonogram_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return sonogram_path
