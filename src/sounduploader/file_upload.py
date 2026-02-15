import os
import numpy as np
from scipy.io import wavfile
from .sonogram import gen_spectrogram, process_spectrogram


def is_wav(filename: str) -> bool:
    """Check if the uploaded file has a .wav extension."""
    return filename.lower().endswith(".wav")


def save_file(file, upload_dir: str = "sounduploader/static/uploads") -> str:
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, file.filename)
    file.save(path)
    return path


def create_sonogram(filepath: str) -> str:
    rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    spec = gen_spectrogram(data, rate, 0.02, 0.5)
    spec = process_spectrogram(spec)
    sonogram_path = filepath.replace(".wav", "_sonogram.png")
    import matplotlib.pyplot as plt
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.axis("off")
    plt.savefig(sonogram_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return sonogram_path
