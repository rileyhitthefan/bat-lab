"""
This file handles device selection.

The get_device function checks whether a GPU is available and returns either
"cuda" or "cpu". This keeps device logic out of the main training and inference
scripts and makes the code easier to read.
"""
def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
