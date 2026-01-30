"""
This file contains helper functions for reproducibility.

The set_seed function sets random seeds for Python, NumPy, and PyTorch so that
experiments are as deterministic as possible. This helps ensure that results can
be reproduced when the model is retrained.

This is especially important for debugging and for comparing model versions.
"""
import random
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
