import random
from pathlib import Path

import numpy as np
import torch


def get_app_root() -> Path:
    r"""Return the absolute path to the root directory of the application.

    Returns:
        Path: resolved absolute path three levels up from the current file.
    """
    return Path(__file__).parent.parent.parent.resolve()


def get_available_device() -> torch.device:
    r"""Return the best available torch device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    r"""Set the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed: seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
