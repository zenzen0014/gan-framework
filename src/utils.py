import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Membuat eksperimen lebih reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """Menentukan device training."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_config == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi GPU tidak tersedia.")

    return torch.device(device_config)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Menyimpan dictionary ke file JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def print_section(title: str) -> None:
    """Membuat judul terminal agar output lebih mudah dibaca."""
    line = "=" * 70
    print(f"\n{line}")
    print(title)
    print(line)
