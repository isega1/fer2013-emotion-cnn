"""FER2013 dataset.

The CSV shipped with the Kaggle competition has three columns:
``emotion`` (0-6), ``pixels`` (2304 space separated ints) and ``Usage``.

The parser mirrors the original coursework code: 48x48 uint8 grayscale,
scaled to [-1, 1]. ``ToTensor()`` followed by ``Normalize([0.5], [0.5])``
is exactly ``(x - 127.5) / 127.5``, so we apply that directly instead of
going through PIL for every sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_pixels(df: pd.DataFrame) -> np.ndarray:
    """Parse the ``pixels`` column into an (N, 48, 48) uint8 array."""
    flat = np.array(" ".join(df["pixels"].astype(str)).split(), dtype=np.uint8)
    return flat.reshape(len(df), 48, 48)


class FER2013Dataset(Dataset):
    def __init__(self, csv_file: str, cache: bool = True):
        df = pd.read_csv(csv_file)
        # Some mirrors write the header as "emotion, Usage, pixels"; strip the
        # stray spaces so both variants resolve to the same column names.
        df.columns = [c.strip() for c in df.columns]
        self.labels = df["emotion"].to_numpy(dtype=np.int64)
        self.images = load_pixels(df) if cache else df["pixels"].tolist()
        self.cache = cache

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.cache:
            pixels = self.images[idx]
        else:
            pixels = np.array(self.images[idx].split(), dtype=np.uint8).reshape(48, 48)
        x = torch.from_numpy(pixels.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return x, int(self.labels[idx])
