from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowIndex:
    sequence_index: int
    target_index: int


class SequenceWindowDataset(Dataset):
    """Windows each tensor independently, so boundaries can never be crossed."""

    def __init__(self, sequences: Sequence[np.ndarray], history_length: int):
        self.sequences = [np.asarray(item, dtype=np.float32) for item in sequences]
        self.history_length = history_length
        self.indices = [
            WindowIndex(sequence_index, target_index)
            for sequence_index, sequence in enumerate(self.sequences)
            for target_index in range(history_length, len(sequence))
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        location = self.indices[index]
        sequence = self.sequences[location.sequence_index]
        start = location.target_index - self.history_length
        x = torch.from_numpy(sequence[start : location.target_index].copy())
        y = torch.from_numpy(sequence[location.target_index].copy())
        padding_mask = torch.zeros(self.history_length, dtype=torch.bool)
        return x, y, padding_mask, location.sequence_index, location.target_index

