from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def select_source_replicate(
    sequences: Sequence[np.ndarray],
    seed: int,
    partition_index: int = 0,
    partition_count: int = 1,
    bootstrap_fraction: float = 1.0,
) -> tuple[list[np.ndarray], list[int], dict]:
    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError("invalid source partition")
    if not 0 < bootstrap_fraction <= 1:
        raise ValueError("bootstrap_fraction must be in (0, 1]")
    boundaries = np.linspace(0, len(sequences), partition_count + 1, dtype=int)
    base_indices = np.arange(boundaries[partition_index], boundaries[partition_index + 1])
    requested = max(1, int(round(len(base_indices) * bootstrap_fraction)))
    if bootstrap_fraction < 1:
        rng = np.random.default_rng(seed)
        selected = rng.choice(base_indices, size=requested, replace=True)
    else:
        selected = base_indices
    indices = selected.tolist()
    metadata = {
        "seed": seed,
        "partition_index": partition_index,
        "partition_count": partition_count,
        "partition_start": int(boundaries[partition_index]),
        "partition_end_exclusive": int(boundaries[partition_index + 1]),
        "bootstrap_fraction": bootstrap_fraction,
        "bootstrap_with_replacement": bootstrap_fraction < 1,
        "selected_sequence_count": len(indices),
        "unique_sequence_count": len(set(indices)),
        "selected_sequence_indices": indices,
        "uses_target_behavior": False,
    }
    return [sequences[index] for index in indices], indices, metadata


def save_selection(path: str | Path, metadata: dict) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

