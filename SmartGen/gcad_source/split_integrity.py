from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path


def sequence_hash(sequence) -> str:
    return hashlib.sha256(pickle.dumps(sequence, protocol=4)).hexdigest()


def split_without_exact_overlap(sequences, ratio: float = 0.8, seed: int = 2024):
    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    split_index = int(len(indices) * ratio)
    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]
    train_hashes = {sequence_hash(sequences[index]) for index in train_indices}
    moved = [index for index in validation_indices if sequence_hash(sequences[index]) in train_hashes]
    validation_indices = [index for index in validation_indices if index not in moved]
    train_indices.extend(moved)
    validation_hashes = {sequence_hash(sequences[index]) for index in validation_indices}
    if train_hashes & validation_hashes:
        raise AssertionError("exact sequence leaked across train/validation")
    report = {
        "seed": seed,
        "requested_ratio": ratio,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "duplicates_moved_to_train": len(moved),
        "exact_overlap_count": 0,
    }
    return train_indices, validation_indices, report


def write_split(sequences, train_indices, validation_indices, train_path: str | Path, validation_path: str | Path):
    with Path(train_path).open("wb") as handle:
        pickle.dump([sequences[index] for index in train_indices], handle)
    with Path(validation_path).open("wb") as handle:
        pickle.dump([sequences[index] for index in validation_indices], handle)
