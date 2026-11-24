"""RaSR-inspired relation augmentation only, without environment inputs.

Paper defaults are in RaSRConfig. Dataset/implementation adaptations are explicit
in the research protocol. No CBRT compatibility or endpoint transport is used.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    from .split import split_sequences
except ImportError:
    from split import split_sequences


@dataclass(frozen=True)
class RaSRConfig:
    window: int = 2
    decay: float = 0.8
    max_events: int = 10
    semantic_variants: int = 2
    statistical_variants: int = 2
    seed: int = 2024
    validation_ratio: float = 0.1

    def __post_init__(self):
        if any(v <= 0 for v in (self.window, self.max_events)):
            raise ValueError("RaSR window and maximum length must be positive")
        if not 0 < self.decay <= 1 or not 0 < self.validation_ratio < 1:
            raise ValueError("invalid decay or validation ratio")
        if (self.semantic_variants, self.statistical_variants) != (2, 2):
            raise ValueError("paper profile uses two semantic and two statistical attempts")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_source(raw_rows, dataset, actions, devices):
    """TSS is performed separately per original record; equal parents stay grouped."""
    names = sorted(actions, key=lambda name: actions[name])
    ids = {actions[name]: i for i, name in enumerate(names)}
    action_devices = [devices.get(name.split(":", 1)[0], -1) for name in names]
    candidates = [i for i, name in enumerate(names)
                  if name.split(":", 1)[0] != "None" and action_devices[i] >= 0]
    records = []
    for parent_index, row in enumerate(raw_rows):
        group = sha(row)
        for fragment_index, fragment in enumerate(split_sequences([row], 9, 24, dataset)):
            valid = bool(fragment) and len(fragment) % 4 == 0
            tokens = []
            for offset in range(0, len(fragment) - 3, 4):
                token = ids.get(fragment[offset + 3])
                valid &= (token is not None and 0 <= fragment[offset] <= 6
                          and 0 <= fragment[offset + 1] <= 7
                          and fragment[offset + 2] == action_devices[token])
                tokens.append(token if token is not None else -1)
            records.append({
                "id": f"{parent_index}:{fragment_index}", "parent_group": group,
                "parent_index": parent_index, "tokens": tokens, "row": list(fragment),
                "valid": bool(valid), "origin": "I",
            })
    return records, names, action_devices, candidates


def split_parent_groups(records, config):
    groups = sorted({r["parent_group"] for r in records})
    random.Random(config.seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError("need at least two distinct original source groups for validation")
    count = min(len(groups) - 1, max(1, round(len(groups) * config.validation_ratio)))
    validation_groups = set(groups[:count])
    train = [r for r in records if r["parent_group"] not in validation_groups]
    validation = [r for r in records if r["parent_group"] in validation_groups]
    return train, validation


def build_relations(records, vectors, config):
    """Window=2 is a two-item directed window, matching paper Eq.3 adjacency.

    Window>2 extends C to the following window-1 positions; T always uses Eq.3
    adjacent pairs weighted by lambda^(one-based source position), not hours.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("semantic vectors must be a finite matrix")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("zero semantic vector")
    normalized = vectors / norm
    n = len(vectors)
    c = np.zeros((n, n), dtype=np.float32)
    t = np.zeros_like(c)
    for record in records:
        if not record["valid"]:
            continue
        sequence = record["tokens"]
        for position, source in enumerate(sequence):
            for j in range(position + 1, min(len(sequence), position + config.window)):
                c[source, sequence[j]] += 1
            if position + 1 < len(sequence):
                t[source, sequence[position + 1]] += config.decay ** (position + 1)
    return {"C": c, "T": t, "S": np.clip(normalized @ normalized.T, -1, 1),
            "semantic_vectors": vectors}


def restore_quadruples(source_row, output_tokens, action_ids, action_devices):
    """Monotone deterministic reuse of I's slots; no target/J or elapsed time."""
    if not output_tokens:
        return []
    source_length = len(source_row) // 4
    if not source_length:
        raise ValueError("cannot restore time without original source slots")
    result = []
    for position, token in enumerate(output_tokens):
        slot = 0 if len(output_tokens) == 1 else (position * (source_length - 1)) // (len(output_tokens) - 1)
        result.extend([source_row[4 * slot], source_row[4 * slot + 1],
                       action_devices[token], action_ids[token]])
    return result


def augment(records, relations, candidates, action_ids, action_devices, config, *, semantic_only=False, random_control=False):
    rng = np.random.RandomState(config.seed)
    available = np.zeros(len(action_ids), dtype=bool)
    available[candidates] = True
    versions, skipped = [], Counter()
    for record in records:
        original = record["tokens"]
        if not record["valid"] or not 0 < len(original) <= config.max_events:
            skipped["invalid_or_overlength_parent"] += 1
            continue
        positions = [p for p, token in enumerate(original) if available[token]]
        if not positions:
            skipped["no_editable_action"] += 1
            continue
        seen = {tuple(original)}
        operations = ["semantic_insert", "semantic_replace"]
        if not semantic_only:
            operations += ["statistical_replace", "statistical_replace"]
        for attempt, operation in enumerate(operations):
            if operation.endswith("insert") and len(original) >= config.max_events:
                skipped["insertion_at_length_limit"] += 1
                continue
            position = int(rng.choice(positions))
            anchor = original[position]
            mask = available.copy()
            mask[anchor] = False
            if operation.startswith("semantic") or semantic_only:
                weights = np.maximum(relations["S"][anchor].copy(), 0)
            else:
                weights = relations["C"][anchor] + relations["T"][anchor]
            weights = np.where(mask, weights, 0).astype(np.float64)
            if weights.sum() <= 0:
                skipped["no_positive_relation"] += 1
                continue
            # Random controls use the same operation kinds and legality rules,
            # but draw uniformly without relation scores. Parent counts are
            # matched by the experiment runner after failed/duplicate edits.
            if random_control:
                weights = mask.astype(np.float64)
            if operation.startswith("statistical") and not random_control:
                replacement = int(np.argmax(weights))
            else:
                replacement = int(rng.choice(len(weights), p=weights / weights.sum()))
            tokens = list(original)
            slots = [record["row"][i:i + 2] for i in range(0, len(record["row"]), 4)]
            if operation.endswith("insert"):
                tokens.insert(position + 1, replacement)
                slots.insert(position + 1, list(slots[position]))
            else:
                tokens[position] = replacement
            if tuple(tokens) in seen:
                skipped["duplicate_or_unchanged_variant"] += 1
                continue
            seen.add(tuple(tokens))
            row = []
            for slot, token in zip(slots, tokens):
                row.extend([*slot, action_devices[token], action_ids[token]])
            versions.append({
                "id": f"{record['id']}:J{attempt}", "parent_id": record["id"],
                "parent_group": record["parent_group"], "origin": "J",
                "tokens": tokens, "row": row, "valid": True,
                "edit": {"operation": operation, "position": position,
                         "old_token": anchor, "new_token": replacement},
            })
    return versions, dict(skipped)
