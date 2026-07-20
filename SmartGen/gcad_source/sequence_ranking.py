from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .ranking_weights import bounded_relation_weights

from .data_boundary import require_roles
from .data_roles import RoleBoundPath


def channels_from_sequence(sequence, action_vocabulary: dict[int, str] | None = None) -> list[str]:
    if sequence and isinstance(sequence[0], dict):
        return [event["action"] if ":" in event["action"] else f"{event['device']}:{event['action']}" for event in sequence]
    if sequence and isinstance(sequence[0], int):
        if action_vocabulary is None or len(sequence) % 4:
            raise ValueError("flat SmartGen sequences require an action vocabulary and quadruple alignment")
        return [action_vocabulary[int(sequence[index])] for index in range(3, len(sequence), 4)]
    return [str(item) for item in sequence]


def score_sequence(channels: Sequence[str], edges: Sequence[dict]) -> dict:
    matched = []
    reversed_edges = []
    unmatched = []
    total_weight = sum(float(edge.get("stable_score", edge.get("strength", 0))) for edge in edges) or 1.0
    support_weight = 0.0
    reverse_weight = 0.0
    for edge in edges:
        source, target = edge["source"], edge["target"]
        lag = max(1, int(round(edge.get("mean_primary_lag", edge.get("primary_lag", 1)) or 1)))
        weight = float(edge.get("stable_score", edge.get("strength", 0)))
        forward = any(
            channels[j] == target
            for i, value in enumerate(channels)
            if value == source
            for j in range(i + 1, min(len(channels), i + lag + 1))
        )
        reverse = any(
            channels[j] == source
            for i, value in enumerate(channels)
            if value == target
            for j in range(i + 1, min(len(channels), i + lag + 1))
        )
        label = f"{source}->{target}"
        if forward:
            matched.append(label)
            support_weight += weight
        elif reverse:
            reversed_edges.append(label)
            reverse_weight += weight
        else:
            unmatched.append(label)
    support = support_weight / total_weight
    conflict = reverse_weight / total_weight
    coverage = (len(matched) + len(reversed_edges)) / max(1, len(edges))
    consistency = support - conflict
    return {
        "stable_relation_support": support,
        "reverse_direction_conflict": conflict,
        "relation_coverage": coverage,
        "weighted_consistency": consistency,
        "matched_edges": matched,
        "reversed_edges": reversed_edges,
        "unmatched_edges": unmatched,
    }


def rank_sequences(
    sequences: Sequence,
    stable_relation: dict,
    action_vocabulary: dict[int, str] | None = None,
    ranking_weight: float = 1.0,
    artifacts: Sequence[RoleBoundPath] | None = None,
) -> dict:
    if artifacts:
        require_roles("ranking", artifacts)
    rows = []
    for index, sequence in enumerate(sequences):
        score = score_sequence(channels_from_sequence(sequence, action_vocabulary), stable_relation.get("edges", []))
        score.update({"sequence_index": index, "original_rank": index + 1})
        rows.append(score)
    rows.sort(key=lambda item: (-ranking_weight * item["weighted_consistency"], item["original_rank"]))
    raw_weights = np.exp(np.array([item["weighted_consistency"] for item in rows]) * ranking_weight)
    raw_weights /= raw_weights.sum() if len(raw_weights) else 1
    for rank, (row, weight) in enumerate(zip(rows, raw_weights), 1):
        row["new_rank"] = rank
        row["sampling_weight"] = float(weight)
    training_weights, diagnostics = bounded_relation_weights(rows)
    if training_weights is not None:
        for row in rows:
            row["training_weight"] = training_weights[row["sequence_index"]]
    else:
        for row in rows:
            row["training_weight"] = 1.0
    return {
        "ranking_mode": "per_sample_weighted_loss",
        "deprecated_sampling_weight_present_for_audit": True,
        "hard_filter": False,
        "sequence_count": len(sequences),
        "weight_diagnostics": diagnostics,
        "ranking": rows,
    }


def rank_sequence_file(sequences, stable_relation_path: str | Path, output_path: str | Path, **kwargs):
    stable = json.loads(Path(stable_relation_path).read_text(encoding="utf-8"))
    result = rank_sequences(sequences, stable, **kwargs)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
