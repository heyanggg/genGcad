from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    flat = matrix.ravel()
    order = np.argsort(-flat, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(flat), dtype=float)
    return ranks.reshape(matrix.shape)


def build_stable_relation(
    matrices: Sequence[np.ndarray],
    vocabulary: Sequence[str],
    primary_lags: Sequence[np.ndarray] | None = None,
    occurrence_threshold: float = 0.5,
    direction_consistency_threshold: float = 0.5,
    stability_threshold: float = 0.25,
    seeds: Sequence[int] | None = None,
    split_ids: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    if not matrices:
        raise ValueError("at least one relation matrix is required")
    stack = np.stack([np.asarray(matrix, dtype=float) for matrix in matrices])
    presence = stack > 0
    occurrence = presence.mean(axis=0)
    mean_strength = stack.mean(axis=0)
    std_strength = stack.std(axis=0)
    direction = np.stack([matrix > matrix.T for matrix in stack]).mean(axis=0)
    ranks = np.stack([_rank_matrix(matrix) for matrix in stack])
    rank_stability = 1.0 - ranks.std(axis=0) / max(1, stack.shape[1] * stack.shape[2] - 1)
    mean_lag = np.stack(primary_lags).mean(axis=0) if primary_lags else np.zeros_like(mean_strength)
    stable_score = occurrence * mean_strength * np.clip(rank_stability, 0, 1)
    keep = (
        (occurrence >= occurrence_threshold)
        & (direction >= direction_consistency_threshold)
        & (stable_score >= stability_threshold)
    )
    stable = np.where(keep, stable_score, 0.0)
    np.fill_diagonal(stable, 0.0)
    edges = []
    for i, j in zip(*np.nonzero(stable)):
        edges.append(
            {
                "source": vocabulary[i],
                "target": vocabulary[j],
                "source_index": int(i),
                "target_index": int(j),
                "occurrence_rate": float(occurrence[i, j]),
                "mean_strength": float(mean_strength[i, j]),
                "std_strength": float(std_strength[i, j]),
                "direction_consistency": float(direction[i, j]),
                "rank_stability": float(rank_stability[i, j]),
                "mean_primary_lag": float(mean_lag[i, j]),
                "stable_score": float(stable[i, j]),
                "seed_count": len(set(seeds or [])) or len(matrices),
                "split_count": len(set(split_ids or [])) or len(matrices),
            }
        )
    edges.sort(key=lambda item: (-item["stable_score"], item["source"], item["target"]))
    payload = {
        "relation_type": "GCAD-style predictive directional relation",
        "formula": "occurrence_rate * mean_strength * rank_stability",
        "replicate_count": len(matrices),
        "seeds": list(seeds or []),
        "split_ids": list(split_ids or []),
        "thresholds": {
            "occurrence_rate": occurrence_threshold,
            "direction_consistency": direction_consistency_threshold,
            "stable_score": stability_threshold,
        },
        "edge_count": len(edges),
        "edges": edges,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "relation_matrix.npy", stable)
        (output / "stable_relation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (output / "stable_relation.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(edges[0]) if edges else ["source", "target", "stable_score"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(edges)
    return {"matrix": stable, "edges": edges, "payload": payload}

