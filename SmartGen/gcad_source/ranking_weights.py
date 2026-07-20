from __future__ import annotations

import math

import numpy as np


def bounded_relation_weights(
    rows: list[dict],
    lower: float = 0.9,
    upper: float = 1.1,
    cv_bypass_threshold: float = 1e-6,
) -> tuple[dict[int, float] | None, dict]:
    if not 0 < lower <= 1 <= upper:
        raise ValueError("weight bounds must contain one")
    ordered = sorted(rows, key=lambda row: int(row["sequence_index"]))
    scores = np.asarray([float(row["weighted_consistency"]) for row in ordered], dtype=float)
    if not len(scores):
        return None, _diagnostics(np.ones(0), True, cv_bypass_threshold, lower, upper)
    centered = scores - scores.mean()
    maximum = float(np.max(np.abs(centered)))
    if maximum <= 1e-12:
        weights = np.ones_like(scores)
    else:
        scale = min(upper - 1.0, 1.0 - lower)
        weights = 1.0 + scale * centered / maximum
    weights /= weights.mean()
    weights = np.clip(weights, lower, upper)
    weights /= weights.mean()
    cv = float(weights.std() / weights.mean())
    absent = cv < cv_bypass_threshold
    diagnostics = _diagnostics(weights, absent, cv_bypass_threshold, lower, upper)
    if absent:
        return None, diagnostics
    return {int(row["sequence_index"]): float(weight) for row, weight in zip(ordered, weights)}, diagnostics


def _diagnostics(weights: np.ndarray, absent: bool, threshold: float, lower: float, upper: float) -> dict:
    if not len(weights):
        return {
            "ranking_signal_absent": True, "count": 0, "mean": 1.0, "min": 1.0, "max": 1.0,
            "coefficient_of_variation": 0.0, "effective_sample_size": 0.0,
            "normalized_weight_entropy": 1.0, "cv_bypass_threshold": threshold,
            "lower_bound": lower, "upper_bound": upper,
        }
    probabilities = weights / weights.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-15)))
    normalized_entropy = entropy / math.log(len(weights)) if len(weights) > 1 else 1.0
    return {
        "ranking_signal_absent": absent,
        "count": len(weights),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "coefficient_of_variation": float(weights.std() / weights.mean()),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "normalized_weight_entropy": normalized_entropy,
        "cv_bypass_threshold": threshold,
        "lower_bound": lower,
        "upper_bound": upper,
    }
