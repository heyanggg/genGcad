from __future__ import annotations

import copy
import json
from pathlib import Path


def _legal_action(value: object) -> bool:
    if not isinstance(value, str) or value.count(":") != 1:
        return False
    device, action = (part.strip() for part in value.split(":", 1))
    forbidden = {"", "none", "location", "unknown", "padding"}
    return device.lower() not in forbidden and action.lower() not in forbidden


def _relation_scores(relation: dict) -> dict[tuple[str, str], float]:
    scores = {}
    for edge in relation.get("edges", []):
        source = edge.get("source", edge.get("source_channel"))
        target = edge.get("target", edge.get("target_channel"))
        if not _legal_action(source) or not _legal_action(target):
            continue
        score = edge.get("score", edge.get("stable_score", edge.get("mean_score", 0.0)))
        scores[(source, target)] = max(0.0, float(score))
    return scores


def rerank_existing_gss(original_gss: dict, relation: dict | None, alpha: float = 0.2) -> dict:
    """Rerank existing GSS edges only; disabled mode is byte-content equivalent."""
    if relation is None:
        return original_gss
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    result = copy.deepcopy(original_gss)
    scores = _relation_scores(relation)
    for source, payload in result.items():
        transitions = payload.get("transitions", [])
        if len(transitions) < 2:
            continue
        original_max = max(float(item.get("count", 0)) for item in transitions) or 1.0
        relation_max = max((scores.get((source, item.get("next_action")), 0.0) for item in transitions), default=0.0)

        def blended(item):
            original = float(item.get("count", 0)) / original_max
            gcad = scores.get((source, item.get("next_action")), 0.0) / relation_max if relation_max else 0.0
            return (1 - alpha) * original + alpha * gcad

        payload["transitions"] = sorted(
            transitions,
            key=lambda item: (-blended(item), -float(item.get("count", 0)), str(item.get("next_action", ""))),
        )
    return result


def load_and_rerank_gss(original_gss: dict, relation_path: str | Path | None, alpha: float = 0.2) -> dict:
    if not relation_path:
        return original_gss
    relation = json.loads(Path(relation_path).read_text(encoding="utf-8"))
    return rerank_existing_gss(original_gss, relation, alpha)

