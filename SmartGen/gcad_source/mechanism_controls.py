from __future__ import annotations

import copy
import json
import random
from pathlib import Path


def random_directed_control(relation: dict, seed: int = 2024) -> dict:
    """Return a deterministic edge-count/out-degree matched directed control."""
    result = copy.deepcopy(relation)
    edges = result.get("edges", [])
    nodes = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    rng = random.Random(seed)
    used: set[tuple[str, str]] = set()
    controlled = []
    for edge in edges:
        candidates = [
            target for target in nodes
            if target != edge["source"] and (edge["source"], target) not in used
        ]
        if not candidates:
            raise ValueError("not enough distinct nodes to construct directed control")
        target = rng.choice(candidates)
        used.add((edge["source"], target))
        item = copy.deepcopy(edge)
        item["target"] = target
        item.pop("target_index", None)
        controlled.append(item)
    result.update({
        "relation_type": "random directed mechanism control",
        "control_seed": seed,
        "control_properties": ["edge_count_matched", "source_out_degree_matched", "self_loop_free"],
        "edge_count": len(controlled),
        "edges": controlled,
    })
    return result


def symmetric_control(relation: dict) -> dict:
    """Mirror every relation while retaining the strongest duplicate edge."""
    result = copy.deepcopy(relation)
    by_pair: dict[tuple[str, str], dict] = {}
    for edge in result.get("edges", []):
        for source, target in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
            if source == target:
                continue
            item = copy.deepcopy(edge)
            item["source"], item["target"] = source, target
            item.pop("source_index", None)
            item.pop("target_index", None)
            key = (source, target)
            if key not in by_pair or item.get("stable_score", 0) > by_pair[key].get("stable_score", 0):
                by_pair[key] = item
    controlled = [by_pair[key] for key in sorted(by_pair)]
    result.update({
        "relation_type": "symmetric relation mechanism control",
        "control_properties": ["reverse_edge_completed", "self_loop_free"],
        "edge_count": len(controlled),
        "edges": controlled,
    })
    return result


def write_control(relation_path: str | Path, output_path: str | Path, mode: str, seed: int = 2024) -> dict:
    relation = json.loads(Path(relation_path).read_text(encoding="utf-8"))
    if mode == "random_directed":
        result = random_directed_control(relation, seed)
    elif mode == "symmetric":
        result = symmetric_control(relation)
    else:
        raise ValueError(f"unknown mechanism control: {mode}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
