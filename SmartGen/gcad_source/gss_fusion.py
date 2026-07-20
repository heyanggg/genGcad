from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .data_boundary import require_roles
from .data_roles import RoleBoundPath


def _device(channel: str) -> str:
    return channel.split(":", 1)[0]


def _legal(channel: str, metadata: Mapping[str, list[str]]) -> bool:
    if ":" not in channel:
        return channel in metadata
    device, action = channel.split(":", 1)
    allowed = metadata.get(device)
    return allowed is not None and (action in allowed or channel in allowed)


def fuse_gss(
    original_gss: dict,
    stable_relation: dict,
    target_static_device_metadata: Mapping[str, list[str]],
    alpha: float = 0.2,
    mode: str = "rerank_existing",
    max_new_edges_per_source: int = 1,
) -> tuple[dict, dict]:
    if not 0 <= alpha <= 0.3:
        raise ValueError("alpha must remain in [0, 0.3]")
    if mode not in {"rerank_existing", "allow_stable_new_edges"}:
        raise ValueError("unsupported fusion mode")
    stable_edges = {(edge["source"], edge["target"]): edge for edge in stable_relation.get("edges", [])}
    fused = deepcopy(original_gss)
    details = []
    original_pairs = set()
    for source, record in fused.items():
        transitions = record.get("transitions", [])
        maximum_count = max([item.get("count", 0) for item in transitions] or [1])
        for original_rank, transition in enumerate(transitions, 1):
            target = transition["next_action"]
            original_pairs.add((source, target))
            count = float(transition.get("count", 0))
            gss_score = count / maximum_count if maximum_count else 0.0
            relation = stable_edges.get((source, target))
            gcad_score = float(relation.get("stable_score", 0.0)) if relation else 0.0
            legal = _legal(source, target_static_device_metadata) and _legal(target, target_static_device_metadata)
            if not legal:
                gcad_score = 0.0
            fused_score = (1 - alpha) * gss_score + alpha * gcad_score
            transition.update(
                {
                    "original_count": count,
                    "normalized_gss_score": gss_score,
                    "gcad_score": gcad_score,
                    "fused_score": fused_score,
                }
            )
            details.append(
                {
                    "source": source,
                    "target": target,
                    "original_score": count,
                    "gcad_score": gcad_score,
                    "fused_score": fused_score,
                    "original_rank": original_rank,
                    "new_rank": None,
                    "added": False,
                    "ignored": relation is None or not legal,
                    "reason": "no_stable_relation" if relation is None else ("illegal_target_metadata" if not legal else "used"),
                }
            )
        transitions.sort(key=lambda item: (-item["fused_score"], item["next_action"]))
        for rank, transition in enumerate(transitions, 1):
            next(item for item in details if item["source"] == source and item["target"] == transition["next_action"])["new_rank"] = rank

    if mode == "allow_stable_new_edges":
        per_source: dict[str, int] = {}
        for edge in sorted(stable_relation.get("edges", []), key=lambda item: -item.get("stable_score", 0)):
            source, target = edge["source"], edge["target"]
            if (source, target) in original_pairs:
                continue
            reason = None
            if source == target:
                reason = "self_loop"
            elif not (_legal(source, target_static_device_metadata) and _legal(target, target_static_device_metadata)):
                reason = "illegal_target_metadata"
            elif per_source.get(source, 0) >= max_new_edges_per_source:
                reason = "new_edge_limit"
            elif source not in fused:
                reason = "source_absent_from_original_gss"
            if reason:
                details.append({"source": source, "target": target, "added": False, "ignored": True, "reason": reason})
                continue
            score = alpha * float(edge["stable_score"])
            fused[source].setdefault("transitions", []).append(
                {"next_action": target, "count": 0, "original_count": 0, "normalized_gss_score": 0, "gcad_score": edge["stable_score"], "fused_score": score}
            )
            per_source[source] = per_source.get(source, 0) + 1
            details.append({"source": source, "target": target, "added": True, "ignored": False, "reason": "stable_new_edge", "fused_score": score})

    report = {
        "mode": mode,
        "alpha": alpha,
        "original_edge_count": len(original_pairs),
        "stable_edge_count": len(stable_edges),
        "fused_existing_edge_count": sum(item.get("reason") == "used" for item in details),
        "new_edge_count": sum(item.get("added", False) for item in details),
        "details": details,
        "uses_target_behavior": False,
    }
    return fused, report


def fuse_gss_files(
    original_gss_path: str | Path,
    stable_relation_path: str | Path,
    target_metadata_path: str | Path,
    output_dir: str | Path,
    alpha: float = 0.2,
    mode: str = "rerank_existing",
    artifacts: list[RoleBoundPath] | None = None,
) -> tuple[dict, dict]:
    if artifacts:
        require_roles("fuse_gss", artifacts)
    original = json.loads(Path(original_gss_path).read_text(encoding="utf-8"))
    stable = json.loads(Path(stable_relation_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(target_metadata_path).read_text(encoding="utf-8"))
    fused, report = fuse_gss(original, stable, metadata, alpha, mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "original_gss.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    (output / "fused_gss.json").write_text(json.dumps(fused, indent=2), encoding="utf-8")
    (output / "fusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# GCAD-GSS fusion report", "", f"- Mode: `{mode}`", f"- Alpha: `{alpha}`", f"- Existing edges adjusted: `{report['fused_existing_edge_count']}`", f"- New edges: `{report['new_edge_count']}`"]
    (output / "fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fused, report

