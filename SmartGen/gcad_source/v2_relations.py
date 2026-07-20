from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

from .semantic_channels import require_valid_semantic_channels


def require_prediction_gate(gate: dict) -> None:
    if not gate.get("passed", False):
        raise RuntimeError("source prediction gate failed; formal relation extraction is blocked")


def per_output_lag_gradients(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return [lag,input_channel,output_channel] gradients for positive targets."""
    inputs = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor(y, dtype=torch.float32)
    logits = model(inputs)
    history, channels = x.shape[1], x.shape[2]
    result = np.zeros((history, channels, channels), dtype=np.float64)
    for output in range(channels):
        positive = targets[:, output] > 0
        if not positive.any():
            continue
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[positive, output], targets[positive, output], reduction="sum"
        )
        gradient = torch.autograd.grad(loss, inputs, retain_graph=True)[0][positive].abs().mean(dim=0)
        result[:, :, output] = gradient.detach().numpy()
    return result


def supported_relation_edges(
    gradients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: list[str],
    *,
    min_source_activation: int,
    min_target_activation: int,
    min_cooccurrence: int,
    min_asymmetric_margin: float,
) -> list[dict]:
    require_valid_semantic_channels(vocabulary, "GCAD v2 channel vocabulary")
    source_activation = (x > 0).any(axis=1).sum(axis=0)
    target_activation = (y > 0).sum(axis=0)
    cooccurrence = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)
    for source in range(len(vocabulary)):
        present = (x[:, :, source] > 0).any(axis=1)
        cooccurrence[source] = y[present].sum(axis=0)
    strengths = gradients.sum(axis=0)
    edges = []
    for source, source_name in enumerate(vocabulary):
        for target, target_name in enumerate(vocabulary):
            if source == target:
                continue
            reverse = float(strengths[target, source])
            strength = float(strengths[source, target])
            margin = strength - reverse
            if (
                source_activation[source] < min_source_activation
                or target_activation[target] < min_target_activation
                or cooccurrence[source, target] < min_cooccurrence
                or margin < min_asymmetric_margin
            ):
                continue
            lag_values = gradients[:, source, target]
            edges.append({
                "source_channel": source_name,
                "target_channel": target_name,
                "strength": strength,
                "reverse_strength": reverse,
                "asymmetric_margin": margin,
                "main_lag": int(np.argmax(lag_values)) + 1,
                "lag_distribution": lag_values.tolist(),
                "source_activation_count": int(source_activation[source]),
                "target_activation_count": int(target_activation[target]),
                "cooccurrence_count": int(cooccurrence[source, target]),
            })
    return sorted(edges, key=lambda edge: (-edge["asymmetric_margin"], edge["source_channel"], edge["target_channel"]))


def stable_edges(per_run_edges: list[list[dict]], minimum_occurrence_rate: float) -> list[dict]:
    run_count = len(per_run_edges)
    grouped = defaultdict(list)
    for edges in per_run_edges:
        for rank, edge in enumerate(edges, start=1):
            grouped[(edge["source_channel"], edge["target_channel"])].append((edge, rank))
    stable = []
    for (source, target), observations in grouped.items():
        occurrence_rate = len(observations) / max(1, run_count)
        if occurrence_rate < minimum_occurrence_rate:
            continue
        margins = [edge["asymmetric_margin"] for edge, _ in observations]
        lags = [edge["main_lag"] for edge, _ in observations]
        ranks = [rank for _, rank in observations]
        stable.append({
            "source_channel": source,
            "target_channel": target,
            "occurrence_rate": occurrence_rate,
            "direction_consistency": float(sum(margin > 0 for margin in margins) / len(margins)),
            "mean_asymmetric_margin": float(np.mean(margins)),
            "main_lag": Counter(lags).most_common(1)[0][0],
            "lag_stability": float(Counter(lags).most_common(1)[0][1] / len(lags)),
            "rank_variance": float(np.var(ranks)),
            "minimum_source_activation": min(edge["source_activation_count"] for edge, _ in observations),
            "minimum_target_activation": min(edge["target_activation_count"] for edge, _ in observations),
            "minimum_cooccurrence": min(edge["cooccurrence_count"] for edge, _ in observations),
        })
    require_valid_semantic_channels(
        [node for edge in stable for node in (edge["source_channel"], edge["target_channel"])],
        "stable relation v2",
    )
    return sorted(stable, key=lambda edge: (-edge["occurrence_rate"], -edge["mean_asymmetric_margin"]))


def frequency_graph(x: np.ndarray, y: np.ndarray, vocabulary: list[str], edge_count: int) -> dict:
    edges = []
    for source, source_name in enumerate(vocabulary):
        present = (x[:, :, source] > 0).any(axis=1)
        for target, target_name in enumerate(vocabulary):
            if source == target:
                continue
            support = int(y[present, target].sum())
            edges.append({"source": source_name, "target": target_name, "stable_score": float(support)})
    edges.sort(key=lambda edge: (-edge["stable_score"], edge["source"], edge["target"]))
    return {"edges": edges[:edge_count], "control": "frequency_cooccurrence"}


def gss_effect(original_edges: list[dict], fused_edges: list[dict], top_k: int = 3) -> dict:
    def group(edges):
        result = defaultdict(list)
        for edge in edges:
            result[edge["source"]].append(edge)
        for source in result:
            result[source].sort(key=lambda edge: (-edge["score"], edge["target"]))
        return result
    original, fused = group(original_edges), group(fused_edges)
    score_changes = 0
    rank_changes = 0
    top1_changes = 0
    topk_changes = 0
    affected = 0
    for source in sorted(set(original) | set(fused)):
        before = original.get(source, [])
        after = fused.get(source, [])
        before_scores = {(edge["source"], edge["target"]): edge["score"] for edge in before}
        after_scores = {(edge["source"], edge["target"]): edge["score"] for edge in after}
        changed = sum(before_scores.get(key) != after_scores.get(key) for key in set(before_scores) | set(after_scores))
        score_changes += changed
        affected += int(changed > 0)
        before_order = [edge["target"] for edge in before]
        after_order = [edge["target"] for edge in after]
        rank_changes += sum(a != b for a, b in zip(before_order, after_order))
        top1_changes += int(before_order[:1] != after_order[:1])
        topk_changes += int(before_order[:top_k] != after_order[:top_k])
    return {
        "affected_source_nodes": affected,
        "score_change_edge_count": score_changes,
        "rank_change_position_count": rank_changes,
        "top_1_changed_node_count": top1_changes,
        "top_3_order_changed_node_count": topk_changes,
    }
