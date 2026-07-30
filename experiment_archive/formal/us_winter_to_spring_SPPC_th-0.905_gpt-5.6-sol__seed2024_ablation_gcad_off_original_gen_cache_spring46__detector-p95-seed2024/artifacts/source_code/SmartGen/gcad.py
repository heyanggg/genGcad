from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import dictionary
except ImportError:
    from SmartGen import dictionary


SCHEMA_VERSION = 2
METHOD_NAME = "GCAD-derived lagged directional dependencies"


class MixerBlock(nn.Module):
    """A small TSMixer-style block over behavior positions and action channels."""

    def __init__(self, history: int, action_count: int, hidden_size: int):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(action_count)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(history, history),
            nn.ReLU(),
            nn.Linear(history, history),
        )
        self.feature_norm = nn.LayerNorm(action_count)
        self.feature_mlp = nn.Sequential(
            nn.Linear(action_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, histories):
        temporal = self.temporal_norm(histories).transpose(1, 2)
        histories = histories + self.temporal_mlp(temporal).transpose(1, 2)
        return histories + self.feature_mlp(self.feature_norm(histories))


class ActionPredictor(nn.Module):
    """Predict every next-action channel from a multichannel history window."""

    def __init__(
        self,
        history: int,
        action_count: int,
        hidden_size: int = 64,
        mixer_blocks: int = 2,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *(MixerBlock(history, action_count, hidden_size) for _ in range(mixer_blocks))
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history * action_count, action_count),
            nn.Sigmoid(),
        )

    def forward(self, histories):
        return self.output(self.blocks(histories))


def decode_normal_sequences(raw_sequences, dataset: str):
    if dataset not in {"fr", "sp", "us"}:
        raise ValueError("GCAD supports the fr, sp, and us datasets")

    devices = getattr(dictionary, f"{dataset}_devices_dict")
    actions = getattr(dictionary, f"{dataset}_actions")
    id_to_device = {value: key for key, value in devices.items()}
    id_to_action = {value: key for key, value in actions.items()}
    sequences = []

    for raw_sequence in raw_sequences:
        sequence = []
        for offset in range(0, len(raw_sequence) - 3, 4):
            device = id_to_device.get(int(raw_sequence[offset + 2]))
            action = id_to_action.get(int(raw_sequence[offset + 3]))
            if (
                device
                and action
                and device != "None"
                and not action.startswith("None:")
                and action.startswith(f"{device}:")
            ):
                sequence.append(action)
        if sequence:
            sequences.append(sequence)
    return sequences


def build_prediction_windows(sequences, history: int, vocabulary=None):
    """Encode event positions as one-hot multichannel input and target tensors."""
    if history < 1:
        raise ValueError("history must be at least 1")
    if vocabulary is None:
        vocabulary = sorted({action for sequence in sequences for action in sequence})
    else:
        vocabulary = list(vocabulary)
    if not vocabulary:
        raise ValueError("normal behavior data has no valid actions")

    action_to_index = {action: index for index, action in enumerate(vocabulary)}
    histories = []
    targets = []
    for sequence in sequences:
        encoded = [action_to_index[action] for action in sequence if action in action_to_index]
        for target_position in range(history, len(encoded)):
            histories.append(encoded[target_position - history : target_position])
            targets.append(encoded[target_position])

    if not histories:
        raise ValueError("normal behavior data does not contain enough events for GCAD windows")
    history_indices = np.asarray(histories, dtype=np.int64)
    action_count = len(vocabulary)
    one_hot_histories = np.eye(action_count, dtype=np.float32)[history_indices]
    one_hot_targets = np.eye(action_count, dtype=np.float32)[np.asarray(targets, dtype=np.int64)]
    return one_hot_histories, one_hot_targets, vocabulary


def _window_count(sequence, history):
    return max(0, len(sequence) - history)


def split_train_validation_sequences(sequences, history: int, validation_ratio: float, seed: int):
    """Hold out complete sequences; use a chronological tail only for a single sequence."""
    usable = [sequence for sequence in sequences if len(sequence) > history]
    if not usable:
        raise ValueError("normal behavior data does not contain usable GCAD sequences")

    vocabulary = sorted({action for sequence in usable for action in sequence})
    if len(usable) == 1:
        histories, targets, _ = build_prediction_windows(usable, history, vocabulary)
        validation_count = max(1, int(round(len(histories) * validation_ratio)))
        validation_count = min(validation_count, len(histories) - 1)
        return (
            histories[:-validation_count],
            targets[:-validation_count],
            histories[-validation_count:],
            targets[-validation_count:],
            vocabulary,
            "chronological_window_tail",
        )

    indices = list(range(len(usable)))
    random.Random(seed).shuffle(indices)
    target_validation_windows = max(
        1,
        int(round(sum(_window_count(sequence, history) for sequence in usable) * validation_ratio)),
    )
    validation_indices = []
    validation_windows = 0
    while len(indices) > 1 and validation_windows < target_validation_windows:
        index = indices.pop()
        validation_indices.append(index)
        validation_windows += _window_count(usable[index], history)

    training_sequences = [usable[index] for index in indices]
    validation_sequences = [usable[index] for index in validation_indices]
    train_histories, train_targets, _ = build_prediction_windows(
        training_sequences, history, vocabulary
    )
    validation_histories, validation_targets, _ = build_prediction_windows(
        validation_sequences, history, vocabulary
    )
    return (
        train_histories,
        train_targets,
        validation_histories,
        validation_targets,
        vocabulary,
        "sequence_holdout",
    )


def train_predictor(
    train_histories,
    train_targets,
    validation_histories,
    validation_targets,
    *,
    history: int,
    action_count: int,
    epochs: int,
    seed: int,
    batch_size: int = 128,
    patience: int = 8,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ActionPredictor(history, action_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_histories), torch.from_numpy(train_targets)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_x = torch.from_numpy(validation_histories)
    validation_y = torch.from_numpy(validation_targets)
    best_state = None
    best_loss = math.inf
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        for batch_histories, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_histories), batch_targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.mse_loss(model(validation_x), validation_y).item()
        epochs_trained = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("GCAD predictor did not produce a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()

    prevalence = torch.from_numpy(train_targets.mean(axis=0, keepdims=True))
    baseline = prevalence.expand(len(validation_targets), -1)
    baseline_loss = nn.functional.mse_loss(baseline, validation_y).item()
    return model, {
        "validation_mse": best_loss,
        "baseline_mse": baseline_loss,
        "relative_improvement": (baseline_loss - best_loss) / max(baseline_loss, 1e-12),
        "epochs_trained": epochs_trained,
    }


def discover_gradient_influences(model, histories, targets, batch_size: int = 64):
    """Build the complete source x target x lag tensor from channel-wise squared errors."""
    action_count = histories.shape[2]
    history = histories.shape[1]
    totals = np.zeros((action_count, action_count, history), dtype=np.float64)
    source_counts = np.zeros((action_count, history), dtype=np.int64)
    target_counts = targets.sum(axis=0).astype(np.int64)

    def predict_single(sample):
        return model(sample.unsqueeze(0)).squeeze(0)

    jacobian = torch.func.jacrev(predict_single)
    for start in range(0, len(histories), batch_size):
        batch_x = torch.from_numpy(histories[start : start + batch_size])
        batch_y = torch.from_numpy(targets[start : start + batch_size])
        with torch.no_grad():
            predictions = model(batch_x)
        output_jacobians = torch.func.vmap(jacobian)(batch_x)
        error_scale = 2.0 * (predictions - batch_y)
        loss_gradients = (
            error_scale[:, :, None, None] * output_jacobians
        ).abs()
        # batch,target,lag_position,source -> source,target,lag (1 is nearest)
        totals += (
            loss_gradients.sum(dim=0)
            .permute(2, 0, 1)
            .flip(dims=(2,))
            .detach()
            .numpy()
        )
        source_counts += batch_x.sum(dim=0).transpose(0, 1).flip(dims=(1,)).numpy().astype(np.int64)

    totals /= len(histories)
    influences = {}
    supports = {}
    for source in range(action_count):
        for target in range(action_count):
            for lag_index in range(history):
                lag = lag_index + 1
                key = (source, target, lag)
                influences[key] = float(totals[source, target, lag_index])
                supports[key] = int(min(source_counts[source, lag_index], target_counts[target]))
    return influences, supports


def sparsify_directional_graph(
    influences,
    supports,
    vocabulary,
    *,
    min_support: int = 3,
    sparsity_quantile: float = 0.75,
    top_k_per_source: int = 3,
    max_relationships: int = 50,
    min_directional_margin: float = 1e-8,
):
    """Integrate lags, apply max(0, A-A^T), then retain the sparse graph."""
    if not 0 <= sparsity_quantile <= 1:
        raise ValueError("sparsity_quantile must be in [0, 1]")

    lagged_influences = defaultdict(dict)
    lagged_supports = defaultdict(dict)
    for (source, target, lag), influence in influences.items():
        support = supports.get((source, target, lag), 0)
        if source != target and support >= min_support and np.isfinite(influence):
            lagged_influences[(source, target)][lag] = float(influence)
            lagged_supports[(source, target)][lag] = int(support)

    integrated = {pair: sum(by_lag.values()) for pair, by_lag in lagged_influences.items()}
    candidates = []
    for (source, target), forward in integrated.items():
        reverse = integrated.get((target, source), 0.0)
        margin = forward - reverse
        if margin > min_directional_margin:
            typical_lag = max(
                lagged_influences[(source, target)],
                key=lagged_influences[(source, target)].get,
            )
            candidates.append(
                {
                    "source_index": source,
                    "target_index": target,
                    "typical_lag": typical_lag,
                    "raw_strength": margin,
                    "support": lagged_supports[(source, target)][typical_lag],
                    "lag_strengths": lagged_influences[(source, target)],
                }
            )
    if not candidates:
        return []

    cutoff = float(np.quantile([item["raw_strength"] for item in candidates], sparsity_quantile))
    by_source = defaultdict(list)
    for candidate in candidates:
        if candidate["raw_strength"] >= cutoff:
            by_source[candidate["source_index"]].append(candidate)

    retained = []
    for source_candidates in by_source.values():
        retained.extend(
            sorted(source_candidates, key=lambda item: -item["raw_strength"])[
                :top_k_per_source
            ]
        )
    retained.sort(
        key=lambda item: (
            -item["raw_strength"],
            vocabulary[item["source_index"]],
            vocabulary[item["target_index"]],
        )
    )
    return retained[:max_relationships]


def aggregate_stable_relationships(
    seed_graphs,
    vocabulary,
    stable_seed_fraction: float,
    total_seed_count: int | None = None,
):
    accepted_seed_count = len(seed_graphs)
    if accepted_seed_count == 0:
        return []
    total_seed_count = total_seed_count or accepted_seed_count
    required_seeds = max(1, math.ceil(total_seed_count * stable_seed_fraction))
    observations = defaultdict(list)
    for seed, graph in seed_graphs:
        for edge in graph:
            observations[(edge["source_index"], edge["target_index"])].append((seed, edge))

    retained = []
    for (source, target), values in observations.items():
        if len(values) < required_seeds:
            continue
        raw_strength = float(np.median([edge["raw_strength"] for _, edge in values]))
        lag_votes = Counter(edge["typical_lag"] for _, edge in values)
        typical_lag = min(
            (lag for lag, count in lag_votes.items() if count == max(lag_votes.values())),
            key=lambda lag: -np.median(
                [edge["lag_strengths"].get(lag, 0.0) for _, edge in values]
            ),
        )
        retained.append(
            {
                "source_action": vocabulary[source],
                "target_action": vocabulary[target],
                "typical_lag": int(typical_lag),
                "raw_strength": raw_strength,
                "support": int(np.median([edge["support"] for _, edge in values])),
                "stable_seeds": len(values),
                "total_seeds": total_seed_count,
                "stability": len(values) / total_seed_count,
            }
        )
    retained.sort(key=lambda item: (-item["raw_strength"], item["source_action"], item["target_action"]))
    if not retained:
        return []
    maximum = retained[0]["raw_strength"]
    for item in retained:
        item["strength"] = round(item["raw_strength"] / maximum, 6)
        item["raw_strength"] = round(item["raw_strength"], 10)
        item["stability"] = round(item["stability"], 6)
    return retained


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(source_hash, dataset, config):
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD_NAME,
            "source_sha256": source_hash,
            "dataset": dataset,
            "config": config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_directional_relationships(
    source_path: str | Path,
    dataset: str,
    output_path: str | Path,
    *,
    history: int = 4,
    epochs: int = 50,
    seeds=(2024, 2025, 2026),
    split_seed: int = 2024,
    validation_ratio: float = 0.2,
    min_total_windows: int = 30,
    min_validation_windows: int = 8,
    min_relative_improvement: float = 0.0,
    min_support: int = 3,
    sparsity_quantile: float = 0.75,
    top_k_per_source: int = 3,
    max_relationships: int = 50,
    stable_seed_fraction: float = 2 / 3,
    force: bool = False,
):
    """Extract, validate, freeze, and reuse GCAD-derived generation hints."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one GCAD seed is required")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not 0 < stable_seed_fraction <= 1:
        raise ValueError("stable_seed_fraction must be in (0, 1]")

    config = {
        "requested_history": history,
        "epochs": epochs,
        "seeds": list(seeds),
        "split_seed": split_seed,
        "validation_ratio": validation_ratio,
        "min_total_windows": min_total_windows,
        "min_validation_windows": min_validation_windows,
        "min_relative_improvement": min_relative_improvement,
        "min_support": min_support,
        "sparsity_quantile": sparsity_quantile,
        "top_k_per_source": top_k_per_source,
        "max_relationships": max_relationships,
        "stable_seed_fraction": stable_seed_fraction,
    }
    source_hash = _file_sha256(source_path)
    fingerprint = _artifact_fingerprint(source_hash, dataset, config)
    if output_path.exists() and not force:
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            if cached.get("artifact_fingerprint") == fingerprint:
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    with source_path.open("rb") as handle:
        raw_sequences = pickle.load(handle)
    sequences = decode_normal_sequences(raw_sequences, dataset)

    prepared = None
    history_diagnostics = []
    for effective_history in range(history, 0, -1):
        total_windows = sum(_window_count(sequence, effective_history) for sequence in sequences)
        diagnostic = {"history": effective_history, "total_windows": total_windows}
        if total_windows < min_total_windows:
            diagnostic["accepted"] = False
            diagnostic["reason"] = "insufficient_total_windows"
            history_diagnostics.append(diagnostic)
            continue
        try:
            split = split_train_validation_sequences(
                sequences, effective_history, validation_ratio, split_seed
            )
        except ValueError:
            diagnostic["accepted"] = False
            diagnostic["reason"] = "unable_to_create_holdout"
            history_diagnostics.append(diagnostic)
            continue
        train_x, train_y, validation_x, validation_y, vocabulary, split_method = split
        diagnostic.update(
            {
                "train_windows": len(train_x),
                "validation_windows": len(validation_x),
                "accepted": len(validation_x) >= min_validation_windows and len(train_x) >= 2,
            }
        )
        if not diagnostic["accepted"]:
            diagnostic["reason"] = "insufficient_holdout_windows"
            history_diagnostics.append(diagnostic)
            continue
        history_diagnostics.append(diagnostic)
        prepared = (
            effective_history,
            train_x,
            train_y,
            validation_x,
            validation_y,
            vocabulary,
            split_method,
        )
        break

    base_result = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "artifact_fingerprint": fingerprint,
        "source": {
            "path": str(source_path),
            "sha256": source_hash,
            "dataset": dataset,
        },
        "config": config,
        "lagged_behavior_relations": [],
    }

    if prepared is None:
        result = {
            **base_result,
            "status": "disabled",
            "disabled_reason": "insufficient_data_for_a_held_out_GCAD_extraction",
            "diagnostics": {"history_candidates": history_diagnostics},
        }
    else:
        (
            effective_history,
            train_x,
            train_y,
            validation_x,
            validation_y,
            vocabulary,
            split_method,
        ) = prepared
        seed_graphs = []
        seed_diagnostics = []
        for seed in seeds:
            model, metrics = train_predictor(
                train_x,
                train_y,
                validation_x,
                validation_y,
                history=effective_history,
                action_count=len(vocabulary),
                epochs=epochs,
                seed=seed,
            )
            passed = (
                np.isfinite(metrics["validation_mse"])
                and metrics["relative_improvement"] >= min_relative_improvement
            )
            graph = []
            if passed:
                influences, supports = discover_gradient_influences(
                    model, validation_x, validation_y
                )
                graph = sparsify_directional_graph(
                    influences,
                    supports,
                    vocabulary,
                    min_support=min_support,
                    sparsity_quantile=sparsity_quantile,
                    top_k_per_source=top_k_per_source,
                    max_relationships=max_relationships,
                )
                seed_graphs.append((seed, graph))
            seed_diagnostics.append(
                {
                    "seed": seed,
                    **{key: round(value, 8) if isinstance(value, float) else value for key, value in metrics.items()},
                    "passed_quality_gate": bool(passed),
                    "sparse_edges": len(graph),
                }
            )

        relationships = aggregate_stable_relationships(
            seed_graphs,
            vocabulary,
            stable_seed_fraction,
            total_seed_count=len(seeds),
        )
        passed_seeds = len(seed_graphs)
        if passed_seeds == 0:
            status = "disabled"
            disabled_reason = "no_predictor_outperformed_the_validation_baseline"
        elif not relationships:
            status = "disabled"
            disabled_reason = "no_directional_edge_was_stable_across_accepted_seeds"
        else:
            status = "ready"
            disabled_reason = None
        result = {
            **base_result,
            "status": status,
            "effective_history": effective_history,
            "lag_unit": "subsequent_behavior_positions",
            "lagged_behavior_relations": relationships,
            "diagnostics": {
                "sequence_count": len(sequences),
                "vocabulary_size": len(vocabulary),
                "train_windows": len(train_x),
                "validation_windows": len(validation_x),
                "split_method": split_method,
                "history_candidates": history_diagnostics,
                "seed_runs": seed_diagnostics,
                "accepted_seed_count": passed_seeds,
            },
        }
        if disabled_reason:
            result["disabled_reason"] = disabled_reason

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
