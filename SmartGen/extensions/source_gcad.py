from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from SmartGen import dictionary
except ImportError:  # Supports running from inside SmartGen/.
    import dictionary


class TinyMixer(nn.Module):
    def __init__(self, history: int, channels: int, hidden: int = 32):
        super().__init__()
        self.history = history
        self.channels = channels
        self.temporal = nn.Linear(history, history)
        self.features = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
        self.output = nn.Linear(history, 1)

    def forward(self, x):
        temporal = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        mixed = x + torch.relu(temporal)
        mixed = mixed + self.features(mixed)
        return self.output(mixed.transpose(1, 2)).squeeze(-1)


def _dataset_maps(dataset: str):
    if dataset not in {"fr", "sp", "us"}:
        raise ValueError("dataset must be fr, sp, or us")
    devices = getattr(dictionary, f"{dataset}_devices_dict")
    actions = getattr(dictionary, f"{dataset}_actions")
    return {value: key for key, value in devices.items()}, {value: key for key, value in actions.items()}


def _legal_sequences(raw_sequences, dataset: str):
    devices, actions = _dataset_maps(dataset)
    parsed = []
    invalid = []
    vocabulary = set()
    for source_index, raw in enumerate(raw_sequences):
        row = []
        for offset in range(0, len(raw), 4):
            if offset + 3 >= len(raw):
                invalid.append({"source_index": source_index, "offset": offset, "reason": "incomplete_event"})
                break
            _, _, device_id, action_id = (int(value) for value in raw[offset : offset + 4])
            device, action = devices.get(device_id), actions.get(action_id)
            if not device or not action or device == "None" or action.startswith("None:") or not action.startswith(device + ":"):
                invalid.append({"source_index": source_index, "offset": offset, "device": device, "action": action})
                continue
            row.append(action)
            vocabulary.add(action)
        parsed.append(row)
    return parsed, invalid, sorted(vocabulary)


def _split_sources(sequences, validation_ratio: float, seed: int):
    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_ratio))
    validation = set(indices[:validation_count])
    return (
        [sequence for index, sequence in enumerate(sequences) if index not in validation],
        [sequence for index, sequence in enumerate(sequences) if index in validation],
    )


def _windows(sequences, vocabulary, history: int):
    index = {action: position for position, action in enumerate(vocabulary)}
    x, y = [], []
    for sequence in sequences:
        encoded = np.zeros((len(sequence), len(vocabulary)), dtype=np.float32)
        for position, action in enumerate(sequence):
            encoded[position, index[action]] = 1.0
        for target in range(history, len(encoded)):
            x.append(encoded[target - history : target])
            y.append(encoded[target])
    if not x:
        return np.empty((0, history, len(vocabulary)), np.float32), np.empty((0, len(vocabulary)), np.float32)
    return np.stack(x), np.stack(y)


def _train(train_x, train_y, history: int, channels: int, seed: int, epochs: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = TinyMixer(history, channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = train_y.sum(axis=0)
    pos_weight = torch.tensor(np.clip((len(train_y) - positives) / np.maximum(1, positives), 1, 20), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def _gradient_edges(model, x, y, vocabulary, top_k, min_activation):
    inputs = torch.tensor(x, requires_grad=True)
    targets = torch.tensor(y)
    logits = model(inputs)
    channels = len(vocabulary)
    gradient = np.zeros((channels, channels), dtype=np.float64)
    for target in range(channels):
        positive = targets[:, target] > 0
        if not positive.any():
            continue
        loss = nn.functional.binary_cross_entropy_with_logits(logits[positive, target], targets[positive, target], reduction="sum")
        values = torch.autograd.grad(loss, inputs, retain_graph=True)[0][positive].abs().mean(dim=(0, 1))
        gradient[:, target] = values.detach().numpy()
    source_support = (x > 0).any(axis=1).sum(axis=0)
    target_support = y.sum(axis=0)
    result = []
    for source in range(channels):
        candidates = []
        for target in range(channels):
            if source == target or source_support[source] < min_activation or target_support[target] < min_activation:
                continue
            margin = max(0.0, gradient[source, target] - gradient[target, source])
            if margin:
                candidates.append((target, margin))
        for target, margin in sorted(candidates, key=lambda item: (-item[1], vocabulary[item[0]]))[:top_k]:
            result.append({
                "source": vocabulary[source],
                "target": vocabulary[target],
                "score": float(margin),
                "source_support": int(source_support[source]),
                "target_support": int(target_support[target]),
            })
    return result


def learn_source_relations(
    source_path: str | Path,
    dataset: str,
    *,
    history: int = 2,
    seeds=(2024, 2025, 2026),
    epochs: int = 20,
    top_k: int = 3,
    min_activation: int = 3,
):
    with Path(source_path).open("rb") as handle:
        raw_sequences = pickle.load(handle)
    sequences, invalid, vocabulary = _legal_sequences(raw_sequences, dataset)
    per_seed = []
    for seed in seeds:
        train, validation = _split_sources(sequences, 0.2, seed)
        train_x, train_y = _windows(train, vocabulary, history)
        validation_x, validation_y = _windows(validation, vocabulary, history)
        if not len(train_x) or not len(validation_x):
            raise ValueError("not enough source windows for GCAD relation learning")
        model = _train(train_x, train_y, history, len(vocabulary), seed, epochs)
        per_seed.append(_gradient_edges(model, validation_x, validation_y, vocabulary, top_k, min_activation))
    grouped = defaultdict(list)
    for edges in per_seed:
        for edge in edges:
            grouped[(edge["source"], edge["target"])].append(edge)
    required = math.ceil(len(seeds) * 2 / 3)
    stable = []
    for pair, observations in grouped.items():
        if len(observations) < required:
            continue
        stable.append({
            "source": pair[0],
            "target": pair[1],
            "score": float(np.mean([edge["score"] for edge in observations])),
            "seed_support": len(observations),
            "source_support": min(edge["source_support"] for edge in observations),
            "target_support": min(edge["target_support"] for edge in observations),
        })
    stable.sort(key=lambda edge: (-edge["score"], edge["source"], edge["target"]))
    return {
        "metadata": {
            "method": "source_only_tiny_mixer_gradient",
            "source_path": str(source_path),
            "history": history,
            "seeds": list(seeds),
            "source_sequence_count": len(raw_sequences),
            "channel_count": len(vocabulary),
            "invalid_event_count": len(invalid),
            "target_behavior_used": False,
        },
        "edges": stable,
    }


def main():
    parser = argparse.ArgumentParser(description="Learn minimal source-only GCAD relations")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True, choices=("fr", "sp", "us"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    relation = learn_source_relations(args.source, args.dataset, history=args.history, epochs=args.epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(relation, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(relation['edges'])} stable source-only relations to {output}")


if __name__ == "__main__":
    main()
