from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mixer_predictor import SourceGCADMixer


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)


def sparse_multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = _clip(probabilities)
    predicted = (probabilities >= 0.5).astype(np.int64)
    positive_mask = y_true == 1
    negative_mask = ~positive_mask
    losses = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities))
    tp = (predicted * y_true).sum(axis=0)
    fp = (predicted * (1 - y_true)).sum(axis=0)
    fn = ((1 - predicted) * y_true).sum(axis=0)
    precision = float(tp.sum() / max(1, tp.sum() + fp.sum()))
    recall = float(tp.sum() / max(1, tp.sum() + fn.sum()))
    micro_f1 = float(2 * precision * recall / max(1e-12, precision + recall))
    channel_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    channel_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    channel_f1 = np.divide(
        2 * channel_precision * channel_recall,
        channel_precision + channel_recall,
        out=np.zeros_like(channel_precision),
        where=(channel_precision + channel_recall) > 0,
    )
    frequencies = y_true.sum(axis=0)
    active = frequencies > 0
    active_frequencies = frequencies[active]
    rare_cutoff = float(np.quantile(active_frequencies, 0.25)) if active.any() else 0.0
    rare = active & (frequencies <= rare_cutoff)
    return {
        "overall_bce": float(losses.mean()),
        "positive_bce": float(losses[positive_mask].mean()) if positive_mask.any() else None,
        "negative_bce": float(losses[negative_mask].mean()) if negative_mask.any() else None,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": float(channel_f1[active].mean()) if active.any() else 0.0,
        "per_channel_recall": channel_recall.tolist(),
        "rare_channel_recall": float(tp[rare].sum() / max(1, frequencies[rare].sum())),
        "activation_weighted_recall": float((channel_recall * frequencies).sum() / max(1, frequencies.sum())),
        "all_zero_prediction_rate": float((predicted.sum(axis=1) == 0).mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "true_positive_rate": float(y_true.mean()),
        "active_validation_channel_count": int(active.sum()),
        "rare_validation_channel_count": int(rare.sum()),
    }


def all_zero_predict(y_shape: tuple[int, int]) -> np.ndarray:
    return np.full(y_shape, 1e-6, dtype=np.float64)


def frequency_predict(train_y: np.ndarray, count: int) -> np.ndarray:
    probabilities = (train_y.sum(axis=0) + 1) / (len(train_y) + 2)
    return np.repeat(probabilities[None, :], count, axis=0)


def persistence_predict(x: np.ndarray) -> np.ndarray:
    return x[:, -1, :] * 0.98 + 0.01


def _signature(row: np.ndarray) -> tuple[int, ...]:
    return tuple(np.flatnonzero(row > 0).tolist())


def fit_ngram(x: np.ndarray, y: np.ndarray, order: int):
    totals = defaultdict(lambda: np.zeros(y.shape[1], dtype=float))
    counts = defaultdict(int)
    for history, target in zip(x, y):
        key = tuple(_signature(row) for row in history[-order:])
        totals[key] += target
        counts[key] += 1
    prior = (y.sum(axis=0) + 1) / (len(y) + 2)
    return totals, counts, prior


def predict_ngram(model, x: np.ndarray, order: int) -> np.ndarray:
    totals, counts, prior = model
    rows = []
    for history in x:
        key = tuple(_signature(row) for row in history[-order:])
        rows.append((totals[key] + prior) / (counts[key] + 1) if counts[key] else prior)
    return np.asarray(rows)


def simple_baseline_predictions(train_x, train_y, validation_x, validation_y) -> dict:
    predictions = {
        "all_zero": all_zero_predict(validation_y.shape),
        "frequency": frequency_predict(train_y, len(validation_y)),
        "persistence": persistence_predict(validation_x),
    }
    for order, name in ((1, "first_order_markov"), (2, "ngram_2"), (3, "ngram_3")):
        predictions[name] = predict_ngram(fit_ngram(train_x, train_y, order), validation_x, order)
    return predictions


def evaluate_simple_baselines(train_x, train_y, validation_x, validation_y) -> dict:
    return {
        name: sparse_multilabel_metrics(validation_y, probabilities)
        for name, probabilities in simple_baseline_predictions(train_x, train_y, validation_x, validation_y).items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def training_loss(logits, targets, mode: str, pos_weight: torch.Tensor) -> torch.Tensor:
    if mode == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if mode == "positive_weighted_bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    if mode == "masked_bce":
        raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = torch.where(targets > 0, torch.ones_like(targets), torch.full_like(targets, 0.1))
        return (raw * weights).sum() / weights.sum()
    raise ValueError(f"unknown loss mode: {mode}")


@dataclass
class MixerRun:
    metrics: dict
    model: SourceGCADMixer
    history: list[dict]


def train_mixer(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    capacity: str,
    loss_mode: str,
    epochs: int = 40,
    patience: int = 7,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.15,
) -> MixerRun:
    set_seed(seed)
    specs = {"tiny": (16, 1), "small": (32, 2)}
    if capacity not in specs:
        raise ValueError("capacity must be tiny or small")
    hidden_size, layers = specs[capacity]
    model = SourceGCADMixer(train_x.shape[1], train_x.shape[2], hidden_size, layers, dropout, "relu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    pos_weight = torch.from_numpy(np.clip(negatives / np.maximum(1, positives), 1, 20).astype(np.float32))
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation_tensor = torch.from_numpy(validation_x)
    history_rows = []
    best_score = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(x_batch), y_batch, loss_mode, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(validation_tensor)).numpy()
        metrics = sparse_multilabel_metrics(validation_y, probabilities)
        selection_score = metrics["macro_f1"]
        history_rows.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "validation_overall_bce": metrics["overall_bce"],
            "validation_macro_f1": metrics["macro_f1"],
        })
        if selection_score > best_score + 1e-10:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Mixer training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(torch.from_numpy(train_x))).numpy()
        validation_probabilities = torch.sigmoid(model(validation_tensor)).numpy()
    train_metrics = sparse_multilabel_metrics(train_y, train_probabilities)
    validation_metrics = sparse_multilabel_metrics(validation_y, validation_probabilities)
    result_metrics = {
        **validation_metrics,
        "seed": seed,
        "capacity": capacity,
        "loss_mode": loss_mode,
        "parameter_count": parameter_count(model),
        "parameter_to_train_window_ratio": parameter_count(model) / max(1, len(train_x)),
        "train_window_count": len(train_x),
        "validation_window_count": len(validation_x),
        "best_epoch": best_epoch,
        "early_stopping_epoch": history_rows[-1]["epoch"],
        "train_macro_f1": train_metrics["macro_f1"],
        "train_validation_macro_f1_gap": train_metrics["macro_f1"] - validation_metrics["macro_f1"],
        "train_overall_bce": train_metrics["overall_bce"],
    }
    return MixerRun(result_metrics, model, history_rows)


def prediction_gate(results_by_seed: dict, primary_metric: str, max_gap: float) -> dict:
    seed_checks = {}
    mixer_wins = 0
    mixer_values = []
    baseline_values = []
    for seed, payload in sorted(results_by_seed.items()):
        simple = payload["simple_baselines"]
        mixer = payload["selected_mixer"]
        best_name, best_metrics = max(simple.items(), key=lambda item: (item[1][primary_metric], item[0]))
        wins = mixer[primary_metric] > best_metrics[primary_metric]
        mixer_wins += int(wins)
        mixer_values.append(mixer[primary_metric])
        baseline_values.append(best_metrics[primary_metric])
        seed_checks[str(seed)] = {
            "best_simple_baseline": best_name,
            "best_simple_value": best_metrics[primary_metric],
            "mixer_value": mixer[primary_metric],
            "mixer_wins": wins,
            "not_all_zero": mixer["all_zero_prediction_rate"] < 1.0,
            "rare_recall_positive": mixer["rare_channel_recall"] > 0,
            "train_validation_gap_ok": mixer["train_validation_macro_f1_gap"] <= max_gap,
        }
    mean_mixer = float(np.mean(mixer_values))
    mean_baseline = float(np.mean(baseline_values))
    passed = (
        mixer_wins >= 2
        and mean_mixer > mean_baseline
        and all(check["not_all_zero"] for check in seed_checks.values())
        and all(check["rare_recall_positive"] for check in seed_checks.values())
        and all(check["train_validation_gap_ok"] for check in seed_checks.values())
    )
    return {
        "passed": passed,
        "primary_metric": primary_metric,
        "required_seed_wins": 2,
        "actual_seed_wins": mixer_wins,
        "mean_mixer_primary": mean_mixer,
        "mean_best_simple_primary": mean_baseline,
        "seed_checks": seed_checks,
        "uses_target_behavior": False,
        "on_failure": "block_relation_extraction_and_B5_generation",
    }

