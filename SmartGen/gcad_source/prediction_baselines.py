from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .window_dataset import SequenceWindowDataset


def _clip(probabilities):
    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = _clip(np.asarray(probabilities, dtype=np.float64))
    predictions = (probabilities >= 0.5).astype(np.int64)
    loss = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)).mean()
    tp = (predictions * y_true).sum(axis=0)
    fp = (predictions * (1 - y_true)).sum(axis=0)
    fn = ((1 - predictions) * y_true).sum(axis=0)
    channel_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    micro_f1 = float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()))
    active = y_true.sum(axis=0) > 0
    macro_f1 = float(channel_f1[active].mean()) if active.any() else 0.0
    positive_rows = y_true.sum(axis=1) > 0
    top1 = probabilities.argmax(axis=1)
    top1_accuracy = float(y_true[np.arange(len(y_true)), top1][positive_rows].mean()) if positive_rows.any() else 0.0
    k = min(3, y_true.shape[1])
    topk = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    topk_hit = np.array([row[indexes].any() for row, indexes in zip(y_true, topk)])
    frequencies = y_true.sum(axis=0)
    positive_frequencies = frequencies[frequencies > 0]
    rare_cutoff = np.quantile(positive_frequencies, 0.25) if len(positive_frequencies) else 0
    rare = (frequencies > 0) & (frequencies <= rare_cutoff)
    rare_recall = float(tp[rare].sum() / max(1, frequencies[rare].sum()))
    per_channel_loss = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)).mean(axis=0)
    return {
        "validation_loss": float(loss),
        "element_accuracy": float((predictions == y_true).mean()),
        "top1": top1_accuracy,
        "top3": float(topk_hit[positive_rows].mean()) if positive_rows.any() else 0.0,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "rare_channel_recall": rare_recall,
        "per_channel_loss": per_channel_loss.tolist(),
    }


def collect_windows(sequences: Sequence[np.ndarray], history_length: int):
    dataset = SequenceWindowDataset(sequences, history_length)
    x = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
    y = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
    return x, y


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


def predict_ngram(model, x: np.ndarray, order: int):
    totals, counts, prior = model
    result = []
    for history in x:
        key = tuple(_signature(row) for row in history[-order:])
        result.append((totals[key] + prior) / (counts[key] + 1) if counts[key] else prior)
    return np.asarray(result)


def compare_baselines(
    model,
    train_sequences: Sequence[np.ndarray],
    validation_sequences: Sequence[np.ndarray],
    history_length: int,
    output_dir: str | Path,
    ngram_order: int = 2,
    device: str = "cpu",
) -> dict:
    train_x, train_y = collect_windows(train_sequences, history_length)
    validation_x, validation_y = collect_windows(validation_sequences, history_length)
    results = {"persistence": multilabel_metrics(validation_y, persistence_predict(validation_x))}
    markov = fit_ngram(train_x, train_y, 1)
    results["first_order_markov"] = multilabel_metrics(validation_y, predict_ngram(markov, validation_x, 1))
    ngram = fit_ngram(train_x, train_y, ngram_order)
    results[f"ngram_{ngram_order}"] = multilabel_metrics(
        validation_y, predict_ngram(ngram, validation_x, ngram_order)
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(validation_x).to(device)).cpu().numpy()
    results["gcad_mixer"] = multilabel_metrics(validation_y, 1 / (1 + np.exp(-logits)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_baseline_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Source prediction baseline comparison", "", "| model | BCE | micro F1 | macro F1 | top-1 | rare recall |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['validation_loss']:.6f} | {metrics['micro_f1']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['top1']:.6f} | {metrics['rare_channel_recall']:.6f} |"
        )
    (output / "source_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results

