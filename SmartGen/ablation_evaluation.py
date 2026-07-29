"""Balanced, repeated detector evaluation for generation ablations.

This module does not change SmartGen generation or the main detector protocol.
It provides a stricter analysis protocol for comparing two already-generated
datasets:

* equal generated sample counts;
* disjoint training and validation splits for every target environment;
* repeated detector seeds;
* thresholded metrics plus AUROC/AUPRC from continuous anomaly scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from . import baseline1
    from .models1 import (
        TimeSeriesDataset2,
        TimeSeriesDataset3,
        TimeSeriesDataset4,
        TransformerAutoencoder,
    )
except ImportError:
    import baseline1
    from models1 import (
        TimeSeriesDataset2,
        TimeSeriesDataset3,
        TimeSeriesDataset4,
        TransformerAutoencoder,
    )


DATASET_CLASSES = {
    "spring": TimeSeriesDataset2,
    "night": TimeSeriesDataset3,
    "multiple": TimeSeriesDataset4,
}
ATTACK_FILENAMES = {
    "spring": "labeled_{dataset}_spring_attack_heater.pkl",
    "night": "labeled_{dataset}_night_attack_time.pkl",
    "multiple": "labeled_{dataset}_multiple_attack_tv.pkl",
}
METRIC_NAMES = (
    "f1",
    "accuracy",
    "precision",
    "recall",
    "auroc",
    "auprc",
    "threshold",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequences(path: Path, *, labeled: bool = False) -> list[list[int]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if labeled:
        return [list(item[0]) for item in data]
    return [list(item) for item in data]


def balanced_train_validation_split(
    sequences: list[list[int]],
    *,
    sample_count: int,
    train_ratio: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if sample_count > len(sequences):
        raise ValueError("sample_count exceeds available sequences")
    if sample_count < 2:
        raise ValueError("at least two balanced sequences are required")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")

    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    selected = [list(sequences[index]) for index in indices[:sample_count]]
    split_index = int(sample_count * train_ratio)
    split_index = min(max(split_index, 1), sample_count - 1)
    return selected[:split_index], selected[split_index:]


def pad_sequences(sequences: list[list[int]], vocab_size: int) -> np.ndarray:
    padded = []
    for sequence in sequences:
        item = list(sequence[:40])
        item.extend([vocab_size - 1] * (40 - len(item)))
        padded.append(item)
    return np.asarray(padded)


def score_sequences(
    environment: str,
    vocab_size: int,
    sequences: list[list[int]],
    model_path: Path,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    dataset = DATASET_CLASSES[environment](
        vocab_size, pad_sequences(sequences, vocab_size)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = TransformerAutoencoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).to(baseline1.device)
    model.load_state_dict(torch.load(model_path, map_location=baseline1.device))
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []
    with torch.no_grad():
        for source, padding_mask, loss_mask in loader:
            source = source.to(baseline1.device)
            padding_mask = padding_mask.to(baseline1.device)
            loss_mask = loss_mask.to(baseline1.device)
            output = model(source, src_key_padding_mask=padding_mask)
            loss = criterion(
                output.reshape(-1, vocab_size), source.long().reshape(-1)
            )
            loss = loss.reshape(source.shape[0], -1) * loss_mask
            denominator = loss_mask.sum(dim=1).clamp_min(1)
            losses.extend((loss.sum(dim=1) / denominator).cpu().numpy().tolist())
    return np.asarray(losses)


def calculate_metrics(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.concatenate(
        [
            np.zeros(len(normal_scores), dtype=int),
            np.ones(len(attack_scores), dtype=int),
        ]
    )
    scores = np.concatenate([normal_scores, attack_scores])
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for name in METRIC_NAMES:
        values = [float(run["metrics"][name]) for run in runs]
        summary[name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def write_checksums(output_dir: Path) -> None:
    """Write hashes for every analysis artifact except the checksum file itself."""
    checksum_path = output_dir / "checksums.sha256"
    records = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != checksum_path:
            records.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            )
    temporary = checksum_path.with_suffix(".sha256.tmp")
    temporary.write_text("".join(records), encoding="utf-8")
    temporary.replace(checksum_path)


def evaluate_mode(
    mode: str,
    sequences: list[list[int]],
    *,
    environment: str,
    vocab_size: int,
    sample_count: int,
    train_ratio: float,
    percentile: float,
    seeds: list[int],
    epochs: int,
    normal_sequences: list[list[int]],
    attack_sequences: list[list[int]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    runs = []
    for seed in seeds:
        run_dir = output_dir / "artifacts" / mode / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_data, validation_data = balanced_train_validation_split(
            sequences,
            sample_count=sample_count,
            train_ratio=train_ratio,
            seed=seed,
        )
        train_path = run_dir / "train.pkl"
        validation_path = run_dir / "validation.pkl"
        model_path = run_dir / "detector.pth"
        write_pickle(train_path, train_data)
        write_pickle(validation_path, validation_data)

        baseline1.setup_seed(seed)
        baseline1.train(
            environment,
            vocab_size,
            epochs,
            train_path,
            model_path,
            10,
        )
        validation_scores = score_sequences(
            environment, vocab_size, validation_data, model_path
        )
        normal_scores = score_sequences(
            environment, vocab_size, normal_sequences, model_path
        )
        attack_scores = score_sequences(
            environment, vocab_size, attack_sequences, model_path
        )
        threshold = float(np.percentile(validation_scores, percentile))
        metrics = calculate_metrics(normal_scores, attack_scores, threshold)
        np.savez_compressed(
            run_dir / "scores.npz",
            validation=validation_scores,
            normal=normal_scores,
            attack=attack_scores,
        )
        result = {
            "mode": mode,
            "seed": seed,
            "balanced_sample_count": sample_count,
            "training_sequence_count": len(train_data),
            "validation_sequence_count": len(validation_data),
            "metrics": metrics,
        }
        atomic_json(run_dir / "metrics.json", result)
        runs.append(result)
    return runs


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balanced repeated detector evaluation for an on/off ablation"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", choices=baseline1.vocab_dic, required=True)
    parser.add_argument(
        "--environment", choices=DATASET_CLASSES, required=True
    )
    parser.add_argument("--on-data", type=Path, required=True)
    parser.add_argument("--off-data", type=Path, required=True)
    parser.add_argument("--normal-data", type=Path)
    parser.add_argument("--attack-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--percentile", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_args_parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("detector seeds must be unique")
    if not 0 <= args.percentile <= 100:
        raise ValueError("percentile must be between zero and one hundred")

    smartgen_root = Path(__file__).resolve().parent
    normal_path = args.normal_data or (
        smartgen_root / "IoT_data" / args.dataset / args.environment / "split_test.pkl"
    )
    attack_path = args.attack_data or (
        smartgen_root
        / "attack"
        / args.dataset
        / ATTACK_FILENAMES[args.environment].format(dataset=args.dataset)
    )
    inputs = {
        "on": args.on_data.resolve(),
        "off": args.off_data.resolve(),
        "normal": normal_path.resolve(),
        "attack": attack_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    on_sequences = load_sequences(inputs["on"])
    off_sequences = load_sequences(inputs["off"])
    normal_sequences = load_sequences(inputs["normal"])
    attack_sequences = load_sequences(inputs["attack"], labeled=True)
    sample_count = min(len(on_sequences), len(off_sequences))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "name": args.name,
        "dataset": args.dataset,
        "environment": args.environment,
        "on_sequence_count": len(on_sequences),
        "off_sequence_count": len(off_sequences),
        "balanced_sample_count": sample_count,
        "train_ratio": args.train_ratio,
        "percentile": args.percentile,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    atomic_json(output_dir / "config.json", config)

    common = {
        "environment": args.environment,
        "vocab_size": baseline1.vocab_dic[args.dataset],
        "sample_count": sample_count,
        "train_ratio": args.train_ratio,
        "percentile": args.percentile,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "normal_sequences": normal_sequences,
        "attack_sequences": attack_sequences,
        "output_dir": output_dir,
    }
    on_runs = evaluate_mode("on", on_sequences, **common)
    off_runs = evaluate_mode("off", off_sequences, **common)
    paired_deltas = {
        name: [
            float(on_run["metrics"][name]) - float(off_run["metrics"][name])
            for on_run, off_run in zip(on_runs, off_runs)
        ]
        for name in METRIC_NAMES
    }
    result = {
        "config": config,
        "on": {"runs": on_runs, "summary": summarize_runs(on_runs)},
        "off": {"runs": off_runs, "summary": summarize_runs(off_runs)},
        "paired_on_minus_off": {
            name: {
                "values": values,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
            for name, values in paired_deltas.items()
        },
    }
    atomic_json(output_dir / "summary.json", result)
    write_checksums(output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
