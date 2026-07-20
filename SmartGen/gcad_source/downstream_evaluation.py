from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from anomaly_detection_pipeline.Anomaly_Detection_pipeline_model import (
    evaluate,
    find_threshold,
    setup_seed,
    train,
    vocab_dic,
)

from .data_boundary import require_roles
from .data_roles import DataRole, RoleBoundPath
from .ranking_weights import bounded_relation_weights
from .split_integrity import split_without_exact_overlap, write_split


ATTACK_NAMES = {
    "spring": "spring_attack_heater",
    "night": "night_attack_time",
    "multiple": "multiple_attack_tv",
}


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array), "min": float(array.min()), "mean": float(array.mean()),
        "median": float(np.median(array)), "max": float(array.max()),
        "p50": float(np.percentile(array, 50)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "p95_5": float(np.percentile(array, 95.5)),
        "p99": float(np.percentile(array, 99)), "std": float(array.std()),
    }


def prepare_generated_detector(
    generated_file: str | Path,
    dataset: str,
    context: str,
    output_dir: str | Path,
    percentile: float,
    epochs: int = 15,
    ranking_path: str | Path | None = None,
) -> dict:
    """Freeze detector and threshold without accepting any target role."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = RoleBoundPath.build(generated_file, DataRole.SOURCE_NORMAL)
    require_roles("threshold", [generated])
    with generated.path.open("rb") as handle:
        sequences = pickle.load(handle)
    train_indices, validation_indices, split_report = split_without_exact_overlap(sequences, seed=2024)
    train_file = output / "generated_train.pkl"
    validation_file = output / "generated_validation.pkl"
    write_split(sequences, train_indices, validation_indices, train_file, validation_file)
    train_weights = None
    weight_diagnostics = None
    if ranking_path is not None:
        ranking = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
        if ranking.get("hard_filter") is not False:
            raise ValueError("ranking must not delete sequences")
        weights_by_index, weight_diagnostics = bounded_relation_weights(ranking["ranking"])
        covered = {int(row["sequence_index"]) for row in ranking["ranking"]}
        if covered != set(range(len(sequences))):
            raise ValueError("ranking indices do not exactly cover generated sequences")
        if weights_by_index is not None:
            train_weights = [weights_by_index[index] for index in train_indices]
    setup_seed(2024)
    vocabulary_size = vocab_dic[dataset]
    sequence_length = 10
    model_path = output / "downstream_model.pth"
    history = train(
        context, vocabulary_size, epochs, str(train_file), str(model_path), sequence_length,
        sample_weights=train_weights,
    )
    threshold, validation_losses = find_threshold(
        context, vocabulary_size, str(validation_file), str(model_path), sequence_length,
        percentile, return_losses=True,
    )
    report = {
        "dataset": dataset, "context": context, "generated_file": str(generated.path),
        "model_path": str(model_path.resolve()), "threshold_source": "generated_validation",
        "threshold_percentile": percentile, "threshold": float(threshold), "epochs": epochs,
        "training_loss_by_epoch": history,
        "validation_losses": [float(value) for value in validation_losses],
        "validation_loss_summary": _summary(validation_losses),
        "split_integrity": split_report,
        "ranking_requested": ranking_path is not None,
        "ranking_applied": train_weights is not None,
        "ranking_signal_absent": bool(weight_diagnostics and weight_diagnostics["ranking_signal_absent"]),
        "ranking_weight_diagnostics": weight_diagnostics,
        "ranking_policy": "per_sample_weighted_loss" if train_weights is not None else "uniform_full_set",
        "all_training_samples_covered_each_epoch": True,
        "hard_sequence_deletion": False,
        "uses_target_behavior": False,
        "frozen_before_target_evaluation": True,
    }
    (output / "training_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_prepared_detector(
    prepared_dir: str | Path,
    dataset: str,
    context: str,
    repository_root: str | Path = ".",
) -> dict:
    """The only function in this module that opens target behavior artifacts."""
    root = Path(repository_root).resolve()
    output = Path(prepared_dir)
    if "baseline_source_semantic" in output.parts:
        from SmartGen.generation_backends.reconstruction_health import require_final_evaluation_gates

        require_final_evaluation_gates(output)
    prepared = json.loads((output / "training_diagnostics.json").read_text(encoding="utf-8"))
    attack = RoleBoundPath.build(
        root / "anomaly_detection_pipeline" / "attack" / dataset / f"labeled_{dataset}_{ATTACK_NAMES[context]}.pkl",
        DataRole.TARGET_ATTACK_EVAL,
    )
    target_normal = RoleBoundPath.build(
        root / "anomaly_detection_pipeline" / "test" / dataset / context / "test.pkl",
        DataRole.TARGET_NORMAL_EVAL,
    )
    require_roles("final_evaluation", [attack, target_normal])
    recall, precision, accuracy, f1, details = evaluate(
        context, vocab_dic[dataset], str(attack.path), str(target_normal.path),
        prepared["model_path"], 10, prepared["threshold"], return_details=True,
    )
    report = {
        **prepared,
        "target_normal_role": target_normal.role.value,
        "target_attack_role": attack.role.value,
        "target_data_first_used_at": "final_evaluation",
        "metrics_not_used_for_selection": True,
        "recall": float(recall), "precision": float(precision), "accuracy": float(accuracy), "f1": float(f1),
        **details,
        "normal_score_summary": _summary(details["normal_scores"]),
        "attack_score_summary": _summary(details["attack_scores"]),
    }
    (output / "downstream_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_generated_sequences(
    generated_file: str | Path,
    dataset: str,
    context: str,
    output_dir: str | Path,
    percentile: float,
    epochs: int = 15,
    repository_root: str | Path = ".",
    ranking_path: str | Path | None = None,
) -> dict:
    prepare_generated_detector(generated_file, dataset, context, output_dir, percentile, epochs, ranking_path)
    return evaluate_prepared_detector(output_dir, dataset, context, repository_root)
