from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

from anomaly_detection_pipeline.Anomaly_Detection_pipeline_model import (
    evaluate,
    find_threshold,
    setup_seed,
    split_random,
    train,
    vocab_dic,
)

from .data_boundary import require_roles
from .data_roles import DataRole, RoleBoundPath


ATTACK_NAMES = {
    "spring": "spring_attack_heater",
    "night": "night_attack_time",
    "multiple": "multiple_attack_tv",
}


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
    """Freeze source-only threshold first, then read target data for final metrics."""
    root = Path(repository_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = RoleBoundPath.build(generated_file, DataRole.SOURCE_NORMAL)
    attack = RoleBoundPath.build(
        root / "anomaly_detection_pipeline" / "attack" / dataset / f"labeled_{dataset}_{ATTACK_NAMES[context]}.pkl",
        DataRole.TARGET_ATTACK_EVAL,
    )
    target_normal = RoleBoundPath.build(
        root / "anomaly_detection_pipeline" / "test" / dataset / context / "test.pkl",
        DataRole.TARGET_NORMAL_EVAL,
    )
    # This is the single stage where target roles are accepted.
    require_roles("final_evaluation", [generated, attack, target_normal])
    train_file = output / "generated_train.pkl"
    validation_file = output / "generated_validation.pkl"
    model_path = output / "downstream_model.pth"
    train_weights = None
    if ranking_path is None:
        split_random(str(generated.path), str(train_file), str(validation_file), seed=2024)
    else:
        ranking_file = Path(ranking_path).resolve()
        ranking = json.loads(ranking_file.read_text(encoding="utf-8"))
        if ranking.get("hard_filter") is not False or ranking.get("ranking_mode") != "soft":
            raise ValueError("downstream ranking must be soft and must not delete sequences")
        with generated.path.open("rb") as handle:
            sequences = pickle.load(handle)
        weights_by_index = {
            int(row["sequence_index"]): float(row["sampling_weight"])
            for row in ranking["ranking"]
        }
        if set(weights_by_index) != set(range(len(sequences))):
            raise ValueError("ranking indices do not exactly cover generated sequences")
        indices = list(range(len(sequences)))
        random.Random(2024).shuffle(indices)
        split_index = int(len(indices) * 0.8)
        train_indices, validation_indices = indices[:split_index], indices[split_index:]
        with train_file.open("wb") as handle:
            pickle.dump([sequences[index] for index in train_indices], handle)
        with validation_file.open("wb") as handle:
            pickle.dump([sequences[index] for index in validation_indices], handle)
        train_weights = [weights_by_index[index] for index in train_indices]
    setup_seed(2024)
    vocabulary_size = vocab_dic[dataset]
    sequence_length = 10
    train(
        context,
        vocabulary_size,
        epochs,
        str(train_file),
        str(model_path),
        sequence_length,
        sample_weights=train_weights,
    )
    # Threshold is fully fixed from held-out generated/source-only data.
    threshold = find_threshold(
        context, vocabulary_size, str(validation_file), str(model_path), sequence_length, percentile
    )
    recall, precision, accuracy, f1 = evaluate(
        context,
        vocabulary_size,
        str(attack.path),
        str(target_normal.path),
        str(model_path),
        sequence_length,
        threshold,
    )
    report = {
        "dataset": dataset,
        "context": context,
        "generated_file": str(generated.path),
        "threshold_source": "generated_validation",
        "threshold_percentile": percentile,
        "threshold": float(threshold),
        "epochs": epochs,
        "target_normal_role": target_normal.role.value,
        "target_attack_role": attack.role.value,
        "target_data_first_used_at": "final_evaluation",
        "metrics_not_used_for_selection": True,
        "ranking_applied": ranking_path is not None,
        "ranking_path": str(Path(ranking_path).resolve()) if ranking_path is not None else None,
        "ranking_policy": "weighted_sampling_with_replacement" if ranking_path is not None else "uniform_full_set",
        "hard_sequence_deletion": False,
        "recall": float(recall),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
    (output / "downstream_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
