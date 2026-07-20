from __future__ import annotations

import json
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
    split_random(str(generated.path), str(train_file), str(validation_file), seed=2024)
    setup_seed(2024)
    vocabulary_size = vocab_dic[dataset]
    sequence_length = 10
    train(context, vocabulary_size, epochs, str(train_file), str(model_path), sequence_length)
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
        "recall": float(recall),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
    (output / "downstream_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

