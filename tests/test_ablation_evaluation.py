import numpy as np
import pytest

from SmartGen.ablation_evaluation import (
    balanced_cross_validation_splits,
    balanced_train_validation_split,
    calculate_metrics,
    robust_mad_threshold,
    robust_standardize,
    sha256_file,
    summarize_runs,
    write_checksums,
)


def test_balanced_split_is_disjoint_deterministic_and_sized():
    sequences = [[index] for index in range(10)]
    train_a, validation_a = balanced_train_validation_split(
        sequences, sample_count=6, train_ratio=0.8, seed=2024
    )
    train_b, validation_b = balanced_train_validation_split(
        sequences, sample_count=6, train_ratio=0.8, seed=2024
    )
    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(train_a) == 4
    assert len(validation_a) == 2
    assert not {tuple(item) for item in train_a} & {
        tuple(item) for item in validation_a
    }


def test_balanced_split_rejects_invalid_requests():
    with pytest.raises(ValueError, match="exceeds"):
        balanced_train_validation_split(
            [[1], [2]], sample_count=3, train_ratio=0.8, seed=2024
        )
    with pytest.raises(ValueError, match="at least two"):
        balanced_train_validation_split(
            [[1]], sample_count=1, train_ratio=0.8, seed=2024
        )


def test_cross_validation_splits_cover_each_selected_sequence_once():
    selected, folds = balanced_cross_validation_splits(
        [[index] for index in range(12)],
        sample_count=10,
        folds=5,
        seed=2024,
    )
    assert len(selected) == 10
    validation_indices = [
        index for _, validation in folds for index in validation
    ]
    assert sorted(validation_indices) == list(range(10))
    for training, validation in folds:
        assert set(training).isdisjoint(validation)
        assert len(training) == 8
        assert len(validation) == 2


def test_cross_validation_rejects_too_many_folds():
    with pytest.raises(ValueError, match="exceeds"):
        balanced_cross_validation_splits(
            [[1], [2]], sample_count=2, folds=3, seed=2024
        )


def test_metrics_include_threshold_free_and_thresholded_results():
    metrics = calculate_metrics(
        np.asarray([0.1, 0.2]),
        np.asarray([0.8, 0.9]),
        threshold=0.5,
    )
    assert metrics["TP"] == 2
    assert metrics["TN"] == 2
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_robust_mad_threshold_is_not_controlled_by_single_extreme_score():
    threshold = robust_mad_threshold(
        np.asarray([1.0, 2.0, 3.0, 100.0]), multiplier=3.5
    )
    assert threshold == pytest.approx(2.5 + 3.5 * 1.4826)
    assert threshold < 100.0


def test_robust_standardize_uses_calibration_fold_scale():
    standardized = robust_standardize(
        np.asarray([1.0, 2.0, 3.0, 100.0]),
        np.asarray([2.5, 2.5 + 3.5 * 1.4826]),
    )
    assert standardized[0] == pytest.approx(0.0)
    assert standardized[1] == pytest.approx(3.5)


def test_run_summary_uses_sample_standard_deviation():
    runs = [
        {"metrics": {name: 0.0 for name in (
            "f1", "accuracy", "precision", "recall", "auroc", "auprc", "threshold"
        )}},
        {"metrics": {name: 1.0 for name in (
            "f1", "accuracy", "precision", "recall", "auroc", "auprc", "threshold"
        )}},
    ]
    summary = summarize_runs(runs)
    assert summary["f1"]["mean"] == 0.5
    assert summary["f1"]["std"] == pytest.approx(2 ** -0.5)


def test_checksums_cover_artifacts_and_exclude_themselves(tmp_path):
    artifact = tmp_path / "artifacts" / "metrics.json"
    artifact.parent.mkdir()
    artifact.write_text('{"f1": 1.0}\n', encoding="utf-8")
    write_checksums(tmp_path)
    checksum_path = tmp_path / "checksums.sha256"
    assert checksum_path.read_text(encoding="utf-8") == (
        f"{sha256_file(artifact)}  artifacts/metrics.json\n"
    )
    write_checksums(tmp_path)
    assert "checksums.sha256" not in checksum_path.read_text(encoding="utf-8")
