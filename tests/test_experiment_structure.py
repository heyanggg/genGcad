import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from SmartGen.experiment import ExperimentConfig, ExperimentRun
from SmartGen.archiving import (
    ArchiveFile,
    archive_files,
    promote_archive,
    rebuild_registry,
    verify_archive,
)


def make_config(**overrides):
    values = {
        "dataset": "fr",
        "original_environment": "winter",
        "target_environment": "spring",
        "method": "SPPC",
        "threshold": 0.918,
        "model": "gpt-5.6-sol",
        "run_id": "run1",
        "experiment_seed": 2024,
        "gcad_seeds": (2024, 2025, 2026),
        "gcad_history": 4,
        "gcad_epochs": 50,
        "gcad_mode": "auto",
        "codex_reasoning_effort": "none",
        "prompt_profile": "environment-aware",
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_experiment_config_isolates_generation_replicates():
    config = make_config(run_id="run2")
    assert config.artifact_model == "gpt-5.6-sol__run2"
    assert "run2" in config.experiment_name


def test_experiment_config_rejects_bad_environment_pair_and_path_label():
    with pytest.raises(ValueError, match="unsupported environment transition"):
        make_config(target_environment="night")
    with pytest.raises(ValueError, match="run_id"):
        make_config(run_id="../overwrite")


def test_experiment_config_rejects_invalid_or_duplicate_gcad_settings():
    with pytest.raises(ValueError, match="unique"):
        make_config(gcad_seeds=(2024, 2024, 2025))
    with pytest.raises(ValueError, match="gcad_history"):
        make_config(gcad_history=0)
    with pytest.raises(ValueError, match="threshold"):
        make_config(threshold=1.1)
    with pytest.raises(ValueError, match="GCAD mode"):
        make_config(gcad_mode="sometimes")


def test_run_manifest_records_resumable_categories(tmp_path):
    run = ExperimentRun(make_config(), runs_root=tmp_path)
    output = tmp_path / "category.pkl"
    run.record_category("0_0", output, "prompt-hash")
    stored = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert stored["status"] == "generating"
    assert stored["completed_categories"] == ["0_0"]
    assert stored["category_outputs"]["0_0"] == str(output)


def test_cached_generation_response_must_be_parseable(tmp_path, monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("main", None)
    from main import load_resumable_response

    path = tmp_path / "response.pkl"
    with path.open("wb") as handle:
        pickle.dump("<seq [['day', 'hour', 'device', 'action']] seq>", handle)
    assert load_resumable_response(path) == "<seq [['day', 'hour', 'device', 'action']] seq>"
    with path.open("wb") as handle:
        pickle.dump("broken", handle)
    assert load_resumable_response(path) is None


def test_empty_source_days_do_not_create_generation_categories(tmp_path, monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("find_categories", None)
    from find_categories import Find_categories

    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "IoT_data/fr/winter"
    source_dir.mkdir(parents=True)
    for day in range(7):
        with (source_dir / f"trn_day_{day}_SPPC_th=0.918.pkl").open("wb") as handle:
            pickle.dump([[day]] if day == 3 else [], handle)

    assert Find_categories("fr", "winter", "SPPC", 0.918) == [3]


def test_anomaly_detection_uses_isolated_run_directory(tmp_path, monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("baseline1", None)
    import baseline1

    monkeypatch.chdir(tmp_path)
    final_data = (
        tmp_path
        / "filter_data/fr/spring/fr_spring_generation_SPPC_th=0.918_model__run1_seq_filter_true.pkl"
    )
    final_data.parent.mkdir(parents=True)
    sequences = [[0, 0, 1, 2] for _ in range(10)]
    with final_data.open("wb") as handle:
        pickle.dump(sequences, handle)

    original_train = tmp_path / "IoT_data/fr/spring/trn.pkl"
    original_train.parent.mkdir(parents=True)
    original_train.write_bytes(b"original-target-domain-data")
    test_file = tmp_path / "IoT_data/fr/spring/split_test.pkl"
    with test_file.open("wb") as handle:
        pickle.dump([[0, 0, 1, 2]], handle)
    attack_file = tmp_path / "attack/fr/labeled_fr_spring_attack_heater.pkl"
    attack_file.parent.mkdir(parents=True)
    with attack_file.open("wb") as handle:
        pickle.dump([([0, 0, 1, 2], 1)], handle)

    monkeypatch.setattr(baseline1, "train", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline1, "find_threshold", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        baseline1,
        "evaluate",
        lambda *args, **kwargs: (
            np.int64(1), np.int64(1), np.int64(0), np.int64(0),
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ),
    )
    result = baseline1.Anomaly_detection(
        "fr",
        "spring",
        0.918,
        "SPPC",
        "model__run1",
        95.5,
        seed=2025,
    )

    assert original_train.read_bytes() == b"original-target-domain-data"
    run_dir = tmp_path / "anomaly_runs/fr_spring_model__run1_SPPC_th-0.918_p-95.5_seed2025"
    assert (run_dir / "train.pkl").exists()
    assert (run_dir / "validation.pkl").exists()
    assert json.loads((run_dir / "metrics.json").read_text())["metrics"]["TP"] == 1
    assert result["seed"] == 2025


def test_security_filter_removes_stale_candidate_files(tmp_path, monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("security_check", None)
    import security_check

    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "filter_data/fr/spring"
    output_dir.mkdir(parents=True)
    data_path = output_dir / "fr_spring_generation_SPPC_th=0.918_model_seq.pkl"
    sequences = [[0, 0, 1, 2]]
    with data_path.open("wb") as handle:
        pickle.dump(sequences, handle)
    stale = output_dir / "fr_spring_generation_SPPC_th=0.918_model_seq_filter_out_99.pkl"
    stale.write_bytes(b"stale")

    monkeypatch.setattr(security_check, "setup_seed", lambda seed: None)
    monkeypatch.setattr(security_check, "train", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        security_check,
        "check_outlier",
        lambda *args, **kwargs: (sequences, []),
    )
    monkeypatch.setattr(security_check, "save_outliers", lambda *args: 0)

    result = security_check.security_check(
        "fr", "spring", 0.918, "SPPC", "model", seed=2024
    )
    assert not stale.exists()
    assert result["final_count"] == 1


def test_active_smartgen_pipeline_has_no_unconditional_cuda_calls():
    for filename in ["baseline1.py", "baseline2.py", "security_check.py"]:
        source = (Path("SmartGen") / filename).read_text(encoding="utf-8")
        assert ".cuda()" not in source


def test_archive_is_checksummed_and_registered(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = source_root / "result.pkl"
    artifact.write_bytes(b"immutable-result")
    archive_root = tmp_path / "archive"

    path = archive_files(
        "fr_test_seed2024",
        "completed",
        {"dataset": "fr", "experiment_seed": 2024},
        [ArchiveFile(artifact, Path("tof/final.pkl"))],
        metrics={"metrics": {"f1_score": 0.99, "accuracy": 0.98}},
        archive_root=archive_root,
    )

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_count"] == 1
    assert manifest["files"][0]["path"] == "artifacts/tof/final.pkl"
    assert (path / "artifacts/tof/final.pkl").read_bytes() == b"immutable-result"
    registry = rebuild_registry(archive_root)
    assert registry[0]["archive_id"] == "fr_test_seed2024"
    assert registry[0]["f1_score"] == 0.99
    assert verify_archive(path) == []

    promoted = promote_archive("fr_test_seed2024", archive_root=archive_root)
    assert promoted.parent.name == "verified_candidates"
    assert not path.exists()
    assert verify_archive(promoted) == []
    with pytest.raises(ValueError, match="already exists under"):
        archive_files(
            "fr_test_seed2024",
            "completed",
            {"dataset": "fr", "experiment_seed": 2024},
            [ArchiveFile(artifact, Path("tof/final.pkl"))],
            archive_root=archive_root,
        )


def test_archive_rejects_unsafe_destination(tmp_path):
    artifact = tmp_path / "result.pkl"
    artifact.write_bytes(b"result")
    with pytest.raises(ValueError, match="safe relative path"):
        archive_files(
            "safe-id",
            "completed",
            {},
            [ArchiveFile(artifact, Path("../escape.pkl"))],
            archive_root=tmp_path / "archive",
        )
