from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .baseline2 import Train
    from .dayse import Dayse
    from .experiment import SMARTGEN_ROOT, atomic_write_json, validate_environment_pair
    from .split import Split
    from .sppc import SPPC_select
except ImportError:  # Direct execution from SmartGen/.
    from baseline2 import Train
    from dayse import Dayse
    from experiment import SMARTGEN_ROOT, atomic_write_json, validate_environment_pair
    from split import Split
    from sppc import SPPC_select

VOCABULARY_SIZE = {"fr": 223, "us": 269, "sp": 235}
CACHE_SCHEMA_VERSION = 1
CACHE_KIND = "original_smartgen_pre_llm_sppc_selection"
PREPROCESSING_CODE_FILES = (
    "baseline2.py",
    "dayse.py",
    "dictionary.py",
    "models1.py",
    "split.py",
    "sppc.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_text(threshold: float) -> str:
    return format(float(threshold), "g")


def _code_fingerprints(smartgen_root: Path) -> dict[str, str]:
    fingerprints = {}
    for relative in PREPROCESSING_CODE_FILES:
        path = smartgen_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing original Gen preprocessing code: {path}")
        fingerprints[relative] = sha256_file(path)
    return fingerprints


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_archive_source(cache_dir: Path, smartgen_root: Path) -> None:
    repository_root = smartgen_root.parent.resolve()
    archive_root = (repository_root / "experiment_archive").resolve()
    if _is_within(cache_dir.resolve(), archive_root):
        raise ValueError(
            "original Gen cache cannot be read from experiment_archive; "
            "GCAD-on and historical experiment archives are prohibited cache sources"
        )


def build_original_gen_cache(
    *,
    dataset: str,
    original_environment: str,
    target_environment: str,
    threshold: float,
    experiment_seed: int,
    cache_dir: Path,
    smartgen_root: Path = SMARTGEN_ROOT,
) -> dict[str, Any]:
    """Freshly execute the original pre-LLM SSC path and save a reusable cache."""
    if dataset not in VOCABULARY_SIZE:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    validate_environment_pair(original_environment, target_environment)
    smartgen_root = smartgen_root.resolve()
    cache_dir = cache_dir.resolve()
    _reject_archive_source(cache_dir, smartgen_root)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"original Gen cache directory is not empty: {cache_dir}")

    source_dir = smartgen_root / "IoT_data" / dataset / original_environment
    input_path = source_dir / "trn.pkl"
    if not input_path.is_file():
        raise FileNotFoundError(f"missing original Gen input: {input_path}")

    # These calls intentionally do not import, execute, or read GCAD.
    Split(dataset, original_environment, 1)
    Dayse(dataset, original_environment)
    Train(
        dataset,
        original_environment,
        VOCABULARY_SIZE[dataset],
        seed=experiment_seed,
    )
    SPPC_select(
        dataset,
        original_environment,
        VOCABULARY_SIZE[dataset],
        threshold,
        seed=experiment_seed,
    )

    threshold_text = _threshold_text(threshold)
    selected_dir = cache_dir / "selected_sequences"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selection_sha256 = {}
    for day in range(7):
        filename = f"trn_day_{day}_SPPC_th={threshold_text}.pkl"
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"fresh original Gen SSC output is missing: {source}")
        destination = selected_dir / filename
        shutil.copy2(source, destination)
        selection_sha256[filename] = sha256_file(destination)

    checkpoint_source = (
        smartgen_root
        / "IoT_model"
        / f"Transformer_{dataset}_{original_environment}_15epoch.pth"
    )
    if not checkpoint_source.is_file():
        raise FileNotFoundError(
            f"fresh original Gen SSC checkpoint is missing: {checkpoint_source}"
        )
    checkpoint_destination = cache_dir / "model.pth"
    shutil.copy2(checkpoint_source, checkpoint_destination)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": CACHE_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_provenance": {
            "mode": "fresh_original_gen_upstream_execution",
            "source_directory": str(source_dir.resolve()),
            "gcad_module_executed": False,
            "llm_generation_executed": False,
            "experiment_archive_used": False,
        },
        "config": {
            "dataset": dataset,
            "original_environment": original_environment,
            "target_environment": target_environment,
            "method": "SPPC",
            "threshold": float(threshold),
            "experiment_seed": int(experiment_seed),
            "split_interval_threshold_hours": 9,
            "split_total_threshold_hours": 24,
        },
        "input_sha256": {
            "trn.pkl": sha256_file(input_path),
        },
        "preprocessing_code_sha256": _code_fingerprints(smartgen_root),
        "checkpoint": {
            "path": "model.pth",
            "sha256": sha256_file(checkpoint_destination),
        },
        "selection_sha256": selection_sha256,
    }
    atomic_write_json(cache_dir / "manifest.json", manifest)
    return manifest


def validate_original_gen_cache(
    cache_dir: Path,
    *,
    dataset: str,
    original_environment: str,
    target_environment: str,
    threshold: float,
    experiment_seed: int,
    smartgen_root: Path = SMARTGEN_ROOT,
) -> dict[str, Any]:
    smartgen_root = smartgen_root.resolve()
    cache_dir = cache_dir.resolve()
    _reject_archive_source(cache_dir, smartgen_root)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"original Gen cache manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported original Gen cache schema version")
    if manifest.get("cache_kind") != CACHE_KIND:
        raise ValueError("cache is not an original SmartGen pre-LLM SSC cache")

    expected_config = {
        "dataset": dataset,
        "original_environment": original_environment,
        "target_environment": target_environment,
        "method": "SPPC",
        "threshold": float(threshold),
        "experiment_seed": int(experiment_seed),
        "split_interval_threshold_hours": 9,
        "split_total_threshold_hours": 24,
    }
    if manifest.get("config") != expected_config:
        raise ValueError(
            "original Gen cache configuration mismatch: "
            f"expected {expected_config}, found {manifest.get('config')}"
        )
    provenance = manifest.get("source_provenance", {})
    if (
        provenance.get("mode") != "fresh_original_gen_upstream_execution"
        or provenance.get("gcad_module_executed") is not False
        or provenance.get("llm_generation_executed") is not False
        or provenance.get("experiment_archive_used") is not False
    ):
        raise ValueError("original Gen cache provenance is incomplete or unsafe")

    input_path = (
        smartgen_root / "IoT_data" / dataset / original_environment / "trn.pkl"
    )
    expected_input_hash = manifest.get("input_sha256", {}).get("trn.pkl")
    if not input_path.is_file() or sha256_file(input_path) != expected_input_hash:
        raise ValueError("original Gen cache input data fingerprint mismatch")
    if manifest.get("preprocessing_code_sha256") != _code_fingerprints(smartgen_root):
        raise ValueError("original Gen cache preprocessing code fingerprint mismatch")

    checkpoint = manifest.get("checkpoint", {})
    checkpoint_path = cache_dir / checkpoint.get("path", "")
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise ValueError("original Gen cache checkpoint fingerprint mismatch")

    threshold_text = _threshold_text(threshold)
    expected_selection_names = {
        f"trn_day_{day}_SPPC_th={threshold_text}.pkl" for day in range(7)
    }
    selection_sha256 = manifest.get("selection_sha256", {})
    if set(selection_sha256) != expected_selection_names:
        raise ValueError("original Gen cache selection file set mismatch")
    for filename, expected_hash in selection_sha256.items():
        path = cache_dir / "selected_sequences" / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(
                f"original Gen cache selection fingerprint mismatch: {filename}"
            )
    return manifest


def reuse_original_gen_cache(
    cache_dir: Path,
    *,
    dataset: str,
    original_environment: str,
    target_environment: str,
    threshold: float,
    experiment_seed: int,
    smartgen_root: Path = SMARTGEN_ROOT,
) -> dict[str, Any]:
    smartgen_root = smartgen_root.resolve()
    cache_dir = cache_dir.resolve()
    manifest = validate_original_gen_cache(
        cache_dir,
        dataset=dataset,
        original_environment=original_environment,
        target_environment=target_environment,
        threshold=threshold,
        experiment_seed=experiment_seed,
        smartgen_root=smartgen_root,
    )
    destination_dir = smartgen_root / "IoT_data" / dataset / original_environment
    for filename in sorted(manifest["selection_sha256"]):
        shutil.copy2(cache_dir / "selected_sequences" / filename, destination_dir / filename)
    checkpoint_destination = (
        smartgen_root
        / "IoT_model"
        / f"Transformer_{dataset}_{original_environment}_15epoch.pth"
    )
    checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_dir / manifest["checkpoint"]["path"], checkpoint_destination)
    return {
        "method": "SPPC",
        "training_mode": "reused_original_gen_cache",
        "selection_source_dir": str(cache_dir),
        "cache_manifest_path": str((cache_dir / "manifest.json").resolve()),
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "selection_sha256": manifest["selection_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "validated_input_sha256": manifest["input_sha256"],
        "validated_preprocessing_code_sha256": manifest[
            "preprocessing_code_sha256"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Build a fresh original SmartGen SSC cache")
    parser.add_argument("--dataset", required=True, choices=sorted(VOCABULARY_SIZE))
    parser.add_argument("--ori-env", required=True)
    parser.add_argument("--new-env", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--experiment-seed", default=2024, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_original_gen_cache(
        dataset=args.dataset,
        original_environment=args.ori_env,
        target_environment=args.new_env,
        threshold=args.threshold,
        experiment_seed=args.experiment_seed,
        cache_dir=args.output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
