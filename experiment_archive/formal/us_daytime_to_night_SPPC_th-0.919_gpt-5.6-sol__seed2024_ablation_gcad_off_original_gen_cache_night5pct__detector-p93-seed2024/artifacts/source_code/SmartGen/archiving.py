from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .experiment import (
        REPOSITORY_ROOT,
        SMARTGEN_ROOT,
        ExperimentConfig,
        atomic_write_json,
        atomic_write_text,
    )
except ImportError:  # Direct execution from SmartGen/.
    from experiment import (
        REPOSITORY_ROOT,
        SMARTGEN_ROOT,
        ExperimentConfig,
        atomic_write_json,
        atomic_write_text,
    )


ARCHIVE_ROOT = REPOSITORY_ROOT / "experiment_archive"
ARCHIVE_STATUSES = {
    "completed",
    "verified_candidates",
    "formal",
    "diagnostic",
    "failed",
}
SAFE_ARCHIVE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ArchiveFile:
    source: Path
    destination: Path
    required: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(path: Path) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive destination must be a safe relative path: {path}")


def _portable_source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def rebuild_registry(archive_root: Path = ARCHIVE_ROOT) -> list[dict[str, Any]]:
    entries = []
    for status in sorted(ARCHIVE_STATUSES):
        status_root = archive_root / status
        if not status_root.exists():
            continue
        for manifest_path in sorted(status_root.glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            relative_archive = manifest_path.parent.relative_to(archive_root)
            metrics = manifest.get("metrics", {}).get("metrics", {})
            entries.append(
                {
                    "archive_id": manifest["archive_id"],
                    "status": manifest["status"],
                    "archive_path": str(relative_archive),
                    "dataset": manifest.get("config", {}).get("dataset"),
                    "original_environment": manifest.get("config", {}).get(
                        "original_environment"
                    ),
                    "target_environment": manifest.get("config", {}).get(
                        "target_environment"
                    ),
                    "experiment_seed": manifest.get("config", {}).get(
                        "experiment_seed"
                    ),
                    "f1_score": metrics.get("f1_score"),
                    "accuracy": metrics.get("accuracy"),
                    "reconstructed": manifest.get("reconstructed", False),
                    "artifact_count": manifest["artifact_count"],
                    "artifact_bytes": manifest["artifact_bytes"],
                }
            )
    entries.sort(key=lambda item: (item["status"], item["archive_id"]))
    atomic_write_json(archive_root / "registry.json", entries)
    return entries


def verify_archive(archive_path: Path) -> list[str]:
    """Return archive integrity errors; an empty list means the archive is intact."""
    manifest_path = archive_path / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        try:
            _validate_relative_path(relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        path = archive_path / relative
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if path.stat().st_size != record["size"]:
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != record["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
    return errors


def promote_archive(
    archive_id: str,
    *,
    source_status: str = "completed",
    target_status: str = "verified_candidates",
    archive_root: Path = ARCHIVE_ROOT,
) -> Path:
    if not SAFE_ARCHIVE_ID.fullmatch(archive_id):
        raise ValueError(f"unsafe archive id: {archive_id!r}")
    if source_status not in ARCHIVE_STATUSES or target_status not in ARCHIVE_STATUSES:
        raise ValueError("unsupported archive status")
    if source_status == target_status:
        raise ValueError("source and target archive statuses must differ")
    source = archive_root / source_status / archive_id
    target = archive_root / target_status / archive_id
    if not source.is_dir():
        raise FileNotFoundError(f"archive does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"archive target already exists: {target}")
    errors = verify_archive(source)
    if errors:
        raise ValueError("archive integrity check failed: " + "; ".join(errors))
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = target_status
    manifest["promoted_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    readme_path = target / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        readme = re.sub(
            r"^Status: `[^`]+`$",
            f"Status: `{target_status}`",
            readme,
            count=1,
            flags=re.MULTILINE,
        )
        atomic_write_text(readme_path, readme)
    rebuild_registry(archive_root)
    return target


def archive_files(
    archive_id: str,
    status: str,
    config: dict[str, Any],
    files: Iterable[ArchiveFile],
    *,
    metrics: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    missing_artifacts: Iterable[str] = (),
    reconstructed: bool = False,
    storage_mode: str = "copy",
    archive_root: Path = ARCHIVE_ROOT,
) -> Path:
    """Create an immutable, checksummed experiment archive."""
    if not SAFE_ARCHIVE_ID.fullmatch(archive_id):
        raise ValueError(f"unsafe archive id: {archive_id!r}")
    if status not in ARCHIVE_STATUSES:
        raise ValueError(f"unsupported archive status: {status!r}")
    if storage_mode not in {"copy", "hardlink"}:
        raise ValueError("storage_mode must be 'copy' or 'hardlink'")

    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / status / archive_id
    existing_locations = [
        archive_root / candidate / archive_id
        for candidate in ARCHIVE_STATUSES
        if (archive_root / candidate / archive_id).exists()
    ]
    if existing_locations and target not in existing_locations:
        raise ValueError(
            f"archive {archive_id!r} already exists under "
            f"{existing_locations[0].parent.name!r}"
        )
    if target.exists():
        existing = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if (
            existing.get("archive_id") != archive_id
            or existing.get("status") != status
            or existing.get("config") != config
        ):
            raise ValueError(f"archive collision at {target}")
        errors = verify_archive(target)
        if errors:
            raise ValueError("existing archive integrity check failed: " + "; ".join(errors))
        return target

    normalized: dict[Path, ArchiveFile] = {}
    for item in files:
        source = item.source if item.source.is_absolute() else SMARTGEN_ROOT / item.source
        destination = Path("artifacts") / item.destination
        _validate_relative_path(destination)
        if destination in normalized and normalized[destination].source != source:
            raise ValueError(f"duplicate archive destination: {destination}")
        normalized[destination] = ArchiveFile(source, destination, item.required)

    staging = Path(tempfile.mkdtemp(prefix=f".{archive_id}-", dir=archive_root))
    records = []
    try:
        for destination, item in sorted(normalized.items(), key=lambda pair: str(pair[0])):
            if not item.source.is_file():
                if item.required:
                    raise FileNotFoundError(f"required archive artifact is missing: {item.source}")
                continue
            output = staging / destination
            output.parent.mkdir(parents=True, exist_ok=True)
            if storage_mode == "hardlink":
                os.link(item.source, output)
            else:
                shutil.copy2(item.source, output)
            records.append(
                {
                    "path": str(destination),
                    "source": _portable_source_path(item.source),
                    "size": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            )

        manifest = {
            "schema_version": 1,
            "archive_id": archive_id,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reconstructed": reconstructed,
            "storage_mode": storage_mode,
            "config": config,
            "metrics": metrics,
            "provenance": provenance or {},
            "missing_artifacts": sorted(set(missing_artifacts)),
            "artifact_count": len(records),
            "artifact_bytes": sum(record["size"] for record in records),
            "files": records,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        checksums = "".join(
            f"{record['sha256']}  {record['path']}\n" for record in records
        )
        (staging / "checksums.sha256").write_text(checksums, encoding="utf-8")
        metric_values = (metrics or {}).get("metrics", {})
        readme = (
            f"# {archive_id}\n\n"
            f"Status: `{status}`\n\n"
            f"Reconstructed metadata: `{str(reconstructed).lower()}`\n\n"
            f"Artifacts: `{len(records)}` files, `{manifest['artifact_bytes']}` bytes\n\n"
            f"F1: `{metric_values.get('f1_score', 'n/a')}`\n\n"
            f"Accuracy: `{metric_values.get('accuracy', 'n/a')}`\n\n"
            "Heavy artifacts are local and ignored by Git. Use `checksums.sha256` "
            "to verify them. See `manifest.json` for source-to-archive mappings.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    rebuild_registry(archive_root)
    return target


def _add_if_present(
    files: list[ArchiveFile], source: Path, destination: Path, required: bool = False
) -> None:
    files.append(ArchiveFile(source, destination, required))


def collect_current_experiment_files(
    config: ExperimentConfig,
    generation_manifest: dict[str, Any],
    anomaly_result: dict[str, Any],
) -> list[ArchiveFile]:
    files: list[ArchiveFile] = []
    for path in sorted(SMARTGEN_ROOT.glob("*.py")):
        _add_if_present(
            files,
            path,
            Path("source_code/SmartGen") / path.name,
            True,
        )
    tests_root = REPOSITORY_ROOT / "tests"
    for path in sorted(tests_root.glob("*.py")):
        _add_if_present(
            files,
            path,
            Path("source_code/tests") / path.name,
            True,
        )
    source_dir = Path("IoT_data") / config.dataset / config.original_environment
    for name in ("trn.pkl", "vld.pkl", "split_trn.pkl", "split_vld.pkl", "rs_trn.pkl", "rs_vld.pkl"):
        _add_if_present(files, source_dir / name, Path("source_snapshot") / name)
    for day in range(7):
        _add_if_present(
            files,
            source_dir / f"trn_day_{day}.pkl",
            Path("source_snapshot/day_groups") / f"trn_day_{day}.pkl",
        )
    threshold = format(config.threshold, "g")
    for path in sorted(source_dir.glob(f"trn_day_*_{config.method}_th={threshold}*.pkl")):
        _add_if_present(files, path, Path("ssc/selected_sequences") / path.name)
    _add_if_present(
        files,
        source_dir / "action_transitions.json",
        Path("gss/action_transitions.json"),
        required=True,
    )
    _add_if_present(
        files,
        Path("IoT_model")
        / f"Transformer_{config.dataset}_{config.original_environment}_15epoch.pth",
        Path("ssc/model.pth"),
    )
    compression = generation_manifest.get("compression", {})
    if compression.get("training_mode") == "reused_original_gen_cache":
        cache_dir = Path(compression["selection_source_dir"])
        _add_if_present(
            files,
            cache_dir / "manifest.json",
            Path("original_gen_cache/manifest.json"),
            True,
        )
        _add_if_present(
            files,
            cache_dir / "model.pth",
            Path("original_gen_cache/model.pth"),
            True,
        )
        for filename in sorted(compression.get("selection_sha256", {})):
            _add_if_present(
                files,
                cache_dir / "selected_sequences" / filename,
                Path("original_gen_cache/selected_sequences") / filename,
                True,
            )

    gcad_path = generation_manifest.get("gcad", {}).get("path")
    if gcad_path:
        _add_if_present(files, Path(gcad_path), Path("gcad/gcad_hints.json"), True)

    run_root = Path("runs") / config.experiment_name
    for path in sorted((SMARTGEN_ROOT / run_root).rglob("*")):
        if path.is_file():
            relative = path.relative_to(SMARTGEN_ROOT / run_root)
            _add_if_present(files, path, Path("generation/run_record") / relative, True)
    for category, output_name in generation_manifest.get("category_outputs", {}).items():
        output = Path(output_name)
        _add_if_present(
            files,
            output,
            Path("generation/raw_categories") / f"category_{category}.pkl",
            True,
        )
        parsed = output.with_name(output.stem + "_seq.pkl")
        _add_if_present(
            files,
            parsed,
            Path("generation/parsed_categories") / f"category_{category}.pkl",
            True,
        )

    generated = Path(generation_manifest["outputs"]["generated_numeric"])
    for path in sorted((SMARTGEN_ROOT / generated.parent).glob(generated.stem + "*.pkl")):
        _add_if_present(files, path, Path("tof") / path.name, True)
    _add_if_present(
        files,
        Path("check_model")
        / f"best_{config.dataset}_{config.artifact_model}_{config.method}.pth",
        Path("tof/model.pth"),
        True,
    )

    metrics_path = Path(anomaly_result["metrics_path"])
    anomaly_root = SMARTGEN_ROOT / metrics_path.parent
    for path in sorted(anomaly_root.glob("*")):
        if path.is_file():
            _add_if_present(
                files,
                path,
                Path("anomaly_detection") / path.name,
                True,
            )
    return files


def archive_completed_experiment(
    config: ExperimentConfig,
    generation_manifest: dict[str, Any],
    anomaly_result: dict[str, Any],
    *,
    status: str = "completed",
    storage_mode: str = "copy",
    automatic_archive: bool = True,
    archive_root: Path = ARCHIVE_ROOT,
) -> Path:
    percentile = format(float(anomaly_result["validation_percentile"]), "g")
    archive_id = (
        f"{config.experiment_name}__detector-p{percentile}-"
        f"seed{anomaly_result['seed']}"
    )
    return archive_files(
        archive_id,
        status,
        {
            **asdict(config),
            "gcad_seeds": list(config.gcad_seeds),
            "artifact_model": config.artifact_model,
            "validation_percentile": anomaly_result["validation_percentile"],
        },
        collect_current_experiment_files(config, generation_manifest, anomaly_result),
        metrics=anomaly_result,
        provenance={
            "generation_manifest": generation_manifest,
            "automatic_archive": automatic_archive,
        },
        missing_artifacts=(
            ["tof_candidate_scratch_files_not_preserved"]
            if generation_manifest.get("tof", {}).get("intermediates_preserved") is False
            else []
        ),
        storage_mode=storage_mode,
        archive_root=archive_root,
    )


def _find_archive(archive_id: str, archive_root: Path) -> Path:
    matches = [
        archive_root / status / archive_id
        for status in ARCHIVE_STATUSES
        if (archive_root / status / archive_id).is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one archive named {archive_id!r}, found {len(matches)}"
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SmartGen experiment archive manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive_id", nargs="?")
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("archive_id")
    promote_parser.add_argument("--from-status", default="completed", choices=sorted(ARCHIVE_STATUSES))
    promote_parser.add_argument(
        "--to-status", default="verified_candidates", choices=sorted(ARCHIVE_STATUSES)
    )

    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(rebuild_registry(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "promote":
        print(
            promote_archive(
                args.archive_id,
                source_status=args.from_status,
                target_status=args.to_status,
            )
        )
        return 0

    paths = (
        [_find_archive(args.archive_id, ARCHIVE_ROOT)]
        if args.archive_id
        else [
            path.parent
            for status in ARCHIVE_STATUSES
            for path in sorted((ARCHIVE_ROOT / status).glob("*/manifest.json"))
        ]
    )
    failed = False
    for path in paths:
        errors = verify_archive(path)
        print(f"{'FAIL' if errors else 'OK'} {path}")
        for error in errors:
            print(f"  {error}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
