from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMARTGEN_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SMARTGEN_ROOT.parent
VALID_ENVIRONMENT_PAIRS = {
    "winter": "spring",
    "daytime": "night",
    "single": "multiple",
}
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def enter_smartgen_root() -> Path:
    """Make original SmartGen relative paths stable from any launch directory."""
    os.chdir(SMARTGEN_ROOT)
    return SMARTGEN_ROOT


def validate_label(value: str, name: str) -> str:
    if not SAFE_LABEL.fullmatch(value):
        raise ValueError(
            f"{name} must contain only letters, numbers, '.', '_' and '-': {value!r}"
        )
    return value


def validate_environment_pair(original: str, target: str) -> None:
    expected = VALID_ENVIRONMENT_PAIRS.get(original)
    if expected != target:
        raise ValueError(
            f"unsupported environment transition {original!r} -> {target!r}; "
            f"expected {original!r} -> {expected!r}"
        )


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_pickle_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    original_environment: str
    target_environment: str
    method: str
    threshold: float
    model: str
    run_id: str
    experiment_seed: int
    gcad_seeds: tuple[int, ...]
    gcad_history: int
    gcad_epochs: int
    gcad_mode: str
    gcad_prompt_max_relationships: int
    gcad_prompt_max_per_target: int
    codex_reasoning_effort: str
    prompt_profile: str

    def __post_init__(self) -> None:
        if self.dataset not in {"fr", "sp", "us"}:
            raise ValueError(f"unsupported dataset: {self.dataset!r}")
        if self.method not in {"SPPC", "similarity", "instance"}:
            raise ValueError(f"unsupported compression method: {self.method!r}")
        validate_environment_pair(self.original_environment, self.target_environment)
        validate_label(self.model, "model")
        validate_label(self.run_id, "run_id")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.gcad_history <= 0:
            raise ValueError("gcad_history must be positive")
        if self.gcad_epochs <= 0:
            raise ValueError("gcad_epochs must be positive")
        if not self.gcad_seeds:
            raise ValueError("at least one GCAD seed is required")
        if len(set(self.gcad_seeds)) != len(self.gcad_seeds):
            raise ValueError("GCAD seeds must be unique")
        if self.gcad_mode not in {"auto", "off", "require"}:
            raise ValueError("unsupported GCAD mode")
        if self.gcad_prompt_max_relationships <= 0:
            raise ValueError("GCAD prompt relationship limit must be positive")
        if self.gcad_prompt_max_per_target <= 0:
            raise ValueError("GCAD per-target prompt relationship limit must be positive")
        if self.codex_reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported Codex reasoning effort")
        if self.prompt_profile not in {"original", "environment-aware", "legacy"}:
            raise ValueError("unsupported prompt profile")

    @property
    def artifact_model(self) -> str:
        return f"{self.model}__{self.run_id}"

    @property
    def experiment_name(self) -> str:
        threshold = format(self.threshold, "g")
        return (
            f"{self.dataset}_{self.original_environment}_to_{self.target_environment}_"
            f"{self.method}_th-{threshold}_{self.artifact_model}"
        )


class ExperimentRun:
    """Filesystem layout and manifest for one immutable generation run."""

    def __init__(self, config: ExperimentConfig, runs_root: Path | None = None):
        self.config = config
        self.root = (runs_root or SMARTGEN_ROOT / "runs") / config.experiment_name
        self.prompts = self.root / "prompts"
        self.responses = self.root / "responses"
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema_version": 2,
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "config": {
                **asdict(config),
                "gcad_seeds": list(config.gcad_seeds),
                "artifact_model": config.artifact_model,
            },
            "completed_categories": [],
            "outputs": {},
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing_config = existing.get("config", {})
            expected_config = {
                **asdict(config),
                "gcad_seeds": list(config.gcad_seeds),
                "artifact_model": config.artifact_model,
            }
            if existing_config != expected_config:
                raise ValueError(
                    f"run_id {config.run_id!r} already exists with different settings; "
                    "choose a new run_id"
                )
            self.manifest = existing

    def update(self, status: str | None = None, **values: Any) -> None:
        if status is not None:
            self.manifest["status"] = status
        self.manifest.update(values)
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.manifest_path, self.manifest)

    def record_category(
        self, category: str, output_path: Path, prompt_sha256: str
    ) -> None:
        completed = set(str(item) for item in self.manifest["completed_categories"])
        completed.add(str(category))
        self.manifest["completed_categories"] = sorted(completed)
        self.manifest.setdefault("category_outputs", {})[str(category)] = str(output_path)
        self.manifest.setdefault("prompt_sha256", {})[str(category)] = prompt_sha256
        self.update(status="generating")

    def prompt_matches(self, category: str, prompt: str) -> bool:
        expected = self.manifest.get("prompt_sha256", {}).get(str(category))
        return expected == hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def prompt_path(self, category: str) -> Path:
        return self.prompts / f"category_{category}.txt"

    def response_path(self, category: str) -> Path:
        return self.responses / f"category_{category}.txt"
