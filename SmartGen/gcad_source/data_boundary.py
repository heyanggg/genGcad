from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .data_roles import DataRole, RoleBoundPath


class DataBoundaryError(PermissionError):
    """Raised before a forbidden role is opened by a pipeline stage."""


STAGE_ROLES: dict[str, frozenset[DataRole]] = {
    "tensorize": frozenset({DataRole.SOURCE_NORMAL}),
    "train": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION}),
    "extract_relations": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION}),
    "stability": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION}),
    "fuse_gss": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION, DataRole.TARGET_METADATA}),
    "prompt": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION, DataRole.TARGET_METADATA}),
    "generation": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION, DataRole.TARGET_METADATA}),
    "tof": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION, DataRole.TARGET_METADATA}),
    "ranking": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION, DataRole.TARGET_METADATA}),
    "threshold": frozenset({DataRole.SOURCE_NORMAL, DataRole.SOURCE_VALIDATION}),
    "final_evaluation": frozenset(DataRole),
}


def require_roles(stage: str, artifacts: Iterable[RoleBoundPath]) -> None:
    if stage not in STAGE_ROLES:
        raise ValueError(f"unknown data-boundary stage: {stage}")
    allowed = STAGE_ROLES[stage]
    forbidden = [item for item in artifacts if item.role not in allowed]
    if forbidden:
        details = ", ".join(f"{item.role.value}:{item.path}" for item in forbidden)
        raise DataBoundaryError(f"{stage} refuses forbidden data role(s): {details}")


def guarded_open(artifact: RoleBoundPath, stage: str, mode: str = "rb"):
    require_roles(stage, [artifact])
    return artifact.path.open(mode)


def boundary_declaration() -> dict[str, bool]:
    return {
        "uses_target_behavior": False,
        "uses_target_normal": False,
        "uses_target_attack": False,
        "uses_target_labels": False,
    }


def assert_no_target_behavior_paths(stage: str, artifacts: Iterable[RoleBoundPath]) -> None:
    """Explicit alias used by adapters immediately before filesystem access."""
    require_roles(stage, artifacts)

