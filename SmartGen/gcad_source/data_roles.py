from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DataRole(str, Enum):
    SOURCE_NORMAL = "source_normal"
    SOURCE_VALIDATION = "source_validation"
    TARGET_METADATA = "target_metadata"
    TARGET_NORMAL_EVAL = "target_normal_eval"
    TARGET_ATTACK_EVAL = "target_attack_eval"


@dataclass(frozen=True)
class RoleBoundPath:
    path: Path
    role: DataRole

    @classmethod
    def build(cls, path: str | Path, role: str | DataRole) -> "RoleBoundPath":
        return cls(Path(path).expanduser().resolve(), DataRole(role))

