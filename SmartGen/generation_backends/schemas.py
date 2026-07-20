from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "1.0"
ALLOWED_METHODS = {"baseline", "gcad_gss", "ranking", "both", "random", "symmetric"}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def sequence_fingerprint(events: list[dict]) -> str:
    canonical = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_generation_method(method: str) -> str:
    if method in {"baseline", "ranking"}:
        return "baseline"
    if method in {"gcad_gss", "both"}:
        return "gcad_gss"
    return method


@dataclass(frozen=True)
class ValidationFailure:
    request_id: str | None
    sequence_id: str | None
    category: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

