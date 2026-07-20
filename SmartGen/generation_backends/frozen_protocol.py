from __future__ import annotations

import hashlib
from pathlib import Path


REPLICATE_2_REQUESTS_SHA256 = "7c0fd1766598b4cfacdbcfac31fc1f672416dbc80c652ed4ddd8f4d9439c6860"


def verify_frozen_requests(path: str | Path, expected_sha256: str = REPLICATE_2_REQUESTS_SHA256) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"frozen generation requests SHA256 mismatch: expected {expected_sha256}, got {digest}")
    return digest
