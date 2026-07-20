from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_relation(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".npy":
        return {"matrix": np.load(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def relation_checksum(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

