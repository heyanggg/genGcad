from __future__ import annotations

from pathlib import Path
from typing import Protocol


class GenerationBackend(Protocol):
    backend_type: str

    def export(self, output_dir: str | Path, **kwargs) -> list[dict]: ...

    def validate(self, output_dir: str | Path, **kwargs) -> dict: ...

    def convert(self, output_dir: str | Path, **kwargs) -> Path: ...

