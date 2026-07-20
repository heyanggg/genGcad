from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path, reject_markdown: bool = True) -> list[dict]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if reject_markdown and ("```" in stripped or stripped.startswith("#")):
            raise ValueError(f"Markdown or explanatory text at line {line_number}")
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain one JSON object")
        records.append(value)
    return records

