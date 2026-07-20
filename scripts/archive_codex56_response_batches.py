#!/usr/bin/env python3
"""Archive already-authored Codex response records without changing event content."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    raw = args.directory / "generation_responses_raw.jsonl"
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_dir = args.directory / "agent_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(records, 1):
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        path = batch_dir / f"batch_{index:02d}_{record['request_id']}.jsonl"
        path.write_text(payload, encoding="utf-8")
        manifest.append({
            "batch_index": index,
            "request_id": record["request_id"],
            "group_id": record["group_id"],
            "sequence_count": len(record["sequences"]),
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "event_content_modified": False,
        })
    (args.directory / "agent_batch_checksums.json").write_text(
        json.dumps({"batches": manifest, "uses_target_behavior": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = args.directory / "checksums.sha256"
    entries = []
    for path in sorted(item for item in args.directory.rglob("*") if item.is_file() and item != checksum_path):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.directory)}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
