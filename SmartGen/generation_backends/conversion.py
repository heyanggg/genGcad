from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .response_loader import load_jsonl


def convert_responses_to_smartgen(
    validated_responses_path: str | Path,
    output_dir: str | Path,
    day_mapping: Mapping[str, int],
    hour_mapping: Mapping[str, int],
    device_mapping: Mapping[str, int],
    action_mapping: Mapping[str, int],
) -> Path:
    responses = load_jsonl(validated_responses_path)
    flattened = []
    for response in responses:
        for sequence in response["sequences"]:
            values = []
            for event in sequence["events"]:
                action = event["action"] if ":" in event["action"] else f"{event['device']}:{event['action']}"
                try:
                    values.extend([day_mapping[event["day"]], hour_mapping[event["hour"]], device_mapping[event["device"]], action_mapping[action]])
                except KeyError as exc:
                    raise ValueError(f"event cannot be converted to SmartGen integer schema: {exc}") from exc
            flattened.append(values)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "generated_sequences.pkl"
    with path.open("wb") as handle:
        pickle.dump(flattened, handle)
    metadata = {
        "sequence_count": len(flattened),
        "format": "flat SmartGen integer quadruples [day,hour,device,action]",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(validated_responses_path).resolve()),
        "external_api_called": False,
        "target_behavior_read": False,
    }
    (output / "generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    checksum_lines = []
    for artifact in [Path(validated_responses_path), path, output / "generation_metadata.json"]:
        checksum_lines.append(f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact.name}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return path

