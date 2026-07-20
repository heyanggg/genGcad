from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from SmartGen import dictionary


FROZEN_CONFIG_SHA256 = "57160eac5399f2ed095d889af85d85565ac44bbcb1e8196ed5f12f2c8ec05dd7"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_frozen_config(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    actual = sha256_file(path)
    if actual != FROZEN_CONFIG_SHA256:
        raise ValueError(
            f"source-copy-safe-v1 config SHA256 mismatch: expected {FROZEN_CONFIG_SHA256}, got {actual}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != "source-copy-safe-v1" or config.get("uses_target_behavior") is not False:
        raise ValueError("invalid source-copy-safe-v1 config")
    return _freeze(config)


def action_sequence_fingerprint(events: list[dict]) -> str:
    actions = [
        f"{event['device']}:{event['action'].split(':', 1)[-1]}"
        for event in events
    ]
    canonical = json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def numeric_action_sequence(sequence: list[int], dataset: str) -> list[str]:
    inverse = {value: key for key, value in getattr(dictionary, f"{dataset}_actions").items()}
    return [inverse[int(sequence[index])] for index in range(3, len(sequence), 4)]


def build_source_denylist(
    group_plan: dict,
    *,
    dataset: str,
    source_context: str,
    compression_threshold: float,
    source_root: str | Path,
) -> dict:
    entries = []
    for group in group_plan["groups"]:
        group_id = str(group["group_id"])
        source_path = (
            Path(source_root) / dataset / source_context
            / f"trn_day_{group_id}_SPPC_th={compression_threshold}.pkl"
        )
        with source_path.open("rb") as handle:
            sequences = pickle.load(handle)
        for index, sequence in enumerate(sequences):
            actions = numeric_action_sequence(sequence, dataset)
            events = [
                {"device": channel.split(":", 1)[0], "action": channel.split(":", 1)[1]}
                for channel in actions
            ]
            entries.append({
                "group_id": group_id,
                "representative_index": index,
                "actions": actions,
                "action_fingerprint": action_sequence_fingerprint(events),
            })
    return {
        "version": "source-copy-safe-v1",
        "match_unit": "ordered device:action sequence, ignoring generated day/hour",
        "source_representative_count": len(entries),
        "unique_action_fingerprint_count": len({item["action_fingerprint"] for item in entries}),
        "entries": entries,
        "uses_target_behavior": False,
    }


def write_source_denylist(output_dir: str | Path, payload: dict) -> tuple[Path, str]:
    path = Path(output_dir) / "source_representative_denylist.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, sha256_file(path)


def validate_replacement_mapping(mapping: list[dict], config: Mapping[str, Any]) -> None:
    maximum = int(config["maximum_replacement_candidates"])
    if len(mapping) > maximum:
        raise ValueError(f"replacement count {len(mapping)} exceeds frozen maximum {maximum}")
    allowed = set(config["allowed_automatic_replacement_categories"])
    for item in mapping:
        reason = item.get("category")
        if reason not in allowed:
            raise ValueError(f"replacement category is not allowed: {reason}")
        material = json.dumps(item, ensure_ascii=False).lower()
        if "semantic" in material or "target_result" in material or "target result" in material:
            raise ValueError("semantic-score and target-result replacements are forbidden")


def verify_source_copy_safe_artifacts(directory: str | Path) -> dict:
    directory = Path(directory)
    protocol_path = directory / "source_copy_safe_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config_path = Path(protocol["config_path"])
    load_frozen_config(config_path)
    checks = {
        "generation_requests.jsonl": sha256_file(directory / "generation_requests.jsonl"),
        "source_copy_safe_protocol.json": sha256_file(protocol_path),
        "source_copy_safe_config": sha256_file(config_path),
        "source_representative_denylist.json": sha256_file(directory / "source_representative_denylist.json"),
    }
    expected = json.loads((directory / "pre_generation_checksums.json").read_text(encoding="utf-8"))
    if checks != expected["sha256"]:
        raise ValueError("source-copy-safe-v1 pre-generation artifact SHA256 mismatch")
    return checks
