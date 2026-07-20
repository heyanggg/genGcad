from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_boundary import guarded_open
from .data_roles import DataRole, RoleBoundPath


@dataclass(frozen=True)
class Event:
    day: int
    hour_bin: int
    device_id: int
    action_id: int


@dataclass
class TensorizedData:
    sequences: list[np.ndarray]
    vocabulary: list[str]
    valid_window_masks: list[np.ndarray]
    skipped_sequences: list[dict]
    metadata: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceEventTensorizer:
    def __init__(
        self,
        channel_mode: str = "device_action",
        time_bin: float = 3.0,
        occurrence_mode: str = "binary",
        history_length: int = 4,
        device_names: Mapping[int, str] | None = None,
        action_names: Mapping[int, str] | None = None,
        unknown_policy: str = "error",
    ):
        if channel_mode not in {"device", "device_action"}:
            raise ValueError("channel_mode must be device or device_action")
        if occurrence_mode not in {"binary", "count"}:
            raise ValueError("occurrence_mode must be binary or count")
        if time_bin <= 0:
            raise ValueError("time_bin must be positive")
        if unknown_policy not in {"error", "skip"}:
            raise ValueError("unknown_policy must be error or skip")
        self.channel_mode = channel_mode
        self.time_bin = float(time_bin)
        self.occurrence_mode = occurrence_mode
        self.history_length = int(history_length)
        self.device_names = dict(device_names or {})
        self.action_names = dict(action_names or {})
        self.unknown_policy = unknown_policy

    @staticmethod
    def parse_sequence(sequence: Sequence[int]) -> list[Event]:
        if len(sequence) % 4:
            raise ValueError("SmartGen sequence length must be divisible by four")
        return [Event(*(int(value) for value in sequence[index : index + 4])) for index in range(0, len(sequence), 4)]

    def _channel(self, event: Event) -> str:
        if self.device_names and event.device_id not in self.device_names:
            raise KeyError(f"unknown device id {event.device_id}")
        if self.action_names and event.action_id not in self.action_names:
            raise KeyError(f"unknown action id {event.action_id}")
        device = self.device_names.get(event.device_id, f"device_{event.device_id}")
        if self.channel_mode == "device":
            return device
        action = self.action_names.get(event.action_id, f"action_{event.action_id}")
        return action if ":" in action else f"{device}:{action}"

    def _prepare(self, raw_sequences: Sequence[Sequence[int]]):
        parsed: list[tuple[int, list[Event]]] = []
        skipped: list[dict] = []
        channels: set[str] = set()
        for sequence_index, raw in enumerate(raw_sequences):
            try:
                events = self.parse_sequence(raw)
                event_channels = [self._channel(event) for event in events]
            except (KeyError, TypeError, ValueError) as exc:
                if self.unknown_policy == "error":
                    raise
                skipped.append({"sequence_index": sequence_index, "reason": str(exc)})
                continue
            if not events:
                skipped.append({"sequence_index": sequence_index, "reason": "empty_sequence"})
                continue
            parsed.append((sequence_index, events))
            channels.update(event_channels)
        return parsed, skipped, sorted(channels)

    def tensorize(self, raw_sequences: Sequence[Sequence[int]]) -> TensorizedData:
        parsed, skipped, vocabulary = self._prepare(raw_sequences)
        channel_index = {name: index for index, name in enumerate(vocabulary)}
        tensors: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        kept_indices: list[int] = []
        week_hours = 7 * 24
        for original_index, events in parsed:
            absolute_hours: list[float] = []
            offset = 0.0
            previous = None
            for event in events:
                current = event.day * 24 + event.hour_bin * 3 + offset
                if previous is not None and current < previous:
                    offset += week_hours
                    current += week_hours
                absolute_hours.append(current)
                previous = current
            first_slot = int(np.floor(absolute_hours[0] / self.time_bin))
            slots = [int(np.floor(value / self.time_bin)) - first_slot for value in absolute_hours]
            tensor = np.zeros((max(slots) + 1, len(vocabulary)), dtype=np.float32)
            for event, slot in zip(events, slots):
                column = channel_index[self._channel(event)]
                if self.occurrence_mode == "binary":
                    tensor[slot, column] = 1.0
                else:
                    tensor[slot, column] += 1.0
            tensors.append(tensor)
            masks.append(np.arange(len(tensor)) >= self.history_length)
            kept_indices.append(original_index)
        valid_windows = sum(max(0, len(item) - self.history_length) for item in tensors)
        metadata = {
            "channel_mode": self.channel_mode,
            "time_bin": self.time_bin,
            "occurrence_mode": self.occurrence_mode,
            "history_length": self.history_length,
            "sequence_count": len(tensors),
            "channel_count": len(vocabulary),
            "valid_window_count": valid_windows,
            "kept_sequence_indices": kept_indices,
            "padding_mask_semantics": "False=observed; windows never cross a sequence boundary",
        }
        return TensorizedData(tensors, vocabulary, masks, skipped, metadata)

    def tensorize_pickle(self, source: RoleBoundPath, output_dir: str | Path) -> TensorizedData:
        if source.role is not DataRole.SOURCE_NORMAL:
            raise ValueError("tensorizer input must have source_normal role")
        with guarded_open(source, "tensorize", "rb") as handle:
            raw_sequences = pickle.load(handle)
        result = self.tensorize(raw_sequences)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "tensorized_sequences.npz", **{f"seq_{i:06d}": x for i, x in enumerate(result.sequences)})
        np.savez_compressed(output / "valid_window_masks.npz", **{f"seq_{i:06d}": x for i, x in enumerate(result.valid_window_masks)})
        (output / "channel_vocabulary.json").write_text(json.dumps(result.vocabulary, indent=2), encoding="utf-8")
        (output / "tensor_metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
        (output / "skipped_sequences.json").write_text(json.dumps(result.skipped_sequences, indent=2), encoding="utf-8")
        manifest = {
            "input_path": str(source.path),
            "input_role": source.role.value,
            "input_sha256": sha256_file(source.path),
            "uses_target_behavior": False,
            "sequence_count": result.metadata["sequence_count"],
            "valid_window_count": result.metadata["valid_window_count"],
        }
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return result


def load_tensorized(path: str | Path) -> tuple[list[np.ndarray], list[str], dict]:
    directory = Path(path)
    with np.load(directory / "tensorized_sequences.npz") as archive:
        sequences = [archive[key] for key in sorted(archive.files)]
    vocabulary = json.loads((directory / "channel_vocabulary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "tensor_metadata.json").read_text(encoding="utf-8"))
    return sequences, vocabulary, metadata

