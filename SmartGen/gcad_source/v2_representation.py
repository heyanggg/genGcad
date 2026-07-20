from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .semantic_channels import is_valid_semantic_channel


@dataclass(frozen=True)
class CanonicalEvent:
    source_id: str
    position: int
    day: int
    hour_bin: int
    device: str
    action: str

    @property
    def channel(self) -> str:
        return self.action


@dataclass
class RepresentationRecord:
    record_id: str
    source_id: str
    day: int | None
    values: np.ndarray
    observed_slot_mask: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def canonicalize_sequences(
    raw_sequences: Sequence[Sequence[int]],
    device_names: Mapping[int, str],
    action_names: Mapping[int, str],
) -> tuple[list[list[CanonicalEvent]], list[dict], list[str]]:
    """Parse explicit SmartGen fields and retain only legal device-action events."""
    parsed: list[list[CanonicalEvent]] = []
    invalid: list[dict] = []
    channels: set[str] = set()
    for sequence_index, raw in enumerate(raw_sequences):
        source_id = f"source_{sequence_index:06d}"
        events: list[CanonicalEvent] = []
        if len(raw) % 4:
            invalid.append({
                "source_id": source_id,
                "reason": "field_count_not_divisible_by_four",
                "value_count": len(raw),
            })
            parsed.append(events)
            continue
        for position in range(0, len(raw), 4):
            day, hour_bin, device_id, action_id = (int(value) for value in raw[position : position + 4])
            device = device_names.get(device_id)
            action = action_names.get(action_id)
            reason = None
            if device is None:
                reason = "unknown_device"
            elif action is None:
                reason = "unknown_action"
            elif not is_valid_semantic_channel(action):
                reason = "forbidden_semantic_field"
            elif not action.startswith(f"{device}:"):
                reason = "device_action_mismatch"
            if reason:
                invalid.append({
                    "source_id": source_id,
                    "position": position // 4,
                    "day": day,
                    "hour_bin": hour_bin,
                    "device_id": device_id,
                    "action_id": action_id,
                    "device": device,
                    "action": action,
                    "reason": reason,
                })
                continue
            event = CanonicalEvent(source_id, position // 4, day, hour_bin, device, action)
            events.append(event)
            channels.add(event.channel)
        parsed.append(events)
    return parsed, invalid, sorted(channels)


def event_position_representation(
    event_sequences: Sequence[Sequence[CanonicalEvent]], vocabulary: Sequence[str]
) -> list[RepresentationRecord]:
    index = {channel: offset for offset, channel in enumerate(vocabulary)}
    records: list[RepresentationRecord] = []
    for sequence_index, events in enumerate(event_sequences):
        if not events:
            continue
        values = np.zeros((len(events), len(vocabulary)), dtype=np.float32)
        for position, event in enumerate(events):
            values[position, index[event.channel]] = 1.0
        records.append(RepresentationRecord(
            record_id=f"event_position_{sequence_index:06d}",
            source_id=events[0].source_id,
            day=None,
            values=values,
            observed_slot_mask=np.ones(len(values), dtype=bool),
        ))
    return records


def continuous_event_time_representation(
    event_sequences: Sequence[Sequence[CanonicalEvent]],
    vocabulary: Sequence[str],
    time_bin_hours: int,
    native_bin_hours: int = 3,
) -> list[RepresentationRecord]:
    """Build true-empty-slot multi-hot records without crossing source or day."""
    if time_bin_hours < native_bin_hours or time_bin_hours % native_bin_hours:
        raise ValueError("time_bin_hours must be a positive multiple of the native 3-hour timestamp bin")
    index = {channel: offset for offset, channel in enumerate(vocabulary)}
    records: list[RepresentationRecord] = []
    for sequence_index, events in enumerate(event_sequences):
        by_day: dict[int, list[CanonicalEvent]] = defaultdict(list)
        for event in events:
            by_day[event.day].append(event)
        for day, day_events in sorted(by_day.items()):
            slots = [(event.hour_bin * native_bin_hours) // time_bin_hours for event in day_events]
            first, last = min(slots), max(slots)
            values = np.zeros((last - first + 1, len(vocabulary)), dtype=np.float32)
            occupied = np.zeros(len(values), dtype=bool)
            for event, slot in zip(day_events, slots):
                values[slot - first, index[event.channel]] = 1.0
                occupied[slot - first] = True
            records.append(RepresentationRecord(
                record_id=f"continuous_{time_bin_hours}h_{sequence_index:06d}_day{day}",
                source_id=day_events[0].source_id,
                day=day,
                values=values,
                # Every row is an observed physical slot; False would mean model padding.
                observed_slot_mask=np.ones(len(values), dtype=bool),
            ))
    return records


def window_count(records: Sequence[RepresentationRecord], history: int) -> int:
    return sum(max(0, len(record.values) - history) for record in records)


def window_origins(records: Sequence[RepresentationRecord], history: int) -> list[dict]:
    return [
        {
            "record_id": record.record_id,
            "source_id": record.source_id,
            "day": record.day,
            "target_position": target,
        }
        for record in records
        for target in range(history, len(record.values))
    ]


def split_by_source(
    records: Sequence[RepresentationRecord], validation_ratio: float, seed: int
) -> tuple[list[RepresentationRecord], list[RepresentationRecord], dict]:
    source_ids = sorted({record.source_id for record in records})
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(source_ids, dtype=object)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(source_ids) * validation_ratio)))
    validation_sources = set(shuffled[:validation_count].tolist())
    train = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    train_sources = {record.source_id for record in train}
    validation_sources_actual = {record.source_id for record in validation}
    overlap = sorted(train_sources & validation_sources_actual)
    manifest = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "train_source_ids": sorted(train_sources),
        "validation_source_ids": sorted(validation_sources_actual),
        "source_overlap": overlap,
        "source_overlap_count": len(overlap),
        "passed": not overlap,
        "uses_target_behavior": False,
    }
    return train, validation, manifest


def collect_windows(records: Sequence[RepresentationRecord], history: int):
    x, y, origins = [], [], []
    for record in records:
        for target in range(history, len(record.values)):
            x.append(record.values[target - history : target])
            y.append(record.values[target])
            origins.append({
                "record_id": record.record_id,
                "source_id": record.source_id,
                "day": record.day,
                "target_position": target,
            })
    if not x:
        channels = records[0].values.shape[1] if records else 0
        return (
            np.empty((0, history, channels), dtype=np.float32),
            np.empty((0, channels), dtype=np.float32),
            origins,
        )
    return np.stack(x), np.stack(y), origins


def representation_statistics(records: Sequence[RepresentationRecord], history: int) -> dict:
    lengths = np.asarray([len(record.values) for record in records], dtype=int)
    occupied = sum(int((record.values.sum(axis=1) > 0).sum()) for record in records)
    total_slots = int(lengths.sum())
    windows = window_count(records, history)
    return {
        "record_count": len(records),
        "source_count": len({record.source_id for record in records}),
        "history": history,
        "window_count": windows,
        "record_length_min": int(lengths.min()) if len(lengths) else 0,
        "record_length_max": int(lengths.max()) if len(lengths) else 0,
        "record_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "empty_slot_ratio": float((total_slots - occupied) / total_slots) if total_slots else 0.0,
        "average_active_channels_per_slot": float(
            sum(float(record.values.sum()) for record in records) / total_slots
        ) if total_slots else 0.0,
        "short_record_count": sum(len(record.values) <= history for record in records),
    }


def map_tss_fragments_to_sources(
    raw_sequences: Sequence[Sequence[int]], split_sequences: Sequence[Sequence[int]]
) -> list[int]:
    mapping: list[int] = []
    split_index = 0
    for source_index, raw in enumerate(raw_sequences):
        reconstructed: list[int] = []
        while split_index < len(split_sequences) and len(reconstructed) < len(raw):
            reconstructed.extend(split_sequences[split_index])
            mapping.append(source_index)
            split_index += 1
        if reconstructed != list(raw):
            raise ValueError(f"TSS fragment mapping diverged at source sequence {source_index}")
    if split_index != len(split_sequences):
        raise ValueError("unmapped TSS fragments remain")
    return mapping


def v1_slot_lengths(raw_sequences: Sequence[Sequence[int]]) -> list[int]:
    lengths = []
    week_hours = 7 * 24
    for raw in raw_sequences:
        absolute = []
        offset = 0
        previous = None
        for position in range(0, len(raw), 4):
            day, hour_bin = int(raw[position]), int(raw[position + 1])
            current = day * 24 + hour_bin * 3 + offset
            if previous is not None and current < previous:
                offset += week_hours
                current += week_hours
            absolute.append(current)
            previous = current
        lengths.append(int((max(absolute) - min(absolute)) / 3) + 1 if absolute else 0)
    return lengths


def write_json(path: str | Path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def save_representation(path: str | Path, records: Sequence[RepresentationRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{record.record_id: record.values for record in records})
