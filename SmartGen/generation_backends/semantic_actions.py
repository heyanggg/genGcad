from __future__ import annotations

from collections import Counter
from typing import Iterable

from SmartGen import dictionary


INVALID_ATOMS = frozenset({
    "", "none", "location", "unknown", "padding", "context", "date", "day", "timestamp",
    "sequence_id", "user_id", "room metadata",
})


def static_legal_actions(metadata: dict[str, list[str]]) -> set[str]:
    legal = set()
    for device, actions in metadata.items():
        for action in actions:
            canonical = canonical_semantic_action(device, action, metadata=None)
            if canonical is not None:
                legal.add(canonical)
    return legal


def canonical_semantic_action(
    device: object,
    action: object,
    metadata: dict[str, list[str]] | None = None,
) -> str | None:
    if not isinstance(device, str) or not isinstance(action, str):
        return None
    device = device.strip()
    action = action.strip()
    if ":" in action:
        action_device, action = action.split(":", 1)
        if device and action_device != device:
            return None
        device = action_device
    if device.lower() in INVALID_ATOMS or action.lower() in INVALID_ATOMS:
        return None
    if not device or not action:
        return None
    canonical = f"{device}:{action}"
    if metadata is not None and canonical not in static_legal_actions(metadata):
        return None
    return canonical


def numeric_semantic_actions(
    sequence: list[int],
    dataset: str,
    metadata: dict[str, list[str]],
) -> list[str]:
    inverse = {value: key for key, value in getattr(dictionary, f"{dataset}_actions").items()}
    actions = []
    for value in sequence[3::4]:
        channel = inverse.get(int(value))
        if not channel or ":" not in channel:
            continue
        device, action = channel.split(":", 1)
        canonical = canonical_semantic_action(device, action, metadata)
        if canonical is not None:
            actions.append(canonical)
    return actions


def event_semantic_actions(events: Iterable[dict], metadata: dict[str, list[str]]) -> list[str]:
    actions = []
    for event in events:
        if not isinstance(event, dict):
            continue
        canonical = canonical_semantic_action(event.get("device"), event.get("action"), metadata)
        if canonical is not None:
            actions.append(canonical)
    return actions


def classify_action_spaces(
    source_sequences: list[list[int]],
    dataset: str,
    metadata: dict[str, list[str]],
) -> dict:
    legal = static_legal_actions(metadata)
    source_sequence_support = Counter()
    source_token_support = Counter()
    for sequence in source_sequences:
        actions = numeric_semantic_actions(sequence, dataset, metadata)
        source_token_support.update(actions)
        source_sequence_support.update(set(actions))
    observed = set(source_token_support)
    return {
        "source_observed_target_legal_actions": sorted(observed),
        "target_metadata_only_actions": sorted(legal - observed),
        "invalid_actions": [],
        "source_token_support": dict(sorted(source_token_support.items())),
        "source_sequence_support": dict(sorted(source_sequence_support.items())),
        "static_legal_action_count": len(legal),
    }
