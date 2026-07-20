from __future__ import annotations

from collections.abc import Iterable, Mapping


FORBIDDEN_SEMANTIC_TOKENS = frozenset({
    "", "none", "location", "unknown", "padding", "context", "date", "day",
    "time", "sequence_id", "room", "room metadata",
})


def is_valid_semantic_channel(channel: object) -> bool:
    if not isinstance(channel, str) or not channel.strip():
        return False
    parts = [part.strip().lower() for part in channel.split(":")]
    if len(parts) > 2 or any(part in FORBIDDEN_SEMANTIC_TOKENS for part in parts):
        return False
    return True


def require_valid_semantic_channels(channels: Iterable[object], artifact: str = "relation") -> None:
    invalid = sorted({str(channel) for channel in channels if not is_valid_semantic_channel(channel)})
    if invalid:
        raise ValueError(f"{artifact} contains invalid semantic channel(s): {', '.join(invalid)}")


def validate_relation_nodes(relation: Mapping, artifact: str = "stable relation") -> None:
    channels = []
    for edge in relation.get("edges", []):
        channels.extend((edge.get("source"), edge.get("target")))
    require_valid_semantic_channels(channels, artifact)


def sanitize_static_metadata(metadata: Mapping[str, list[str]]) -> dict[str, list[str]]:
    result = {}
    for device, actions in metadata.items():
        if not is_valid_semantic_channel(device):
            continue
        legal = [action for action in actions if is_valid_semantic_channel(f"{device}:{action}")]
        if legal:
            result[device] = legal
    return result
