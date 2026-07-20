from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GCADConfig:
    enabled: bool = True
    channel_mode: str = "device_action"
    time_bin: float = 3.0
    occurrence_mode: str = "binary"
    history_length: int = 4
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 50
    patience: int = 7
    seeds: list[int] = field(default_factory=lambda: [2024, 2025, 2026])
    bootstrap_count: int = 1
    validation_ratio: float = 0.2
    gradient_clip: float = 1.0
    gradient_aggregation: str = "mean"
    lag_aggregation: str = "max"
    stability_threshold: float = 0.25
    occurrence_threshold: float = 0.5
    direction_consistency_threshold: float = 0.5
    edge_threshold: float = 0.001
    top_k: int = 5
    fusion_alpha: float = 0.2
    fusion_mode: str = "rerank_existing"
    allow_new_edges: bool = False
    max_new_edges_per_source: int = 1
    ranking_enabled: bool = True
    ranking_weight: float = 1.0
    device: str = "cuda"
    num_workers: int = 0

    def validate(self) -> None:
        if self.channel_mode not in {"device", "device_action"}:
            raise ValueError("channel_mode must be device or device_action")
        if self.occurrence_mode not in {"binary", "count"}:
            raise ValueError("occurrence_mode must be binary or count")
        if self.time_bin <= 0 or self.history_length < 1:
            raise ValueError("time_bin and history_length must be positive")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("validation_ratio must be in [0, 1)")
        if not 0 <= self.fusion_alpha <= 0.3:
            raise ValueError("fusion_alpha must be in [0, 0.3]")
        if self.fusion_mode not in {"rerank_existing", "allow_stable_new_edges"}:
            raise ValueError("unsupported fusion_mode")
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu or cuda[:index]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GCADConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        values = raw.get("gcad_source", raw)
        known = {key: value for key, value in values.items() if key in cls.__dataclass_fields__}
        config = cls(**known)
        config.validate()
        return config

