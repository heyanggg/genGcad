from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GCADConfig
from .data_boundary import require_roles
from .data_roles import DataRole, RoleBoundPath
from .mixer_predictor import SourceGCADMixer
from .window_dataset import SequenceWindowDataset


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device=cpu explicitly and record the reason")
    return torch.device(requested)


def split_sequences(sequences: Sequence[np.ndarray], validation_ratio: float, seed: int):
    indices = np.arange(len(sequences))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_ratio))) if validation_ratio else 0
    validation_indices = set(indices[:validation_count].tolist())
    train = [item for index, item in enumerate(sequences) if index not in validation_indices]
    validation = [item for index, item in enumerate(sequences) if index in validation_indices]
    return train, validation, sorted(set(range(len(sequences))) - validation_indices), sorted(validation_indices)


@dataclass
class TrainingResult:
    model: SourceGCADMixer
    metrics: dict
    checkpoint_path: Path
    train_sequences: list[np.ndarray]
    validation_sequences: list[np.ndarray]


def _loader(sequences, config: GCADConfig, shuffle: bool, seed: int):
    dataset = SequenceWindowDataset(sequences, config.history_length)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
    )


def _evaluate(model, loader, device):
    model.eval()
    total = 0.0
    windows = 0
    per_channel = None
    with torch.no_grad():
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            losses = model.per_channel_loss(model(x), y)
            total += losses.sum().item()
            windows += len(x)
            values = losses.sum(dim=0).cpu()
            per_channel = values if per_channel is None else per_channel + values
    if windows == 0:
        raise ValueError("no valid windows; reduce history_length or provide longer source sequences")
    return total / (windows * per_channel.numel()), (per_channel / windows).tolist()


def train_model(
    sequences: Sequence[np.ndarray],
    config: GCADConfig,
    output_dir: str | Path,
    seed: int,
    source_artifacts: Sequence[RoleBoundPath] | None = None,
) -> TrainingResult:
    require_roles("train", source_artifacts or [RoleBoundPath(Path("<memory>"), DataRole.SOURCE_NORMAL)])
    config.validate()
    set_deterministic_seed(seed)
    device = resolve_device(config.device)
    train_sequences, validation_sequences, train_indices, validation_indices = split_sequences(
        sequences, config.validation_ratio, seed
    )
    train_loader = _loader(train_sequences, config, True, seed)
    validation_loader = _loader(validation_sequences, config, False, seed)
    channels = sequences[0].shape[1]
    model = SourceGCADMixer(
        config.history_length,
        channels,
        config.hidden_size,
        config.num_layers,
        config.dropout,
        config.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        elements = 0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.per_channel_loss(model(x), y)
            loss = losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running += losses.detach().sum().item()
            elements += losses.numel()
        train_loss = running / elements
        validation_loss, validation_per_channel = _evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_validation_loss, validation_per_channel = _evaluate(model, validation_loader, device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": config.to_dict(),
            "channels": channels,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_per_channel_loss": validation_per_channel,
        "history": history,
        "train_sequence_indices": train_indices,
        "validation_sequence_indices": validation_indices,
        "train_window_count": len(train_loader.dataset),
        "validation_window_count": len(validation_loader.dataset),
        "config": config.to_dict(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "data_sha256": hashlib.sha256(b"".join(x.tobytes() for x in sequences)).hexdigest(),
    }
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainingResult(model, metrics, checkpoint_path, train_sequences, validation_sequences)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[SourceGCADMixer, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GCADConfig(**{k: v for k, v in payload["config"].items() if k in GCADConfig.__dataclass_fields__})
    model = SourceGCADMixer(
        config.history_length,
        payload["channels"],
        config.hidden_size,
        config.num_layers,
        config.dropout,
        config.activation,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload
