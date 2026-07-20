from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_boundary import require_roles
from .data_roles import DataRole, RoleBoundPath
from .window_dataset import SequenceWindowDataset


def extract_gradient_relation(
    model,
    sequences: Sequence[np.ndarray],
    history_length: int,
    output_dir: str | Path,
    vocabulary: Sequence[str],
    batch_size: int = 64,
    sample_aggregation: str = "mean",
    lag_aggregation: str = "max",
    device: str = "cpu",
    source_artifacts: Sequence[RoleBoundPath] | None = None,
) -> dict:
    require_roles("extract_relations", source_artifacts or [RoleBoundPath(Path("<memory>"), DataRole.SOURCE_NORMAL)])
    if sample_aggregation not in {"mean", "median"} or lag_aggregation not in {"max", "sum"}:
        raise ValueError("unsupported relation aggregation")
    loader = DataLoader(SequenceWindowDataset(sequences, history_length), batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    batches = []
    for x, y, *_ in loader:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        logits = model(x)
        losses = model.per_channel_loss(logits, y)
        output_gradients = []
        for output_channel in range(losses.shape[1]):
            gradient = torch.autograd.grad(
                losses[:, output_channel].sum(), x, retain_graph=output_channel + 1 < losses.shape[1]
            )[0]
            output_gradients.append(gradient.detach().abs().cpu())
        batches.append(torch.stack(output_gradients, dim=-1).numpy())
    if not batches:
        raise ValueError("no windows available for gradient extraction")
    all_gradients = np.concatenate(batches, axis=0)  # N,H,Cin,Cout
    if sample_aggregation == "mean":
        lag_chronological = all_gradients.mean(axis=0)
    else:
        lag_chronological = np.median(all_gradients, axis=0)
    lag_relation = lag_chronological[::-1].copy()  # [lag-1,...,lag-H]
    raw_relation = lag_relation.max(axis=0) if lag_aggregation == "max" else lag_relation.sum(axis=0)
    raw_relation = np.nan_to_num(raw_relation, nan=0.0, posinf=0.0, neginf=0.0)
    primary_lag = lag_relation.argmax(axis=0) + 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "raw_relation_matrix.npy", raw_relation)
    np.save(output / "lag_relation_matrix.npy", lag_relation)
    np.save(output / "primary_lag_matrix.npy", primary_lag)
    edges = [
        {
            "source": vocabulary[i],
            "target": vocabulary[j],
            "source_index": i,
            "target_index": j,
            "strength": float(raw_relation[i, j]),
            "primary_lag": int(primary_lag[i, j]),
        }
        for i in range(len(vocabulary))
        for j in range(len(vocabulary))
    ]
    payload = {"shape": list(raw_relation.shape), "edges": edges}
    (output / "raw_relation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "channel_vocabulary.json").write_text(json.dumps(list(vocabulary), indent=2), encoding="utf-8")
    metadata = {
        "window_count": int(all_gradients.shape[0]),
        "history_length": history_length,
        "sample_aggregation": sample_aggregation,
        "lag_aggregation": lag_aggregation,
        "loss": "per-output-channel BCEWithLogitsLoss",
        "relation_name": "GCAD-style predictive directional relation",
    }
    (output / "relation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"raw_relation": raw_relation, "lag_relation": lag_relation, "primary_lag": primary_lag, "metadata": metadata}
