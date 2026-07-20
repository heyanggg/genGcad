from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def asymmetric_difference(
    relation: np.ndarray,
    edge_threshold: float = 0.0,
    top_k: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    relation = np.asarray(relation, dtype=float)
    if relation.ndim != 2 or relation.shape[0] != relation.shape[1]:
        raise ValueError("relation must be a square matrix")
    result = np.maximum(0.0, relation - relation.T)
    np.fill_diagonal(result, 0.0)
    maximum = result.max(initial=0.0)
    if normalize and maximum > 0:
        result = result / maximum
    result[result < edge_threshold] = 0.0
    if top_k is not None and top_k >= 0:
        for source in range(len(result)):
            keep = np.argsort(result[source])[-top_k:] if top_k else np.array([], dtype=int)
            mask = np.ones(len(result), dtype=bool)
            mask[keep] = False
            result[source, mask] = 0.0
    return result


def save_asymmetric_relation(
    relation: np.ndarray,
    vocabulary: Sequence[str],
    output_dir: str | Path,
    edge_threshold: float,
    top_k: int,
    primary_lag: np.ndarray | None = None,
) -> dict:
    filtered = asymmetric_difference(relation, edge_threshold, top_k)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "asymmetric_relation_matrix.npy", filtered)
    edges = []
    for i, j in zip(*np.nonzero(filtered)):
        edges.append(
            {
                "source": vocabulary[i],
                "target": vocabulary[j],
                "source_index": int(i),
                "target_index": int(j),
                "strength": float(filtered[i, j]),
                "primary_lag": int(primary_lag[i, j]) if primary_lag is not None else None,
            }
        )
    edges.sort(key=lambda item: (-item["strength"], item["source"], item["target"]))
    payload = {"relation_type": "GCAD-style predictive directional relation", "edges": edges}
    (output / "asymmetric_relation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = {
        "edge_threshold": edge_threshold,
        "top_k": top_k,
        "edge_count": len(edges),
        "self_loops": int(np.count_nonzero(np.diag(filtered))),
        "normalized": True,
    }
    (output / "threshold_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "top_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "source_index", "target_index", "strength", "primary_lag"])
        writer.writeheader()
        writer.writerows(edges)
    return {"matrix": filtered, "edges": edges, "report": report}

