from __future__ import annotations

import numpy as np


def relation_summary(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix, dtype=float)
    nonzero = values[values > 0]
    return {
        "edge_count": int(nonzero.size),
        "density": float(nonzero.size / max(1, values.size - len(values))),
        "mean_strength": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_strength": float(nonzero.max()) if nonzero.size else 0.0,
        "self_loop_count": int(np.count_nonzero(np.diag(values))),
    }

