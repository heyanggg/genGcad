"""Explicit, provenance-preserving I / I+J input for existing SSC.

The learned SSC is the existing 15-epoch autoencoder, NOT a behavior generator.
All arms share its source-only checkpoint to isolate augmentation from capacity.
Greedy cosine compression is identical to SPPC within a provenance stratum;
with a finite prompt budget, its exact retained prefix can be computed lazily.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def compose_ssc_input(originals, augmented=(), *, mode="I"):
    if mode not in {"I", "I+J"}:
        raise ValueError("this augmentation-only protocol supports I or I+J, not regeneration")
    rows = list(originals)
    if any(r["origin"] != "I" for r in rows):
        raise ValueError("original SSC input must contain only real I records")
    if mode == "I+J":
        parents = {r["id"] for r in rows}
        for r in augmented:
            if r["origin"] != "J" or r["parent_id"] not in parents:
                raise ValueError("augmentation must reference a real source parent")
            rows.append(r)
    return rows


@torch.no_grad()
def compress_prefix(records, model, vocab_size, threshold, budget, *, batch_size=128):
    """Exact first `budget` greedily retained examples, no quadratic matrix.

    Original SPPC selection left the module in training mode (dropout on).
    This interface fixes eval mode for deterministic, shared frozen features.
    It records this necessary reproducibility adjustment in the protocol.
    """
    if budget <= 0:
        return [], {"scanned": 0, "eligible": len(records), "retained": 0}
    model.eval()
    device = next(model.parameters()).device
    selected, representations, scanned = [], [], 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for r in batch:
            if not r["row"] or len(r["row"]) % 4 or len(r["row"]) > 40:
                raise ValueError("SSC requires complete quadruples of at most ten events")
        source = torch.full((len(batch), 40), vocab_size - 1, dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            source[i, :len(r["row"])] = torch.tensor(r["row"], device=device)
        memory = model(source, src_key_padding_mask=source.eq(vocab_size - 1))
        vectors = F.normalize(memory.flatten(1), dim=1)
        for r, vector in zip(batch, vectors):
            scanned += 1
            if representations and bool((torch.stack(representations) @ vector > threshold).any()):
                continue
            selected.append(r)
            representations.append(vector)
            if len(selected) >= budget:
                return selected, {"scanned": scanned, "eligible": len(records), "retained": len(selected)}
    return selected, {"scanned": scanned, "eligible": len(records), "retained": len(selected)}


def select_prompt_examples(pool, model, vocab_size, threshold, *, day, examples=8, synthetic=False):
    """Equal total prompt budget; augmented arms reserve half for J.

    Provenance stratification is an experiment-budget control, not RaSR itself.
    Failure to meet quotas is explicit; no silent J -> I substitution.
    """
    if examples < 2 or examples % 2:
        raise ValueError("use a positive even prompt example budget >= 2")
    eligible = [r for r in pool if r["row"][0] == day and 0 < len(r["row"]) <= 40]
    quotas = {"I": examples // 2, "J": examples // 2} if synthetic else {"I": examples}
    selected, audit = [], {}
    for origin, count in quotas.items():
        subset = [r for r in eligible if r["origin"] == origin]
        retained, audit[origin] = compress_prefix(subset, model, vocab_size, threshold, count)
        if len(retained) != count:
            raise ValueError(f"SSC day={day} origin={origin}: retained {len(retained)}, quota {count}")
        selected.extend(retained)
    audit["actual_prompt_origin_counts"] = dict(Counter(r["origin"] for r in selected))
    audit["actual_prompt_ids"] = [r["id"] for r in selected]
    return selected, audit
