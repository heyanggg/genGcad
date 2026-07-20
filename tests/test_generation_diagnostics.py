import math

import numpy as np

from SmartGen.generation_backends.generation_diagnostics import (
    effective_count,
    near_duplicate_metrics,
    normalized_ngram_entropy,
    sequence_metrics,
)


def test_bigram_and_trigram_entropy_are_normalized_correctly():
    sequences = [["a", "b", "c"], ["a", "b", "d"]]
    bigram_count, bigram_entropy = normalized_ngram_entropy(sequences, 2)
    trigram_count, trigram_entropy = normalized_ngram_entropy(sequences, 3)
    assert bigram_count == 3
    expected = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)) / math.log(3)
    assert np.isclose(bigram_entropy, expected)
    assert trigram_count == 2 and np.isclose(trigram_entropy, 1.0)


def test_effective_sequence_count_and_template_collapse_metrics():
    sequences = [["a", "b"], ["a", "b"], ["a", "c"], ["d", "e", "f"]]
    metrics = sequence_metrics(sequences)
    assert np.isclose(effective_count([("a",), ("a",), ("b",), ("b",)]), 2.0)
    assert metrics["unique_sequence_ratio"] == 0.75
    assert metrics["exact_duplicate_count"] == 1
    assert metrics["top_1_template_share"] == 0.5
    assert metrics["length"]["distinct_count"] == 2


def test_near_duplicate_templates_are_reported_without_counting_exact_duplicates():
    sequences = [
        ("a", "b", "c", "d", "e"),
        ("a", "b", "c", "d", "f"),
        ("a", "b", "c", "d", "e"),
        ("x", "y"),
    ]
    pair_count, sequence_count = near_duplicate_metrics(sequences)
    assert pair_count == 1
    assert sequence_count == 2
