import numpy as np

from SmartGen.gcad_source.source_resampling import select_source_replicate


def test_partitions_are_disjoint_and_cover_source():
    sequences = [np.array([[i]]) for i in range(10)]
    _, left, _ = select_source_replicate(sequences, 1, 0, 2)
    _, right, _ = select_source_replicate(sequences, 1, 1, 2)
    assert set(left).isdisjoint(right)
    assert sorted(left + right) == list(range(10))


def test_bootstrap_is_deterministic_and_audited():
    sequences = [np.array([[i]]) for i in range(20)]
    _, first, metadata = select_source_replicate(sequences, 7, 0, 1, 0.8)
    _, second, _ = select_source_replicate(sequences, 7, 0, 1, 0.8)
    assert first == second
    assert metadata["bootstrap_with_replacement"] is True
    assert metadata["uses_target_behavior"] is False
