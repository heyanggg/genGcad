from SmartGen.gcad_source.split_integrity import split_without_exact_overlap


def test_train_validation_split_keeps_exact_duplicates_together():
    sequences = [[1, 2], [3, 4], [1, 2], [5, 6], [7, 8]]
    train, validation, report = split_without_exact_overlap(sequences, ratio=0.6, seed=4)
    train_values = {tuple(sequences[index]) for index in train}
    validation_values = {tuple(sequences[index]) for index in validation}
    assert not train_values & validation_values
    assert report["exact_overlap_count"] == 0


def test_unique_sequence_split_is_deterministic_and_preserves_all_indices():
    sequences = [[index] for index in range(10)]
    first = split_without_exact_overlap(sequences, seed=2024)
    second = split_without_exact_overlap(sequences, seed=2024)
    assert first == second
    assert sorted(first[0] + first[1]) == list(range(10))
