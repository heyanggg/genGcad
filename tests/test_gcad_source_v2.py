import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from SmartGen.gcad_source.mechanism_controls import random_directed_control, symmetric_control
from SmartGen.gcad_source.prompt_adapter import adapt_prompt
from SmartGen.gcad_source.v2_prediction import (
    all_zero_predict,
    evaluate_simple_baselines,
    fit_ngram,
    frequency_predict,
    parameter_count,
    persistence_predict,
    predict_ngram,
    prediction_gate,
    sparse_multilabel_metrics,
    training_loss,
)
from SmartGen.gcad_source.v2_relations import (
    frequency_graph,
    gss_effect,
    per_output_lag_gradients,
    require_prediction_gate,
    stable_edges,
    supported_relation_edges,
)
from SmartGen.gcad_source.v2_representation import (
    CanonicalEvent,
    canonicalize_sequences,
    collect_windows,
    continuous_event_time_representation,
    event_position_representation,
    map_tss_fragments_to_sources,
    representation_statistics,
    split_by_source,
    v1_slot_lengths,
    window_count,
    window_origins,
)


VOCAB = ["Light:switch on", "Light:switch off", "Blind:windowShade open"]


def events(source, rows):
    return [CanonicalEvent(source, i, *row) for i, row in enumerate(rows)]


def sample_events():
    return [
        events("s0", [(0, 0, "Light", VOCAB[0]), (0, 2, "Light", VOCAB[1]), (1, 0, "Blind", VOCAB[2])]),
        events("s1", [(0, 1, "Light", VOCAB[0]), (0, 2, "Blind", VOCAB[2]), (0, 3, "Light", VOCAB[1])]),
    ]


def synthetic_prediction():
    train_x = np.asarray([
        [[1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0]],
    ], dtype=np.float32)
    train_y = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    return train_x, train_y


def test_v1_165_root_cause_slot_count():
    lengths = v1_slot_lengths([[0, 0, 1, 1], [0, 0, 1, 1, 0, 4, 1, 1]])
    assert lengths == [1, 5]
    assert sum(max(0, length - 4) for length in lengths) == 1


def test_tss_fragment_mapping_recovers_original_sources():
    raw = [[0, 0, 1, 1, 0, 1, 1, 2], [1, 0, 1, 1]]
    fragments = [[0, 0, 1, 1], [0, 1, 1, 2], [1, 0, 1, 1]]
    assert map_tss_fragments_to_sources(raw, fragments) == [0, 0, 1]


def test_tss_fragment_mapping_rejects_nonmatching_data():
    with pytest.raises(ValueError):
        map_tss_fragments_to_sources([[0, 0, 1, 1]], [[0, 1, 1, 1]])


def test_continuous_representation_never_crosses_day():
    records = continuous_event_time_representation(sample_events(), VOCAB, 3)
    assert all(len({record.day}) == 1 for record in records)
    assert len([record for record in records if record.source_id == "s0"]) == 2


def test_continuous_representation_never_crosses_source_session():
    records = continuous_event_time_representation(sample_events(), VOCAB, 3)
    assert {record.source_id for record in records} == {"s0", "s1"}


def test_continuous_empty_slots_are_real_observed_slots():
    records = continuous_event_time_representation([sample_events()[0][:2]], VOCAB, 3)
    assert records[0].values.shape[0] == 3
    assert records[0].values[1].sum() == 0
    assert records[0].observed_slot_mask.all()


def test_continuous_multihot_same_slot():
    row = events("s", [(0, 0, "Light", VOCAB[0]), (0, 0, "Blind", VOCAB[2])])
    values = continuous_event_time_representation([row], VOCAB, 3)[0].values
    assert values.shape == (1, 3)
    assert values.sum() == 2


def test_continuous_requires_native_bin_multiple():
    with pytest.raises(ValueError):
        continuous_event_time_representation(sample_events(), VOCAB, 4)


def test_multiple_time_bins_change_record_lengths():
    row = events("s", [(0, 0, "Light", VOCAB[0]), (0, 4, "Light", VOCAB[1])])
    assert len(continuous_event_time_representation([row], VOCAB, 3)[0].values) == 5
    assert len(continuous_event_time_representation([row], VOCAB, 6)[0].values) == 3
    assert len(continuous_event_time_representation([row], VOCAB, 12)[0].values) == 2


def test_event_position_does_not_join_sequences():
    records = event_position_representation(sample_events(), VOCAB)
    assert len(records) == 2
    assert window_count(records, 2) == 2


def test_event_position_retains_original_order():
    values = event_position_representation([sample_events()[1]], VOCAB)[0].values
    assert values.argmax(axis=1).tolist() == [0, 2, 1]


@pytest.mark.parametrize("history,expected", [(2, 2), (3, 0), (4, 0), (6, 0)])
def test_multiple_history_window_counts(history, expected):
    assert window_count(event_position_representation(sample_events(), VOCAB), history) == expected


def test_window_origins_include_source_and_position():
    origins = window_origins(event_position_representation(sample_events(), VOCAB), 2)
    assert origins[0]["source_id"] == "s0"
    assert origins[0]["target_position"] == 2


def test_collect_windows_keeps_shape_and_origin():
    x, y, origins = collect_windows(event_position_representation(sample_events(), VOCAB), 2)
    assert x.shape == (2, 2, 3)
    assert y.shape == (2, 3)
    assert len(origins) == 2


def test_canonical_channel_filter_rejects_none_location():
    raw = [[0, 0, 18, 96, 0, 1, 13, 78]]
    parsed, invalid, vocabulary = canonicalize_sequences(
        raw, {18: "Other", 13: "Light"}, {96: "None:location", 78: "Light:switch on"}
    )
    assert vocabulary == ["Light:switch on"]
    assert invalid[0]["reason"] == "forbidden_semantic_field"
    assert len(parsed[0]) == 1


def test_canonical_channel_filter_rejects_device_action_mismatch():
    parsed, invalid, vocabulary = canonicalize_sequences(
        [[0, 0, 13, 28]], {13: "Light"}, {28: "Blind:windowShade open"}
    )
    assert not parsed[0] and not vocabulary
    assert invalid[0]["reason"] == "device_action_mismatch"


def test_canonical_channel_filter_records_unknown_device():
    _, invalid, _ = canonicalize_sequences([[0, 0, 99, 1]], {}, {1: "Light:switch on"})
    assert invalid[0]["reason"] == "unknown_device"


def test_source_split_has_no_provenance_overlap():
    records = event_position_representation(sample_events(), VOCAB)
    _, _, manifest = split_by_source(records, 0.5, 2024)
    assert manifest["source_overlap_count"] == 0
    assert manifest["passed"]


def test_source_split_is_deterministic():
    records = event_position_representation(sample_events(), VOCAB)
    assert split_by_source(records, 0.5, 9)[2] == split_by_source(records, 0.5, 9)[2]


def test_representation_statistics_reports_empty_ratio():
    row = events("s", [(0, 0, "Light", VOCAB[0]), (0, 2, "Light", VOCAB[1])])
    stats = representation_statistics(continuous_event_time_representation([row], VOCAB, 3), 1)
    assert stats["empty_slot_ratio"] == pytest.approx(1 / 3)


def test_all_zero_baseline_is_all_zero_after_threshold():
    y = np.asarray([[1, 0], [0, 1]])
    metrics = sparse_multilabel_metrics(y, all_zero_predict(y.shape))
    assert metrics["all_zero_prediction_rate"] == 1
    assert metrics["micro_recall"] == 0


def test_frequency_predictor_uses_training_prior():
    y = np.asarray([[1, 0], [1, 0], [0, 1]])
    prediction = frequency_predict(y, 2)
    assert prediction.shape == (2, 2)
    assert prediction[0, 0] > prediction[0, 1]


def test_persistence_predictor_uses_latest_position():
    x, _ = synthetic_prediction()
    assert persistence_predict(x)[0].argmax() == 1


@pytest.mark.parametrize("order", [1, 2, 3])
def test_markov_and_ngram_predictions_have_common_shape(order):
    x, y = synthetic_prediction()
    prediction = predict_ngram(fit_ngram(x, y, order), x, order)
    assert prediction.shape == y.shape


def test_all_simple_baselines_use_sparse_metrics():
    x, y = synthetic_prediction()
    results = evaluate_simple_baselines(x, y, x, y)
    assert set(results) == {"all_zero", "frequency", "persistence", "first_order_markov", "ngram_2", "ngram_3"}
    assert all("macro_f1" in metrics for metrics in results.values())


def test_sparse_metrics_report_positive_and_negative_bce():
    y = np.asarray([[1, 0], [0, 1]])
    metrics = sparse_multilabel_metrics(y, np.asarray([[0.8, 0.1], [0.2, 0.9]]))
    assert metrics["positive_bce"] < metrics["negative_bce"] + 1
    assert metrics["micro_precision"] == 1


def test_sparse_metrics_report_rare_channel_recall():
    y = np.asarray([[1, 0], [1, 0], [0, 1]])
    metrics = sparse_multilabel_metrics(y, y * 0.98 + 0.01)
    assert metrics["rare_channel_recall"] == 1


def test_mixer_parameter_count_is_positive():
    from SmartGen.gcad_source.mixer_predictor import SourceGCADMixer
    assert parameter_count(SourceGCADMixer(2, 3, 8, 1)) > 0


@pytest.mark.parametrize("mode", ["bce", "positive_weighted_bce", "masked_bce"])
def test_all_registered_losses_are_finite(mode):
    logits = torch.zeros((2, 3))
    target = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    assert torch.isfinite(training_loss(logits, target, mode, torch.ones(3)))


def test_failed_prediction_gate_blocks_relations():
    with pytest.raises(RuntimeError):
        require_prediction_gate({"passed": False})


def test_passed_prediction_gate_allows_relations():
    require_prediction_gate({"passed": True})


def test_prediction_gate_requires_two_seed_wins():
    def payload(mixer, baseline):
        metrics = {"macro_f1": mixer, "all_zero_prediction_rate": 0.2, "rare_channel_recall": 0.1, "train_validation_macro_f1_gap": 0.1}
        return {"simple_baselines": {"markov": {"macro_f1": baseline}}, "selected_mixer": metrics}
    gate = prediction_gate({2024: payload(.5, .4), 2025: payload(.3, .4), 2026: payload(.2, .4)}, "macro_f1", .3)
    assert not gate["passed"]


def test_per_output_gradient_keeps_lag_dimension():
    class Linear(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1)
    x = np.asarray([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.float32)
    y = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    assert per_output_lag_gradients(Linear(), x, y).shape == (2, 2, 2)


def test_per_output_gradient_is_independent_by_output():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, -1]
    x = np.asarray([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=np.float32)
    y = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    gradients = per_output_lag_gradients(Identity(), x, y)
    assert gradients[:, 1, 0].sum() == 0


def test_low_support_relations_are_filtered():
    gradients = np.ones((2, 2, 2))
    x = np.asarray([[[1, 0], [0, 0]]], dtype=np.float32)
    y = np.asarray([[0, 1]], dtype=np.float32)
    edges = supported_relation_edges(gradients, x, y, VOCAB[:2], min_source_activation=2, min_target_activation=1, min_cooccurrence=1, min_asymmetric_margin=0)
    assert edges == []


def test_stable_relation_rejects_illegal_nodes():
    rows = [[{"source_channel": "None:location", "target_channel": VOCAB[0], "asymmetric_margin": 1, "main_lag": 1, "source_activation_count": 10, "target_activation_count": 10, "cooccurrence_count": 5}]]
    with pytest.raises(ValueError):
        stable_edges(rows, 1.0)


def test_stable_relation_computes_occurrence_and_lag_stability():
    edge = {"source_channel": VOCAB[0], "target_channel": VOCAB[1], "asymmetric_margin": 1, "main_lag": 2, "source_activation_count": 10, "target_activation_count": 9, "cooccurrence_count": 4}
    result = stable_edges([[edge], [edge], []], 2 / 3)
    assert result[0]["occurrence_rate"] == pytest.approx(2 / 3)
    assert result[0]["lag_stability"] == 1


def test_random_graph_is_reproducible():
    relation = {"edges": [{"source": "a", "target": "b", "stable_score": 1}]}
    assert random_directed_control(relation, 2) == random_directed_control(relation, 2)


def test_symmetric_graph_contains_reverse():
    relation = {"edges": [{"source": "a", "target": "b", "stable_score": 1}]}
    pairs = {(edge["source"], edge["target"]) for edge in symmetric_control(relation)["edges"]}
    assert pairs == {("a", "b"), ("b", "a")}


def test_frequency_graph_uses_cooccurrence_support():
    x, y = synthetic_prediction()
    graph = frequency_graph(x, y, VOCAB, 2)
    assert len(graph["edges"]) == 2
    assert graph["control"] == "frequency_cooccurrence"


def test_gss_effect_detects_score_and_rank_changes():
    before = [{"source": "a", "target": "b", "score": 2}, {"source": "a", "target": "c", "score": 1}]
    after = [{"source": "a", "target": "c", "score": 3}, {"source": "a", "target": "b", "score": 2}]
    report = gss_effect(before, after)
    assert report["rank_change_position_count"] == 2
    assert report["top_1_changed_node_count"] == 1


def test_gss_unchanged_ranking_is_detected():
    edges = [{"source": "a", "target": "b", "score": 1}]
    assert gss_effect(edges, edges)["rank_change_position_count"] == 0


def test_disabled_gcad_prompt_is_byte_equivalent():
    prompt = "baseline bytes\n"
    assert adapt_prompt(prompt, False) == prompt


def test_paired_protocol_reuses_v5_counts_but_not_sequences():
    import yaml
    cfg = yaml.safe_load(Path("configs/gcad_source_v2/paired_experiment.yaml").read_text())
    assert cfg["candidate_pool_size"] == 160 and cfg["final_sequence_count"] == 137
    assert cfg["reuse_a5_sequences"] is False


def test_paired_protocol_keeps_official_detector_and_seeds():
    import yaml
    cfg = yaml.safe_load(Path("configs/gcad_source_v2/paired_experiment.yaml").read_text())
    assert cfg["official_detector"] == "anomaly_detection_pipeline/models1.py"
    assert cfg["paired_seeds"] == [2024, 2025, 2026]


def test_ranking_remains_disabled():
    import yaml
    cfg = yaml.safe_load(Path("configs/gcad_source_v2/gss_fusion.yaml").read_text())
    assert cfg["ranking_enabled"] is False


def test_target_data_access_is_final_only():
    import yaml
    cfg = yaml.safe_load(Path("configs/gcad_source_v2/paired_experiment.yaml").read_text())
    assert cfg["target_behavior_access_stage"] == "final_paired_evaluation_only"


def test_replicate5_frozen_commit_is_unchanged():
    import subprocess
    value = subprocess.check_output(["git", "rev-parse", "replicate5-codex-baseline-frozen^{}"], text=True).strip()
    assert value == "07ed86a3fcbc55218197d1e0b6957fb008c51fe1"


def test_official_detector_file_is_unchanged_from_frozen_commit():
    import subprocess
    current = hashlib.sha256(Path("anomaly_detection_pipeline/models1.py").read_bytes()).hexdigest()
    frozen = subprocess.check_output(["git", "show", "07ed86a:anomaly_detection_pipeline/models1.py"])
    assert current == hashlib.sha256(frozen).hexdigest()


def test_cpu_smoke():
    assert torch.device("cpu").type == "cpu"
