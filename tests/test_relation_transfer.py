"""Transport algebra, semantic compatibility and additive integration checks."""
import json
from copy import deepcopy

import pytest

from SmartGen.action_semantics import build_action_semantics, canonical_sha256
from SmartGen.dictionary import fr_actions, fr_devices_dict
from SmartGen.relation_guidance import render_relation_guidance
from SmartGen.relation_transfer import (
    build_action_mappings, compatibility, load_embeddings, transfer_relations,
)
from test_relation_guidance import A, B, C, adapt, event, graph, import_main, raw_rows


TV, PROJECTOR = "Television:switch on", "Projector:switch on"


def fixture_embeddings(semantics):
    names = sorted(semantics["records"])
    value = {
        "dataset": "fr", "dictionary_sha256": semantics["dictionary_sha256"],
        "semantics_sha256": canonical_sha256(semantics), "action_names": names,
        "vectors": [[1.0, 0.0] for _ in names],
        "model": {"model_id": "unit-test-fixture-not-a-real-encoder"},
    }
    value["artifact_fingerprint"] = canonical_sha256(value)
    return value


def test_mask_blocks_opposite_states_telemetry_unknown_devices_and_aliases():
    labels = build_action_semantics(fr_actions, fr_devices_dict)["records"]
    assert compatibility(labels[A], labels[B]) == (False, "opposite_or_different_state")
    assert compatibility(labels["Camera:notification"], labels["Camera:refresh refresh"])[0] is False
    assert compatibility(labels["SmartPlug:switch on"], labels[A])[0] is False
    assert compatibility(labels[A], labels["Fan:switch on"])[0] is False
    assert compatibility(labels[TV], labels[PROJECTOR])[0] is True


def test_maximal_semantic_similarity_cannot_bypass_compatibility_mask():
    semantics = build_action_semantics(fr_actions, fr_devices_dict)
    result = build_action_mappings(semantics, fixture_embeddings(semantics))
    assert result["mappings"]
    assert set(result["statistics"]["function_families"]) == {"video_entertainment"}
    assert not any(m["source_action"] == A for m in result["mappings"])
    assert result["statistics"]["target_environment_specific_mappings"] == 0
    for row in result["sparse_rows"].values():
        assert sum(m["weight"] for m in row) == pytest.approx(1.0)


def test_embedding_loader_rejects_modified_vectors_and_dictionary(tmp_path):
    semantics = build_action_semantics(fr_actions, fr_devices_dict)
    value = fixture_embeddings(semantics)
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(value))
    assert load_embeddings(path, semantics, "fr") == value
    value["vectors"][0] = [0.0, 1.0]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fingerprint"):
        load_embeddings(path, semantics, "fr")
    value["artifact_fingerprint"] = canonical_sha256({k: v for k, v in value.items() if k != "artifact_fingerprint"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="dataset"):
        load_embeddings(path, semantics, "us")


@pytest.mark.parametrize("bad_vector", [[0.0, 0.0], [float("inf"), 0.0], [True, 0.0], [1.0]])
def test_bad_embedding_vectors_are_rejected(tmp_path, bad_vector):
    semantics = build_action_semantics(fr_actions, fr_devices_dict)
    value = fixture_embeddings(semantics)
    value["vectors"][0] = bad_vector
    value.pop("artifact_fingerprint")
    if bad_vector[0] == float("inf"):
        # Non-finite JSON fails canonicalization before a fingerprint can exist.
        with pytest.raises(ValueError):
            canonical_sha256(value)
        return
    value["artifact_fingerprint"] = canonical_sha256(value)
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="vectors"):
        load_embeddings(path, semantics, "fr")


def prepare(rows, gcad=None):
    source = adapt(rows, gcad, mode="typed")
    mappings = build_action_mappings(source["action_semantics"], fixture_embeddings(source["action_semantics"]))
    return source, mappings


def test_one_endpoint_transport_produces_new_candidate_not_target_observation():
    rows = raw_rows(A, TV)
    source, mappings = prepare(rows)
    before = deepcopy((source, mappings, rows))
    result = transfer_relations(source, mappings, rows)
    candidate = next(c for c in result["selected_candidates"] if c["source_action"] == A and c["target_action"] == PROJECTOR)
    assert candidate["ranking_score"] == pytest.approx(0.5)
    assert candidate["source_match_count"] == 0
    assert candidate["target_observation_count"] is None
    assert candidate["contributions"][0]["source_evidence"]["count"] == 5
    assert candidate["contributions"][0]["q_environment_catalogue_gate"] == 1.0
    assert candidate["environment_preference_evidence"] == "unavailable"
    assert (source, mappings, rows) == before


def test_existing_source_relation_is_not_reported_as_novel():
    rows = raw_rows(A, TV) + raw_rows(A, PROJECTOR)
    source, mappings = prepare(rows)
    result = transfer_relations(source, mappings, rows)
    assert result["all_candidates"]
    assert result["selected_candidates"] == []
    assert all(not c["novel_against_source_events"] for c in result["all_candidates"])


def test_gss_and_gcad_transport_keep_separate_scores_and_lag_units():
    rows = raw_rows(A, TV)
    source, mappings = prepare(rows, graph(A, TV, lag=2))
    result = transfer_relations(source, mappings, rows)
    by_kind = {c["relation_type"]: c for c in result["selected_candidates"]}
    assert by_kind["observed_transition"]["source_lag_prior"] == 1
    assert by_kind["observed_transition"]["lag_unit"] == "raw_event_positions"
    assert by_kind["predictive_lag"]["source_lag_prior"] == 2
    assert by_kind["predictive_lag"]["lag_unit"] == "valid_device_behavior_positions"
    assert by_kind["predictive_lag"]["ranking_score"] == pytest.approx(1 / 3)
    assert by_kind["observed_transition"]["ranking_score"] == pytest.approx(1 / 2)


def test_endpoint_limit_avoids_self_loops_and_double_transfer_by_default():
    rows = raw_rows(PROJECTOR, TV)
    source, mappings = prepare(rows)
    assert transfer_relations(source, mappings, rows)["all_candidates"] == []
    expanded = transfer_relations(source, mappings, rows, max_changed_endpoints=2)
    assert expanded["selected_candidates"][0]["source_action"] == TV
    assert expanded["selected_candidates"][0]["target_action"] == PROJECTOR
    assert expanded["selected_candidates"][0]["ranking_score"] == pytest.approx(0.25)


@pytest.mark.parametrize("kind", ["gss", "gcad"])
def test_transfer_source_ablation_uses_only_requested_evidence(kind):
    rows = raw_rows(A, TV)
    source, mappings = prepare(rows, graph(A, TV))
    result = transfer_relations(source, mappings, rows, transfer_sources=kind)
    expected = "observed_transition" if kind == "gss" else "predictive_lag"
    assert all(c["relation_type"] == expected for c in result["all_candidates"])


def test_bad_time_bin_does_not_shift_upstream_gcad_event_positions():
    row = event(A) + event(B, hour=99) + event(C)
    source = adapt([row] * 5, graph(A, C, lag=2))
    gcad = next(e for r in source["relationships"] for e in r["evidence"] if e["source"] == "GCAD-derived")
    assert gcad["lag"] == 2
    assert gcad["target_hour_counts"] == {"3": 5}
    assert source["source_audit"]["excluded_event_reasons"] == {"invalid_day_or_hour_bin": 5}


@pytest.mark.parametrize("profile", ["original", "environment-aware"])
def test_removing_supplement_restores_byte_identical_baseline_prompt(monkeypatch, profile):
    main = import_main(monkeypatch)
    artifact = adapt(raw_rows(A, B), graph())
    inputs = ("catalog", "environment", [["source"]], {"existing-gss": 123}, graph())
    baseline = main.build_prompt(*inputs, target_environment="spring", prompt_profile=profile)
    augmented = main.build_prompt(*inputs, target_environment="spring", prompt_profile=profile, relation_guidance=artifact)
    supplement = " " + render_relation_guidance(artifact) + " "
    assert augmented.replace(supplement, "", 1) == baseline


def test_empty_supplement_leaves_baseline_prompt_byte_identical(monkeypatch):
    main = import_main(monkeypatch)
    artifact = adapt(raw_rows(A))
    args = ("catalog", "env", [], {}, {"status": "disabled"})
    assert render_relation_guidance(artifact) == ""
    assert main.build_prompt(*args, target_environment="spring", relation_guidance=artifact) == main.build_prompt(*args, target_environment="spring")


def test_environment_deferral_is_not_silently_used_as_source_graph_selection():
    rows = raw_rows(A, TV)
    source, mappings = prepare(rows)
    source["mode"] = "adapted"
    with pytest.raises(ValueError, match="typed source evidence"):
        transfer_relations(source, mappings, rows)
