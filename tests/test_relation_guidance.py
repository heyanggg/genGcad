"""Behavior and integration checks for typed, environment-aware guidance."""
import hashlib
import json
import pickle
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from SmartGen.action_semantics import build_action_semantics
from SmartGen.dictionary import fr_actions, fr_devices_dict
from SmartGen.experiment import ExperimentRun
from SmartGen.relation_adapter import adapt_relations
from SmartGen.relation_guidance import render_relation_guidance
from SmartGen.relation_stage import prepare_relation_stage
from SmartGen.relation_preview import main as preview_main
from SmartGen.text_translation_matrix import build_transition_guidance
from test_experiment_structure import make_config


A, B, C = "Light:switch on", "Light:switch off", "Heater:switch on"
UNKNOWN = "Heater:setHeatingSetpoint"


def event(action, hour=3, day=0):
    return [day, hour, fr_devices_dict[action.split(":")[0]], fr_actions[action]]


def raw_rows(*actions, repeats=5):
    return [sum((event(action) for action in actions), []) for _ in range(repeats)]


def gss_for(rows):
    return build_transition_guidance([row[3:40:4] for row in rows], fr_actions)


def graph(source=A, target=B, lag=1, source_hash="source-hash"):
    return {
        "status": "ready", "source": {"sha256": source_hash, "dataset": "fr"},
        "lag_unit": "subsequent_behavior_positions",
        "config": {"stable_seed_fraction": 2 / 3},
        "lagged_behavior_relations": [{
            "source_action": source, "target_action": target, "typical_lag": lag,
            "support": 5, "stable_seeds": 2, "total_seeds": 3,
            "raw_strength": 0.1, "strength": 1.0,
        }],
    }


def adapt(rows, gcad=None, **kwargs):
    defaults = dict(dataset="fr", source_environment="winter",
                    target_environment="spring", source_sha256="source-hash")
    defaults.update(kwargs)
    semantics = build_action_semantics(fr_actions, fr_devices_dict)
    artifact = adapt_relations(
        gss_for(rows), gcad or {"status": "disabled"}, semantics, rows, **defaults,
    )
    artifact["action_semantics"] = semantics
    return artifact


def test_action_labels_distinguish_opposite_states_and_abstain_on_missing_values():
    labels = build_action_semantics(fr_actions, fr_devices_dict)["records"]
    assert labels[A]["desired_state"] == "on"
    assert labels[B]["desired_state"] == "off"
    assert labels[UNKNOWN]["semantic_status"] == "unresolved"
    assert labels["None:location"]["valid_device_action"] is False
    assert labels["SmartPlug:switch on"]["function"] == "unspecified_control"
    assert labels[A]["provenance"]["review_status"] == "not_independently_reviewed"


def test_combined_action_pair_keeps_independent_lags_units_and_supports():
    artifact = adapt(raw_rows(A, B), graph(lag=2))
    assert len(artifact["relationships"]) == 1
    evidence = artifact["relationships"][0]["evidence"]
    assert [(e["relation_type"], e["lag"], e["lag_unit"]) for e in evidence] == [
        ("observed_transition", 1, "raw_event_positions"),
        ("predictive_lag", 2, "valid_device_behavior_positions"),
    ]
    assert evidence[0]["count"] == evidence[1]["support"] == 5
    assert evidence[1]["cross_environment_stability"] == "not_established"


def test_nondevice_event_does_not_create_gss_shortcut_but_gcad_uses_filtered_positions():
    artifact = adapt(raw_rows(A, "None:location", B), graph())
    assert len(artifact["relationships"]) == 1
    evidence = artifact["relationships"][0]["evidence"]
    assert len(evidence) == 1 and evidence[0]["relation_type"] == "predictive_lag"
    assert evidence[0]["target_hour_counts"] == {"3": 5}
    assert artifact["source_audit"]["invalid_event_count"] == 5


def test_conditional_probability_uses_all_outgoing_events_not_top_five_only():
    alternatives = [B, C, "Fan:switch on", "Camera:switch on",
                    "WaterValve:valve open", "RobotCleaner:setRobotCleanerMovement cleaning"]
    rows = [event(A) + event(target) for target in alternatives for _ in range(3)]
    artifact = adapt(rows, mode="typed", max_per_target=10)
    assert len(artifact["relationships"]) == 5
    assert all(r["evidence"][0]["conditional_probability"] == pytest.approx(1 / 6)
               for r in artifact["relationships"])


def test_gss_context_uses_same_ten_event_cap_as_upstream_atm():
    rows = raw_rows(*([A] * 10 + [B]))
    artifact = adapt(rows)
    assert all(r["target_action"] != B for r in artifact["relationships"])
    assert artifact["relationships"][0]["evidence"][0]["count"] == 45


def test_no_pairs_cross_sequence_boundaries():
    artifact = adapt(raw_rows(A) + raw_rows(B))
    assert artifact["status"] == "empty"
    assert artifact["relationships"] == []


def test_night_retimes_preferences_without_reinterpreting_lag_as_hours():
    artifact = adapt(raw_rows(A, B), graph(), source_environment="daytime", target_environment="night")
    relation = artifact["relationships"][0]
    assert relation["policy"]["decision"] == "retime_preference"
    assert relation["policy"]["target_preferred_hour_bins"] == [6, 7, 0, 1]
    assert relation["evidence"][0]["target_hour_counts"] == {"3": 5}
    assert relation["evidence"][1]["lag"] == 1
    assert "not hours" in render_relation_guidance(artifact)


def test_spring_climate_relation_is_conditional_not_heater_prohibition():
    relation = adapt(raw_rows(A, C))["relationships"][0]
    assert relation["policy"]["decision"] == "conditional_reference"
    assert "do not require or prohibit" in relation["policy"]["instruction"]


def test_multiple_occupants_does_not_invent_person_identity_or_duplicate_habits():
    artifact = adapt(raw_rows(A, B), source_environment="single", target_environment="multiple")
    assert "do not duplicate" in artifact["relationships"][0]["policy"]["instruction"]


def test_unresolved_commands_are_deferred_only_in_adapted_mode():
    rows = raw_rows(A, UNKNOWN)
    adapted, typed = adapt(rows), adapt(rows, mode="typed")
    assert adapted["relationships"] == []
    assert adapted["deferred_relationships"][0]["target_action"] == UNKNOWN
    assert typed["relationships"][0]["target_action"] == UNKNOWN
    assert UNKNOWN not in render_relation_guidance(adapted)


@pytest.mark.parametrize("change", [
    {"sha256": "wrong"}, {"dataset": "us"}, {},
])
def test_ready_gcad_requires_source_hash_and_dataset(change):
    gcad = graph()
    gcad["source"] = change
    with pytest.raises(ValueError, match="does not match"):
        adapt(raw_rows(A, B), gcad)


def test_gcad_disabled_status_cannot_leak_supplied_edges_into_prompt():
    gcad = graph(A, C)
    gcad["status"] = "disabled"
    artifact = adapt(raw_rows(A, B), gcad)
    assert all(e["source"] == "GSS" for r in artifact["relationships"] for e in r["evidence"])


def test_unstable_gcad_is_rejected_without_losing_gss():
    gcad = graph()
    gcad["lagged_behavior_relations"][0]["stable_seeds"] = 1
    artifact = adapt(raw_rows(A, B), gcad)
    assert len(artifact["relationships"][0]["evidence"]) == 1
    assert artifact["selection"]["rejected_evidence_count"] == 1


@pytest.mark.parametrize("field,value", [("typical_lag", 0), ("raw_strength", float("nan")),
                                         ("stable_seeds", 4), ("support", True)])
def test_malformed_gcad_evidence_fails_explicitly(field, value):
    gcad = graph()
    gcad["lagged_behavior_relations"][0][field] = value
    with pytest.raises(ValueError, match="malformed"):
        adapt(raw_rows(A, B), gcad)


def test_stale_gss_counts_are_rejected():
    rows = raw_rows(A, B)
    gss = gss_for(rows)
    gss[A]["transitions"][0]["count"] += 1
    with pytest.raises(ValueError, match="GSS count"):
        adapt_relations(gss, {"status": "disabled"}, build_action_semantics(fr_actions, fr_devices_dict),
                        rows, dataset="fr", source_environment="winter",
                        target_environment="spring", source_sha256="source-hash")


def test_device_action_id_mismatch_is_audited_without_rewriting_source_or_support():
    rows = raw_rows(A, B) + raw_rows(C, A)
    for row in rows[5:]:
        row[2] = 999  # Same real action ID, inconsistent device ID.
    before = deepcopy(rows)
    artifact = adapt(rows)
    assert rows == before
    assert artifact["source_audit"]["excluded_event_reasons"] == {"device_action_id_mismatch": 5}
    assert [r["source_action"] for r in artifact["relationships"]] == [A]
    assert any(e["reason"] == "source_pair_contains_invalid_events" for e in artifact["rejected_evidence"])


def test_selection_reserves_both_evidence_types_and_obeys_target_and_total_limits():
    rows = raw_rows(A, B, C, A, C, B, A)
    artifact = adapt(rows, graph(B, C, lag=2), mode="typed", max_relationships=3, max_per_target=1)
    assert len(artifact["relationships"]) == 3
    targets = Counter(r["target_action"] for r in artifact["relationships"])
    assert max(targets.values()) == 1
    assert {e["relation_type"] for r in artifact["relationships"] for e in r["evidence"]} == {
        "observed_transition", "predictive_lag",
    }


def test_preparation_freezes_inputs_and_rejects_changed_resume_evidence(tmp_path):
    source = tmp_path / "IoT_data/fr/winter/split_trn.pkl"
    source.parent.mkdir(parents=True)
    rows = raw_rows(A, B)
    source.write_bytes(pickle.dumps(rows))
    output = tmp_path / "run/relations.json"
    gss = gss_for(rows)
    kwargs = dict(mode="adapted", min_support=3, max_relationships=12, max_per_target=4)
    args = (source, "fr", "winter", "spring", fr_actions, fr_devices_dict,
            gss, {"status": "disabled"}, output)
    before = source.read_bytes()
    first = prepare_relation_stage(*args, **kwargs)
    assert first == prepare_relation_stage(*args, **kwargs)
    assert source.read_bytes() == before
    assert first["inputs"]["source_training"]["sha256"] == hashlib.sha256(before).hexdigest()
    with pytest.raises(ValueError, match="changed"):
        prepare_relation_stage(*args, **{**kwargs, "min_support": 4})


def import_main(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("SmartGen").resolve()))
    sys.modules.pop("main", None)
    import main
    return main


def test_additional_prompt_preserves_original_gss_gcad_including_deferred_actions(monkeypatch):
    main = import_main(monkeypatch)
    artifact = adapt(raw_rows(A, B) + raw_rows(A, UNKNOWN), graph())
    prompt = main.build_prompt("catalog", "environment", [],
                               {"RAW_GSS_SENTINEL": "deferred evidence"},
                               {**graph(), "lagged_behavior_relations": [{"RAW_GCAD_SENTINEL": 1}]},
                               target_environment="spring", relation_guidance=artifact)
    assert "Additional typed behavior relationship audit" in prompt
    assert "RAW_GSS_SENTINEL" in prompt and "RAW_GCAD_SENTINEL" in prompt
    assert "Directional behavior relationship guidance" in prompt


def test_off_mode_default_preserves_prompt_and_artifact_names(monkeypatch):
    main = import_main(monkeypatch)
    args = main.get_args_parser().parse_args([])
    assert args.relation_mode == "off"
    assert main.experiment_config_from_args(args).artifact_model == "gpt-5.6-sol__run1"
    assert main.build_prompt("catalog", "env", [], {}, {"status": "disabled"}) == main.compose_smartgen_prompt("catalog", "env", [], {})
    assert "__rel-adapted" in make_config(relation_mode="adapted").artifact_model
    assert make_config(relation_mode="typed").experiment_name != make_config(relation_mode="adapted").experiment_name


@pytest.mark.parametrize("overrides", [{"relation_mode": "bad"}, {"relation_min_support": 0},
                                       {"relation_max_relationships": -1}, {"relation_max_per_target": True}])
def test_invalid_relation_configuration_fails(overrides):
    with pytest.raises(ValueError, match="relation"):
        make_config(**overrides)


def test_legacy_off_run_can_resume_but_new_settings_cannot(tmp_path):
    run = ExperimentRun(make_config(), runs_root=tmp_path)
    run.update(status="preparing")
    legacy = json.loads(run.manifest_path.read_text())
    for key in list(legacy["config"]):
        if key.startswith("relation_"):
            legacy["config"].pop(key)
    run.manifest_path.write_text(json.dumps(legacy))
    assert ExperimentRun(make_config(), runs_root=tmp_path).manifest["status"] == "preparing"
    with pytest.raises(ValueError, match="different settings"):
        ExperimentRun(make_config(relation_min_support=9), runs_root=tmp_path)


@pytest.mark.parametrize("mode", ["typed", "adapted"])
def test_generation_pipeline_wires_fresh_gss_relations_and_provenance(tmp_path, monkeypatch, mode):
    main = import_main(monkeypatch)
    monkeypatch.chdir(tmp_path)
    args = main.get_args_parser().parse_args([
        "--method", "instance", "--need-generate", "true", "--relation-mode", mode,
        "--gcad-mode", "auto", "--run-id", "integration",
    ])
    config = main.experiment_config_from_args(args)
    source_dir = Path("IoT_data/fr/winter")
    source_dir.mkdir(parents=True)
    rows = raw_rows(A, B) + raw_rows(A, UNKNOWN)
    source = source_dir / "split_trn.pkl"
    source.write_bytes(pickle.dumps(rows))
    gcad = graph(source_hash=hashlib.sha256(source.read_bytes()).hexdigest())
    gcad_path = tmp_path / "gcad.json"
    gcad_path.write_text(json.dumps(gcad))
    source_text = source_dir / "trn_day_0_instance_th=0.918_text.pkl"
    source_text.write_bytes(pickle.dumps([["source representative"]]))
    Path("fr_keys_best.txt").write_text("catalog")
    monkeypatch.setattr(main, "ExperimentRun", lambda c: ExperimentRun(c, runs_root=tmp_path / "runs"))
    monkeypatch.setattr(main, "Split", lambda *args: None)
    monkeypatch.setattr(main, "prepare_gcad_stage", lambda *args: (gcad, gcad_path, True))
    monkeypatch.setattr(main, "Dayse", lambda *args: None)
    monkeypatch.setattr(main, "Find_categories", lambda *args: [0])
    monkeypatch.setattr(main, "Transtext", lambda *args: None)
    captured = []

    class Client:
        generation_protocol = {"test_double": True}

        def __init__(self, **kwargs):
            pass

        def generate(self, prompt):
            captured.append(prompt)
            return "<seq [['Monday', '(0~3)', 'Light', 'Light:switch on']] seq>"

    monkeypatch.setattr(main, "CodexClient", Client)
    monkeypatch.setattr(main, "Extract", lambda *args: None)
    numeric = Path("filter_data/fr/spring") / f"fr_spring_generation_instance_th=0.918_{config.artifact_model}_seq.pkl"

    def convert(*args):
        numeric.write_bytes(pickle.dumps(rows))

    monkeypatch.setattr(main, "Transnum", convert)
    monkeypatch.setattr(main, "security_check", lambda *args, **kwargs: {
        "input_count": len(rows), "first_pass_count": len(rows),
        "final_count": len(rows), "output_path": str(numeric),
    })
    manifest = main.run_generation(args, config)
    assert manifest["status"] == "completed" and len(captured) == 1
    artifact = json.loads(Path(manifest["relation"]["path"]).read_text())
    assert artifact["gss_snapshot"] == json.loads((source_dir / "action_transitions.json").read_text())
    assert manifest["gcad"]["guidance_enabled"] is True
    assert manifest["gcad"]["prompt_relationship_count"] == 1
    assert "Additional typed behavior relationship audit" in captured[0]
    assert "Directional behavior relationship guidance" in captured[0]
    assert UNKNOWN in captured[0]  # The unchanged original GSS still includes it.
    assert (UNKNOWN in render_relation_guidance(artifact)) == (mode == "typed")
    assert source.read_bytes() == pickle.dumps(rows)


def test_relation_artifacts_are_required_archive_inputs(tmp_path):
    from SmartGen.archiving import collect_current_experiment_files

    manifest = {
        "relation": {"path": str(tmp_path / "relation_artifact.json"),
                     "guidance_path": str(tmp_path / "guidance.txt")},
        "outputs": {"generated_numeric": str(tmp_path / "generated_seq.pkl")},
    }
    files = collect_current_experiment_files(
        make_config(relation_mode="adapted"), manifest,
        {"metrics_path": str(tmp_path / "detector/metrics.json")},
    )
    relations = [item for item in files if item.destination.parts[0] == "relation"]
    assert {item.destination.as_posix() for item in relations} == {
        "relation/relation_artifact.json", "relation/guidance.txt",
    }
    assert all(item.required for item in relations)


def test_preview_reads_source_only_and_keeps_source_bytes_unchanged(tmp_path, monkeypatch):
    project = tmp_path / "upstream"
    source = project / "SmartGen/IoT_data/fr/winter/split_trn.pkl"
    source.parent.mkdir(parents=True)
    source.write_bytes(pickle.dumps(raw_rows(A, B)))
    target = project / "SmartGen/IoT_data/fr/spring"
    target.mkdir(parents=True)
    for filename in ("trn.pkl", "vld.pkl", "test.pkl"):
        (target / filename).write_bytes(b"must never read target data")
    before = source.read_bytes()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if target in path.resolve().parents:
            pytest.fail("preview attempted to open target data")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output = tmp_path / "preview"
    assert preview_main([
        "--source-project", str(project), "--output", str(output),
        "--dataset", "fr", "--ori-env", "winter", "--new-env", "spring",
    ]) == 0
    assert source.read_bytes() == before
    result = json.loads((output / "summary.json").read_text())
    assert result["generation_calls"] == 0


def test_preview_refuses_output_inside_upstream_project(tmp_path):
    with pytest.raises(SystemExit) as result:
        preview_main([
            "--source-project", str(tmp_path), "--output", str(tmp_path / "unsafe-output"),
            "--dataset", "fr", "--ori-env", "winter", "--new-env", "spring",
        ])
    assert result.value.code == 2
