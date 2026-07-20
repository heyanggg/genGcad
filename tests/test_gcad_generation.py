import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from SmartGen.codex_backend import CodexClient, CodexGenerationError
from SmartGen.gcad import (
    ActionPredictor,
    aggregate_stable_relationships,
    build_prediction_windows,
    decode_normal_sequences,
    discover_gradient_influences,
    extract_directional_relationships,
    sparsify_directional_graph,
    split_train_validation_sequences,
)


def test_normal_parser_excludes_non_device_actions():
    raw = [[0, 0, 17, 96, 0, 1, 13, 78]]
    assert decode_normal_sequences(raw, "fr") == [["Light:switch on"]]


def test_prediction_windows_stay_inside_each_sequence():
    histories, targets, vocabulary = build_prediction_windows(
        [["a:on", "b:on", "c:on"], ["a:on", "b:on", "c:on"]],
        history=2,
    )
    assert histories.shape == (2, 2, 3)
    assert targets.shape == (2, 3)
    assert targets.argmax(axis=1).tolist() == [
        vocabulary.index("c:on"), vocabulary.index("c:on")
    ]


def test_predictor_is_multichannel_tsmixer_style():
    model = ActionPredictor(history=3, action_count=4)
    output = model(torch.zeros(2, 3, 4))
    assert output.shape == (2, 4)
    assert len(model.blocks) == 2


def test_gradient_discovery_preserves_lag_positions():
    model = nn.Sequential(nn.Flatten(), nn.Linear(6, 3, bias=False))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].weight[2, 0] = 2.0
        model[1].weight[2, 4] = 5.0
    histories = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    targets = np.asarray([[0, 0, 1]], dtype=np.float32)
    influences, supports = discover_gradient_influences(model, histories, targets)
    assert influences[(1, 2, 1)] > influences[(0, 2, 2)] > 0
    assert influences[(1, 2, 1)] / influences[(0, 2, 2)] == pytest.approx(2.5)
    assert supports[(0, 2, 2)] == 1


def test_sparsification_keeps_stronger_one_way_edge_and_best_lag():
    influences = {
        (0, 1, 1): 0.8,
        (0, 1, 2): 0.9,
        (1, 0, 1): 0.2,
        (0, 2, 1): 0.3,
        (2, 0, 1): 0.25,
        (2, 1, 1): 1.0,
    }
    supports = {key: 5 for key in influences}
    result = sparsify_directional_graph(
        influences,
        supports,
        ["light:on", "curtain:open", "camera:on"],
        sparsity_quantile=0.5,
    )
    assert result[0]["source_index"] == 0
    assert result[0]["target_index"] == 1
    assert result[0]["typical_lag"] == 2
    assert result[0]["raw_strength"] == pytest.approx(1.5)
    assert result[1]["source_index"] == 2
    assert result[1]["target_index"] == 1
    assert result[1]["raw_strength"] == pytest.approx(1.0)


def test_sequence_holdout_has_no_sequence_overlap():
    sequences = [
        ["a", "b", "c", "d"],
        ["a", "c", "b", "d"],
        ["d", "c", "b", "a"],
    ]
    train_x, _, validation_x, _, _, method = split_train_validation_sequences(
        sequences, history=2, validation_ratio=0.34, seed=7
    )
    assert method == "sequence_holdout"
    assert len(train_x) == 4
    assert len(validation_x) == 2


def test_stability_filter_requires_recurrence_across_seeds():
    stable = {
        "source_index": 0,
        "target_index": 1,
        "typical_lag": 2,
        "raw_strength": 0.5,
        "support": 9,
        "lag_strengths": {2: 0.7},
    }
    unstable = {**stable, "source_index": 1, "target_index": 2}
    result = aggregate_stable_relationships(
        [(1, [stable, unstable]), (2, [stable]), (3, [stable])],
        ["a", "b", "c"],
        stable_seed_fraction=2 / 3,
    )
    assert len(result) == 1
    assert result[0]["source_action"] == "a"
    assert result[0]["target_action"] == "b"
    assert result[0]["stable_seeds"] == 3
    assert result[0]["strength"] == 1.0


def test_failed_seed_cannot_reduce_stability_requirement():
    edge = {
        "source_index": 0,
        "target_index": 1,
        "typical_lag": 1,
        "raw_strength": 0.2,
        "support": 5,
        "lag_strengths": {1: 0.2},
    }
    result = aggregate_stable_relationships(
        [(2024, [edge])],
        ["a", "b"],
        stable_seed_fraction=2 / 3,
        total_seed_count=3,
    )
    assert result == []


def test_extractor_disables_tiny_data_and_reuses_frozen_artifact(tmp_path, monkeypatch):
    source = tmp_path / "split_trn.pkl"
    output = tmp_path / "gcad_hints.json"
    with source.open("wb") as handle:
        pickle.dump([[0, 0, 17, 96, 0, 1, 17, 96]], handle)
    first = extract_directional_relationships(
        source,
        "fr",
        output,
        min_total_windows=30,
    )
    assert first["status"] == "disabled"
    assert first["lagged_behavior_relations"] == []
    monkeypatch.setattr("SmartGen.gcad.pickle.load", lambda _: pytest.fail("cache missed"))
    second = extract_directional_relationships(
        source,
        "fr",
        output,
        min_total_windows=30,
    )
    assert second == first


def test_codex_client_invokes_gpt56_without_exchange_files(monkeypatch, tmp_path):
    monkeypatch.setattr("SmartGen.codex_backend.shutil.which", lambda _: "/usr/bin/codex")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "prefix <seq [['ok']] seq> suffix", "")

    monkeypatch.setattr("SmartGen.codex_backend.subprocess.run", fake_run)
    client = CodexClient(working_directory=tmp_path)
    assert client.generate("prompt") == "<seq [['ok']] seq>"
    assert captured["input"] == "prompt"
    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-5.6-sol"
    assert captured["command"][captured["command"].index("--sandbox") + 1] == "read-only"
    assert 'model_reasoning_effort="medium"' in captured["command"]


def test_codex_client_rejects_unparseable_output(monkeypatch, tmp_path):
    monkeypatch.setattr("SmartGen.codex_backend.shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "SmartGen.codex_backend.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "not a sequence", ""),
    )
    with pytest.raises(CodexGenerationError):
        CodexClient(working_directory=tmp_path).generate("prompt")


def test_main_adds_gcad_as_soft_guidance_without_reranking_gss(monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("main", None)
    from main import build_prompt

    gss = {"a:on": {"transitions": [{"next_action": "b:on", "count": 4}]}}
    gcad = {"status": "ready", "lagged_behavior_relations": [{
        "source_action": "a:on", "target_action": "c:on", "strength": 0.8,
        "typical_lag": 2
    }]}
    prompt = build_prompt("devices", "environment", [["sequence"]], gss, gcad)
    assert json.dumps(gss, ensure_ascii=False) in prompt
    assert '"source_action": "a:on"' in prompt
    assert "artifact_fingerprint" not in prompt
    assert "Immediate transition patterns" in prompt
    assert "Lagged directional dependency patterns" in prompt
    assert "soft guidance" in prompt
    assert "Do not force every pattern" in prompt
    assert "rerank" not in Path("SmartGen/main.py").read_text(encoding="utf-8").lower()


def test_main_boolean_arguments_parse_false(monkeypatch):
    smartgen = str(Path("SmartGen").resolve())
    monkeypatch.syspath_prepend(smartgen)
    sys.modules.pop("main", None)
    from main import get_args_parser

    args = get_args_parser().parse_args(["--need_generate", "False", "--need_test", "False"])
    assert args.need_generate is False
    assert args.need_test is False


def test_main_trains_gcad_from_tss_output_before_ssc():
    source = Path("SmartGen/main.py").read_text(encoding="utf-8")
    split = source.index("        Split(args.dataset")
    gcad = source.index("        gcad_relationships = extract_directional_relationships", split)
    ssc = source.index("        Dayse(args.dataset", gcad)
    assert split < gcad < ssc
    assert "split_trn.pkl" in source[gcad:ssc]
