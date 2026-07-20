import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from SmartGen.extensions.codex_file import CodexFileBackend, MissingCodexResponse
from SmartGen.extensions.gss_rerank import load_and_rerank_gss, rerank_existing_gss
from SmartGen.extensions.source_gcad import TinyMixer, _legal_sequences, _windows


def sample_gss():
    return {
        "Light:switch on": {
            "message": "The most common action following it",
            "transitions": [
                {"next_action": "Light:switch off", "count": 10},
                {"next_action": "Blind:windowShade open", "count": 9},
            ],
        }
    }


def test_gcad_disabled_returns_same_object_and_content():
    original = sample_gss()
    assert rerank_existing_gss(original, None) is original
    assert rerank_existing_gss(original, None) == original


def test_gcad_reranks_existing_edges_only():
    original = sample_gss()
    relation = {"edges": [{"source": "Light:switch on", "target": "Blind:windowShade open", "score": 1.0}]}
    result = rerank_existing_gss(original, relation, alpha=0.8)
    assert result["Light:switch on"]["transitions"][0]["next_action"] == "Blind:windowShade open"
    assert {row["next_action"] for row in result["Light:switch on"]["transitions"]} == {
        "Light:switch off", "Blind:windowShade open"
    }
    assert original["Light:switch on"]["transitions"][0]["next_action"] == "Light:switch off"


def test_gcad_does_not_add_relation_only_edges():
    relation = {"edges": [{"source": "Light:switch on", "target": "Camera:notification", "score": 99}]}
    result = rerank_existing_gss(sample_gss(), relation, alpha=1)
    assert all(row["next_action"] != "Camera:notification" for row in result["Light:switch on"]["transitions"])


def test_illegal_relation_node_has_no_effect():
    relation = {"edges": [{"source": "None:location", "target": "Light:switch off", "score": 99}]}
    assert rerank_existing_gss(sample_gss(), relation, alpha=1) == sample_gss()


def test_alpha_bounds_are_enforced():
    with pytest.raises(ValueError):
        rerank_existing_gss(sample_gss(), {"edges": []}, alpha=1.1)


def test_relation_file_loader_is_optional(tmp_path):
    original = sample_gss()
    assert load_and_rerank_gss(original, None) is original
    path = tmp_path / "relation.json"
    path.write_text('{"edges": []}')
    assert load_and_rerank_gss(original, path) == original


def test_codex_export_writes_exact_prompt_and_manifest(tmp_path):
    backend = CodexFileBackend(tmp_path, "export")
    assert backend.generate("day_0", "exact prompt") is None
    assert (tmp_path / "prompts/day_0.txt").read_text() == "exact prompt"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["external_api_used"] is False
    assert manifest["requests"]["day_0"]["prompt_sha256"] == hashlib.sha256(b"exact prompt").hexdigest()


def test_codex_consume_reads_authored_response(tmp_path):
    backend = CodexFileBackend(tmp_path, "consume")
    (tmp_path / "responses/day_1.txt").write_text("  authored response  ")
    assert backend.generate("day_1", "prompt") == "authored response"


def test_codex_consume_requires_response(tmp_path):
    with pytest.raises(MissingCodexResponse):
        CodexFileBackend(tmp_path, "consume").generate("day_2", "prompt")


def test_codex_request_id_cannot_escape_directory(tmp_path):
    backend = CodexFileBackend(tmp_path, "export")
    backend.generate("../../day 3", "prompt")
    assert (tmp_path / "prompts/day_3.txt").is_file()


def test_source_parser_excludes_none_location():
    raw = [[0, 0, 18, 96, 0, 1, 13, 78]]
    sequences, invalid, vocabulary = _legal_sequences(raw, "fr")
    assert sequences == [["Light:switch on"]]
    assert vocabulary == ["Light:switch on"]
    assert len(invalid) == 1


def test_event_position_windows_do_not_cross_sequences():
    sequences = [["a:x", "b:y", "c:z"], ["a:x", "b:y", "c:z"]]
    x, y = _windows(sequences, ["a:x", "b:y", "c:z"], history=2)
    assert x.shape == (2, 2, 3)
    assert y.shape == (2, 3)


def test_tiny_mixer_shape_and_gradient():
    model = TinyMixer(history=2, channels=3)
    x = torch.zeros((4, 2, 3), requires_grad=True)
    output = model(x)
    assert output.shape == (4, 3)
    output.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("path", [
    "SmartGen/baseline1.py",
    "SmartGen/baseline2.py",
    "SmartGen/security_check.py",
    "anomaly_detection_pipeline/Anomaly_Detection_pipeline_model.py",
    "anomaly_detection_pipeline/models1.py",
])
def test_official_training_and_detector_files_are_unchanged(path):
    current = Path(path).read_bytes()
    official = subprocess.check_output(["git", "show", f"c2ed36c:{path}"])
    assert current == official


def test_main_no_longer_imports_or_calls_openai():
    source = Path("SmartGen/main.py").read_text()
    assert "from openai import OpenAI" not in source
    assert "chat.completions.create" not in source
    assert "CodexFileBackend" in source


def test_ranking_extension_is_absent():
    assert not Path("SmartGen/gcad_source/sequence_ranking.py").exists()
    assert not Path("SmartGen/extensions/ranking.py").exists()

