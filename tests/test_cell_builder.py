import json
import pickle

from SmartGen.gcad_source.cell_builder import build_cell_prompts


def test_cell_builder_uses_only_source_representatives_and_preserves_baseline(tmp_path):
    source = tmp_path / "data/fr/winter"
    source.mkdir(parents=True)
    sequence = [0, 0, 13, 76, 0, 1, 13, 77]
    for day in range(7):
        (source / f"trn_day_{day}_SPPC_th=0.5.pkl").write_bytes(pickle.dumps([sequence]))
    (source / "action_transitions.json").write_text("{}")
    stable = tmp_path / "stable.json"; stable.write_text('{"edges": []}')
    fused = tmp_path / "fused.json"; fused.write_text("{}")
    devices = tmp_path / "devices.txt"; devices.write_text("Light: switch on, switch off")
    result = build_cell_prompts(
        dataset="fr", source_context="winter", target_context="spring", compression_threshold=0.5,
        stable_relation_path=stable, fused_gss_path=fused, output_dir=tmp_path / "out",
        source_root=tmp_path / "data", device_control_path=devices,
    )
    baseline = (tmp_path / "out/baseline_prompt.txt").read_text()
    enhanced = (tmp_path / "out/gcad_gss_prompt.txt").read_text()
    assert enhanced.startswith(baseline)
    assert "source environment" in enhanced[len(baseline):]
    assert result["uses_target_behavior"] is False
