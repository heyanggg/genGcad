from SmartGen.gcad_source.gss_fusion import fuse_gss
from SmartGen.gcad_source.prompt_adapter import adapt_prompt
from SmartGen.gcad_source.sequence_ranking import rank_sequences


def fixtures():
    gss = {
        "Light:switch on": {"message": "x", "transitions": [
            {"next_action": "Light:switch off", "count": 10},
            {"next_action": "Blind:windowShade open", "count": 5},
        ]}
    }
    stable = {"edges": [
        {"source": "Light:switch on", "target": "Blind:windowShade open", "stable_score": 1.0, "mean_primary_lag": 2},
        {"source": "Blind:windowShade open", "target": "Light:switch on", "stable_score": 0.9, "mean_primary_lag": 1},
    ]}
    metadata = {"Light": ["switch on", "switch off"], "Blind": ["windowShade open"]}
    return gss, stable, metadata


def test_rerank_existing_formula_and_no_new_edges():
    gss, stable, metadata = fixtures()
    fused, report = fuse_gss(gss, stable, metadata, alpha=0.2)
    transitions = fused["Light:switch on"]["transitions"]
    assert {item["next_action"] for item in transitions} == {"Light:switch off", "Blind:windowShade open"}
    enhanced = next(item for item in transitions if item["next_action"].startswith("Blind"))
    assert enhanced["fused_score"] == (0.8 * 0.5) + (0.2 * 1.0)
    assert report["new_edge_count"] == 0


def test_illegal_action_cannot_influence_fusion():
    gss, stable, metadata = fixtures()
    metadata.pop("Blind")
    fused, _ = fuse_gss(gss, stable, metadata, alpha=0.2)
    enhanced = next(item for item in fused["Light:switch on"]["transitions"] if item["next_action"].startswith("Blind"))
    assert enhanced["gcad_score"] == 0


def test_disabled_prompt_is_byte_equivalent():
    baseline = "exact baseline prompt\nwith spacing"
    assert adapt_prompt(baseline, enabled=False) == baseline


def test_ranking_is_soft_deterministic_and_shared_data():
    _, stable, _ = fixtures()
    sequences = [
        [{"device": "Light", "action": "switch on"}, {"device": "Blind", "action": "windowShade open"}],
        [{"device": "Blind", "action": "windowShade open"}, {"device": "Light", "action": "switch on"}],
    ]
    first = rank_sequences(sequences, stable)
    second = rank_sequences(sequences, stable)
    assert first == second
    assert first["hard_filter"] is False
    assert first["sequence_count"] == len(sequences)
    assert sorted(row["sequence_index"] for row in first["ranking"]) == [0, 1]

