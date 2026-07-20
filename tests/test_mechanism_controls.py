from SmartGen.gcad_source.mechanism_controls import random_directed_control, symmetric_control


def relation():
    return {"edges": [
        {"source": "a", "target": "b", "stable_score": 0.8},
        {"source": "a", "target": "c", "stable_score": 0.7},
        {"source": "b", "target": "c", "stable_score": 0.6},
    ]}


def test_random_directed_control_is_deterministic_degree_matched_and_loop_free():
    first = random_directed_control(relation(), seed=9)
    second = random_directed_control(relation(), seed=9)
    assert first == second
    assert len(first["edges"]) == 3
    assert [edge["source"] for edge in first["edges"]].count("a") == 2
    assert all(edge["source"] != edge["target"] for edge in first["edges"])


def test_symmetric_control_has_every_reverse_edge():
    result = symmetric_control(relation())
    pairs = {(edge["source"], edge["target"]) for edge in result["edges"]}
    assert pairs == {("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("b", "c"), ("c", "b")}
