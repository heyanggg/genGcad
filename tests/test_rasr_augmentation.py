"""Augmentation-only RaSR: equations, leakage barriers, legality and SSC exposure."""
import ast
import inspect
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SmartGen"))
from SmartGen import rasr
from SmartGen.rasr import (RaSRConfig, augment, build_relations, prepare_source,
                          restore_quadruples, split_parent_groups)
from SmartGen.ssc_input import compose_ssc_input, compress_prefix, select_prompt_examples


def record(tokens, name="p", group=None, origin="I"):
    row = [v for p, t in enumerate(tokens) for v in [0, p % 8, t + 10, t + 20]]
    return {"id": name, "parent_group": group or name, "tokens": list(tokens),
            "row": row, "valid": True, "origin": origin}


def test_paper_defaults_and_no_independent_generator():
    cfg = RaSRConfig()
    assert (cfg.window, cfg.decay, cfg.semantic_variants, cfg.statistical_variants) == (2, .8, 2, 2)
    assert not hasattr(rasr, "RelationGenerator")
    assert not hasattr(cfg, "epochs")
    assert "environment" not in inspect.signature(augment).parameters


def test_eq3_and_eq4_exact_direction_and_one_based_decay():
    relations = build_relations([record([0, 1, 2]), record([0, 1], "q")], np.eye(3), RaSRConfig())
    assert relations["C"][0, 1] == 2
    assert relations["C"][1, 2] == 1
    assert relations["C"][0, 2] == relations["C"][1, 0] == 0
    assert relations["T"][0, 1] == pytest.approx(1.6)
    assert relations["T"][1, 2] == pytest.approx(.64)
    np.testing.assert_allclose(relations["S"], np.eye(3))


def test_split_by_original_parent_including_fragments_and_duplicates_before_graph():
    source = [record([0, 1], f"{group}:{i}", group=group) for group in ("a", "b", "c") for i in range(2)]
    train, val = split_parent_groups(source, RaSRConfig())
    assert not {r["parent_group"] for r in train} & {r["parent_group"] for r in val}
    train_rel = build_relations(train, np.eye(2), RaSRConfig())
    assert train_rel["C"].sum() == len(train)


def test_insert_replace_preserve_time_and_dictionary_device_without_state_gate():
    config = RaSRConfig()
    source = record([0, 1])
    relations = {"C": np.array([[0, 0, 2], [3, 0, 0], [0, 0, 0]]),
                 "T": np.zeros((3, 3)), "S": np.ones((3, 3))}
    before = deepcopy(source)
    variants, _ = augment([source], relations, [0, 1, 2], [20, 21, 22], [10, 11, 12], config)
    assert source == before
    assert 1 <= len(variants) <= 4
    assert {v["edit"]["operation"] for v in variants} >= {"semantic_insert", "semantic_replace"}
    for v in variants:
        assert v["parent_id"] == "p"
        for k, token in enumerate(v["tokens"]):
            assert v["row"][4*k+2:4*k+4] == [10+token, 20+token]
            assert v["row"][4*k:4*k+2] in [[0, 0], [0, 1]]


def test_no_relation_does_not_force_variants():
    relations = {key: np.zeros((2, 2)) for key in ("C", "T", "S")}
    variants, skipped = augment([record([0])], relations, [0, 1], [20, 21], [10, 11], RaSRConfig())
    assert variants == []
    assert skipped["no_positive_relation"] == 4


def test_random_control_is_seeded_and_never_creates_illegal_actions():
    relations = {key: np.ones((4, 4)) for key in ("C", "T", "S")}
    args = ([record([0, 1])], relations, [0, 1, 2], [20, 21, 22, 23], [10, 11, 12, -1], RaSRConfig())
    one, _ = augment(*args, random_control=True)
    two, _ = augment(*args, random_control=True)
    assert one == two
    assert all(r["edit"]["new_token"] in {0, 1, 2} for r in one)
    assert all(r["edit"]["new_token"] != r["edit"]["old_token"] for r in one)


def test_invalid_parent_is_preserved_but_not_augmented_or_counted():
    source = record([0, 1])
    source["valid"] = False
    relations = build_relations([source], np.ones((2, 3)), RaSRConfig())
    assert not relations["C"].any() and not relations["T"].any()
    variants, skipped = augment([source], relations, [0, 1], [20, 21], [10, 11], RaSRConfig())
    assert variants == [] and skipped["invalid_or_overlength_parent"] == 1
    assert compose_ssc_input([source], mode="I") == [source]


def test_semantic_only_never_uses_statistical_strategy():
    relations = {key: np.ones((3, 3)) for key in ("C", "T", "S")}
    variants, _ = augment([record([0, 1])], relations, [0, 1, 2], [20, 21, 22], [10, 11, 12],
                          RaSRConfig(), semantic_only=True)
    assert len(variants) <= 2
    assert all(r["edit"]["operation"].startswith("semantic_") for r in variants)


def test_statistical_replacement_uses_argmax_c_plus_t():
    relations = {key: np.zeros((3, 3)) for key in ("C", "T", "S")}
    relations["C"][0, 1] = 2
    relations["T"][0, 2] = 3
    variants, _ = augment([record([0])], relations, [0, 1, 2], [20, 21, 22], [10, 11, 12], RaSRConfig())
    assert len(variants) == 1  # repeated deterministic variant is skipped, not counted twice
    assert variants[0]["tokens"] == [2]


def test_semantic_replacement_can_cross_state_and_device_boundaries():
    relations = {key: np.zeros((2, 2)) for key in ("C", "T", "S")}
    relations["S"][0, 1] = 1  # e.g. light:on -> fan:off; no functional-equivalence gate
    variants, _ = augment([record([0])], relations, [0, 1], [20, 21], [10, 11], RaSRConfig())
    assert [1] in [v["tokens"] for v in variants]


def test_length_cap_skips_insertion_not_all_augmentation():
    relations = {key: np.ones((2, 2)) for key in ("C", "T", "S")}
    variants, skipped = augment([record([0]*10)], relations, [0, 1], [20, 21], [10, 11], RaSRConfig())
    assert skipped["insertion_at_length_limit"] == 1
    assert all(len(v["tokens"]) == 10 for v in variants)


def test_time_restoration_is_deterministic_and_uses_only_original_slots():
    row = [1, 2, 10, 20, 1, 5, 11, 21]
    result = restore_quadruples(row, [2, 1, 0, 2], [20, 21, 22], [10, 11, 12])
    assert result[0::4] == [1]*4
    assert result[1::4] == [2, 2, 2, 5]
    assert result[2::4] == [12, 11, 10, 12]


class IdentitySSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))

    def forward(self, source, src_key_padding_mask):
        return torch.nn.functional.one_hot(source, 50).float()


def test_explicit_ssc_input_provenance_and_regeneration_disabled():
    original = record([0])
    j = {**record([1], "j", origin="J"), "parent_id": "p"}
    assert compose_ssc_input([original], [j], mode="I+J") == [original, j]
    assert compose_ssc_input([original], [j], mode="I") == [original]
    with pytest.raises(ValueError, match="not regeneration"):
        compose_ssc_input([original], [], mode="I+R")
    with pytest.raises(ValueError, match="real source parent"):
        compose_ssc_input([original], [{**j, "parent_id": "unknown"}], mode="I+J")


def test_ssc_greedy_prefix_matches_full_cosine_selection():
    source = [record([0]), record([0], "duplicate"), record([1], "next")]
    selected, audit = compress_prefix(source, IdentitySSC(), 50, .99999, 2)
    assert [r["id"] for r in selected] == ["p", "next"]
    assert audit["scanned"] == 3


def test_actual_prompt_exposure_counts_and_no_silent_fill():
    i = record([0])
    j = {**record([1], "j", origin="J"), "parent_id": "p"}
    selected, audit = select_prompt_examples([i, j], IdentitySSC(), 50, .99, day=0, examples=2, synthetic=True)
    assert len(selected) == 2
    assert audit["actual_prompt_origin_counts"] == {"I": 1, "J": 1}
    with pytest.raises(ValueError, match="quota"):
        select_prompt_examples([i], IdentitySSC(), 50, .99, day=0, examples=2, synthetic=True)


def test_new_modules_do_not_import_cbtr_or_environment_adapter():
    for module in (rasr,):
        imports = [n.module for n in ast.walk(ast.parse(inspect.getsource(module))) if isinstance(n, ast.ImportFrom)]
        assert not any("relation_transfer" in (s or "") or "relation_adapter" in (s or "") for s in imports)
