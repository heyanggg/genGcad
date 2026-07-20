from SmartGen.gcad_source.cli import parser, target_metadata
from SmartGen.gcad_source.config import GCADConfig


def test_config_and_cli_are_runnable():
    config = GCADConfig.from_yaml("configs/gcad_source/fr.yaml")
    assert config.epochs > 2
    assert config.device == "cpu"
    assert parser().parse_args(["validate", "--directory", "x"]).command == "validate"
    assert parser().parse_args([
        "evaluate-generated", "--generated", "x.pkl", "--dataset", "fr", "--context", "spring",
        "--output", "out", "--percentile", "95.5"
    ]).command == "evaluate-generated"


def test_target_metadata_comes_from_static_dictionary():
    metadata = target_metadata("fr")
    assert "Light" in metadata
    assert "switch on" in metadata["Light"]


def test_continue_pipeline_can_explicitly_enable_final_evaluation():
    args = parser().parse_args([
        "continue-pipeline", "--directory", "out", "--dataset", "fr", "--context", "spring",
        "--evaluate-output", "evaluation", "--percentile", "95.5", "--apply-ranking-to-downstream",
    ])
    assert args.evaluate_output == "evaluation"
    assert args.apply_ranking_to_downstream is True


def test_preparation_and_final_evaluation_are_separate_commands():
    prepare = parser().parse_args([
        "prepare-generated", "--generated", "x.pkl", "--dataset", "fr", "--context", "spring",
        "--output", "prepared", "--percentile", "95.5",
    ])
    final = parser().parse_args([
        "evaluate-prepared", "--prepared", "prepared", "--dataset", "fr", "--context", "spring",
    ])
    assert prepare.command == "prepare-generated"
    assert final.command == "evaluate-prepared"


def test_source_semantic_gate_requires_only_explicit_source_input():
    args = parser().parse_args([
        "gate-source-semantics", "--directory", "out", "--dataset", "fr",
        "--source", "SmartGen/IoT_data/fr/winter/split_trn.pkl",
    ])
    assert args.command == "gate-source-semantics"
    assert not hasattr(args, "target")


def test_reconstruction_health_gate_accepts_only_generated_diagnostics():
    args = parser().parse_args([
        "gate-reconstruction-health", "--directory", "out", "--diagnostics",
        "seed_2024.json", "seed_2025.json", "seed_2026.json",
    ])
    assert args.command == "gate-reconstruction-health"
    assert len(args.diagnostics) == 3
    assert not hasattr(args, "target")
