from SmartGen.gcad_source.cli import parser, target_metadata
from SmartGen.gcad_source.config import GCADConfig


def test_config_and_cli_are_runnable():
    config = GCADConfig.from_yaml("configs/gcad_source/fr.yaml")
    assert config.epochs > 2
    assert config.device == "cpu"
    assert parser().parse_args(["validate", "--directory", "x"]).command == "validate"


def test_target_metadata_comes_from_static_dictionary():
    metadata = target_metadata("fr")
    assert "Light" in metadata
    assert "switch on" in metadata["Light"]

