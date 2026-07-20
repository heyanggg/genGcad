from pathlib import Path

import pytest

from SmartGen.gcad_source.data_boundary import DataBoundaryError, require_roles
from SmartGen.gcad_source.data_roles import DataRole, RoleBoundPath


def artifact(role: DataRole) -> RoleBoundPath:
    return RoleBoundPath(Path("/tmp/example"), role)


@pytest.mark.parametrize("stage", ["train", "generation", "threshold", "tof", "ranking"])
def test_pre_eval_stages_reject_target_normal(stage):
    with pytest.raises(DataBoundaryError):
        require_roles(stage, [artifact(DataRole.TARGET_NORMAL_EVAL)])


@pytest.mark.parametrize("stage", ["train", "generation", "threshold", "tof", "ranking"])
def test_pre_eval_stages_reject_target_attack(stage):
    with pytest.raises(DataBoundaryError):
        require_roles(stage, [artifact(DataRole.TARGET_ATTACK_EVAL)])


def test_target_attack_only_enters_final_evaluation():
    require_roles("final_evaluation", [artifact(DataRole.TARGET_ATTACK_EVAL)])

