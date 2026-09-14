import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mix_credit_resources", ROOT / "verl/utils/credit_resources.py")
RES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RES)


def test_sixteen_total_single_node():
    cfg = dict(nnodes=1, n_gpus_per_node=16, agent_gpus_per_node=[15])
    assert RES.agent_world_size(cfg) == 15
    RES.validate_pool_layout({"head": {"GPU": 16}}, {"agents": [15]}, "head")
    with pytest.raises(ValueError):
        RES.validate_pool_layout({"head": {"GPU": 16}}, {"agents": [16]}, "head")


def test_two_eight_gpu_nodes_and_same_eval_world_size():
    cfg = dict(nnodes=2, n_gpus_per_node=8, agent_gpus_per_node=[7, 8])
    assert RES.agent_world_size(cfg) == 15
    nodes = {"head": {"GPU": 8}, "worker": {"GPU": 8}}
    RES.validate_pool_layout(nodes, {"agents": [7, 8]}, "head")
    RES.validate_pool_layout(nodes, {"agents": [7, 8]})  # evaluation keeps fifteen ranks
    with pytest.raises(ValueError):
        RES.validate_pool_layout(nodes, {"agents": [8, 8]}, "head")


def test_layout_checks_nodes_not_only_gpu_sum():
    with pytest.raises(ValueError):
        RES.validate_pool_layout({"a": {"GPU": 16}}, {"agents": [7, 8]})
    with pytest.raises(ValueError):
        RES.agent_gpu_layout(dict(nnodes=2, n_gpus_per_node=8, agent_gpus_per_node=[15]))
    with pytest.raises(ValueError):
        RES.agent_gpu_layout(dict(nnodes=1, n_gpus_per_node=8, agent_gpus_per_node=[True]))


def test_legacy_homogeneous_layout_unchanged():
    assert RES.agent_gpu_layout(dict(nnodes=2, n_gpus_per_node=8)) == [8, 8]
