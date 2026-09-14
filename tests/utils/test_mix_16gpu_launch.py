"""Validate actual shell/Hydra arguments and mocked Ray allocation on CPU."""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def ray_method(class_name, method_name, namespace):
    path = ROOT / "verl/single_controller/ray/base.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[method_name]


def test_heterogeneous_ray_groups_reserve_eight_before_seven():
    events = []
    def create(**kwargs):
        count = len(kwargs["bundles"])
        events.append(("create", count))
        return SimpleNamespace(bundle_count=count, ready=lambda: ("ready", count))
    method = ray_method("RayResourcePool", "get_placement_groups", {
        "placement_group": create, "ray": SimpleNamespace(get=lambda value: events.append(value))})
    pool = SimpleNamespace(pgs=None, name_prefix="test", _store=[7, 8], max_colocate_count=1,
                           use_gpu=True, accelerator_type=None, detached=False)
    groups = method(pool)
    assert events == [("create", 8), ("ready", 8), ("create", 7), ("ready", 7)]
    assert [pg.bundle_count for pg in groups] == [7, 8]
    assert method(pool) is groups


@pytest.mark.parametrize("node_counts", [(7, 8), (8, 7), (15,)])
def test_worker_rank_initialization_uses_each_actual_node_count(node_counts):
    created = []
    class Factory:
        cls = object
        def update_options(self, options):
            if "runtime_env" in options:
                self.env = options["runtime_env"]["env_vars"]
        def __call__(self, **kwargs):
            created.append(dict(self.env))
            return object()
    pg = [SimpleNamespace(bundle_count=n) for n in node_counts]
    rank_info = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "9000"}
    fake_ray = SimpleNamespace(get=lambda value: value,
                              get_actor=lambda name: SimpleNamespace(get_rank_zero_info=SimpleNamespace(remote=lambda: rank_info)))
    method = ray_method("RayWorkerGroup", "_init_with_resource_pool", {
        "sort_placement_group_by_node_ip": lambda groups: groups,
        "ray": fake_ray, "list_named_actors": lambda: ["test_register_center"],
        "time": time, "logging": logging})
    pool = SimpleNamespace(use_gpu=True, world_size=15, max_colocate_count=1,
                           get_placement_groups=lambda **kwargs: pg)
    wg = SimpleNamespace(device_name="cuda", name_prefix="test", _workers=[], _worker_names=[],
                         _ray_wait_register_center_timeout=1)
    method(wg, pool, Factory(), True, False)
    assert len(created) == 15
    assert [int(env["RANK"]) for env in created] == list(range(15))
    offset = 0
    for count in node_counts:
        local = created[offset:offset + count]
        assert {int(env["RAY_LOCAL_WORLD_SIZE"]) for env in local} == {count}
        assert [int(env["RAY_LOCAL_RANK"]) for env in local] == list(range(count))
        offset += count


def bash_path():
    candidate = shutil.which("bash") or ("E:/Git/bin/bash.exe" if Path("E:/Git/bin/bash.exe").is_file() else None)
    if candidate is None:
        pytest.skip("Bash unavailable")
    return candidate


@pytest.mark.parametrize("nodes,mode", [(1, "train"), (2, "train"), (1, "eval"), (2, "eval")])
def test_launcher_dry_run_and_hydra_config(tmp_path, nodes, mode):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    data = tmp_path / "data with spaces.parquet"
    data.touch()
    value = tmp_path / "prefix_value.pt"
    value.touch()
    resume = tmp_path / "global_step_50"
    resume.mkdir()
    env = {**os.environ, "NNODES": str(nodes), "RAY_ADDRESS": "auto", "DRY_RUN": "1",
           "VALUE_CHECKPOINT": value.as_posix(), "TRAIN_DATA": data.as_posix(), "VAL_DATA": data.as_posix(),
           "RESUME_FROM": resume.as_posix() if mode == "eval" else "", "TRAIN_BATCH_SIZE": "30", "GROUP_SIZE": "8"}
    result = subprocess.run([bash_path(), "examples/drmas_trainer/run_math_16gpu.sh", mode],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    arguments = shlex.split(result.stdout.strip())
    assert arguments[:3] == ["python3", "-m", "verl.trainer.main_ppo"]
    with hydra.initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = hydra.compose(config_name="ppo_trainer", overrides=arguments[3:])
    OmegaConf.resolve(cfg)
    assert list(cfg.trainer.agent_gpus_per_node) == ([15] if nodes == 1 else [7, 8])
    assert cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node == 16
    assert cfg.trainer.val_only == (mode == "eval")
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert cfg.algorithm.entropy_credit.sparse.enable and cfg.algorithm.entropy_credit.control.enabled
    assert cfg.algorithm.advantage_recovery.enable
    assert cfg.algorithm.advantage_recovery.curriculum.enabled
    assert cfg.actor_rollout_ref.actor.advantage_recovery.enable
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer_cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    validate = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_advantage_recovery_config")
    spec = importlib.util.spec_from_file_location("launch_recovery", ROOT / "verl/utils/advantage_recovery.py")
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    namespace = {"AdvantageEstimator": SimpleNamespace(GRPO="grpo"),
                 "validate_advantage_recovery_config": recovery.validate_advantage_recovery_config}
    exec(compile(ast.Module(body=[validate], type_ignores=[]), "actual_recovery_launch_validation", "exec"), namespace)
    namespace["_validate_advantage_recovery_config"](SimpleNamespace(config=cfg))
    different = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True))
    different.actor.advantage_recovery.enable = False
    overridden = SimpleNamespace(config=cfg, wg_to_agents_mapping={"verifier": [
        {"config_actor_rollout_ref": different}]})
    with pytest.raises(ValueError, match="every Agent"):
        namespace["_validate_advantage_recovery_config"](overridden)
    different.actor.advantage_recovery.enable = True
    different.actor.ulysses_sequence_parallel_size = 2
    with pytest.raises(ValueError, match="sequence parallel"):
        namespace["_validate_advantage_recovery_config"](overridden)
    assert cfg.data.train_batch_size % 15 == 0
    # Worker normalizes this bootstrap setting before adaptive minibatches exist.
    local_mini = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n // 15
    for micro in cfg.agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu:
        assert local_mini > 0 and local_mini % micro == 0
    assert cfg.data.train_files == data.as_posix()
    if mode == "eval":
        assert cfg.algorithm.entropy_credit.value.initial_checkpoint is None
        assert cfg.trainer.resume_from_path == resume.as_posix()


def test_ray_cpu_preflight_accounts_for_value_and_coordinator():
    spec = importlib.util.spec_from_file_location("mix_launch_resources", ROOT / "verl/utils/credit_resources.py")
    resource = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resource)
    with pytest.raises(ValueError, match="resources"):
        resource.validate_pool_layout({"a": {"GPU": 8, "CPU": 8}, "b": {"GPU": 8, "CPU": 8}},
                                      {"agents": [7, 8]}, "a", cpus_per_gpu=1, reserved_cpus=2)
    resource.validate_pool_layout({"a": {"GPU": 8, "CPU": 10}, "b": {"GPU": 8, "CPU": 8}},
                                  {"agents": [7, 8]}, "a", cpus_per_gpu=1, reserved_cpus=2)
