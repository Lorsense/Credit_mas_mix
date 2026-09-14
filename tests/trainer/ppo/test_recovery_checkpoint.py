"""Run production trainer checkpoint methods with the real question buffer."""

import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("recovery_checkpoint_curriculum", ROOT / "verl/utils/question_curriculum.py")
CURRICULUM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRICULUM)


def _methods(**overrides):
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    names = {"_recovery_state", "_save_checkpoint", "_load_checkpoint"}
    selected = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    environment = {"torch": torch, "OmegaConf": OmegaConf, "os": os, "json": json,
                   "ray": SimpleNamespace(get=lambda value: value),
                   "find_latest_ckpt_path": lambda directory: None}
    environment.update(overrides)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "production_recovery_checkpoint", "exec"), environment)
    return environment


def _config(path):
    return OmegaConf.create({
        "algorithm": {
            "advantage_recovery": {"enable": True, "success_weight": .2, "failure_weight": .1,
                                   "allow_missing_resume": False,
                                   "curriculum": {"enabled": True, "seed": 41}},
            "entropy_credit": {"value": {"allow_missing_resume": False}}},
        "trainer": {"default_local_dir": str(path), "default_hdfs_dir": None,
                    "resume_mode": "resume_path", "resume_from_path": str(path / "global_step_3"),
                    "del_local_ckpt_after_load": False, "val_only": False}})


def _trainer(path, *, warm=False, methods=None):
    methods = _methods() if methods is None else methods
    cfg = _config(path)
    curriculum = CURRICULUM.QuestionCurriculum(
        OmegaConf.to_container(cfg.algorithm.advantage_recovery.curriculum),
        dataset_size=100, dataset_fingerprint="ordered-training-questions-v1")
    if warm:
        for step in range(3):
            curriculum.observe({**{i: [1] * 8 for i in range(20)},
                                **{i: [0] * 8 for i in range(20, 30)},
                                **{i: [0, 1] * 4 for i in range(30, 40)}}, step)
        curriculum.select(list(range(20)), 3)
    events = []

    def save_actor(path, *args, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        events.append(("save_actor", str(path)))

    trainer = SimpleNamespace(
        config=cfg, global_steps=3, advantage_recovery_enabled=True,
        question_curriculum=curriculum, value_scorer=None, entropy_controller=None, use_critic=False,
        wg_to_agents_mapping={"solver": [], "verifier": []},
        actor_rollout_wg={role: SimpleNamespace(save_checkpoint=save_actor,
            load_checkpoint=lambda path, **kwargs: events.append(("load_actor", str(path))))
            for role in ("solver", "verifier")},
        train_dataloader=SimpleNamespace(state_dict=lambda: {"position": 3, "epoch": 0},
                                        load_state_dict=lambda state: events.append(("load_data", state))),
        events=events)
    for name in ("_recovery_state", "_save_checkpoint", "_load_checkpoint"):
        setattr(trainer, name, MethodType(methods[name], trainer))
    return trainer


def _json_state(trainer):
    return json.loads(json.dumps(trainer._recovery_state(), allow_nan=False))


def _state_path(path):
    return path / "global_step_3" / "advantage_recovery.json"


def test_real_checkpoint_roundtrip_restores_buffer_rng_and_next_selection(tmp_path):
    uninterrupted = _trainer(tmp_path, warm=True)
    expected = _json_state(uninterrupted)
    uninterrupted._save_checkpoint()
    assert json.loads(_state_path(tmp_path).read_text(encoding="utf-8")) == expected
    resumed = _trainer(tmp_path)
    assert _json_state(resumed) != expected
    resumed._load_checkpoint()
    assert _json_state(resumed) == expected
    assert resumed.global_steps == 3
    assert [kind for kind, _ in resumed.events] == ["load_actor", "load_actor", "load_data"]
    assert resumed.events[-1] == ("load_data", {"position": 3, "epoch": 0})
    # State equality alone can miss a broken tuple/list RNG deserializer.
    for step in (4, 5, 6):
        actual = resumed.question_curriculum.select(list(range(20)), step)
        expected_selection = uninterrupted.question_curriculum.select(list(range(20)), step)
        assert actual == expected_selection
    assert _json_state(resumed) == _json_state(uninterrupted)


def test_allow_missing_resume_is_not_part_of_algorithm_identity(tmp_path):
    trainer = _trainer(tmp_path, warm=True)
    original = _json_state(trainer)
    trainer.config.algorithm.advantage_recovery.allow_missing_resume = True
    assert _json_state(trainer) == original
    assert "allow_missing_resume" not in original["config"]


def test_old_checkpoint_without_recovery_state_fails_before_actor_loading(tmp_path):
    trainer = _trainer(tmp_path)
    with pytest.raises(FileNotFoundError, match="advantage recovery state"):
        trainer._load_checkpoint()
    assert trainer.events == []


def test_explicit_allow_missing_initializes_curriculum_for_old_actor_checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    initial = _json_state(trainer)
    trainer.config.algorithm.advantage_recovery.allow_missing_resume = True
    trainer._load_checkpoint()
    assert _json_state(trainer) == initial
    assert [kind for kind, _ in trainer.events] == ["load_actor", "load_actor"]


@pytest.mark.parametrize("change", ["version", "weight", "curriculum_config"])
def test_checkpoint_algorithm_mismatch_is_rejected_even_with_allow_missing(tmp_path, change):
    trainer = _trainer(tmp_path, warm=True)
    trainer._save_checkpoint()
    payload = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    if change == "version":
        payload["version"] += 1
    elif change == "weight":
        payload["config"]["success_weight"] = .7
    else:
        payload["config"]["curriculum"]["seed"] += 1
    _state_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")
    resumed = _trainer(tmp_path)
    initial = _json_state(resumed)
    resumed.config.algorithm.advantage_recovery.allow_missing_resume = True
    with pytest.raises(ValueError, match="configuration does not match"):
        resumed._load_checkpoint()
    assert resumed.events == []
    assert _json_state(resumed) == initial


def test_dataset_identity_mismatch_is_checked_by_real_buffer_loader(tmp_path):
    trainer = _trainer(tmp_path, warm=True)
    trainer._save_checkpoint()
    resumed = _trainer(tmp_path)
    resumed.question_curriculum.dataset_fingerprint = "different-question-order"
    before = _json_state(resumed)
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        resumed._load_checkpoint()
    assert _json_state(resumed) == before
    assert resumed.events == []


@pytest.mark.parametrize("state", ["missing", "mismatch", "invalid_json"])
def test_validation_only_skips_recovery_state_validation(tmp_path, state):
    trainer = _trainer(tmp_path)
    if state != "missing":
        _state_path(tmp_path).parent.mkdir(parents=True)
        _state_path(tmp_path).write_text("{}" if state == "mismatch" else "not-json", encoding="utf-8")
    initial = _json_state(trainer)
    trainer.config.trainer.val_only = True
    trainer._load_checkpoint()
    assert _json_state(trainer) == initial
    assert [kind for kind, _ in trainer.events] == ["load_actor", "load_actor"]


def test_recovery_disabled_does_not_require_new_state(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.advantage_recovery_enabled = False
    trainer._load_checkpoint()
    assert [kind for kind, _ in trainer.events] == ["load_actor", "load_actor"]


def test_recovery_without_curriculum_saves_and_restores_null_state(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.config.algorithm.advantage_recovery.curriculum.enabled = False
    trainer.question_curriculum = None
    trainer._save_checkpoint()
    assert json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))["curriculum"] is None
    trainer._load_checkpoint()
    assert trainer.question_curriculum is None


def test_checkpoint_tracker_is_published_after_complete_new_state_and_dataloader(tmp_path):
    observed = []
    tracker = tmp_path / "latest_checkpointed_iteration.txt"

    def open_checked(path, mode="r", *args, **kwargs):
        path = Path(path)
        if path == tracker and "w" in mode:
            payload = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
            assert payload["version"] == 1
            assert payload["curriculum"]["last_select_step"] == 3
            assert (tmp_path / "global_step_3" / "data.pt").is_file()
            assert not Path(str(_state_path(tmp_path)) + ".tmp").exists()
            observed.append("published")
        return builtins.open(path, mode, *args, **kwargs)

    trainer = _trainer(tmp_path, warm=True, methods=_methods(open=open_checked))
    trainer._save_checkpoint()
    assert observed == ["published"]
    assert tracker.read_text() == "3"


def test_failed_recovery_save_does_not_publish_incomplete_checkpoint(tmp_path):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("2", encoding="utf-8")
    trainer = _trainer(tmp_path, warm=True)

    def failed_state():
        raise ValueError("injected serialization failure")

    trainer._recovery_state = failed_state
    with pytest.raises(ValueError, match="serialization failure"):
        trainer._save_checkpoint()
    assert tracker.read_text(encoding="utf-8") == "2"
    assert not _state_path(tmp_path).exists()
    assert not (tmp_path / "global_step_3" / "data.pt").exists()


def test_malformed_curriculum_does_not_mutate_live_buffer_or_load_actor(tmp_path):
    trainer = _trainer(tmp_path, warm=True)
    trainer._save_checkpoint()
    payload = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    payload["curriculum"]["rng"] = ["invalid RNG"]
    _state_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")
    resumed = _trainer(tmp_path)
    before = _json_state(resumed)
    with pytest.raises((ValueError, TypeError)):
        resumed._load_checkpoint()
    assert _json_state(resumed) == before
    assert resumed.events == []
