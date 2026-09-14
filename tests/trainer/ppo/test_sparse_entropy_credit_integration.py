"""Exercise real sparse helpers and Torch advantages without launching Ray."""
import ast
from collections import defaultdict
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[3]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENTROPY = load_module("pure_test_entropy_credit", "verl/utils/entropy_credit.py")
SPARSE = load_module("pure_test_sparse_entropy_credit", "verl/utils/sparse_entropy_credit.py")


def real_grpo_function():
    path = ROOT / "verl/trainer/ppo/core_algos.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name == "compute_grpo_outcome_advantage")
    # Registration is irrelevant when directly executing the actual function.
    function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    namespace = {"torch": torch, "np": np, "defaultdict": defaultdict}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[function.name]


def trainer_helpers():
    path = ROOT / "verl/trainer/ppo/ray_trainer.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "compute_response_mask", "compute_advantage", "prepare_entropy_credit", "prepare_sparse_entropy_credit_for_batch",
        "finalize_sparse_entropy_credit_for_advantages", "apply_entropy_credit_to_advantages", "_dump_generations",
    }
    namespace = {"np": np, "torch": torch, "os": os, "json": json, "Enum": Enum, "DataProto": object,
                 "core_algos": SimpleNamespace(compute_grpo_outcome_advantage=real_grpo_function()),
                 "normalize_agent_id": lambda role: role.replace(" ", "")}
    for node in source.body:
        if isinstance(node, ast.ImportFrom) and node.module in (
            "verl.utils.entropy_credit", "verl.utils.sparse_entropy_credit"
        ):
            module = ENTROPY if node.module.endswith(".entropy_credit") else SPARSE
            for alias in node.names:
                namespace[alias.asname or alias.name] = getattr(module, alias.name)
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in source.body:
        if isinstance(node, ast.ClassDef):
            selected.extend(method for method in node.body
                            if isinstance(method, ast.FunctionDef) and method.name == "_dump_generations")
    assert {node.name for node in selected} == names
    selected.insert(0, next(node for node in source.body if isinstance(node, ast.ClassDef)
                            and node.name == "AdvantageEstimator"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


HELPERS = trainer_helpers()


class Batch:
    def __init__(self, tensors, metadata):
        self.batch = tensors
        self.non_tensor_batch = metadata
        self.meta_info = {"eos_token_id": 2, "pad_token_id": 0}

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def rows(self, indices):
        return Batch({key: value[indices].clone() for key, value in self.batch.items()},
                     {key: np.asarray(value)[indices].copy() for key, value in self.non_tensor_batch.items()})


def entropy_config(sparse=True):
    return {"action_scale": 0.2, "trajectory_scale": 0.1, "trajectory_deadzone": 0.05,
            "sparse": {"enable": sparse}}


def fixture(success=True, include_verifier=False):
    previous = [.09, .02, .03, .04, .05, .06, .07, .08] if success else [.01, .02, .03, .04, .05, .06, .07, .08]
    current = [.01, .021, .031, .041, .051, .061, .071, .081] if success else [.095, .019, .029, .039, .049, .059, .069, .079]
    rows = []
    roles = ["Solver Agent", "Verifier Agent"] if include_verifier else ["Solver Agent"]
    for trajectory in range(8):
        # Mixed terminal outcomes yield nonzero grouped GRPO advantages. t0 is
        # the only strong entropy transition; all others are near the drift.
        label = success if trajectory == 0 else not success
        for turn in range(2):
            for role in roles:
                rows.append((f"t{trajectory}", role, turn, int(label),
                             previous[trajectory] if turn == 0 else current[trajectory]))
    n, width = len(rows), 8
    lengths = np.array([2 if row[2] == 0 else 3 for row in rows])
    mask = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    responses = torch.where(mask, 11, 0).long()
    responses[torch.arange(n), torch.tensor(lengths) - 1] = 2
    scalar = np.array([1.0 if row[3] else -1.0 for row in rows])
    advantages = torch.tensor(scalar, dtype=torch.float64)[:, None] * mask
    tensors = {"responses": responses, "response_mask": mask.long(),
               "attention_mask": torch.cat((torch.ones(n, 2, dtype=torch.long), mask.long()), dim=1),
               "advantages": advantages, "returns": torch.full((n, width), 7.0, dtype=torch.float64),
               "token_level_rewards": torch.arange(n * width, dtype=torch.float64).reshape(n, width)}
    metadata = {
        "uid": np.array(["prompt"] * n, dtype=object),
        "traj_uid": np.array([row[0] for row in rows], dtype=object),
        "agent_id": np.array([row[1] for row in rows], dtype=object),
        "role_turn_index": np.array([row[2] for row in rows]),
        "pass": np.array([row[3] for row in rows]),
        "is_action_valid": np.ones(n, dtype=bool),
        "top16_entropy_mean": np.array([row[4] for row in rows]),
        "top16_entropy": np.array([{"coverage": 1.0, "effective_support": 2.0} for _ in rows], dtype=object),
    }
    return Batch(tensors, metadata)


def prepared(batch):
    config = entropy_config()
    batch, _ = HELPERS["prepare_entropy_credit"](batch, config)
    batch, metrics = HELPERS["prepare_sparse_entropy_credit_for_batch"](batch, config["sparse"])
    return batch, metrics


def finalized(batch):
    return HELPERS["finalize_sparse_entropy_credit_for_advantages"](batch, entropy_config()["sparse"])


def logical_factors(batch):
    meta = batch.non_tensor_batch
    return {(str(meta["traj_uid"][i]), str(meta["agent_id"][i]), int(meta["role_turn_index"][i])):
            float(meta["entropy_credit_final_multiplier"][i]) for i in range(len(batch))}


@pytest.mark.parametrize("success", [True, False])
def test_real_joint_gate_preserves_stage_one_mass_and_updates_actual_logs(success):
    batch, _ = prepared(fixture(success, include_verifier=True))
    meta = batch.non_tensor_batch
    stage_one = meta["entropy_credit_action_multiplier"].copy()
    np.testing.assert_array_equal(meta["entropy_credit_final_multiplier"], stage_one)
    rewards, returns = batch.batch["token_level_rewards"].clone(), batch.batch["returns"].clone()
    batch, metrics = finalized(batch)
    final = meta["entropy_credit_final_multiplier"]
    selected = (meta["traj_uid"] == "t0") & (meta["agent_id"] == "Solver Agent")
    assert np.any(np.abs(final[selected] - stage_one[selected]) > 1e-9)
    np.testing.assert_array_equal(final[~selected], stage_one[~selected])
    np.testing.assert_array_equal(meta["entropy_credit_action_multiplier"], stage_one)
    lengths = meta["pure_entropy_response_tokens"]
    np.testing.assert_allclose(np.sum(lengths[selected] * final[selected]),
                               np.sum(lengths[selected] * stage_one[selected]), rtol=0, atol=1e-10)
    assert np.all((final >= .8) & (final <= 1.2))
    np.testing.assert_allclose(meta["entropy_credit_trajectory_multiplier"], final / stage_one)
    for role in ("SolverAgent", "VerifierAgent"):
        scope = meta["agent_id"] == role.replace("Agent", " Agent")
        key = f"entropy_credit/{role}/final_multiplier_mean"
        assert key in metrics
        assert metrics[key] == pytest.approx(np.mean(final[scope]))
    HELPERS["apply_entropy_credit_to_advantages"](batch)
    torch.testing.assert_close(batch.batch["token_level_rewards"], rewards)
    torch.testing.assert_close(batch.batch["returns"], returns)


def test_padding_and_reordering_after_prepare_preserve_logical_factors():
    original, _ = prepared(fixture())
    indices = np.array([7, 0, 15, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 0, 1, 1])
    padded = original.rows(indices)
    finalized(original)
    finalized(padded)
    assert logical_factors(padded) == logical_factors(original)
    meta = padded.non_tensor_batch
    for trajectory, role, turn in logical_factors(original):
        selected = (meta["traj_uid"] == trajectory) & (meta["agent_id"] == role) & (meta["role_turn_index"] == turn)
        np.testing.assert_array_equal(meta["entropy_credit_final_multiplier"][selected],
                                      np.full(np.sum(selected), logical_factors(original)[trajectory, role, turn]))


@pytest.mark.parametrize("bad_advantages", [(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0), (1.0, -1.0)])
def test_actual_advantage_sign_or_zero_disables_success_transfer(bad_advantages):
    batch, _ = prepared(fixture(True))
    for row, scalar in enumerate(bad_advantages):
        batch.batch["advantages"][row] = scalar * batch.batch["response_mask"][row]
    expected = batch.non_tensor_batch["entropy_credit_action_multiplier"].copy()
    finalized(batch)
    np.testing.assert_array_equal(batch.non_tensor_batch["entropy_credit_final_multiplier"], expected)


def test_final_multiplier_is_detached_and_applies_only_to_advantages():
    batch, _ = prepared(fixture())
    source = batch.batch["advantages"].clone().requires_grad_(True)
    batch.batch["advantages"] = source
    finalized(batch)
    factor = torch.tensor(batch.non_tensor_batch["entropy_credit_final_multiplier"], dtype=source.dtype)
    HELPERS["apply_entropy_credit_to_advantages"](batch)
    batch.batch["advantages"].sum().backward()
    torch.testing.assert_close(source.grad, factor[:, None].expand_as(source))


def test_sparse_disabled_keeps_exact_legacy_two_stage_factors():
    batch = fixture()
    expected = ENTROPY.compute_entropy_credit_multipliers(
        prompt_group_ids=batch.non_tensor_batch["uid"], trajectory_ids=batch.non_tensor_batch["traj_uid"],
        agent_ids=batch.non_tensor_batch["agent_id"], role_turn_indices=batch.non_tensor_batch["role_turn_index"],
        terminal_success=batch.non_tensor_batch["pass"].astype(bool),
        action_entropies=batch.non_tensor_batch["top16_entropy_mean"],
        action_valid=batch.non_tensor_batch["is_action_valid"], action_scale=.2, trajectory_scale=.1,
        trajectory_deadzone=.05,
    )
    HELPERS["prepare_entropy_credit"](batch, entropy_config(False))
    for name, field in (("action", "action"), ("trajectory", "trajectory"), ("final", "final")):
        np.testing.assert_array_equal(batch.non_tensor_batch[f"entropy_credit_{field}_multiplier"], expected[name])


def test_three_solver_actions_share_one_component_and_preserve_its_token_mass():
    base = fixture(True)
    batch = base.rows(np.repeat(np.arange(0, len(base), 2), 3))
    meta = batch.non_tensor_batch
    meta["role_turn_index"] = np.tile(np.arange(3), 8)
    entropy = []
    for trajectory in range(8):
        entropy.extend([.09, .07, .05] if trajectory == 0 else
                       [.015 + .005 * (trajectory - 1) + .001 * turn for turn in range(3)])
    meta["top16_entropy_mean"] = np.array(entropy)
    lengths = torch.tensor(np.tile([2, 3, 4], 8))
    mask = torch.arange(8)[None, :] < lengths[:, None]
    batch.batch["response_mask"] = mask.long()
    batch.batch["attention_mask"] = torch.cat((torch.ones(len(batch), 2, dtype=torch.long), mask.long()), dim=1)
    batch.batch["responses"] = torch.where(mask, 11, 0).long()
    batch.batch["responses"][torch.arange(len(batch)), lengths - 1] = 2
    batch.batch["advantages"] = torch.tensor(np.where(meta["pass"], 1.0, -1.0))[:, None] * mask
    prepared(batch)
    baseline = meta["entropy_credit_action_multiplier"].copy()
    _, metrics = finalized(batch)
    selected = meta["traj_uid"] == "t0"
    component = meta["pure_entropy_active_component"]
    assert len(np.unique(component[selected])) == 1
    assert component[selected][0] >= 0
    assert metrics["pure_entropy/active_edges"] == 2
    assert metrics["pure_entropy/active_components"] == 1
    np.testing.assert_allclose(np.sum(meta["pure_entropy_response_tokens"][selected] *
                                     meta["entropy_credit_final_multiplier"][selected]),
                               np.sum(meta["pure_entropy_response_tokens"][selected] * baseline[selected]),
                               rtol=0, atol=1e-10)
    np.testing.assert_array_equal(meta["entropy_credit_final_multiplier"][~selected], baseline[~selected])


def test_prepare_derives_lengths_and_conservatively_excludes_full_width_responses():
    batch = fixture()
    # A full-width action is excluded even if it ends with EOS, because the
    # rollout path does not retain a reliable finish_reason.
    batch.batch["response_mask"][0] = 1
    batch.batch["attention_mask"][0] = 1
    batch.batch["responses"][0] = 11
    batch.batch["responses"][0, -1] = 2
    prepared(batch)
    meta = batch.non_tensor_batch
    assert meta["pure_entropy_response_tokens"][0] == 8
    assert bool(meta["pure_entropy_truncated"][0])
    assert meta["pure_entropy_response_tokens"][1] == 3
    assert not bool(meta["pure_entropy_truncated"][1])
    expected = meta["entropy_credit_action_multiplier"].copy()
    finalized(batch)
    np.testing.assert_array_equal(meta["entropy_credit_final_multiplier"], expected)


def test_masked_padding_nan_does_not_change_the_actual_advantage_sign():
    batch, _ = prepared(fixture())
    reference = batch.rows(np.arange(len(batch)))
    batch.batch["advantages"][~batch.batch["response_mask"].bool()] = float("nan")
    finalized(batch)
    finalized(reference)
    assert logical_factors(batch) == logical_factors(reference)


def test_actual_rollout_export_retains_pure_fields_and_logical_actions_in_utf8(tmp_path):
    metadata = {
        "traj_uid": np.array(["t0", "t0", "t0", "t1"], dtype=object),
        "agent_id": np.array(["Solver Agent"] * 4, dtype=object),
        "role_turn_index": np.array([0, 1, 0, 0]),
        "pure_entropy_response_tokens": np.array([2, 3, 2, 4]),
        "pure_entropy_residual": np.array([np.nan, -.04, np.nan, .01]),
        "pure_entropy_selected_gate": np.array([0.0, .4, 0.0, 0.0]),
        "pure_entropy_truncated": np.array([False, False, False, True]),
        "entropy_credit_final_multiplier": np.array([1.1, .9, 1.1, 1.0]),
    }
    HELPERS["_dump_generations"](
        SimpleNamespace(global_steps=17), inputs=["\u4e2d\u6587\u9898\u76ee"] * 4, outputs=["\u540c\u4e00\u4e2a\u89e3\u7b54"] * 4,
        scores=[1.0, 1.0, 1.0, 0.0], reward_extra_infos_dict={"pass": [1, 1, 1, 0]},
        dump_path=str(tmp_path), rollout_metadata=metadata,
    )
    raw = (tmp_path / "17.jsonl").read_bytes().decode("utf-8")
    assert "\u4e2d\u6587\u9898\u76ee" in raw and "\u540c\u4e00\u4e2a\u89e3\u7b54" in raw
    assert any(ord(character) > 127 for character in raw)
    assert "NaN" not in raw
    exported = [json.loads(line) for line in raw.splitlines()]
    assert len(exported) == 3
    assert [(row["traj_uid"], row["role_turn_index"]) for row in exported] == [("t0", 0), ("t0", 1), ("t1", 0)]
    assert exported[0]["output"] == exported[1]["output"]
    assert exported[0]["pure_entropy_residual"] is None
    assert exported[1]["pure_entropy_selected_gate"] == .4
    assert exported[2]["pure_entropy_truncated"] is True
    assert [row["pure_entropy_response_tokens"] for row in exported] == [2, 3, 4]
    assert all(row["step"] == 17 for row in exported)


def test_actual_omegaconf_listconfig_is_accepted_by_sparse_pipeline():
    omegaconf = pytest.importorskip("omegaconf")
    config = omegaconf.OmegaConf.create(entropy_config())
    config.sparse.roles = ["Solver Agent"]
    assert isinstance(config.sparse.roles, omegaconf.ListConfig)
    SPARSE.validate_sparse_entropy_config(config.sparse)
    batch = fixture()
    HELPERS["prepare_entropy_credit"](batch, config)
    HELPERS["prepare_sparse_entropy_credit_for_batch"](batch, config.sparse)
    _, metrics = HELPERS["finalize_sparse_entropy_credit_for_advantages"](batch, config.sparse)
    assert metrics["pure_entropy/active_edges"] == 1


@pytest.mark.parametrize("success", [True, False])
def test_actual_role_grouped_grpo_rewards_feed_sparse_credit_without_changing_returns(success):
    batch, _ = prepared(fixture(success, include_verifier=True))
    meta = batch.non_tensor_batch
    lengths = batch.batch["response_mask"].sum(-1)
    rewards = torch.zeros_like(batch.batch["advantages"])
    rewards[torch.arange(len(batch)), lengths - 1] = torch.tensor(meta["pass"], dtype=rewards.dtype)
    batch.batch["token_level_rewards"] = rewards
    HELPERS["compute_advantage"](batch, HELPERS["AdvantageEstimator"].GRPO,
                                 norm_adv_by_std_in_grpo=True, group_by_agent_id=True)
    original_advantages = batch.batch["advantages"].clone()
    original_returns = batch.batch["returns"].clone()
    targeted = (meta["traj_uid"] == "t0") & (meta["agent_id"] == "Solver Agent")
    expected_sign = 1 if success else -1
    assert torch.all(torch.sign(original_advantages[targeted, 0]) == expected_sign)
    _, metrics = finalized(batch)
    assert metrics["pure_entropy/active_edges"] == 1
    torch.testing.assert_close(
        torch.tensor(meta["pure_entropy_base_advantage"], dtype=torch.float32),
        (original_advantages.float().sum(-1) / lengths),
    )
    HELPERS["apply_entropy_credit_to_advantages"](batch)
    factors = torch.tensor(meta["entropy_credit_final_multiplier"], dtype=original_advantages.dtype)
    torch.testing.assert_close(batch.batch["advantages"], original_advantages * factors[:, None])
    torch.testing.assert_close(batch.batch["returns"], original_returns)
    torch.testing.assert_close(batch.batch["token_level_rewards"], rewards)
