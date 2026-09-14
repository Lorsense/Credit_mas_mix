"""Execute actual trainer functions with real Torch GRPO, without Ray startup."""

import ast
from collections import defaultdict
from copy import deepcopy
from enum import Enum
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[3]
TRAINER_PATH = ROOT / "verl/trainer/ppo/ray_trainer.py"
TRAINER_AST = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECOVERY = load("integration_recovery", "verl/utils/advantage_recovery.py")
CURRICULUM = load("integration_curriculum", "verl/utils/question_curriculum.py")


def execute_nodes(nodes, namespace, path=TRAINER_PATH):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *deepcopy(nodes)], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def helpers():
    core_path = ROOT / "verl/trainer/ppo/core_algos.py"
    grpo = next(n for n in ast.parse(core_path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "compute_grpo_outcome_advantage")
    grpo.decorator_list = []
    core = execute_nodes([grpo], {"torch": torch, "np": np, "defaultdict": defaultdict}, core_path)
    names = {"compute_response_mask", "compute_advantage", "compute_advantage_with_recovery",
             "_select_curriculum_batch", "_observe_curriculum_batch", "_prepare_value_credit",
             "value_scorer_metrics"}
    selected = [n for n in TRAINER_AST.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in TRAINER_AST.body:
        if isinstance(node, ast.ClassDef):
            selected.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    assert {n.name for n in selected} == names
    selected.insert(0, next(n for n in TRAINER_AST.body if isinstance(n, ast.ClassDef) and n.name == "AdvantageEstimator"))
    return execute_nodes(selected, {
        "np": np, "torch": torch, "Enum": Enum, "defaultdict": defaultdict,
        "core_algos": SimpleNamespace(compute_grpo_outcome_advantage=core[grpo.name]),
        "logical_action_indices": RECOVERY.logical_action_indices,
        "recover_collapsed_advantages": RECOVERY.recover_collapsed_advantages,
    })


class Batch:
    def __init__(self, tensors, metadata):
        self.batch, self.non_tensor_batch = tensors, metadata

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def select_idxs(self, indices):
        return Batch({k: v[indices].clone() for k, v in self.batch.items()},
                     {k: np.asarray(v)[indices].copy() for k, v in self.non_tensor_batch.items()})


def fixture(labels=(1, 1, 1, 1), turns=2):
    count = len(labels) * turns
    mask = torch.tensor([[1, 1, 0, 0] if i % 2 else [1, 1, 1, 0] for i in range(count)])
    rewards = torch.zeros(count, 4)
    rewards[:, 0] = torch.tensor([float(label) for label in labels for _ in range(turns)])
    return Batch({"token_level_rewards": rewards, "response_mask": mask}, {
        "uid": np.array(["question"] * count, dtype=object),
        "traj_uid": np.array([f"t{i}" for i in range(len(labels)) for _ in range(turns)], dtype=object),
        "agent_id": np.array(["Solver Agent"] * count, dtype=object),
        "role_turn_index": np.array(list(range(turns)) * len(labels)),
        "pass": np.array([label for label in labels for _ in range(turns)]),
        "is_action_valid": np.ones(count, dtype=bool),
    })


def args(h):
    return {"adv_estimator": h["AdvantageEstimator"].GRPO, "group_by_agent_id": True,
            "norm_adv_by_std_in_grpo": True, "multi_turn": False}


def recovery_cfg():
    return {"enable": True, "success_weight": 0.2, "failure_weight": 0.1,
            "success_mass_cap": 100, "failure_mass_cap": 100}


def test_mixed_groups_match_real_original_grpo_exactly():
    h = helpers()
    batch = fixture((0, 1, 0, 1))
    baseline = h["compute_advantage"](deepcopy(batch), **args(h))
    recovered, metrics = h["compute_advantage_with_recovery"](batch, recovery_cfg(), **args(h))
    torch.testing.assert_close(recovered.batch["advantages"], baseline.batch["advantages"], rtol=0, atol=0)
    torch.testing.assert_close(recovered.batch["returns"], baseline.batch["returns"], rtol=0, atol=0)
    assert set(recovered.non_tensor_batch["recovery_kind"]) == {"mixed"}
    assert metrics["advantage_recovery/zero_variance_group_fraction"] == 0


@pytest.mark.parametrize("labels", [(0, 1, 0, 1), (1, 1, 1, 1)])
def test_multi_turn_loss_mask_is_preserved_for_mixed_and_restored_groups(labels):
    h = helpers()
    batch = fixture(labels)
    batch.batch["responses"] = torch.ones(8, 4, dtype=torch.long)
    mask = batch.batch["response_mask"].clone()
    mask[:, 0] = 0  # response token is outside the Actor's actual loss mask
    batch.batch["loss_mask"] = torch.cat([torch.ones(8, 2), mask], dim=1)
    settings = {**args(h), "multi_turn": True}
    baseline = h["compute_advantage"](deepcopy(batch), **settings)
    result, _ = h["compute_advantage_with_recovery"](batch, recovery_cfg(), **settings)
    assert not result.batch["advantages"][:, 0].any()
    if len(set(labels)) > 1:
        torch.testing.assert_close(result.batch["advantages"], baseline.batch["advantages"], rtol=0, atol=0)
    else:
        assert torch.all(result.batch["advantages"][mask.bool()] > 0)


@pytest.mark.parametrize("label,sign", [(1, 1), (0, -1)])
def test_zero_groups_get_signed_advantages_without_mutating_rewards_or_returns(label, sign):
    h = helpers()
    batch = fixture((label,) * 4)
    baseline = h["compute_advantage"](deepcopy(batch), **args(h))
    rewards = batch.batch["token_level_rewards"].clone()
    recovered, _ = h["compute_advantage_with_recovery"](batch, recovery_cfg(), **args(h))
    assert len(recovered) == 8  # no virtual row or token is ever added
    assert torch.all(recovered.batch["advantages"][recovered.batch["response_mask"].bool()] * sign > 0)
    assert not recovered.batch["advantages"][~recovered.batch["response_mask"].bool()].any()
    torch.testing.assert_close(recovered.batch["returns"], baseline.batch["returns"], rtol=0, atol=0)
    torch.testing.assert_close(recovered.batch["token_level_rewards"], rewards, rtol=0, atol=0)


@pytest.mark.parametrize("label,reward,kind", [(1, .9, "success"), (0, .1, "failure")])
def test_constant_float32_shaped_rewards_do_not_create_spurious_negative_base_advantage(label, reward, kind):
    h = helpers()
    batch = fixture((label,) * 8, turns=1)
    batch.batch["token_level_rewards"][:, 0] = reward
    rewards = batch.batch["token_level_rewards"].clone()
    original = h["compute_advantage"](deepcopy(batch), **args(h))
    result, _ = h["compute_advantage_with_recovery"](batch, recovery_cfg(), **args(h))
    assert set(result.non_tensor_batch["recovery_kind"]) == {kind}
    assert not result.non_tensor_batch["recovery_base_advantage"].any()
    # Correct the spurious policy advantage without changing the original
    # returns tensor or substituting virtual rewards into another learner.
    torch.testing.assert_close(result.batch["returns"], original.batch["returns"], rtol=0, atol=0)
    torch.testing.assert_close(result.batch["token_level_rewards"], rewards, rtol=0, atol=0)


@pytest.mark.parametrize("labels", [(0, 1, 0, 1), (1, 1, 1, 1), (0, 0, 0, 0)])
def test_padding_preserves_logical_advantages_and_returns_and_masks_duplicate_policy_rows(labels):
    h = helpers()
    baseline = fixture(labels)
    order = np.array([7, 0, 1, 2, 3, 4, 5, 6, 0, 3, 7])
    padded = baseline.select_idxs(order)
    baseline, _ = h["compute_advantage_with_recovery"](baseline, recovery_cfg(), **args(h))
    padded, _ = h["compute_advantage_with_recovery"](padded, recovery_cfg(), **args(h))
    for key in ("advantages", "returns"):
        torch.testing.assert_close(padded.batch[key], baseline.batch[key][order], rtol=0, atol=0)
    assert not padded.batch["policy_action_weight"][8:].any()


def test_trainer_executes_recovery_before_pure_finalize_and_credit_application():
    h = helpers()
    fit = next(n for n in ast.walk(TRAINER_AST) if isinstance(n, ast.FunctionDef) and n.name == "fit")
    adv_with = next(n for n in ast.walk(fit) if isinstance(n, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and item.context_expr.args
        and isinstance(item.context_expr.args[0], ast.Constant) and item.context_expr.args[0].value == "adv"
        for item in n.items))
    start = next(i for i, node in enumerate(adv_with.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "norm_adv_by_std_in_grpo" for t in node.targets))
    events = []

    def finalize(batch, cfg):
        assert batch.batch["advantages"].sum() > 0
        assert set(batch.non_tensor_batch["recovery_kind"]) == {"success"}
        events.append("pure_finalize_after_recovery")
        return batch, {}

    def apply(batch):
        events.append("pure_apply")
        batch.batch["advantages"] *= 1.1
        return batch

    class Config(dict):
        __getattr__ = dict.__getitem__

    algorithm = Config(**args(h), advantage_recovery=recovery_cfg(), gamma=1, lam=1,
                       use_pf_ppo=False, pf_ppo=Config(reweight_method="pow", weight_pow=2),
                       gigpo=Config(step_advantage_w=1, mode="mean_std_norm", enable_similarity=False, similarity_thresh=.95))
    namespace = dict(h, self=SimpleNamespace(advantage_recovery_enabled=True,
                     config=Config(algorithm=algorithm, actor_rollout_ref=Config(rollout=Config(n=4, multi_turn=Config(enable=False))))),
                     batch=fixture(), metrics={}, entropy_credit_enabled=True, sparse_entropy_enabled=True,
                     sparse_entropy_config={}, finalize_sparse_entropy_credit_for_advantages=finalize,
                     apply_entropy_credit_to_advantages=apply)
    execute_nodes(adv_with.body[start:], namespace)
    assert events == ["pure_finalize_after_recovery", "pure_apply"]


def test_selection_refetches_dataset_rows_and_keeps_source_metadata():
    h = helpers()
    source = [{"prompt": f"question {i}", "curriculum_dataset_index": i} for i in range(30)]
    calls = []

    def collate(items):
        calls.append(items)
        return {key: np.asarray([item[key] for item in items], dtype=object) for key in items[0]}

    self = SimpleNamespace(global_steps=7, curriculum_dataset=source, train_collate_fn=collate,
                           question_curriculum=SimpleNamespace(select=lambda indices, step:
                               ([0, 21, 2], ["fresh", "hard", "fresh_adaptive"], {"selected": 1})))
    initial = {"curriculum_dataset_index": np.array([0, 1, 2]), "prompt": np.array(["old0", "old1", "old2"])}
    result, metrics = h["_select_curriculum_batch"](self, initial)
    assert result["prompt"].tolist() == ["question 0", "question 21", "question 2"]
    assert result["curriculum_source"].tolist() == ["fresh", "hard", "fresh_adaptive"]
    assert len(calls) == 1 and metrics == {"selected": 1}


def test_curriculum_observation_counts_each_trajectory_once_across_roles_turns_and_padding():
    h = helpers()
    observed = []
    self = SimpleNamespace(global_steps=11, question_curriculum=SimpleNamespace(
        observe=lambda outcomes, step: observed.append((outcomes, step)) or {}))
    batch = fixture((0, 1), turns=3)
    batch.non_tensor_batch["curriculum_dataset_index"] = np.zeros(6, dtype=int)
    batch = batch.select_idxs(np.array([0, 1, 2, 3, 4, 5, 0, 5]))
    h["_observe_curriculum_batch"](self, batch)
    assert observed == [({0: [0, 1]}, 11)]
    batch.non_tensor_batch["pass"][-1] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        h["_observe_curriculum_batch"](self, batch)


def test_curriculum_rejects_one_trajectory_mapped_to_different_dataset_questions():
    h = helpers()
    observed = []
    self = SimpleNamespace(global_steps=11, question_curriculum=SimpleNamespace(
        observe=lambda outcomes, step: observed.append(outcomes) or {}))
    batch = fixture((1,), turns=2)
    batch.non_tensor_batch["curriculum_dataset_index"] = np.array([0, 1])
    with pytest.raises(ValueError, match="inconsistent dataset"):
        h["_observe_curriculum_batch"](self, batch)
    assert not observed


def test_value_scoring_keeps_all_records_but_only_random_fresh_slice_can_train():
    h = helpers()
    batch = fixture((0, 1), turns=2)
    batch.non_tensor_batch["curriculum_source"] = np.array(["hard", "hard", "fresh", "fresh"])
    records = [{"traj_uid": "t0"}, {"traj_uid": "t1"}]
    received = []

    def prepare(rows):
        received.extend(deepcopy(rows))
        return {"ready": True, "values": {r["traj_uid"]: [0.3] for r in rows},
                "metrics": {"version": 5}, "reliability": {"solver": .8}}

    h["_prepare_value_credit"].__globals__.update(
        build_trajectory_records=lambda metadata, turns: (deepcopy(records), {}),
        ray=SimpleNamespace(get=lambda result: result),
        attach_value_predictions=lambda metadata, values, ready: {"score_count": len(values)})
    self = SimpleNamespace(question_curriculum=object(), config=SimpleNamespace(agent=SimpleNamespace(
        orchestra=SimpleNamespace(math=SimpleNamespace(max_loop_num=3)))),
        value_scorer=SimpleNamespace(prepare=SimpleNamespace(remote=prepare)))
    metrics = h["_prepare_value_credit"](self, batch)
    assert received == [{"traj_uid": "t0", "train_eligible": False}, {"traj_uid": "t1", "train_eligible": True}]
    assert metrics["score_count"] == 2
    assert batch.non_tensor_batch["value_credit_scorer_version"].tolist() == [5] * 4
    batch.non_tensor_batch["curriculum_source"][1] = "fresh"
    with pytest.raises(ValueError, match="inconsistent curriculum source"):
        h["_prepare_value_credit"](self, batch)


def test_rollout_copies_source_and_row_index_to_actions():
    path = ROOT / "agent_system/multi_turn_rollout/rollout_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    copy_loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)
                     and [e.value for e in n.iter.elts if isinstance(e, ast.Constant)] ==
                     ["curriculum_dataset_index", "curriculum_source"])
    source = {"curriculum_dataset_index": np.array([4, 4, 9, 9]),
              "curriculum_source": np.array(["fresh", "fresh", "hard", "hard"], dtype=object)}
    agent = SimpleNamespace(non_tensor_batch={})
    execute_nodes([copy_loop], {"gen_batch": SimpleNamespace(non_tensor_batch=source), "agent_batch": agent}, path)
    for key in source:
        np.testing.assert_array_equal(agent.non_tensor_batch[key], source[key])
        assert agent.non_tensor_batch[key] is not source[key]
