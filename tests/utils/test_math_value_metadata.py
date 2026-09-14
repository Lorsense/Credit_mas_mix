"""Exercise actual orchestra control flow with CPU-only agent/DataProto stubs."""
import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


class Batch:
    def __init__(self, n, metadata=None):
        self.n = n
        self.non_tensor_batch = metadata or {}

    def __len__(self):
        return self.n


def load_definitions(path, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + definitions, type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)


class Agent:
    def __init__(self, verifier=False):
        self.verifier = verifier
        self.contexts = []

    def reset(self):
        self.contexts.clear()

    def call(self, gen_batch, env_obs, team_context, actor_rollout_wg, agent_active_mask, step):
        self.contexts.append(list(team_context))
        text = (["<verify>approve</verify>", "<verify>reject</verify>"] if self.verifier
                else ["postprocessed identical solver answer"] * len(gen_batch))
        return Batch(len(gen_batch), {"agent_active_mask": agent_active_mask.copy()}), text

    def update_approved_vector(self, responses, approved, active):
        # This fixture approves only trajectory 0 and keeps trajectory 1 running.
        return np.logical_or(approved, np.array([True, False]))


def orchestra(enabled=True):
    namespace = {"np": np, "DataProto": Batch, "BaseAgent": object}
    load_definitions(ROOT / "agent_system/agent/orchestra/base.py", namespace)
    load_definitions(ROOT / "agent_system/agent/orchestra/math/math_orchestra.py", namespace)
    cls = namespace["MathMultiAgentOrchestra"]
    instance = cls.__new__(cls)
    instance.agents = {cls.SOLVER_AGENT: Agent(), cls.VERIFIER_AGENT: Agent(verifier=True)}
    instance.agents_to_wg_mapping = {role: role for role in instance.agents}
    instance.max_loop_num = 3
    instance.value_metadata_enabled = enabled
    instance.multiagent_batch_buffer = []
    instance._role_turn_counts = {}
    instance._value_action_counts = None
    instance.memory = None
    return instance


def run(instance, gen_batch):
    return instance.run(gen_batch, {"text": ["formatted q0", "formatted q1"]},
                        {role: None for role in instance.agents}, np.ones(2, dtype=bool), 1)


def questions(instance, gen_batch):
    instance.prepare_value_metadata(gen_batch, [
        {"question": "original q0", "ground_truth": "secret 0"},
        {"question": "original q1", "ground_truth": "secret 1"},
    ])


def test_three_solver_turns_early_stop_and_identical_actions_keep_logical_indices():
    instance = orchestra()
    gen_batch = Batch(2)
    questions(instance, gen_batch)
    actions, buffer = run(instance, gen_batch)
    assert len(buffer) == 5
    assert actions == ["postprocessed identical solver answer"] * 2
    expected = [[0, 0], [1, 1], [-1, 2], [-1, 3], [-1, 4]]
    for entry, indices in zip(buffer, expected):
        metadata = entry["batch"].non_tensor_batch
        assert metadata["value_action_index"].tolist() == indices
        assert metadata["value_question"].tolist() == ["original q0", "original q1"]
        assert "ground_truth" not in metadata
        assert metadata["value_max_solver_turns"].tolist() == [3, 3]
    assert buffer[2]["batch"].non_tensor_batch["value_action_text"].tolist() == [
        "", "postprocessed identical solver answer"]
    assert buffer[4]["batch"].non_tensor_batch["role_turn_index"].tolist() == [-1, 2]
    assert buffer[2]["batch"].non_tensor_batch["value_action_text"][1] == buffer[4]["batch"].non_tensor_batch["value_action_text"][1]


def test_disabled_mode_leaves_actions_context_and_metadata_unchanged():
    enabled, disabled = orchestra(), orchestra(False)
    enabled_batch, disabled_batch = Batch(2), Batch(2)
    questions(enabled, enabled_batch)
    # Disabled mode must not require environment kwargs or produce scorer fields.
    disabled.prepare_value_metadata(disabled_batch, None)
    enabled_actions, enabled_buffer = run(enabled, enabled_batch)
    disabled_actions, disabled_buffer = run(disabled, disabled_batch)
    assert enabled_actions == disabled_actions
    assert len(enabled_buffer) == len(disabled_buffer)
    for role in enabled.agents:
        assert enabled.agents[role].contexts == disabled.agents[role].contexts
    assert all(not any(k.startswith("value_") for k in entry["batch"].non_tensor_batch)
               for entry in disabled_buffer)


def test_only_trajectory_reset_resets_global_action_counter():
    instance = orchestra()
    gen_batch = Batch(2)
    questions(instance, gen_batch)
    instance._save_value_metadata(gen_batch, Batch(2), ["a", "b"], [True, True])
    instance.reset_buffer()
    second = Batch(2)
    instance._save_value_metadata(gen_batch, second, ["a", "b"], [True, True])
    assert second.non_tensor_batch["value_action_index"].tolist() == [1, 1]
    instance.reset()
    final = Batch(2)
    instance._save_value_metadata(gen_batch, final, ["a", "b"], [True, True])
    assert final.non_tensor_batch["value_action_index"].tolist() == [0, 0]


def test_missing_original_question_fails_without_using_ground_truth():
    instance = orchestra()
    with pytest.raises(ValueError, match="question"):
        instance.prepare_value_metadata(Batch(2), [{"ground_truth": "x"}] * 2)
    with pytest.raises(ValueError, match="Original questions"):
        instance._save_value_metadata(Batch(2), Batch(2), ["a", "b"], [True, True])
