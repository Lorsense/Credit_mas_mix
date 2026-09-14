"""Behavioral tests for virtual rewards, trajectory weights and padding safety."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "isolated_advantage_recovery", Path(__file__).parents[2] / "verl/utils/advantage_recovery.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
recover = _MODULE.recover_collapsed_advantages
unique_indices = _MODULE.logical_action_indices
validate = _MODULE.validate_advantage_recovery_config


def _group(*, success=True, count=8, turns=1, uid="p", role="Solver Agent"):
    turns = [turns] * count if isinstance(turns, int) else turns
    n = sum(turns)
    return {
        "uid": np.asarray([uid] * n, dtype=object),
        "traj_uid": np.asarray([f"{uid}_t{i}" for i, t in enumerate(turns) for _ in range(t)], dtype=object),
        "agent_id": np.asarray([role] * n, dtype=object),
        "role_turn_index": np.asarray([j for t in turns for j in range(t)], dtype=np.int64),
        "pass": np.full(n, success, dtype=bool),
        "is_action_valid": np.ones(n, dtype=bool),
    }


def _concat(*batches):
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def _take(batch, rows):
    return {key: value[rows].copy() for key, value in batch.items()}


def _run(meta, *, rewards=None, base=None, lengths=None, config=None):
    n = len(next(iter(meta.values())))
    cfg = {"enable": True, "normalization_epsilon": 0,
           "success_weight": 1, "failure_weight": 1,
           "success_mass_cap": 100, "failure_mass_cap": 100}
    cfg.update(config or {})
    return recover(meta, np.asarray(meta["pass"], dtype=float) if rewards is None else rewards,
                   np.zeros(n) if base is None else base,
                   np.full(n, 10) if lengths is None else lengths, cfg)


@pytest.mark.parametrize("success,expected", [(True, 1 / 3), (False, -1 / 3)])
def test_single_virtual_sample_eight_trajectories_sample_std(success, expected):
    fields, metrics = _run(_group(success=success))
    np.testing.assert_allclose(fields["advantages"], expected)
    np.testing.assert_allclose(fields["virtual_advantage"], expected)
    np.testing.assert_allclose(fields["policy_action_weight"], 1)
    assert set(fields["kind"]) == {"success" if success else "failure"}
    assert metrics["advantage_recovery/zero_variance_group_fraction"] == 1


@pytest.mark.parametrize("count", [2, 3, 8, 16])
def test_raw_virtual_advantage_depends_on_real_trajectory_count(count):
    fields, _ = _run(_group(count=count, turns=3, success=False))
    np.testing.assert_allclose(fields["advantages"], -1 / np.sqrt(count + 1))


def test_actual_format_reward_variance_preserves_original_grpo():
    meta = _group()
    meta["is_action_valid"][0] = False
    rewards = np.asarray([.9] + [1.] * 7)
    base = (rewards - rewards.mean()) / rewards.std(ddof=1)
    fields, metrics = _run(meta, rewards=rewards, base=base)
    np.testing.assert_array_equal(fields["advantages"], base)
    np.testing.assert_array_equal(fields["policy_action_weight"], np.ones(8))
    assert set(fields["kind"]) == {"mixed"}
    assert metrics["advantage_recovery/all_success_group_fraction"] == 1
    assert metrics["advantage_recovery/zero_variance_group_fraction"] == 0


def test_same_reward_with_mixed_terminal_outcomes_is_not_fabricated():
    meta = _group()
    meta["pass"][::2] = False
    fields, _ = _run(meta, rewards=np.zeros(8))
    assert not fields["advantages"].any()
    assert set(fields["kind"]) == {"unrecovered"}


def test_nonzero_base_advantage_is_not_overwritten():
    fields, _ = _run(_group(), base=np.full(8, .1))
    np.testing.assert_array_equal(fields["advantages"], np.full(8, .1))
    assert set(fields["kind"]) == {"unrecovered"}


def test_recovery_is_per_question_and_role():
    solver = _group(turns=2)
    verifier = _group(role="Verifier Agent")
    other_question = _group(uid="other", success=False)
    meta = _concat(solver, verifier, other_question)
    fields, metrics = _run(meta)
    np.testing.assert_allclose(fields["advantages"][:24], 1 / 3)
    np.testing.assert_allclose(fields["advantages"][24:], -1 / 3)
    assert metrics["advantage_recovery/groups"] == 3


def test_variable_turns_and_lengths_have_equal_trajectory_token_mass():
    meta = _group(count=3, turns=[1, 2, 3])
    lengths = np.asarray([5, 10, 15, 20, 25, 30])
    fields, _ = _run(meta, lengths=lengths)
    weighted_tokens = fields["policy_action_weight"] * lengths
    for trajectory in set(meta["traj_uid"]):
        assert weighted_tokens[meta["traj_uid"] == trajectory].sum() == pytest.approx(lengths.sum() / 3)
    assert weighted_tokens.sum() == pytest.approx(lengths.sum())
    np.testing.assert_allclose(fields["advantages"], .5)


def test_padding_does_not_change_signal_mass_or_counts():
    meta = _group(count=3, turns=[1, 2, 3])
    lengths = np.arange(6) + 1
    baseline, baseline_metrics = _run(meta, lengths=lengths)
    indices = [0, 1, 2, 3, 4, 5, 0, 0, 2, 5]
    padded = _take(meta, indices)
    fields, metrics = _run(padded, lengths=lengths[indices])
    np.testing.assert_array_equal(fields["advantages"], baseline["advantages"][indices])
    np.testing.assert_array_equal(fields["policy_action_weight"][:6], baseline["policy_action_weight"])
    assert not fields["policy_action_weight"][6:].any()
    assert metrics["advantage_recovery/groups"] == baseline_metrics["advantage_recovery/groups"]
    assert metrics["advantage_recovery/padding_actions"] == 4
    unique, inverse = unique_indices(padded)
    np.testing.assert_array_equal(unique, np.arange(6))
    np.testing.assert_array_equal(inverse, indices)


def test_duplicate_order_is_arbitrary_and_inverse_is_an_ordinal():
    meta = _group(count=3)
    indices = [2, 2, 0, 1, 0]
    unique, inverse = unique_indices(_take(meta, indices))
    np.testing.assert_array_equal(unique, [0, 2, 3])
    np.testing.assert_array_equal(inverse, [0, 0, 1, 2, 1])


@pytest.mark.parametrize("key,value", [
    ("pass", False), ("is_action_valid", False),
    ("top16_entropy_mean", .3), ("value_action_text", "different"),
])
def test_conflicting_padding_metadata_raises(key, value):
    meta = _group(count=2)
    meta["top16_entropy_mean"] = np.asarray([.2, .2])
    meta["value_action_text"] = np.asarray(["same", "same"], dtype=object)
    padded = _take(meta, [0, 1, 0])
    padded[key][-1] = value
    with pytest.raises(ValueError, match="conflicting|consistent"):
        _run(padded)


def test_nested_entropy_metadata_conflicts_are_checked():
    meta = _group(count=2)
    meta["top16_entropy"] = np.asarray([{"coverage": 1., "mean": .2}] * 2, dtype=object)
    padded = _take(meta, [0, 1, 0])
    padded["top16_entropy"][-1] = {"coverage": .1, "mean": .2}
    with pytest.raises(ValueError, match="top16_entropy"):
        _run(padded)


@pytest.mark.parametrize("which", ["rewards", "base", "lengths"])
def test_conflicting_padding_numeric_training_fields_raise(which):
    meta = _take(_group(count=2), [0, 1, 0])
    kwargs = {"rewards": np.ones(3), "base": np.zeros(3), "lengths": np.full(3, 10)}
    kwargs[which][-1] += 1
    with pytest.raises(ValueError, match="conflicting"):
        _run(meta, **kwargs)


def test_derived_control_fields_may_mask_duplicate_rows():
    meta = _take(_group(count=2), [0, 1, 0])
    meta["entropy_control_brake"] = np.asarray([.2, .2, 0.])
    fields, _ = _run(meta)
    assert fields["policy_action_weight"][-1] == 0


def test_single_trajectory_is_not_recovered_even_with_many_turns():
    fields, _ = _run(_group(count=1, turns=8))
    assert not fields["advantages"].any()
    assert set(fields["kind"]) == {"unrecovered"}


def test_no_response_trajectory_does_not_inflate_group_size():
    fields, _ = _run(_group(count=3), lengths=[10, 10, 0])
    np.testing.assert_allclose(fields["advantages"][:2], 1 / np.sqrt(3))
    assert fields["advantages"][-1] == 0
    assert fields["policy_action_weight"][-1] == 0


def test_no_response_actions_are_neutral():
    fields, metrics = _run(_group(count=2), lengths=[0, 0])
    assert not fields["advantages"].any()
    assert not fields["policy_action_weight"].any()
    assert metrics["advantage_recovery/zero_variance_group_fraction"] == 0


@pytest.mark.parametrize("success,reward", [(True, .4), (False, 1.1)])
def test_wrong_side_virtual_reward_does_not_reverse_success_or_failure(success, reward):
    fields, _ = _run(_group(success=success), rewards=np.full(8, reward))
    assert not fields["advantages"].any()
    assert set(fields["kind"]) == {"unrecovered"}


def test_constant_format_penalty_still_recovers_in_correct_direction():
    fields, _ = _run(_group(success=False), rewards=np.full(8, -.1))
    np.testing.assert_allclose(fields["advantages"], -1 / 3)


def test_weights_apply_once_and_not_to_actor_weights():
    fields, _ = _run(_group(), config={"success_weight": .2})
    np.testing.assert_allclose(fields["advantages"], .2 / 3)
    np.testing.assert_allclose(fields["virtual_advantage"], 1 / 3)
    np.testing.assert_allclose(fields["recovery_weight"], .2)
    np.testing.assert_allclose(fields["policy_action_weight"], 1)


@pytest.mark.parametrize("success,cap", [(True, .05), (False, .03)])
def test_role_mass_cap_reserves_pure_factor_bound_without_mixed_groups(success, cap):
    meta = _concat(*[_group(uid=f"p{i}", success=success, turns=[1, 2, 3, 1, 2, 3, 2, 1]) for i in range(12)])
    lengths = np.arange(len(meta["uid"])) % 13 + 1
    branch = "success" if success else "failure"
    fields, metrics = _run(meta, lengths=lengths, config={f"{branch}_mass_cap": cap, "pure_factor_bound": 1.2})
    actual_bound = np.sum(np.abs(fields["advantages"]) * fields["policy_action_weight"] * lengths * 1.2) / lengths.sum()
    assert actual_bound == pytest.approx(cap)
    assert metrics[f"advantage_recovery/solver_agent/{branch}_budget_scale"] < 1
    assert metrics[f"advantage_recovery/solver_agent/{branch}_post_pure_mass_bound"] == pytest.approx(cap)


def test_success_and_failure_caps_are_independent_for_each_role():
    meta = _concat(_group(uid="s"), _group(uid="f", success=False),
                   _group(uid="v", role="Verifier Agent"))
    fields, metrics = _run(meta, config={"success_mass_cap": .05, "failure_mass_cap": .01})
    assert metrics["advantage_recovery/solver_agent/success_post_pure_mass_bound"] == pytest.approx(.05)
    assert metrics["advantage_recovery/solver_agent/failure_post_pure_mass_bound"] == pytest.approx(.01)
    assert metrics["advantage_recovery/verifier_agent/success_post_pure_mass_bound"] == pytest.approx(.05)
    np.testing.assert_allclose(fields["advantages"][:8], .05 * 2 / 1.2)
    np.testing.assert_allclose(fields["advantages"][8:16], -.01 * 2 / 1.2)


def test_more_success_groups_do_not_exceed_mass_cap():
    for count in (1, 2, 25):
        meta = _concat(*[_group(uid=f"p{i}") for i in range(count)])
        fields, _ = _run(meta, config={"success_mass_cap": .05})
        assert np.mean(fields["advantages"] * fields["policy_action_weight"] * 1.2) == pytest.approx(.05)


def test_metrics_preserve_raw_collapse_after_successful_recovery():
    _, metrics = _run(_group())
    assert metrics["advantage_recovery/zero_variance_group_fraction"] == 1
    assert metrics["advantage_recovery/solver_agent/raw_zero_advantage_token_fraction"] == 1
    assert metrics["advantage_recovery/solver_agent/remaining_zero_advantage_token_fraction"] == 0


def test_inputs_are_not_modified():
    meta = _group()
    reward = np.ones(8)
    base = np.zeros(8)
    lengths = np.full(8, 10)
    original = {key: value.copy() for key, value in meta.items()}
    _run(meta, rewards=reward, base=base, lengths=lengths)
    np.testing.assert_array_equal(reward, np.ones(8))
    np.testing.assert_array_equal(base, np.zeros(8))
    np.testing.assert_array_equal(lengths, np.full(8, 10))
    for key, values in original.items():
        np.testing.assert_array_equal(meta[key], values)


def test_disabled_path_needs_no_metadata_and_preserves_advantages_dtype_and_padding():
    base = np.asarray([-.4, .2, 0], dtype=np.float32)
    fields, metrics = recover({}, None, base, None, {"enable": False})
    np.testing.assert_array_equal(fields["advantages"], base)
    assert fields["advantages"].dtype == base.dtype
    np.testing.assert_array_equal(fields["policy_action_weight"], np.ones(3))
    assert metrics == {"advantage_recovery/enabled": 0.0}


@pytest.mark.parametrize("key", ["uid", "traj_uid", "agent_id", "role_turn_index", "pass", "is_action_valid"])
def test_missing_required_metadata_fails_fast(key):
    meta = _group()
    del meta[key]
    with pytest.raises(KeyError, match="missing"):
        _run(meta, rewards=np.ones(8))


@pytest.mark.parametrize("key,value", [("uid", ""), ("traj_uid", None), ("agent_id", np.inf),
                                       ("role_turn_index", -1), ("pass", 2), ("is_action_valid", "yes")])
def test_invalid_metadata_fails_fast(key, value):
    meta = _group()
    meta[key] = meta[key].astype(object)
    meta[key][0] = value
    with pytest.raises(ValueError):
        _run(meta, rewards=np.ones(8))


@pytest.mark.parametrize("which,value", [("rewards", np.nan), ("base", np.inf),
                                         ("lengths", np.nan), ("lengths", -.1), ("lengths", 1.5)])
def test_nonfinite_or_invalid_numeric_inputs_fail_fast(which, value):
    data = np.zeros(8)
    data[0] = value
    with pytest.raises(ValueError):
        _run(_group(), **{which: data})


@pytest.mark.parametrize("key,value", [("success_weight", -1), ("failure_mass_cap", -.1),
                                       ("success_virtual_reward", np.nan), ("pure_factor_bound", .9),
                                       ("enable", "false"), ("normalization_epsilon", True)])
def test_invalid_configuration_fails_fast(key, value):
    with pytest.raises(ValueError):
        validate({key: value})


def test_empty_enabled_batch_is_well_defined():
    meta = {key: values[:0] for key, values in _group().items()}
    fields, metrics = _run(meta)
    assert fields["advantages"].shape == (0,)
    assert metrics["advantage_recovery/unique_actions"] == 0
