"""Behavioral tests for entropy-only gates and bounded coefficient transfer."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "isolated_sparse_entropy", Path(__file__).parents[2] / "verl" / "utils" / "sparse_entropy_credit.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
prepare = _MODULE.prepare_sparse_entropy_credit
finalize = _MODULE.finalize_sparse_entropy_credit
validate = _MODULE.validate_sparse_entropy_config


def _pairs(previous=None, current=None, *, success=True, role="Solver Agent", prompt="p"):
    previous = [.9, .2, .3, .4, .5, .6, .7, .8] if previous is None else previous
    current = [.1, .21, .31, .41, .51, .61, .71, .81] if current is None else current
    n = len(previous)
    return {
        "uid": np.asarray([prompt] * (2 * n), dtype=object),
        "traj_uid": np.asarray([f"{prompt}_t{i}" for i in range(n) for _ in range(2)], dtype=object),
        "agent_id": np.asarray([role] * (2 * n), dtype=object),
        "role_turn_index": np.asarray([0, 1] * n),
        "pass": np.full(2 * n, success, dtype=bool),
        "top16_entropy_mean": np.asarray(list(zip(previous, current))).reshape(-1),
        "top16_entropy": np.asarray([{"coverage": 1.0}] * (2 * n), dtype=object),
        "is_action_valid": np.ones(2 * n, dtype=bool),
        "pure_entropy_response_tokens": np.full(2 * n, 100, dtype=np.int64),
        "pure_entropy_truncated": np.zeros(2 * n, dtype=bool),
        "entropy_credit_action_multiplier": np.ones(2 * n, dtype=np.float64),
    }


def _prepared(meta, config=None):
    fields, metrics = prepare(meta, {} if config is None else config)
    meta.update(fields)
    return meta, metrics


def _concat(*batches):
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def _take(batch, rows):
    return {key: value[rows].copy() for key, value in batch.items()}


def test_joint_gate_uses_midrank_and_absolute_residual_without_forced_quota():
    meta, metrics = _prepared(_pairs())
    assert meta["pure_entropy_correction_score"][1] == (.9375 ** 2)
    expected_gate = (.9375 ** 2 - .8) / .2
    assert meta["pure_entropy_correction_gate"][1] == pytest.approx(expected_gate)
    assert np.count_nonzero(meta["pure_entropy_correction_gate"]) == 1
    assert not meta["pure_entropy_regression_gate"].any()
    assert meta["pure_entropy_group_size"][1] == 8
    assert meta["pure_entropy_residual"][1] < -meta["pure_entropy_residual_threshold"][1]
    assert metrics["pure_entropy/eligible_pairs"] == 8


def test_regression_midrank_threshold_is_impossible_with_six_samples():
    previous = [.1, .2, .3, .4, .5, .6]
    current = [.95, .19, .29, .39, .49, .59]
    meta, _ = _prepared(_pairs(previous, current, success=False))
    assert meta["pure_entropy_pair_reliable"][1]
    assert meta["pure_entropy_regression_score"][1] == pytest.approx((5.5 / 6) ** 2)
    assert not meta["pure_entropy_regression_gate"].any()


def test_local_subtraction_preserves_ranks_but_residual_threshold_makes_drift_operational():
    small = _pairs(
        [.9, .2, .3, .4, .5, .6, .7, .8],
        [.899, .201, .301, .401, .501, .601, .701, .801], prompt="small")
    other = _pairs(
        [.05, .1, .15, .2, .25, .3, .35, .4],
        [.15, .2, .25, .3, .35, .4, .45, .5], prompt="other")
    local, _ = _prepared(_take(small, list(range(16))), {"min_residual": .005})
    pooled, _ = _prepared(_concat(small, other), {"min_residual": .005})
    np.testing.assert_allclose(
        local["pure_entropy_correction_score"], pooled["pure_entropy_correction_score"][:16],
        equal_nan=True)
    assert local["pure_entropy_correction_gate"][1] == 0
    assert pooled["pure_entropy_correction_gate"][1] > 0
    assert abs(pooled["pure_entropy_residual"][1]) > abs(local["pure_entropy_residual"][1])


def test_raw_direction_guard_does_not_call_relative_slow_increase_a_repair():
    meta = _pairs(
        [.8, .1, .2, .3, .4, .5, .6, .7],
        [.81, .13, .23, .33, .43, .53, .63, .73])
    guarded, _ = _prepared(_take(meta, list(range(16))))
    unguarded, _ = _prepared(_take(meta, list(range(16))), {"require_raw_direction": False})
    assert unguarded["pure_entropy_correction_gate"][1] > 0
    assert guarded["pure_entropy_delta"][1] > 0
    assert guarded["pure_entropy_correction_gate"][1] == 0


@pytest.mark.parametrize("case", ["small", "ties", "tiny_delta", "tiny_entropy"])
def test_small_groups_ties_and_numerical_spreads_are_neutral(case):
    config = {"correction_threshold": 0, "regression_threshold": 0, "min_residual": 0}
    if case == "small":
        meta = _pairs([.9, .2, .3, .4], [.1, .21, .31, .41])
    elif case == "ties":
        meta = _pairs([.5] * 8, [.4] * 8)
    elif case == "tiny_delta":
        previous = np.linspace(.2, .9, 8)
        meta = _pairs(previous, previous + np.linspace(-1e-8, 1e-8, 8))
    else:
        previous = .5 + np.arange(8) * 1e-8
        meta = _pairs(previous, previous + np.linspace(-.1, .1, 8))
        config["roles"] = ["Solver Agent"]
        # The correction branch specifically needs spread in H_previous.
    meta, _ = _prepared(meta, config)
    assert not meta["pure_entropy_correction_gate"].any()
    if case != "tiny_entropy":
        assert not meta["pure_entropy_regression_gate"].any()


@pytest.mark.parametrize("quality", ["invalid", "missing", "coverage", "truncated", "zero_tokens"])
def test_bad_endpoint_disables_pair_without_jump_over_it(quality):
    meta = _pairs()
    if quality == "invalid":
        meta["is_action_valid"][0] = False
    elif quality == "missing":
        meta["top16_entropy_mean"][0] = np.nan
    elif quality == "coverage":
        meta["top16_entropy"][0] = {"coverage": .89}
    elif quality == "truncated":
        meta["pure_entropy_truncated"][0] = True
    else:
        meta["pure_entropy_response_tokens"][0] = 0
    prepared, _ = _prepared(meta)
    assert prepared["pure_entropy_pair_observed"][1]
    assert not prepared["pure_entropy_pair_eligible"][1]
    assert prepared["pure_entropy_correction_gate"][1] == 0


def test_missing_role_turn_is_not_connected_to_an_earlier_action():
    meta = _pairs()
    meta["role_turn_index"][1] = 2
    meta, _ = _prepared(meta)
    assert not meta["pure_entropy_pair_observed"][1]
    assert meta["pure_entropy_prev_turn"][1] == -1
    assert meta["pure_entropy_correction_gate"][1] == 0


def test_verifier_requires_explicit_opt_in_and_first_turn_has_no_pair():
    verifier = _pairs(role="Verifier Agent")
    default, _ = _prepared(_take(verifier, list(range(16))))
    opted, _ = _prepared(_take(verifier, list(range(16))), {"roles": ["Verifier Agent"]})
    assert not default["pure_entropy_pair_observed"].any()
    assert opted["pure_entropy_correction_gate"][1] > 0
    firsts, _ = _prepared(_take(verifier, list(range(0, 16, 2))), {"roles": ["Verifier Agent"]})
    assert not firsts["pure_entropy_pair_observed"].any()


def test_success_repairs_and_failure_regressions_transfer_to_current_action():
    success, _ = _prepared(_pairs())
    positive = finalize(success, np.ones(16), {})
    assert positive["final_multipliers"][0] < 1 < positive["final_multipliers"][1]
    failure, _ = _prepared(_pairs(
        [.1, .2, .3, .4, .5, .6, .7, .8],
        [.95, .19, .29, .39, .49, .59, .69, .79], success=False))
    negative = finalize(failure, -np.ones(16), {})
    assert negative["final_multipliers"][0] < 1 < negative["final_multipliers"][1]
    # Both are positive multipliers, so the failure update becomes less
    # negative for the previous action and more negative for the current one.
    assert (-negative["final_multipliers"])[0] > -1 > (-negative["final_multipliers"])[1]
    np.testing.assert_array_equal(positive["final_multipliers"][2:], np.ones(14))
    assert positive["metrics"]["pure_entropy/active_edges"] == 1


@pytest.mark.parametrize("advantages", [
    [-1, -1], [1, -1], [-1, 1], [0, 1], [1, 0], [0, 0],
])
def test_success_branch_requires_both_actual_advantages_positive(advantages):
    meta, _ = _prepared(_pairs())
    actual = np.ones(16)
    actual[:2] = advantages
    result = finalize(meta, actual, {})
    np.testing.assert_array_equal(result["final_multipliers"], np.ones(16))
    assert result["metrics"]["pure_entropy/sign_mismatch_edges"] == 1


def test_terminal_failure_does_not_use_a_repair_gate_even_with_positive_advantages():
    meta, _ = _prepared(_pairs(success=False))
    result = finalize(meta, np.ones(16), {})
    np.testing.assert_array_equal(result["final_multipliers"], np.ones(16))
    assert result["metrics"]["pure_entropy/candidate_edges"] == 0


def _long_chain():
    batches = []
    for turn, (previous, current) in enumerate([
        ([.95, .1, .2, .3, .4, .5, .6, .7], [.8, .11, .21, .31, .41, .51, .61, .71]),
        ([.8, .11, .21, .31, .41, .51, .61, .71], [.1, .12, .22, .32, .42, .52, .62, .72]),
    ]):
        batch = _pairs(previous, current)
        batch["role_turn_index"] += turn
        if turn:
            batch = _take(batch, list(range(1, 16, 2)))
        batches.append(batch)
    return _concat(*batches)


def test_long_chain_accumulates_incoming_minus_outgoing_and_does_not_explode():
    meta, _ = _prepared(_long_chain())
    result = finalize(meta, np.ones(24), {})
    # t0's three actions are at rows 0, 1, 16. Same gate strength gives
    # cancellation on the middle action and a single transfer at each end.
    u = result["metadata"]["pure_entropy_u"][[0, 1, 16]]
    amount = result["metadata"]["pure_entropy_edge_transfer"][1]
    np.testing.assert_allclose(u, [-amount, 0, amount], atol=1e-15)
    assert result["metrics"]["pure_entropy/active_edges"] == 2
    assert result["metrics"]["pure_entropy/active_components"] == 1
    assert abs(result["final_multipliers"][[0, 1, 16]].sum() - 3) < 1e-12
    assert np.max(np.abs(u)) <= .04


@pytest.mark.parametrize("weight_mode", ["response_tokens", "uniform"])
def test_active_component_preserves_weighted_coefficient_mass_and_bounds(weight_mode):
    meta = _pairs()
    meta["entropy_credit_action_multiplier"] = np.linspace(.81, 1.19, 16)
    meta["pure_entropy_response_tokens"] = np.asarray([31, 907] + [100] * 14)
    meta, _ = _prepared(meta)
    result = finalize(meta, np.ones(16), {"eta": .9, "calibration_weight": weight_mode})
    final = result["final_multipliers"]
    weights = meta["pure_entropy_response_tokens"][:2] if weight_mode == "response_tokens" else np.ones(2)
    assert np.dot(weights, final[:2]) == pytest.approx(
        np.dot(weights, meta["entropy_credit_action_multiplier"][:2]), abs=1e-11)
    assert np.all((final >= .8) & (final <= 1.2))
    np.testing.assert_array_equal(final[2:], meta["entropy_credit_action_multiplier"][2:])
    assert result["metrics"]["pure_entropy/weighted_coefficient_mass_error_max"] < 1e-12
    np.testing.assert_allclose(
        result["trajectory_multipliers"], final / meta["entropy_credit_action_multiplier"])


def test_clipped_projection_preserves_mass_even_when_baseline_at_both_bounds():
    meta = _pairs()
    meta["entropy_credit_action_multiplier"][:2] = [.8, 1.2]
    meta, _ = _prepared(meta)
    result = finalize(meta, np.ones(16), {"eta": 1})
    np.testing.assert_allclose(result["final_multipliers"][:2], [.8, 1.2], atol=1e-14)
    assert result["metrics"]["pure_entropy/bound_touch_fraction"] == 1
    assert result["metrics"]["pure_entropy/clipped_action_fraction"] > 0
    assert result["metrics"]["pure_entropy/weighted_coefficient_mass_error_max"] < 1e-12


def test_eta_zero_is_exact_noop_for_arbitrary_baseline():
    meta = _pairs()
    meta["entropy_credit_action_multiplier"] = np.linspace(.8, 1.2, 16)
    meta, _ = _prepared(meta)
    result = finalize(meta, np.ones(16), {"eta": 0})
    np.testing.assert_array_equal(result["final_multipliers"], meta["entropy_credit_action_multiplier"])
    np.testing.assert_array_equal(result["trajectory_multipliers"], np.ones(16))
    assert result["metrics"]["pure_entropy/active_components"] == 0


def test_prepare_and_finalize_are_invariant_to_padding_copies_and_row_shuffle():
    base = _long_chain()
    base["entropy_credit_action_multiplier"] = np.linspace(.85, 1.15, 24)
    base["pure_entropy_response_tokens"] = np.arange(1, 25) * 17
    prepared, original_metrics = _prepared(_take(base, list(range(24))))
    original = finalize(prepared, np.ones(24), {})
    order = np.random.default_rng(7).permutation(24).tolist() + [0, 1, 16, 16, 16]
    padded, metrics = _prepared(_take(base, order))
    for key in prepared:
        if key.startswith("pure_entropy_"):
            np.testing.assert_allclose(padded[key], prepared[key][order], equal_nan=True)
    result = finalize(padded, np.ones(len(order)), {})
    np.testing.assert_allclose(result["final_multipliers"], original["final_multipliers"][order], rtol=0, atol=1e-14)
    for key, values in original["metadata"].items():
        np.testing.assert_allclose(result["metadata"][key], values[order], equal_nan=True)
    assert metrics["pure_entropy/eligible_pairs"] == original_metrics["pure_entropy/eligible_pairs"]
    assert metrics["pure_entropy/padding_copies"] == 5
    assert result["metrics"]["pure_entropy/final_unique_actions"] == 24


def test_conflicting_copies_and_nonfinite_numeric_inputs_fail_loudly():
    base = _pairs()
    copied = _take(base, list(range(16)) + [0])
    copied["top16_entropy_mean"][-1] = .123
    with pytest.raises(ValueError, match="conflicting padding"):
        prepare(copied, {})
    meta, _ = _prepared(_take(base, list(range(16)) + [0]))
    advantages = np.ones(17)
    advantages[-1] = -1
    with pytest.raises(ValueError, match="conflicting padding"):
        finalize(meta, advantages, {})
    meta["entropy_credit_action_multiplier"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finalize(meta, np.ones(17), {})


@pytest.mark.parametrize("config", [
    {"min_group_size": 1}, {"eta": -1}, {"eta": np.inf},
    {"correction_threshold": 1}, {"regression_threshold": -1},
    {"calibration_weight": "gradient"}, {"roles": "Solver Agent"},
    {"min_residual": -1}, {"min_coverage": 1.1},
])
def test_configuration_rejects_invalid_values_at_startup(config):
    with pytest.raises((TypeError, ValueError)):
        validate(config)


def test_reliability_diagnostics_count_unique_actions_and_log_zero_active_scopes():
    meta = _pairs()
    meta["is_action_valid"][0] = False
    meta["pure_entropy_truncated"][2] = True
    meta["pure_entropy_response_tokens"][4] = 0
    meta["top16_entropy_mean"][6] = np.nan
    meta["top16_entropy"][8] = None
    meta["top16_entropy"][10] = {"coverage": .1}
    meta, metrics = _prepared(_take(meta, list(range(16)) + [0, 2, 4, 6, 8, 10]))
    for reason in ("invalid", "truncated", "zero_response", "missing_entropy", "missing_coverage", "low_coverage"):
        assert metrics[f"pure_entropy/{reason}_actions"] == 1
    assert metrics["pure_entropy/group_size_min"] == 2
    assert metrics["pure_entropy/regression_threshold_unreachable_groups"] == 1
    result = finalize(meta, np.ones(22), {})
    assert result["metrics"]["pure_entropy/solver_agent/turn_1/active_edges"] == 0
    assert metrics["pure_entropy/solver_agent/turn_1/correction_gate/max"] == 0


def test_failure_branch_rejects_actual_positive_or_mixed_advantages():
    meta, _ = _prepared(_pairs(
        [.1, .2, .3, .4, .5, .6, .7, .8],
        [.95, .19, .29, .39, .49, .59, .69, .79], success=False))
    for pair in ([1, 1], [-1, 1], [0, -1]):
        advantages = -np.ones(16)
        advantages[:2] = pair
        result = finalize(meta, advantages, {})
        np.testing.assert_array_equal(result["final_multipliers"], np.ones(16))
        assert result["metrics"]["pure_entropy/sign_mismatch_edges"] == 1


def test_invalid_middle_action_breaks_both_neighboring_long_chain_edges():
    meta = _long_chain()
    meta["is_action_valid"][1] = False
    meta, _ = _prepared(meta)
    assert not meta["pure_entropy_pair_eligible"][1]
    assert not meta["pure_entropy_pair_eligible"][16]
    result = finalize(meta, np.ones(24), {})
    np.testing.assert_array_equal(result["metadata"]["pure_entropy_u"][[0, 1, 16]], np.zeros(3))


def test_empty_batch_stays_well_defined_and_missing_metadata_is_rejected():
    meta = _take(_pairs(), [])
    meta, metrics = _prepared(meta)
    result = finalize(meta, [], {})
    assert result["final_multipliers"].size == 0
    assert metrics["pure_entropy/eligible_pairs"] == 0
    assert result["metrics"]["pure_entropy/weighted_coefficient_mass_error_max"] == 0
    del meta["uid"]
    with pytest.raises(KeyError, match="missing fields"):
        prepare(meta, {})


def test_disconnected_active_edges_calibrate_separate_components_within_one_trajectory():
    meta = _long_chain()
    meta["top16_entropy_mean"][16] = .9  # middle edge increases entropy, so no repair gate
    last = _pairs(
        [.9, .12, .22, .32, .42, .52, .62, .72],
        [.1, .13, .23, .33, .43, .53, .63, .73])
    last["role_turn_index"] += 2
    meta = _concat(meta, _take(last, list(range(1, 16, 2))))
    indices = [0, 1, 16, 24]
    meta["entropy_credit_action_multiplier"][indices] = [.95, 1.05, 1.15, .85]
    meta["pure_entropy_response_tokens"][indices] = [10, 900, 300, 15]
    meta, _ = _prepared(meta)
    result = finalize(meta, np.ones(32), {})
    assert result["metrics"]["pure_entropy/active_edges"] == 2
    assert result["metrics"]["pure_entropy/active_components"] == 2
    assert result["metadata"]["pure_entropy_active_component"][0] != result["metadata"]["pure_entropy_active_component"][16]
    for rows in ([0, 1], [16, 24]):
        weights = meta["pure_entropy_response_tokens"][rows]
        assert np.dot(weights, result["final_multipliers"][rows]) == pytest.approx(
            np.dot(weights, meta["entropy_credit_action_multiplier"][rows]), abs=1e-11)
