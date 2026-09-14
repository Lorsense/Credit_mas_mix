import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("value_metadata_test", Path(__file__).resolve().parents[2] / "verl/utils/value_credit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def actions():
    return [{"role": "solver", "text": "s1", "valid": True, "role_turn_index": 0, "entropy_mean": .2},
            {"role": "verifier", "text": "<verify>reject</verify>", "valid": True, "role_turn_index": 0, "entropy_mean": .3},
            {"role": "solver", "text": "s2", "valid": True, "role_turn_index": 1, "entropy_mean": .5}]


def batch():
    a = actions()
    return {"traj_uid": ["t"] * 3, "uid": ["q"] * 3, "value_question": ["question"] * 3,
            "value_action_index": [0, 1, 2], "value_max_solver_turns": [2] * 3,
            "agent_id": [x["role"] for x in a], "role_turn_index": [0, 0, 1],
            "value_action_text": [x["text"] for x in a], "is_action_valid": [True] * 3,
            "pass": [True] * 3, "top16_entropy_mean": [.2, .3, .5]}


def test_entropy_features_are_causal_and_label_free():
    original = actions()
    first = module.prefix_entropy_features(original)
    changed = copy.deepcopy(original)
    changed[-1].update(entropy_mean=.9, text="different future", **{"pass": False})
    second = module.prefix_entropy_features(changed)
    for key in first:
        np.testing.assert_array_equal(first[key][:-1], second[key][:-1])
    assert first["temporal"][-1, 1] == pytest.approx(.3)
    assert first["temporal"][-1, 2] == 0  # excludes current delta from drift


def test_invalid_action_breaks_its_role_temporal_edge():
    a = actions()
    a[0]["valid"] = False
    f = module.prefix_entropy_features(a)
    assert not f["temporal_available"].any()
    assert f["absolute_available"][-1]


def test_mean_only_records_do_not_require_token_quantiles():
    records, metrics = module.build_trajectory_records(batch(), 2)
    assert len(records) == 1
    assert records[0]["actions"][0]["entropy_mean"] == .2
    assert metrics["value_credit/skipped_trajectories"] == 0


def test_partial_predictions_and_invalid_action_are_masked_locally():
    b = batch()
    b["is_action_valid"][0] = False
    predictions = [{"sem": .4, "abs": .4, "full": .4, "absolute_available": False, "temporal_available": False},
                   {"sem": .5, "abs": .45, "full": .45, "absolute_available": False, "temporal_available": False},
                   {"sem": .6, "abs": .55, "full": .5, "absolute_available": True, "temporal_available": True}]
    module.attach_value_predictions(b, {"t": predictions}, True)
    assert b["value_credit_available"].tolist() == [False, True, False]
    assert b["value_credit_delta"][1] == pytest.approx(.05)
    assert b["value_entropy_delta"][1] == pytest.approx(-.05)


def test_no_value_multiplier_is_exposed():
    assert not hasattr(module, "compute_value_credit_multipliers")


def test_mixed_real_budgets_allowed_without_runtime_override():
    b = batch()
    b2 = batch()
    for key in b2:
        if key == "traj_uid":
            b2[key] = ["other"] * 3
    b2["value_max_solver_turns"] = [3] * 3
    # Three-turn budget cannot terminate at second Solver; reject incomplete history.
    merged = {key: b[key] + b2[key] for key in b}
    records, metrics = module.build_trajectory_records(merged)
    assert len(records) == 1
    assert metrics["value_credit/incomplete_trajectories"] == 1
