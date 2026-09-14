"""No GPU required: calibration, value gates, resume, and optional autograd."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "isolated_entropy_control", Path(__file__).parents[2] / "verl/utils/entropy_control.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
EntropyController = _MODULE.EntropyController
hinge = _MODULE.action_entropy_hinge_loss


def metadata(n=8):
    return {
        "traj_uid": np.asarray([f"t{i}" for i in range(n)], dtype=object),
        "agent_id": np.full(n, "Solver Agent", dtype=object),
        "role_turn_index": np.ones(n, dtype=np.int64),
        "is_action_valid": np.ones(n, dtype=bool),
        "pure_entropy_truncated": np.zeros(n, dtype=bool),
        "pure_entropy_response_tokens": np.full(n, 50),
        "value_credit_delta": np.full(n, -.2),
        "value_entropy_delta": np.full(n, -.2),
        "value_credit_available": np.ones(n, dtype=bool),
        "value_entropy_available": np.ones(n, dtype=bool),
    }


def controller(**kwargs):
    return EntropyController({"enabled": True, "ramp_steps": 1, **kwargs})


def calibrated(**kwargs):
    obj = controller(**kwargs)
    obj.prepare(metadata(), np.full(8, 1.0), step=1, ready=False)
    return obj


def test_disabled_controller_has_no_state_or_control_effect():
    obj = EntropyController()
    fields, metrics = obj.prepare({}, [3.0, 4.0], step=1, ready=True)
    assert not fields["entropy_control_weight"].any()
    assert not obj.groups
    assert obj.start_step is None
    assert metrics["entropy_control/enabled"] == 0


def test_reference_uses_current_full_entropy_even_before_value_ready():
    obj = calibrated()
    state = obj.groups[("Solver Agent", 1)]
    assert state["cap"] == pytest.approx(1.2)
    fields, _ = obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=True)
    assert np.all(fields["entropy_control_weight"] == 1)
    assert np.allclose(fields["entropy_control_cap"], 1.2)
    assert state["cap"] == pytest.approx(1.2)


def test_high_entropy_without_negative_progress_is_not_braked():
    obj = calibrated()
    meta = metadata()
    meta["value_credit_delta"][:] = .2
    fields, _ = obj.prepare(meta, np.full(8, 3.0), step=2, ready=True)
    assert not fields["entropy_control_weight"].any()


def test_negative_progress_still_has_base_brake_when_entropy_correction_is_positive():
    obj = calibrated()
    meta = metadata()
    meta["value_entropy_delta"][:] = .2
    fields, _ = obj.prepare(meta, np.full(8, 2.0), step=2, ready=True)
    np.testing.assert_allclose(fields["entropy_control_weight"], .25)


def test_negative_progress_below_deadzone_or_without_entropy_risk_does_not_brake():
    obj = calibrated()
    fields, _ = obj.prepare(metadata(), np.full(8, 1.0), step=2, ready=True)
    assert not fields["entropy_control_weight"].any()
    meta = metadata()
    meta["value_credit_delta"][:] = -.01
    fields, _ = obj.prepare(meta, np.full(8, 3.0), step=3, ready=True)
    assert not fields["entropy_control_weight"].any()


def test_missing_value_prediction_disables_its_control():
    obj = calibrated()
    meta = metadata()
    meta["value_credit_available"][0] = False
    fields, _ = obj.prepare(meta, np.full(8, 2.0), step=2, ready=True)
    assert fields["entropy_control_weight"][0] == 0
    assert fields["entropy_control_weight"][1] > 0


def test_first_action_with_absolute_entropy_can_use_base_brake_without_temporal_edge():
    obj = calibrated()
    meta = metadata()
    meta["value_entropy_available"][:] = False
    meta["value_entropy_delta"][:] = np.nan
    fields, metrics = obj.prepare(meta, np.full(8, 2.0), step=2, ready=True)
    np.testing.assert_allclose(fields["entropy_control_weight"], .25)
    assert metrics["entropy_control/prediction_coverage"] == 1
    assert metrics["entropy_control/temporal_prediction_coverage"] == 0


def test_unqualified_role_temporal_branch_does_not_disable_absolute_base_brake():
    obj = calibrated()
    fields, _ = obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=True,
                            reliability={"solver": 1.}, temporal_reliability={"solver": 0.})
    np.testing.assert_allclose(fields["entropy_control_weight"], .25)


def test_missing_current_action_absolute_entropy_is_not_replaced_by_previous_role_entropy():
    obj = calibrated()
    meta = metadata()
    meta["value_absolute_available"] = np.zeros(8, dtype=bool)
    fields, _ = obj.prepare(meta, np.full(8, 2.0), step=2, ready=True)
    assert not fields["entropy_control_weight"].any()


def test_unready_deployed_head_produces_no_gate_even_with_predictions():
    obj = calibrated()
    fields, _ = obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=False)
    assert not fields["entropy_control_weight"].any()


def test_duplicate_padding_and_invalid_or_truncated_actions_are_excluded():
    obj = calibrated()
    meta = metadata()
    meta = {key: np.concatenate([value, value[:1]]) for key, value in meta.items()}
    meta["is_action_valid"][1] = False
    meta["pure_entropy_truncated"][2] = True
    meta["pure_entropy_response_tokens"][3] = 0
    fields, metrics = obj.prepare(meta, np.full(9, 2.0), step=2, ready=True)
    np.testing.assert_array_equal(fields["entropy_control_valid"], [1, 0, 0, 0, 1, 1, 1, 1, 0])
    assert metrics["entropy_control/valid_unique_actions"] == 5
    assert not fields["entropy_control_weight"][[1, 2, 3, 8]].any()


def test_padding_copies_cannot_satisfy_calibration_sample_count():
    obj = controller()
    meta = metadata(1)
    meta = {key: np.repeat(value, 8) for key, value in meta.items()}
    fields, _ = obj.prepare(meta, np.full(8, 1.0), step=1, ready=True)
    assert obj.groups[("Solver Agent", 1)]["cap"] is None
    assert not fields["entropy_control_weight"].any()


@pytest.mark.parametrize("conflict", ["entropy", "value_credit_delta", "is_action_valid"])
def test_conflicting_padding_duplicates_fail_instead_of_silently_disappearing(conflict):
    obj = calibrated()
    meta = metadata()
    meta = {key: np.concatenate([value, value[:1]]) for key, value in meta.items()}
    entropies = np.full(9, 2.0)
    if conflict == "entropy":
        entropies[-1] = 3.0
    elif conflict == "is_action_valid":
        meta[conflict][-1] = False
    else:
        meta[conflict][-1] = .2
    with pytest.raises(ValueError, match="conflicting"):
        obj.prepare(meta, entropies, step=2, ready=True)


def test_late_group_cannot_calibrate_against_already_inflated_entropy():
    obj = calibrated()
    meta = metadata()
    meta["agent_id"][:] = "Verifier Agent"
    fields, _ = obj.prepare(meta, np.full(8, 6.0), step=2, ready=True)
    assert obj.groups[("Verifier Agent", 1)]["cap"] is None
    assert not fields["entropy_control_weight"].any()


def test_ramp_and_role_reliability_scale_gate_without_using_advantage():
    obj = calibrated(ramp_steps=10)
    meta = metadata()
    # Advantage intentionally absent. A zero GRPO advantage must not disable
    # an otherwise valid independent entropy penalty.
    fields, _ = obj.prepare(meta, np.full(8, 2.0), step=2, ready=True,
                            reliability={"Solver Agent": .5}, temporal_reliability=1.)
    np.testing.assert_allclose(fields["entropy_control_weight"], .05)


def test_reenabling_after_deployed_failure_restarts_gradual_ramp():
    obj = calibrated(ramp_steps=10)
    obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=True)
    obj.prepare(metadata(), np.full(8, 2.0), step=3, ready=False)
    fields, _ = obj.prepare(metadata(), np.full(8, 2.0), step=9, ready=True)
    np.testing.assert_allclose(fields["entropy_control_weight"], .1)


def test_canonical_scorer_role_reliability_matches_rollout_agent_name():
    obj = calibrated()
    fields, _ = obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=True,
                            reliability={"solver": .6, "verifier": 0.}, temporal_reliability=1.)
    np.testing.assert_allclose(fields["entropy_control_weight"], .6)


def test_nonfinite_entropy_and_predictions_do_not_create_nan_gates():
    obj = calibrated()
    meta = metadata()
    meta["value_entropy_delta"][0] = np.nan
    entropy = np.full(8, 2.0)
    entropy[1] = np.nan
    fields, _ = obj.prepare(meta, entropy, step=2, ready=True)
    assert np.isfinite(fields["entropy_control_weight"]).all()
    assert fields["entropy_control_weight"][0] == pytest.approx(.25)
    assert fields["entropy_control_weight"][1] == 0


def test_resume_reproduces_caps_ema_ramp_and_next_gate():
    obj = calibrated(ramp_steps=10)
    obj.prepare(metadata(), np.full(8, 2.0), step=2, ready=True)
    resumed = controller(ramp_steps=10)
    resumed.load_state_dict(json.loads(json.dumps(obj.state_dict())))
    expected, expected_metrics = obj.prepare(metadata(), np.full(8, 1.8), step=3, ready=True)
    actual, actual_metrics = resumed.prepare(metadata(), np.full(8, 1.8), step=3, ready=True)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual_metrics == expected_metrics


def test_resume_rejects_incompatible_scales_and_duplicate_step():
    obj = calibrated()
    with pytest.raises(ValueError, match="mismatch"):
        controller(cap_margin=.9).load_state_dict(obj.state_dict())
    with pytest.raises(ValueError, match="increasing"):
        obj.prepare(metadata(), np.ones(8), step=1, ready=True)


@pytest.mark.parametrize("changes", [
    {"fast_beta": 1.0}, {"fast_beta": .999}, {"delta_scale": 0},
    {"cap_margin": -1}, {"kappa": 0}, {"min_group_actions": 0},
])
def test_invalid_controller_config_is_rejected(changes):
    with pytest.raises(ValueError):
        controller(**changes)


def test_hinge_has_actor_gradient_but_detaches_gate_and_cap():
    torch = pytest.importorskip("torch")
    entropy = torch.tensor([[2.0, 4.0, 100.0], [1.0, 1.0, 1.0]], requires_grad=True)
    weights = torch.tensor([.5, 1.0], requires_grad=True)
    caps = torch.tensor([2.0, 2.0], requires_grad=True)
    loss, _ = hinge(entropy, torch.tensor([[1, 1, 0], [1, 1, 1]]), weights, caps)
    assert loss.item() == pytest.approx(.25)
    loss.backward()
    np.testing.assert_allclose(entropy.grad.numpy(), [[.125, .125, 0], [0, 0, 0]])
    assert weights.grad is None
    assert caps.grad is None


def test_hinge_equal_action_weighting_masks_padding_and_invalid_rows():
    torch = pytest.importorskip("torch")
    entropy = torch.tensor([[3., 3., 3.], [3., 100., 100.], [float("nan")] * 3], requires_grad=True)
    loss, metrics = hinge(entropy, torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),
                          torch.ones(3), torch.ones(3), torch.tensor([1, 1, 0]))
    assert loss.item() == pytest.approx(2.)
    assert metrics["valid_actions"].item() == 2
    loss.backward()
    assert torch.isfinite(entropy.grad).all()
    assert entropy.grad[0].sum() == entropy.grad[1].sum()
    assert not entropy.grad[2].any()


def test_hinge_empty_or_zero_gate_is_differentiable_zero():
    torch = pytest.importorskip("torch")
    for mask, weights in [(torch.ones(2, 3), torch.zeros(2)), (torch.zeros(2, 3), torch.ones(2))]:
        entropy = torch.full((2, 3), 3., requires_grad=True)
        loss, _ = hinge(entropy, mask, weights, torch.ones(2))
        assert loss.item() == 0
        loss.backward()
        assert not entropy.grad.any()


def test_hinge_updates_logits_toward_lower_full_vocabulary_entropy():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[2., 0., -1.], [1., .1, -.5]]], requires_grad=True)
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * logits.log_softmax(-1)).sum(-1)
    loss, _ = hinge(entropy, torch.ones(1, 2), torch.ones(1), torch.zeros(1))
    initial = entropy.mean().item()
    loss.backward()
    with torch.no_grad():
        updated = logits - .1 * logits.grad
        after = -(updated.softmax(-1) * updated.log_softmax(-1)).sum(-1).mean().item()
    assert after < initial
