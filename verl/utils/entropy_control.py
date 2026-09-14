"""Value-gated entropy control, separate from pure's advantage multipliers.

The controller consumes *full-vocabulary* action-mean policy entropies in nats.
Historical Top-16 summaries are features of the value model, never a substitute
for the differentiable Actor entropy used in :func:`action_entropy_hinge_loss`.
This module's controller and checkpoint format do not depend on torch or Ray.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

import numpy as np


DEFAULT_ENTROPY_CONTROL_CONFIG = {
    "enabled": False,
    "fast_beta": 0.9,
    "slow_beta": 0.99,
    "cap_margin": 0.2,
    "cap_std_multiplier": 2.0,
    "risk_scale": 0.2,
    "trend_tolerance": 0.02,
    "trend_scale": 0.1,
    "min_calibration_actions": 8,
    "calibration_steps": 1,
    "min_group_actions": 2,
    "delta_deadzone": 0.02,
    "delta_scale": 0.1,
    "entropy_delta_deadzone": 0.01,
    "entropy_delta_scale": 0.1,
    "kappa": 0.25,
    "ramp_steps": 20,
}


def normalize_entropy_control_config(config=None):
    result = dict(DEFAULT_ENTROPY_CONTROL_CONFIG)
    if config is not None:
        result.update(dict(config))
    for key in ("fast_beta", "slow_beta"):
        if not 0 <= float(result[key]) < 1:
            raise ValueError(f"entropy control {key} must lie in [0, 1)")
    if float(result["fast_beta"]) >= float(result["slow_beta"]):
        raise ValueError("entropy control fast_beta must be smaller than slow_beta")
    for key in ("risk_scale", "trend_scale", "delta_scale", "entropy_delta_scale"):
        if not math.isfinite(float(result[key])) or float(result[key]) <= 0:
            raise ValueError(f"entropy control {key} must be finite and positive")
    for key in ("cap_margin", "cap_std_multiplier", "trend_tolerance", "delta_deadzone", "entropy_delta_deadzone"):
        if not math.isfinite(float(result[key])) or float(result[key]) < 0:
            raise ValueError(f"entropy control {key} must be finite and nonnegative")
    for key in ("min_calibration_actions", "calibration_steps", "min_group_actions", "ramp_steps"):
        if int(result[key]) < 1 or int(result[key]) != float(result[key]):
            raise ValueError(f"entropy control {key} must be a positive integer")
    if not 0 < float(result["kappa"]) < 1:
        raise ValueError("entropy control kappa must lie in (0, 1)")
    return result


def _column(meta, name, size, default, dtype=None):
    value = meta.get(name)
    array = np.full(size, default, dtype=dtype) if value is None else np.asarray(value, dtype=dtype)
    if array.shape != (size,):
        raise ValueError(f"entropy control {name} must have shape ({size},)")
    return array


def _role_confidence(reliability, roles, turns):
    if isinstance(reliability, Mapping):
        confidence = []
        for role, turn in zip(roles, turns):
            role = str(role)
            canonical = {"solver agent": "solver", "verifier agent": "verifier"}.get(role.strip().lower(), role.strip().lower())
            keys = ((role, int(turn)), (canonical, int(turn)), role, canonical)
            confidence.append(next((reliability[key] for key in keys if key in reliability), 0.0))
        confidence = np.asarray(confidence, dtype=np.float64)
    else:
        confidence = np.broadcast_to(np.asarray(reliability, dtype=np.float64), (len(roles),)).copy()
    return np.clip(np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)


class EntropyController:
    """Maintain frozen initial-policy caps and role/turn entropy trend state.

    Call ``prepare`` on the first old-log-prob batch, even when the value head is
    not ready. With the default one-step calibration window, every cap is based
    on the initial Actor. A group with insufficient initial data remains closed;
    it cannot silently redefine a later, inflated entropy as its healthy cap.

    Input rows are individual actions. Repeated ``(traj_uid, agent_id, turn)``
    padding rows do not contribute to calibration, trends, or the Actor penalty.
    ``ready`` describes the deployed head used for this batch, not its candidate.
    """

    VERSION = 1

    def __init__(self, config=None):
        self.config = normalize_entropy_control_config(config)
        self.groups = {}
        self.start_step = None
        self.ready_step = None
        self.last_step = None

    def state_dict(self):
        return {
            "version": self.VERSION,
            "config": dict(self.config),
            "start_step": self.start_step,
            "ready_step": self.ready_step,
            "last_step": self.last_step,
            "groups": [{"role": key[0], "turn": key[1], **dict(value)} for key, value in sorted(self.groups.items())],
        }

    def load_state_dict(self, state):
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported entropy controller checkpoint version")
        # Saved scales/caps must remain compatible on resume. Explicitly starting
        # a new controller is required to change calibration conventions.
        saved_config = normalize_entropy_control_config(state.get("config", {}))
        for key in DEFAULT_ENTROPY_CONTROL_CONFIG:
            if saved_config[key] != self.config[key]:
                raise ValueError(f"entropy controller resume config mismatch: {key}")
        self.start_step = state.get("start_step")
        self.ready_step = state.get("ready_step")
        self.last_step = state.get("last_step")
        self.groups = {}
        for row in state.get("groups", []):
            key = (str(row["role"]), int(row["turn"]))
            self.groups[key] = {k: v for k, v in row.items() if k not in ("role", "turn")}

    def prepare(self, meta, full_entropy_means, step, ready, reliability=1.0, temporal_reliability=None):
        entropies = np.asarray(full_entropy_means, dtype=np.float64)
        if entropies.ndim != 1:
            raise ValueError("full_entropy_means must contain one full-vocabulary entropy per action")
        size = len(entropies)
        output = {
            "entropy_control_weight": np.zeros(size, dtype=np.float32),
            "entropy_control_cap": np.zeros(size, dtype=np.float32),
            "entropy_control_valid": np.zeros(size, dtype=bool),
        }
        metrics = {"entropy_control/enabled": float(bool(self.config["enabled"]))}
        if not self.config["enabled"] or size == 0:
            return output, metrics
        step = int(step)
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("entropy controller prepare requires a strictly increasing training step")
        self.last_step = step
        if self.start_step is None:
            self.start_step = step
        if not ready:
            self.ready_step = None
        elif self.ready_step is None:
            self.ready_step = step
        cfg = self.config
        required = {"agent_id", "traj_uid", "role_turn_index", "is_action_valid", "pure_entropy_response_tokens"}
        missing = required.difference(meta)
        if missing:
            raise ValueError(f"entropy control missing action metadata: {sorted(missing)}")
        roles = _column(meta, "agent_id", size, "unknown", object)
        turns = _column(meta, "role_turn_index", size, 0, np.int64)
        trajectory = _column(meta, "traj_uid", size, "", object)
        valid = _column(meta, "is_action_valid", size, False, bool)
        truncated = _column(meta, "pure_entropy_truncated", size, False, bool)
        tokens = _column(meta, "pure_entropy_response_tokens", size, 0, np.int64)
        d_value = _column(meta, "value_credit_delta", size, np.nan, np.float64)
        q_entropy = _column(meta, "value_entropy_delta", size, np.nan, np.float64)
        credit_available = _column(meta, "value_credit_available", size, False, bool)
        absolute_available = (_column(meta, "value_absolute_available", size, False, bool)
                              if "value_absolute_available" in meta else credit_available)
        entropy_available = _column(meta, "value_entropy_available", size, False, bool)
        available = credit_available & absolute_available & np.isfinite(d_value)
        temporal_available = available & entropy_available & np.isfinite(q_entropy)
        good = valid & ~truncated & (tokens > 0) & np.isfinite(entropies) & (entropies >= 0)
        groups = defaultdict(list)
        seen = {}
        for row in range(size):
            identity = (str(trajectory[row]), str(roles[row]), int(turns[row]))
            if identity in seen:
                prior = seen[identity]
                equal_flags = all(array[row] == array[prior] for array in
                                  (valid, truncated, tokens, credit_available, absolute_available, entropy_available))
                equal_values = all(np.isclose(array[row], array[prior], atol=1e-7, rtol=1e-6, equal_nan=True)
                                   for array in (d_value, q_entropy))
                # Repacking the same Actor forward can introduce small BF16
                # numerical differences, but materially different entropies are
                # not legitimate padding copies.
                equal_entropy = np.isclose(entropies[row], entropies[prior], atol=1e-3, rtol=1e-4, equal_nan=True)
                if not (equal_flags and equal_values and equal_entropy):
                    raise ValueError(f"conflicting entropy control padding duplicate: {identity}")
                continue
            seen[identity] = row
            if good[row]:
                output["entropy_control_valid"][row] = True
                groups[(identity[1], identity[2])].append(row)

        confidence = _role_confidence(reliability, roles, turns)
        temporal_confidence = _role_confidence(reliability if temporal_reliability is None else temporal_reliability, roles, turns)
        ramp = min(1.0, (step - self.ready_step + 1) / cfg["ramp_steps"]) if ready and self.ready_step is not None else 0.0
        calibration_open = step - self.start_step < cfg["calibration_steps"]
        available_rows = 0
        temporal_rows = 0
        for key, rows in groups.items():
            values = entropies[rows]
            state = self.groups.setdefault(key, {"count": 0, "sum": 0.0, "sum_sq": 0.0, "cap": None, "fast": None, "slow": None})
            # Freeze the cap as soon as sufficient *initial-window* data exist.
            if calibration_open and state["cap"] is None:
                state["count"] += len(rows)
                state["sum"] += float(values.sum())
                state["sum_sq"] += float(np.square(values).sum())
                if state["count"] >= cfg["min_calibration_actions"]:
                    reference = state["sum"] / state["count"]
                    std = math.sqrt(max(0.0, state["sum_sq"] / state["count"] - reference * reference))
                    state["cap"] = reference + cfg["cap_margin"] + cfg["cap_std_multiplier"] * std
                    state["fast"] = state["slow"] = reference
            scope = f"entropy_control/{key[0]}/turn_{key[1]}"
            metrics[scope + "/calibrated"] = float(state["cap"] is not None)
            metrics[scope + "/calibration_actions"] = float(state["count"])
            if state["cap"] is None:
                continue
            output["entropy_control_cap"][rows] = state["cap"]
            metrics[scope + "/cap"] = float(state["cap"])
            metrics[scope + "/full_entropy_mean"] = float(values.mean())
            if len(rows) < cfg["min_group_actions"]:
                continue
            current = float(values.mean())
            state["fast"] = cfg["fast_beta"] * state["fast"] + (1 - cfg["fast_beta"]) * current
            state["slow"] = cfg["slow_beta"] * state["slow"] + (1 - cfg["slow_beta"]) * current
            trend = np.clip((state["fast"] - state["slow"] - cfg["trend_tolerance"]) / cfg["trend_scale"], 0, 1)
            excess = np.clip((current - state["cap"]) / cfg["risk_scale"], 0, 1)
            risk = float(max(trend, excess))
            bad = np.clip((-d_value[rows] - cfg["delta_deadzone"]) / cfg["delta_scale"], 0, 1)
            harm = np.clip((-q_entropy[rows] - cfg["entropy_delta_deadzone"]) / cfg["entropy_delta_scale"], 0, 1)
            # A first same-role action has absolute entropy but no temporal
            # edge. Its reliable negative progress can use the kappa base
            # brake; absent/unqualified temporal information adds nothing.
            harm = np.where(temporal_available[rows], harm * temporal_confidence[rows], 0.0)
            gate = confidence[rows] * ramp * risk * bad * (cfg["kappa"] + (1 - cfg["kappa"]) * harm)
            gate = np.where(available[rows] & bool(ready), gate, 0.0)
            output["entropy_control_weight"][rows] = gate.astype(np.float32)
            available_rows += int(available[rows].sum())
            temporal_rows += int(temporal_available[rows].sum())
            metrics[scope + "/risk"] = risk
            metrics[scope + "/fast"] = float(state["fast"])
            metrics[scope + "/slow"] = float(state["slow"])
            metrics[scope + "/mean_weight"] = float(gate.mean())
            predicted = np.asarray(rows, dtype=np.int64)[available[rows]]
            if len(predicted):
                metrics[scope + "/mean_value_delta"] = float(d_value[predicted].mean())
                metrics[scope + "/negative_progress_fraction"] = float((d_value[predicted] < -cfg["delta_deadzone"]).mean())
            temporal_predicted = np.asarray(rows, dtype=np.int64)[temporal_available[rows]]
            if len(temporal_predicted):
                metrics[scope + "/mean_entropy_delta"] = float(q_entropy[temporal_predicted].mean())
        count = max(1, int(output["entropy_control_valid"].sum()))
        weights = output["entropy_control_weight"]
        metrics.update({
            "entropy_control/ready": float(bool(ready)),
            "entropy_control/ramp": float(ramp),
            "entropy_control/valid_unique_actions": float(output["entropy_control_valid"].sum()),
            "entropy_control/prediction_coverage": available_rows / count,
            "entropy_control/temporal_prediction_coverage": temporal_rows / count,
            "entropy_control/active_fraction": float(np.count_nonzero(weights) / count),
            "entropy_control/mean_weight": float(weights.sum() / count),
            "entropy_control/calibrated_groups": float(sum(g["cap"] is not None for g in self.groups.values())),
        })
        return output, metrics


def action_entropy_hinge_loss(token_entropy, response_mask, weights, caps, action_valid=None):
    """Mean over valid unique actions, not tokens or only gated actions.

    ``token_entropy`` retains gradients. All rollout-derived gates, masks and
    caps are detached. Each row is exactly one action; this function must not be
    used on rows concatenating several actions. Returns loss and tensor metrics.
    """
    import torch

    if token_entropy.ndim != 2 or token_entropy.shape != response_mask.shape:
        raise ValueError("token entropy and response mask must have matching [actions, tokens] shapes")
    size = token_entropy.shape[0]
    if weights.shape != (size,) or caps.shape != (size,):
        raise ValueError("entropy control weights/caps must have shape [actions]")
    mask = response_mask.detach().bool()
    count = mask.sum(-1)
    valid = count > 0
    if action_valid is not None:
        if action_valid.shape != (size,):
            raise ValueError("entropy control validity must have shape [actions]")
        valid = valid & action_valid.detach().bool()
    # Exclude entire invalid rows as well as token padding, including any NaNs
    # in those positions. Zero eligible rows still produce a differentiable zero.
    active_mask = mask & valid.unsqueeze(-1)
    entropy = torch.where(active_mask, token_entropy.float(), torch.zeros_like(token_entropy, dtype=torch.float32))
    action_mean = entropy.sum(-1) / count.clamp_min(1)
    gate = torch.where(valid, weights.detach().float().clamp(0, 1), torch.zeros_like(weights, dtype=torch.float32))
    cap = torch.where(valid, caps.detach().float(), torch.zeros_like(caps, dtype=torch.float32))
    excess = torch.relu(action_mean - cap)
    denominator = valid.sum().clamp_min(1)
    loss = (gate * excess).sum() / denominator
    return loss, {
        "hinge_loss": loss.detach(),
        "full_entropy_mean": action_mean.sum().detach() / denominator,
        "mean_gate": gate.sum().detach() / denominator,
        "active_fraction": ((gate > 0) & (excess > 0)).sum().detach() / denominator,
        "valid_actions": valid.sum().detach(),
    }
