"""Restore zero GRPO advantages without inventing trajectories or reward labels.

Only detached action scalars are handled here.  A virtual reward participates
in the sample mean/std, never in the actor batch.  Recovery groups use one
reward per real trajectory; actor weights compensate for different response
lengths so each trajectory has the same weight inside its recovery group.
"""

from __future__ import annotations

import math
import numbers
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULTS = {
    "enable": False,
    "success_virtual_reward": 0.5,
    "failure_virtual_reward": 1.0,
    "success_weight": 0.2,
    "failure_weight": 0.1,
    "success_mass_cap": 0.05,
    "failure_mass_cap": 0.03,
    "reward_std_tolerance": 1e-8,
    "advantage_zero_tolerance": 1e-8,
    "normalization_epsilon": 1e-6,
    "pure_factor_bound": 1.2,
}
_REQUIRED = ("uid", "traj_uid", "agent_id", "role_turn_index", "pass", "is_action_valid")
# Rollout facts must be identical on padding copies.  Derived credit/control
# fields are intentionally excluded: their duplicate entries can be masked.
_OPTIONAL_FACTS = (
    "top16_entropy_mean", "top16_entropy", "pure_entropy_response_tokens",
    "pure_entropy_truncated", "value_question", "value_action_text",
    "value_global_action_index", "value_max_solver_turns", "value_role_turn_index",
)


def _binary(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be bool or numeric 0/1")


def _configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: config.get(key, default) for key, default in DEFAULTS.items()}
    result["enable"] = _binary(result["enable"], "enable")
    for key in DEFAULTS:
        if key == "enable":
            continue
        if isinstance(result[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a finite number")
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be a finite number")
    for key in ("success_weight", "failure_weight", "success_mass_cap", "failure_mass_cap",
                "reward_std_tolerance", "advantage_zero_tolerance", "normalization_epsilon"):
        if result[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if result["pure_factor_bound"] < 1:
        raise ValueError("pure_factor_bound must be at least one")
    return result


def validate_advantage_recovery_config(config: Mapping[str, Any]) -> None:
    """Validate the recovery configuration before expensive workers start."""
    _configuration(config)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric action scalars") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return (isinstance(a, Mapping) and isinstance(b, Mapping)
                and a.keys() == b.keys() and all(_equal(a[key], b[key]) for key in a))
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        try:
            return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
        except TypeError:
            return False
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return bool(a == b) or (math.isnan(float(a)) and math.isnan(float(b)))
    return bool(a == b)


def logical_action_indices(
    meta: Mapping[str, Sequence[Any]], rewards: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first logical-action rows and an inverse map into those rows.

    Identity is (question uid, trajectory uid, role, role-local turn).  Outcome
    and question must be consistent across each trajectory, including roles.
    Conflicting padding copies raise rather than silently changing GRPO stats.
    """
    missing = set(_REQUIRED).difference(meta)
    if missing:
        raise KeyError(f"advantage recovery is missing fields: {sorted(missing)}")
    facts = [key for key in (*_REQUIRED, *_OPTIONAL_FACTS) if key in meta]
    lengths = {len(meta[key]) for key in facts}
    if len(lengths) != 1:
        raise ValueError("advantage recovery metadata fields have inconsistent lengths")
    size = lengths.pop()
    reward_values = None if rewards is None else _vector(rewards, size, "rewards")
    unique: list[int] = []
    identities: dict[tuple[str, str, str, int], int] = {}
    inverse = np.empty(size, dtype=np.int64)
    trajectory_state: dict[str, tuple[str, bool]] = {}
    for row in range(size):
        identifiers = []
        for key in ("uid", "traj_uid", "agent_id"):
            value = meta[key][row]
            if (value is None or not str(value).strip() or isinstance(value, (bool, np.bool_))
                    or (isinstance(value, numbers.Real) and not math.isfinite(float(value)))):
                raise ValueError(f"{key} must contain nonempty finite identities")
            identifiers.append(str(value))
        uid, trajectory, role = identifiers
        turn = meta["role_turn_index"][row]
        if isinstance(turn, (bool, np.bool_)) or not isinstance(turn, numbers.Integral) or turn < 0:
            raise ValueError("role_turn_index must contain nonnegative integers")
        success = _binary(meta["pass"][row], "pass")
        _binary(meta["is_action_valid"][row], "is_action_valid")
        state = (uid, success)
        if trajectory in trajectory_state and trajectory_state[trajectory] != state:
            raise ValueError("question and terminal outcome must be consistent inside a trajectory")
        trajectory_state[trajectory] = state
        identity = (uid, trajectory, role, int(turn))
        if identity not in identities:
            identities[identity] = len(unique)
            unique.append(row)
        ordinal = identities[identity]
        inverse[row] = ordinal
        first = unique[ordinal]
        if first != row:
            for key in facts:
                if not _equal(meta[key][first], meta[key][row]):
                    raise ValueError(f"conflicting padding copies for logical action {identity!r}: {key}")
            if reward_values is not None and reward_values[first] != reward_values[row]:
                raise ValueError(f"conflicting padding copies for logical action {identity!r}: rewards")
    return np.asarray(unique, dtype=np.int64), inverse


def _scope(role: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_") or "role"


def recover_collapsed_advantages(
    meta: Mapping[str, Sequence[Any]],
    rewards: Sequence[float],
    base_advantages: Sequence[float],
    response_lengths: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Recover only truly zero-variance, zero-advantage, same-outcome groups.

    Rewards and base advantages are *actual* training action scalars, after
    any format/reward penalty.  ``response_lengths`` counts actor response-mask
    tokens.  The caller must preserve original rewards and returns, apply pure
    credit only after this function, and pass ``policy_action_weight`` into the
    actor's policy loss (not its independently controlled entropy loss).

    Branch caps bound weighted absolute advantage per real role token even if
    every group collapses.  The bound reserves ``pure_factor_bound`` for the
    subsequent credit multiplier; the caller must enforce that factor bound.
    These are advantage-mass bounds, not claims about gradient-norm bounds.

    Disabled mode bypasses metadata validation and leaves advantages and actor
    weights unchanged.  Enabled mode excludes duplicate padding from all counts
    and policy-loss weights.  No reward, return, or terminal label is modified.
    """
    cfg = _configuration(config)
    original = np.asarray(base_advantages)
    size = len(original)
    arrays = {
        "advantages": original.copy(),
        "kind": np.full(size, "unrecovered", dtype=object),
        "virtual_advantage": np.zeros(size, dtype=np.float64),
        "recovery_weight": np.zeros(size, dtype=np.float64),
        "policy_action_weight": np.ones(size, dtype=np.float64),
    }
    if not cfg["enable"]:
        return arrays, {"advantage_recovery/enabled": 0.0}
    unique, inverse = logical_action_indices(meta, rewards)
    reward = _vector(rewards, size, "rewards")
    base = _vector(base_advantages, size, "base_advantages")
    lengths = _vector(response_lengths, size, "response_lengths")
    if (lengths < 0).any() or (lengths != np.floor(lengths)).any():
        raise ValueError("response_lengths must contain nonnegative integer token counts")
    first_for_row = unique[inverse]
    if not np.array_equal(base, base[first_for_row]):
        raise ValueError("conflicting padding copies: base_advantages")
    if not np.array_equal(lengths, lengths[first_for_row]):
        raise ValueError("conflicting padding copies: response_lengths")
    arrays["advantages"] = base.copy()
    arrays["policy_action_weight"][:] = 0
    live = unique[lengths[unique] > 0]
    arrays["policy_action_weight"][live] = 1
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    role_rows: dict[str, list[int]] = defaultdict(list)
    for row in live:
        role = str(meta["agent_id"][row])
        groups[(str(meta["uid"][row]), role)].append(int(row))
        role_rows[role].append(int(row))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    recovered: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (_, role), row_list in groups.items():
        rows = np.asarray(row_list, dtype=np.int64)
        counter = counts[role]
        counter["groups"] += 1
        total["groups"] += 1
        outcomes = np.asarray([_binary(meta["pass"][row], "pass") for row in rows])
        terminal_kind = "all_success" if outcomes.all() else "all_failure" if not outcomes.any() else "mixed_outcome"
        counter[terminal_kind] += 1
        total[terminal_kind] += 1
        if float(np.std(reward[rows], ddof=0)) > cfg["reward_std_tolerance"]:
            arrays["kind"][rows] = "mixed"
            counter["nonzero_reward_variance"] += 1
            total["nonzero_reward_variance"] += 1
            continue
        counter["zero_variance"] += 1
        total["zero_variance"] += 1
        if (np.abs(base[rows]) > cfg["advantage_zero_tolerance"]).any():
            counter["skipped_nonzero_advantage"] += 1
            continue
        if terminal_kind == "mixed_outcome":
            counter["skipped_mixed_outcome"] += 1
            continue
        branch = "success" if terminal_kind == "all_success" else "failure"
        counter[f"{branch}_zero_variance"] += 1
        total[f"{branch}_zero_variance"] += 1
        trajectories: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            trajectories[str(meta["traj_uid"][row])].append(int(row))
        group_size = len(trajectories)
        if group_size < 2:
            counter["skipped_single_trajectory"] += 1
            continue
        trajectory_rewards = np.asarray([reward[indices].mean() for indices in trajectories.values()])
        virtual = cfg[f"{branch}_virtual_reward"]
        # A format penalty may change the constant reward.  Never produce a
        # success penalty or a failure bonus just because a fixed virtual value
        # is on the wrong side of that actual reward.
        direction_ok = (virtual < trajectory_rewards.min() - cfg["reward_std_tolerance"]
                        if branch == "success" else
                        virtual > trajectory_rewards.max() + cfg["reward_std_tolerance"])
        if not direction_ok:
            counter["skipped_virtual_direction"] += 1
            continue
        augmented = np.append(trajectory_rewards, virtual)
        denominator = float(np.std(augmented, ddof=1)) + cfg["normalization_epsilon"]
        virtual_advantages = (trajectory_rewards - augmented.mean()) / denominator
        group_tokens = float(lengths[rows].sum())
        weight = cfg[f"{branch}_weight"]
        for raw_advantage, trajectory_rows in zip(virtual_advantages, trajectories.values()):
            indices = np.asarray(trajectory_rows, dtype=np.int64)
            arrays["kind"][indices] = branch
            arrays["virtual_advantage"][indices] = raw_advantage
            arrays["recovery_weight"][indices] = weight
            arrays["advantages"][indices] = raw_advantage * weight
            arrays["policy_action_weight"][indices] = group_tokens / (group_size * lengths[indices].sum())
        recovered[(role, branch)].extend(row_list)
        counter[f"{branch}_recovered_groups"] += 1
        total[f"{branch}_recovered_groups"] += 1

    metrics: dict[str, float] = {"advantage_recovery/enabled": 1.0,
                                "advantage_recovery/unique_actions": float(len(unique)),
                                "advantage_recovery/padding_actions": float(size - len(unique))}
    prefix = "advantage_recovery"
    for key, count in total.items():
        metrics[f"{prefix}/{key}"] = float(count)
    groups_count = max(total["groups"], 1)
    for key in ("zero_variance", "all_success", "all_failure", "mixed_outcome",
                "success_zero_variance", "failure_zero_variance"):
        metrics[f"{prefix}/{key}_group_fraction"] = total[key] / groups_count
    for role, rows_list in role_rows.items():
        rows = np.asarray(rows_list, dtype=np.int64)
        role_tokens = float(lengths[rows].sum())
        scope = f"{prefix}/{_scope(role)}"
        for key, count in counts[role].items():
            metrics[f"{scope}/{key}"] = float(count)
        metrics[f"{scope}/real_response_tokens"] = role_tokens
        for branch in ("success", "failure"):
            branch_rows = np.asarray(recovered[(role, branch)], dtype=np.int64)
            token_weights = lengths[branch_rows] * arrays["policy_action_weight"][branch_rows]
            mass_before_cap = float(np.sum(token_weights * np.abs(arrays["advantages"][branch_rows])))
            allowance = cfg[f"{branch}_mass_cap"] * role_tokens / cfg["pure_factor_bound"]
            scale = min(1.0, allowance / mass_before_cap) if mass_before_cap > 0 else 1.0
            arrays["advantages"][branch_rows] *= scale
            arrays["recovery_weight"][branch_rows] *= scale
            mass = mass_before_cap * scale
            metrics[f"{scope}/{branch}_budget_scale"] = scale
            metrics[f"{scope}/{branch}_mass_per_token"] = mass / role_tokens
            metrics[f"{scope}/{branch}_post_pure_mass_bound"] = mass * cfg["pure_factor_bound"] / role_tokens
        metrics[f"{scope}/raw_zero_advantage_token_fraction"] = float(
            np.sum(lengths[rows] * (np.abs(base[rows]) <= cfg["advantage_zero_tolerance"])) / role_tokens)
        metrics[f"{scope}/remaining_zero_advantage_token_fraction"] = float(
            np.sum(lengths[rows] * (np.abs(arrays["advantages"][rows]) <= cfg["advantage_zero_tolerance"])) / role_tokens)
    # Padding carries the corresponding diagnostics/advantage for consistent
    # downstream pure metadata, but never an additional policy-loss weight.
    for key in ("advantages", "kind", "virtual_advantage", "recovery_weight"):
        arrays[key] = arrays[key][first_for_row]
    return arrays, metrics
