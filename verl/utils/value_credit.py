"""Detached entropy-aware prefix-value utilities (standard library and NumPy).

A scorer estimates every chronological prefix, including the empty action
prefix and the final prefix. No terminal label is substituted for a prediction.
Training records carry the terminal label separately from their prefix inputs.
The value model controls a separate entropy penalty; it never replaces or
multiplies the two-stage pure credit coefficients.

Prediction deltas are probability differences, not local correctness labels or
causal entropy effects. Eligibility does not depend on the raw advantage:
the independent actor entropy penalty may act even when that advantage is zero.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np


_RECORD_FIELDS = (
    "value_question", "value_action_text", "value_action_index", "uid",
    "traj_uid", "agent_id", "role_turn_index", "is_action_valid", "pass",
)

# The schema contains action means only: token quantiles cannot be reconstructed
# from historical mean entropy. Every temporal statistic uses completed actions.
FEATURE_SCHEMA = "causal-role-mean-top16-v2"
ABS_FEATURE_NAMES = tuple(f"{role}_{name}" for role in ("solver", "verifier")
                          for name in ("mean", "coverage", "log_tokens", "valid", "truncated", "available"))
TEMP_FEATURE_NAMES = tuple(f"{role}_{name}" for role in ("solver", "verifier")
                           for name in ("previous_mean", "delta", "past_drift", "residual", "log_history", "available"))


def _optional_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("entropy statistics must be numeric or missing")
    return number if math.isfinite(number) else None


def _coverage(value):
    return _optional_float(value.get("coverage")) if isinstance(value, Mapping) else None


def prefix_entropy_features(actions, min_coverage=0.9):
    """Return causal absolute/temporal vectors for h0,...,hL.

    No labels, final length, sibling trajectories or outcome-selected pure gate
    enter this function. Drift is the mean of *earlier* same-role differences;
    the current difference is excluded. Missing actions break temporal edges.
    """
    latest, history, differences = {}, {"solver": [], "verifier": []}, {"solver": [], "verifier": []}
    absolute, temporal, abs_ok, temp_ok = [], [], [], []

    def snapshot():
        a, t = [], []
        for role in ("solver", "verifier"):
            state = latest.get(role)
            if state is None:
                a.extend([0.0] * 6)
                t.extend([0.0] * 6)
            else:
                a.extend(state[0])
                t.extend(state[1])
        absolute.append(a)
        temporal.append(t)
        abs_ok.append(bool(a[5] or a[11]))
        temp_ok.append(bool(t[5] or t[11]))

    snapshot()
    for action in actions:
        role = _role(action["role"])
        entropy = _optional_float(action.get("entropy_mean", action.get("top16_entropy_mean")))
        coverage = _optional_float(action.get("entropy_coverage"))
        count = _optional_float(action.get("entropy_token_count"))
        valid = _binary(action.get("valid", True))
        truncated = _binary(action.get("truncated", False))
        if entropy is not None and not 0 <= entropy <= 1:
            raise ValueError("normalized mean Top-16 entropy must be in [0, 1]")
        if coverage is not None and not 0 <= coverage <= 1:
            raise ValueError("entropy coverage must be in [0, 1]")
        if count is not None and count < 0:
            raise ValueError("entropy token count cannot be negative")
        # A recorded action mean is usable without optional coverage; an
        # explicitly low coverage value is not silently upgraded to reliable.
        available = entropy is not None and valid and not truncated and (coverage is None or coverage >= min_coverage)
        prior = history[role][-1] if history[role] else None
        edge = available and prior is not None
        drift = float(np.mean(differences[role])) if differences[role] else 0.0
        delta = entropy - prior if edge else 0.0
        latest[role] = (
            [entropy if available else 0.0, coverage or 0.0, math.log1p(count or 0),
             float(valid), float(truncated), float(available)],
            [prior if edge else 0.0, delta, drift if edge else 0.0,
             delta - drift if edge else 0.0, math.log1p(len(differences[role])) if edge else 0.0, float(edge)],
        )
        if edge:
            differences[role].append(delta)
        history[role].append(entropy if available else None)
        snapshot()
    return {"absolute": np.asarray(absolute, dtype=np.float32),
            "temporal": np.asarray(temporal, dtype=np.float32),
            "absolute_available": np.asarray(abs_ok, dtype=bool),
            "temporal_available": np.asarray(temp_ok, dtype=bool)}


def _integer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("action and role-turn indices must be nonnegative integers")
    value = int(value)
    if value < 0:
        raise ValueError("action and role-turn indices must be nonnegative integers")
    return value


def _binary(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
        return bool(value)
    raise ValueError("validity and terminal labels must be bool or numeric 0/1")


def _role(value: Any) -> str:
    aliases = {"solver": "solver", "solver agent": "solver",
               "verifier": "verifier", "verifier agent": "verifier"}
    role = aliases.get(str(value).strip().lower())
    if role is None:
        raise ValueError("prefix-value credit only supports Solver and Verifier roles")
    return role


def _row_count(batch: Mapping[str, Sequence[Any]]) -> int:
    for key in ("traj_uid", "uid", "agent_id"):
        if key in batch:
            return len(batch[key])
    return len(next(iter(batch.values()))) if batch else 0


def _collect_trajectories(
    batch: Mapping[str, Sequence[Any]], max_solver_turns: int | None
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Validate whole trajectories before exposing any of their prefixes."""
    size = _row_count(batch)
    metrics: dict[str, float | int] = {
        "value_credit/record_rows": size,
        "value_credit/record_trajectories": 0,
        "value_credit/skipped_trajectories": 0,
        "value_credit/invalid_trajectories": 0,
        "value_credit/padding_rows": 0,
        "value_credit/record_unique_actions": 0,
        "value_credit/missing_metadata": 0,
        "value_credit/budget_mismatch_trajectories": 0,
        "value_credit/incomplete_trajectories": 0,
        "value_credit/budget_checked_trajectories": 0,
    }
    missing = [field for field in _RECORD_FIELDS if field not in batch]
    if missing:
        metrics["value_credit/missing_metadata"] = len(missing)
        if "traj_uid" in batch:
            metrics["value_credit/skipped_trajectories"] = len({str(x) for x in batch["traj_uid"]})
        return [], metrics
    checked_fields = _RECORD_FIELDS + (("value_max_solver_turns",) if "value_max_solver_turns" in batch else ())
    if any(len(batch[field]) != size for field in checked_fields):
        raise ValueError("value trajectory metadata fields have inconsistent lengths")

    groups: dict[str, list[int]] = defaultdict(list)
    for row, trajectory in enumerate(batch["traj_uid"]):
        groups[str(trajectory)].append(row)
    records: list[dict[str, Any]] = []
    for trajectory, rows in groups.items():
        try:
            if batch["traj_uid"][rows[0]] is None or not trajectory:
                raise ValueError("trajectory id is missing")
            budget = max_solver_turns
            if "value_max_solver_turns" in batch:
                logged_budgets = {_integer(batch["value_max_solver_turns"][row]) for row in rows}
                if (len(logged_budgets) != 1 or 0 in logged_budgets
                        or (max_solver_turns is not None and logged_budgets != {max_solver_turns})):
                    metrics["value_credit/budget_mismatch_trajectories"] += 1
                    raise ValueError("logged solver budget is inconsistent with the trajectory or configuration")
                budget = logged_budgets.pop()
            question = batch["value_question"][rows[0]]
            prompt = batch["uid"][rows[0]]
            label = _binary(batch["pass"][rows[0]])
            if not isinstance(question, str) or not question.strip() or prompt is None:
                raise ValueError("question or prompt identity is missing")
            actions: dict[int, dict[str, Any]] = {}
            for row in rows:
                if (batch["value_question"][row] != question
                        or str(batch["uid"][row]) != str(prompt)
                        or _binary(batch["pass"][row]) != label):
                    raise ValueError("question, prompt identity, or label differs inside trajectory")
                index = _integer(batch["value_action_index"][row])
                action = {
                    "role": _role(batch["agent_id"][row]),
                    "text": batch["value_action_text"][row],
                    "valid": _binary(batch["is_action_valid"][row]),
                    "role_turn_index": _integer(batch["role_turn_index"][row]),
                    "entropy_mean": _optional_float(batch.get("top16_entropy_mean", [None] * size)[row]),
                    "entropy_coverage": _coverage(batch.get("top16_entropy", [None] * size)[row]),
                    "entropy_token_count": _optional_float(batch.get("pure_entropy_response_tokens", [None] * size)[row]),
                    "truncated": _binary(batch.get("pure_entropy_truncated", [False] * size)[row]),
                }
                if not isinstance(action["text"], str):
                    raise ValueError("action text is missing")
                if index in actions and actions[index] != action:
                    raise ValueError("conflicting copies of one logical action")
                actions[index] = action
            indices = sorted(actions)
            if indices != list(range(len(actions))):
                raise ValueError("chronological action indices must start at zero without gaps")
            ordered = [actions[index] for index in indices]
            for index, action in enumerate(ordered):
                expected_role = "solver" if index % 2 == 0 else "verifier"
                if action["role"] != expected_role or action["role_turn_index"] != index // 2:
                    raise ValueError("roles or role-turn indices do not follow S,V,S,V chronology")
            if budget is not None and sum(a["role"] == "solver" for a in ordered) > budget:
                metrics["value_credit/budget_mismatch_trajectories"] += 1
                raise ValueError("trajectory exceeds configured solver-turn budget")
            if "value_max_solver_turns" in batch:
                # Replay the actual Math controller stop rule. Format validity
                # does not drive stopping: exact approve wins, exact reject
                # continues, and any unrecognized Verifier output stops.
                metrics["value_credit/budget_checked_trajectories"] += 1
                for index, action in enumerate(ordered):
                    if action["role"] == "solver":
                        terminal = action["role_turn_index"] + 1 == budget
                    else:
                        text = action["text"]
                        terminal = ("<verify>approve</verify>" in text
                                    or "<verify>reject</verify>" not in text)
                    if terminal != (index == len(ordered) - 1):
                        metrics["value_credit/incomplete_trajectories"] += 1
                        raise ValueError("history is truncated or contains actions after controller termination")
            records.append({
                "traj_uid": trajectory,
                "question": question,
                "question_key": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "max_solver_turns": budget,
                "actions": ordered,
                "label": float(label),
            })
            metrics["value_credit/padding_rows"] += len(rows) - len(ordered)
            metrics["value_credit/record_unique_actions"] += len(ordered)
            metrics["value_credit/invalid_trajectories"] += int(any(not a["valid"] for a in ordered))
        except (TypeError, ValueError, OverflowError):
            # Partial, ambiguous, or malformed histories must never masquerade
            # as a valid prefix chain. Other trajectories remain usable.
            metrics["value_credit/skipped_trajectories"] += 1
    metrics["value_credit/record_trajectories"] = len(records)
    return records, metrics


def build_trajectory_records(
    non_tensor_batch: Mapping[str, Sequence[Any]], max_solver_turns: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Reconstruct chronological scorer records; never deduplicate by text.

    Padding is removed by (traj_uid, global action index), with identical
    metadata required for copies. Validity-false actions remain in records so
    the scorer can learn from them; only that action is ineligible for control.
    Structurally incomplete or conflicting
    trajectories are skipped and counted rather than partially reconstructed.
    New logs carrying value_max_solver_turns must match the explicit budget
    and must end exactly where the Math controller would stop. Legacy logs
    without that field retain structural-only validation for compatibility.
    """
    turns = None if max_solver_turns is None else _integer(max_solver_turns)
    if turns == 0:
        raise ValueError("max_solver_turns must be positive")
    return _collect_trajectories(non_tensor_batch, turns)


def attach_value_predictions(non_tensor_batch, values, ready):
    """Attach probability deltas without modifying any pure credit coefficient.

    Predictions may stop at the last complete encodable prefix. An invalid or
    truncated action is masked locally; its text remains in later contexts.
    Each prefix prediction has sem/abs/full and entropy availability flags.
    """
    size = _row_count(non_tensor_batch)
    arrays = {f"value_{branch}_{side}": np.full(size, np.nan)
              for branch in ("sem", "abs", "full") for side in ("before", "after")}
    available, entropy_available = np.zeros(size, bool), np.zeros(size, bool)
    delta, entropy_delta = np.zeros(size), np.zeros(size)
    records, metrics = _collect_trajectories(non_tensor_batch, None)
    record_map = {r["traj_uid"]: r for r in records}
    for row, trajectory in enumerate(non_tensor_batch.get("traj_uid", [])):
        record = record_map.get(str(trajectory))
        prediction = values.get(str(trajectory)) if ready else None
        if record is None or prediction is None:
            continue
        index = int(non_tensor_batch["value_action_index"][row])
        action = record["actions"][index]
        if not action["valid"] or action.get("truncated", False) or index + 1 >= len(prediction):
            continue
        try:
            before, after = prediction[index], prediction[index + 1]
            numbers_ = [float(p[branch]) for p in (before, after) for branch in ("sem", "abs", "full")]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in numbers_):
                continue
            for branch in ("sem", "abs", "full"):
                arrays[f"value_{branch}_before"][row] = before[branch]
                arrays[f"value_{branch}_after"][row] = after[branch]
            available[row] = bool(after.get("action_absolute_available", after.get("absolute_available", False)))
            entropy_available[row] = available[row] and bool(after.get("action_temporal_available", after.get("temporal_available", False)))
            delta[row] = float(after["full"] - before["full"])
            entropy_delta[row] = delta[row] - float(after["abs"] - before["abs"])
        except (KeyError, TypeError, ValueError):
            continue
    non_tensor_batch.update(arrays)
    non_tensor_batch["value_credit_before"] = arrays["value_full_before"]
    non_tensor_batch["value_credit_after"] = arrays["value_full_after"]
    non_tensor_batch["value_credit_delta"] = delta
    non_tensor_batch["value_entropy_delta"] = entropy_delta
    non_tensor_batch["value_credit_available"] = available
    non_tensor_batch["value_absolute_available"] = available.copy()
    non_tensor_batch["value_entropy_available"] = entropy_available
    metrics.update({"value_credit/scorer_ready": int(bool(ready)),
                    "value_credit/prediction_available_rows": int(available.sum()),
                    "value_credit/prediction_coverage": float(available.mean()) if size else 0.0,
                    "value_credit/temporal_coverage": float(entropy_available.mean()) if size else 0.0})
    return metrics


