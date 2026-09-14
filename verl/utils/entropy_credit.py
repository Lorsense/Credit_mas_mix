"""Rollout Top-K entropy statistics and detached multi-agent credit factors.

The functions in this module intentionally use only NumPy and the Python
standard library.  Rollout workers reduce per-token Top-K distributions to a
compact action statistic, and the trainer builds scalar credit factors from
that detached metadata.  No value produced here participates in autograd.
"""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np


def _coerce_logprob(value: Any) -> Optional[float]:
    """Extract a scalar log-probability from common SGLang/vLLM formats."""

    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "logprob"):
        try:
            return float(value.logprob)
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        for key in ("logprob", "log_prob", "value"):
            if key in value:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(value, np.ndarray) and value.size:
        try:
            return float(value.flat[0])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (tuple, list)) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    return None


def _position_logprobs(position: Any, top_k: int) -> Optional[np.ndarray]:
    if position is None:
        return None
    entries = position.values() if isinstance(position, Mapping) else position
    try:
        values = [_coerce_logprob(entry) for entry in entries]
    except TypeError:
        return None
    values = [value for value in values if value is not None]
    if len(values) < top_k:
        return None

    # Some backends may include the sampled token in addition to the requested
    # Top-K candidates.  Sorting and retaining exactly K keeps the definition
    # backend-independent.
    values = sorted(values, reverse=True)[:top_k]
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).any() or np.isposinf(array).any() or not np.isfinite(array).any():
        return None
    return array


def compute_action_topk_entropy(
    top_logprobs_by_position: Any,
    response_mask: Optional[Sequence[int | bool]] = None,
    *,
    top_k: int = 16,
) -> Optional[dict[str, float | int]]:
    """Compute normalized conditional Top-K entropy for one generated action.

    Candidate log-probabilities at every valid response position are
    re-normalized inside the Top-K set, then entropy is divided by ``log(K)``.
    Positions with fewer than K usable candidates are skipped.  Returning
    ``None`` means the action has no complete Top-K position; it must not be
    confused with a genuinely zero-entropy action.
    """

    if top_k < 2:
        raise ValueError("top_k must be at least 2")
    if top_logprobs_by_position is None:
        return None
    try:
        positions = list(top_logprobs_by_position)
    except TypeError:
        return None

    if response_mask is None:
        mask = [True] * len(positions)
    else:
        mask = [bool(value) for value in response_mask]
    valid_response_tokens = int(sum(mask))
    token_entropies: list[float] = []
    log_k = math.log(top_k)

    for position_index, position in enumerate(positions):
        if position_index >= len(mask) or not mask[position_index]:
            continue
        logprobs = _position_logprobs(position, top_k)
        if logprobs is None:
            continue
        finite = np.isfinite(logprobs)
        max_logprob = float(np.max(logprobs[finite]))
        weights = np.exp(logprobs - max_logprob)
        normalizer = float(weights.sum())
        if not math.isfinite(normalizer) or normalizer <= 0:
            continue
        probabilities = weights / normalizer
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive]))) / log_k
        token_entropies.append(float(np.clip(entropy, 0.0, 1.0)))

    if not token_entropies:
        return None

    values = np.asarray(token_entropies, dtype=np.float64)
    mean = float(values.mean())
    num_tokens = int(values.size)
    return {
        "mean": mean,
        "std": float(values.std(ddof=0)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "num_tokens": num_tokens,
        "effective_support": float(top_k**mean),
        "valid_response_tokens": valid_response_tokens,
        "coverage": float(num_tokens / valid_response_tokens) if valid_response_tokens else 0.0,
    }


def _average_rank_scores(values: Sequence[float]) -> np.ndarray:
    """Map ascending average ranks to [-1, 1], sharing ranks across ties."""

    array = np.asarray(values, dtype=np.float64)
    size = int(array.size)
    if size <= 1 or np.all(array == array[0]):
        return np.zeros(size, dtype=np.float64)

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(size, dtype=np.float64)
    start = 0
    while start < size:
        end = start + 1
        while end < size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return 2.0 * ranks / (size - 1) - 1.0


def first_unique_action_indices(
    trajectory_ids: Sequence[Any],
    agent_ids: Sequence[Any],
    role_turn_indices: Sequence[Any],
) -> np.ndarray:
    """Return first occurrences of logical actions after batch padding/copying.

    ``adjust_batch`` can append exact row copies to satisfy distributed batch
    divisors.  Those copies are needed for training but must not appear as new
    actions in trajectory JSONL exports.
    """

    lengths = {len(trajectory_ids), len(agent_ids), len(role_turn_indices)}
    if len(lengths) != 1:
        raise ValueError(f"action identity fields have inconsistent lengths: {sorted(lengths)}")

    seen: set[tuple[str, str, int]] = set()
    keep: list[int] = []
    for row, (trajectory_id, agent_id, turn_index) in enumerate(
        zip(trajectory_ids, agent_ids, role_turn_indices)
    ):
        if isinstance(turn_index, (bool, np.bool_)) or not isinstance(turn_index, numbers.Integral):
            raise ValueError("role_turn_index values must be integers when exporting actions")
        identity = (str(trajectory_id), str(agent_id), int(turn_index))
        if identity not in seen:
            seen.add(identity)
            keep.append(row)
    return np.asarray(keep, dtype=np.int64)


def compute_entropy_credit_multipliers(
    *,
    prompt_group_ids: Sequence[Any],
    trajectory_ids: Sequence[Any],
    agent_ids: Sequence[Any],
    role_turn_indices: Sequence[Any],
    terminal_success: Sequence[bool | int | float],
    action_entropies: Sequence[Optional[float] | Mapping[str, Any]],
    action_valid: Optional[Sequence[bool]] = None,
    action_scale: float = 0.2,
    trajectory_scale: float = 0.1,
    trajectory_deadzone: float = 0.05,
    multiplier_min: float = 0.8,
    multiplier_max: float = 1.2,
    final_multiplier_min: float = 0.8,
    final_multiplier_max: float = 1.2,
    relative_epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Build the two entropy-credit factors and their third-clipped product.

    Stage 1 ranks all valid actions in each
    ``(prompt, role, terminal outcome)`` group.  Stage 2 never compares other
    trajectories or roles: every non-first action is compared only with the
    immediately preceding action of the same role in the same trajectory.
    """

    fields = (
        prompt_group_ids,
        trajectory_ids,
        agent_ids,
        role_turn_indices,
        terminal_success,
        action_entropies,
    )
    if action_valid is not None:
        fields = (*fields, action_valid)
    lengths = {len(field) for field in fields}
    if len(lengths) != 1:
        raise ValueError(f"entropy-credit fields have inconsistent lengths: {sorted(lengths)}")
    size = lengths.pop()
    if not 0 <= action_scale <= 1 or not 0 <= trajectory_scale <= 1:
        raise ValueError("entropy-credit scales must be in [0, 1]")
    if trajectory_deadzone < 0:
        raise ValueError("trajectory_deadzone must be non-negative")
    if multiplier_min <= 0 or multiplier_min > 1 or multiplier_max < 1:
        raise ValueError("individual multiplier bounds must straddle 1 and stay positive")
    if final_multiplier_min <= 0 or final_multiplier_min > 1 or final_multiplier_max < 1:
        raise ValueError("final multiplier bounds must straddle 1 and stay positive")
    numeric_config = (
        action_scale,
        trajectory_scale,
        trajectory_deadzone,
        multiplier_min,
        multiplier_max,
        final_multiplier_min,
        final_multiplier_max,
        relative_epsilon,
    )
    if not all(math.isfinite(value) for value in numeric_config):
        raise ValueError("entropy-credit scales, bounds, deadzone, and epsilon must be finite")
    if relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be positive")

    prompts = np.asarray(prompt_group_ids, dtype=object)
    trajectories = np.asarray(trajectory_ids, dtype=object)
    roles = np.asarray(agent_ids, dtype=object)

    turns = np.empty(size, dtype=np.int64)
    for row, value in enumerate(role_turn_indices):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("role_turn_index values must be non-negative integers")
        turns[row] = int(value)
        if turns[row] < 0:
            raise ValueError("role_turn_index values must be non-negative integers")

    success_values: list[bool] = []
    for value in terminal_success:
        if isinstance(value, (bool, np.bool_)):
            success_values.append(bool(value))
        elif isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
            success_values.append(bool(value))
        else:
            raise ValueError("terminal_success values must be bool or numeric 0/1")
    successes = np.asarray(success_values, dtype=bool)
    valid = (
        np.ones(size, dtype=bool)
        if action_valid is None
        else np.asarray(action_valid, dtype=bool)
    )
    if valid.shape != (size,):
        raise ValueError(f"action_valid must have shape ({size},), got {valid.shape}")

    entropies = np.empty(size, dtype=np.float64)
    for row, value in enumerate(action_entropies):
        if isinstance(value, Mapping):
            value = value.get("mean")
        try:
            entropies[row] = float(value)
        except (TypeError, ValueError):
            entropies[row] = np.nan
        if math.isfinite(entropies[row]) and not 0.0 <= entropies[row] <= 1.0:
            raise ValueError("finite action entropies must be normalized to [0, 1]")
    entropy_valid = valid & np.isfinite(entropies)

    action_multiplier = np.ones(size, dtype=np.float64)
    trajectory_multiplier = np.ones(size, dtype=np.float64)
    relative_change = np.full(size, np.nan, dtype=np.float64)

    # Stage 1: cross-trajectory action-level credit, separated by role and
    # terminal correctness.  Multiple same-role actions from one trajectory
    # remain separate members of the set by design.
    action_groups: dict[tuple[Any, Any, bool], list[int]] = defaultdict(list)
    for row in np.flatnonzero(entropy_valid):
        action_groups[(prompts[row], roles[row], bool(successes[row]))].append(int(row))
    for (_, _, success), rows_in_group in action_groups.items():
        rank_scores = _average_rank_scores(entropies[rows_in_group])
        outcome_direction = 1.0 if success else -1.0
        raw = 1.0 + action_scale * outcome_direction * rank_scores
        action_multiplier[rows_in_group] = np.clip(raw, multiplier_min, multiplier_max)

    # Stage 2: local, same-role, immediately preceding action only.
    trajectory_role_rows: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    for row in range(size):
        trajectory_role_rows[(prompts[row], trajectories[row], roles[row])].append(row)

    for trajectory_key, rows_in_group in trajectory_role_rows.items():
        outcomes = {bool(successes[row]) for row in rows_in_group}
        if len(outcomes) != 1:
            raise ValueError(f"terminal success is inconsistent inside trajectory-role {trajectory_key!r}")
        outcome_direction = 1.0 if outcomes.pop() else -1.0
        rows_in_group.sort(key=lambda row: turns[row])
        ordered_turns = [int(turns[row]) for row in rows_in_group]
        if len(set(ordered_turns)) != len(ordered_turns):
            raise ValueError(f"duplicate role_turn_index inside trajectory-role {trajectory_key!r}")
        if ordered_turns != list(range(len(ordered_turns))):
            raise ValueError(
                f"role_turn_index must be contiguous from zero inside trajectory-role {trajectory_key!r}; "
                f"got {ordered_turns}"
            )

        # The first action of every role intentionally keeps stage 2 neutral.
        for offset in range(1, len(rows_in_group)):
            previous_row = rows_in_group[offset - 1]
            current_row = rows_in_group[offset]
            previous_entropy = entropies[previous_row]
            current_entropy = entropies[current_row]
            if not (
                valid[previous_row]
                and valid[current_row]
                and math.isfinite(previous_entropy)
                and math.isfinite(current_entropy)
            ):
                continue
            denominator = current_entropy + previous_entropy + relative_epsilon
            change = float(np.clip((current_entropy - previous_entropy) / denominator, -1.0, 1.0))
            relative_change[current_row] = change
            if change > trajectory_deadzone:
                change_direction = 1.0
            elif change < -trajectory_deadzone:
                change_direction = -1.0
            else:
                change_direction = 0.0
            raw = 1.0 + trajectory_scale * outcome_direction * change_direction
            trajectory_multiplier[current_row] = float(np.clip(raw, multiplier_min, multiplier_max))

    # Stage 3: independently clip the product once more, so two individually
    # valid factors can never change the original advantage beyond [0.8, 1.2].
    final_multiplier = np.clip(
        action_multiplier * trajectory_multiplier,
        final_multiplier_min,
        final_multiplier_max,
    )
    action_multiplier[~valid] = 1.0
    trajectory_multiplier[~valid] = 1.0
    final_multiplier[~valid] = 1.0

    return {
        "action": action_multiplier,
        "trajectory": trajectory_multiplier,
        "final": final_multiplier,
        "relative_change": relative_change,
        "entropy_valid": entropy_valid,
    }
