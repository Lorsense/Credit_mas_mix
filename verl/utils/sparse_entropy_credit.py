"""Sparse entropy-only temporal credit with bounded coefficient-mass calibration.

Preparation reads detached rollout statistics, never a local correctness
checker or an auxiliary model. Finalization uses the *actual* GRPO advantage
sign and changes only connected actions selected by reliable entropy gates.
Weighted coefficient mass is preserved within each active component; this is
not a claim that PPO gradients, gradient norms, or advantage mass are conserved.
"""

from __future__ import annotations

import math
import numbers
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_DEFAULTS = {
    "roles": ["Solver Agent"],
    "min_group_size": 5,
    "shrinkage": 4.0,
    "min_coverage": 0.9,
    "min_residual": 0.002,
    "residual_mad_multiplier": 1.0,
    "min_entropy_spread": 1e-4,
    "min_delta_spread": 1e-4,
    "require_raw_direction": True,
    "correction_threshold": 0.8,
    "regression_threshold": 0.85,
    "eta": 0.04,
    "calibration_weight": "response_tokens",
    "multiplier_min": 0.8,
    "multiplier_max": 1.2,
}
_REQUIRED = (
    "uid", "traj_uid", "agent_id", "role_turn_index", "pass",
    "top16_entropy_mean", "top16_entropy", "is_action_valid",
    "pure_entropy_response_tokens", "pure_entropy_truncated",
)


def _integer(value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("turn indices, group sizes, and response-token counts must be integers")
    result = int(value)
    if result < minimum:
        raise ValueError(f"integer value must be at least {minimum}")
    return result


def _binary(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0, 1):
        return bool(value)
    raise ValueError("outcome, validity, and truncation flags must be bool or numeric 0/1")


def _configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: config.get(key, default) for key, default in _DEFAULTS.items()}
    roles = result["roles"]
    if isinstance(roles, str) or not isinstance(roles, Sequence) or not all(isinstance(x, str) for x in roles):
        raise ValueError("roles must be a sequence of role names")
    result["roles"] = frozenset(roles)
    result["min_group_size"] = _integer(result["min_group_size"], minimum=2)
    result["require_raw_direction"] = _binary(result["require_raw_direction"])
    numeric = ("shrinkage", "min_coverage", "min_residual", "residual_mad_multiplier",
               "min_entropy_spread", "min_delta_spread", "correction_threshold",
               "regression_threshold", "eta", "multiplier_min", "multiplier_max")
    for key in numeric:
        result[key] = float(result[key])
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    for key in ("shrinkage", "min_residual", "residual_mad_multiplier",
                "min_entropy_spread", "min_delta_spread"):
        if result[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not 0 <= result["min_coverage"] <= 1 or not 0 <= result["eta"] <= 1:
        raise ValueError("min_coverage and eta must be in [0,1]")
    if not all(0 <= result[key] < 1 for key in ("correction_threshold", "regression_threshold")):
        raise ValueError("gate score thresholds must be in [0,1)")
    if not 0 < result["multiplier_min"] <= 1 <= result["multiplier_max"]:
        raise ValueError("positive multiplier bounds must straddle one")
    if result["calibration_weight"] not in ("response_tokens", "uniform"):
        raise ValueError("calibration_weight must be response_tokens or uniform")
    return result


def validate_sparse_entropy_config(config: Mapping[str, Any]) -> None:
    """Validate startup configuration without requiring any rollout metadata."""
    _configuration(config)


def _optional_unit_float(value: Any, name: str) -> float:
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or missing") from exc
    if math.isnan(number):
        return number
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"finite {name} must be in [0,1]")
    return number


def _read_metadata(meta: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    missing = set(_REQUIRED).difference(meta)
    if missing:
        raise KeyError(f"sparse entropy credit is missing fields: {sorted(missing)}")
    lengths = {len(meta[key]) for key in _REQUIRED}
    if len(lengths) != 1:
        raise ValueError("sparse entropy metadata fields have inconsistent lengths")
    size = lengths.pop()
    for key in ("uid", "traj_uid", "agent_id"):
        for value in meta[key]:
            if (value is None or not str(value).strip()
                    or (isinstance(value, numbers.Real) and not math.isfinite(float(value)))):
                raise ValueError(f"{key} must contain nonempty finite identities")
    columns = {
        "prompt": np.asarray([str(x) for x in meta["uid"]], dtype=object),
        "trajectory": np.asarray([str(x) for x in meta["traj_uid"]], dtype=object),
        "role": np.asarray([str(x) for x in meta["agent_id"]], dtype=object),
        "turn": np.asarray([_integer(x) for x in meta["role_turn_index"]], dtype=np.int64),
        "success": np.asarray([_binary(x) for x in meta["pass"]], dtype=bool),
        "valid": np.asarray([_binary(x) for x in meta["is_action_valid"]], dtype=bool),
        "tokens": np.asarray([_integer(x) for x in meta["pure_entropy_response_tokens"]], dtype=np.int64),
        "truncated": np.asarray([_binary(x) for x in meta["pure_entropy_truncated"]], dtype=bool),
        "entropy": np.asarray([_optional_unit_float(x, "entropy") for x in meta["top16_entropy_mean"]]),
        "coverage": np.asarray([
            _optional_unit_float(x.get("coverage") if isinstance(x, Mapping) else None, "coverage")
            for x in meta["top16_entropy"]
        ]),
    }
    first_of_row = np.empty(size, dtype=np.int64)
    identities: dict[tuple[str, str, int], int] = {}
    trajectory_state: dict[str, tuple[str, bool]] = {}
    for row in range(size):
        trajectory = columns["trajectory"][row]
        state = (columns["prompt"][row], bool(columns["success"][row]))
        if trajectory in trajectory_state and trajectory_state[trajectory] != state:
            raise ValueError("prompt and terminal outcome must be consistent inside a trajectory")
        trajectory_state[trajectory] = state
        identity = (trajectory, columns["role"][row], int(columns["turn"][row]))
        first = identities.setdefault(identity, row)
        first_of_row[row] = first
        if first != row:
            for name, values in columns.items():
                a, b = values[first], values[row]
                equal = bool(a == b) or (name in ("entropy", "coverage") and np.isnan(a) and np.isnan(b))
                if not equal:
                    raise ValueError(f"conflicting padding copies for logical action {identity!r}: {name}")
    columns.update(size=size, first=first_of_row,
                   unique=np.asarray(list(identities.values()), dtype=np.int64), identities=identities)
    return columns


def _midranks(values: np.ndarray) -> np.ndarray:
    """Ascending mid-percentiles; ties share their average zero-based rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (0.5 * (start + end - 1) + 0.5) / len(values)
        start = end
    return ranks


def _scope_key(role: str, turn: int) -> str:
    return re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_") + f"/turn_{turn}"


def _summarize(metrics: dict, prefix: str, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite):
        for name, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            metrics[f"{prefix}/{name}"] = float(fn(finite))


def prepare_sparse_entropy_credit(
    non_tensor_batch: Mapping[str, Sequence[Any]], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Prepare unsigned correction/regression gates before GRPO or padding.

    Same-role transitions are grouped by (prompt, role, current role turn).
    Subtracting a common drift does not affect percentile ranks; the absolute
    residual threshold is deliberately retained so drift affects activation.
    No tail quota is enforced. Low sample counts or tiny spreads remain neutral.
    """
    cfg = _configuration(config)
    data = _read_metadata(non_tensor_batch)
    size, unique = data["size"], data["unique"]
    prepared = {
        "pure_entropy_pair_observed": np.zeros(size, dtype=bool),
        "pure_entropy_pair_eligible": np.zeros(size, dtype=bool),
        "pure_entropy_pair_reliable": np.zeros(size, dtype=bool),
        "pure_entropy_prev_turn": np.full(size, -1, dtype=np.int64),
        "pure_entropy_group_size": np.zeros(size, dtype=np.int64),
    }
    for key in ("delta", "drift", "residual", "residual_threshold",
                "correction_score", "regression_score"):
        prepared["pure_entropy_" + key] = np.full(size, np.nan, dtype=np.float64)
    for key in ("correction_gate", "regression_gate"):
        prepared["pure_entropy_" + key] = np.zeros(size, dtype=np.float64)
    good = (data["valid"] & ~data["truncated"] & (data["tokens"] > 0)
            & np.isfinite(data["entropy"]) & np.isfinite(data["coverage"])
            & (data["coverage"] >= cfg["min_coverage"]))
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    pools: dict[tuple[str, int], list[int]] = defaultdict(list)
    previous: dict[int, int] = {}
    for row in unique:
        role, turn = data["role"][row], int(data["turn"][row])
        if role not in cfg["roles"] or turn == 0:
            continue
        prior = data["identities"].get((data["trajectory"][row], role, turn - 1))
        if prior is None:
            continue
        prepared["pure_entropy_pair_observed"][row] = True
        prepared["pure_entropy_prev_turn"][row] = turn - 1
        if not (good[row] and good[prior]):
            continue
        previous[row] = prior
        prepared["pure_entropy_pair_eligible"][row] = True
        prepared["pure_entropy_delta"][row] = data["entropy"][row] - data["entropy"][prior]
        groups[(data["prompt"][row], role, turn)].append(int(row))
        pools[(role, turn)].append(int(row))

    small_groups = low_delta_groups = 0
    unreachable_correction = unreachable_regression = 0
    delta = prepared["pure_entropy_delta"]
    for (_, role, turn), rows in groups.items():
        changes = delta[rows]
        local = float(np.median(changes))
        pooled = float(np.median(delta[pools[(role, turn)]]))
        amount = len(rows) / (len(rows) + cfg["shrinkage"])
        drift = amount * local + (1.0 - amount) * pooled
        residual = changes - drift
        threshold = max(cfg["min_residual"], cfg["residual_mad_multiplier"] * 1.4826
                        * float(np.median(np.abs(changes - local))))
        prepared["pure_entropy_group_size"][rows] = len(rows)
        prepared["pure_entropy_drift"][rows] = drift
        prepared["pure_entropy_residual"][rows] = residual
        prepared["pure_entropy_residual_threshold"][rows] = threshold
        previous_entropy = data["entropy"][[previous[row] for row in rows]]
        current_entropy = data["entropy"][rows]
        plus_score = _midranks(previous_entropy) * _midranks(-residual)
        minus_score = _midranks(current_entropy) * _midranks(residual)
        prepared["pure_entropy_correction_score"][rows] = plus_score
        prepared["pure_entropy_regression_score"][rows] = minus_score
        maximum_score = (1.0 - 0.5 / len(rows)) ** 2
        unreachable_correction += int(maximum_score <= cfg["correction_threshold"])
        unreachable_regression += int(maximum_score <= cfg["regression_threshold"])
        enough = len(rows) >= cfg["min_group_size"]
        delta_spread = float(np.ptp(changes))
        varied = delta_spread > 0 and delta_spread >= cfg["min_delta_spread"]
        small_groups += int(not enough)
        low_delta_groups += int(not varied)
        if not (enough and varied):
            continue
        prepared["pure_entropy_pair_reliable"][rows] = True
        plus_reliable = ((residual < -threshold)
                         & (np.ptp(previous_entropy) > 0)
                         & (np.ptp(previous_entropy) >= cfg["min_entropy_spread"]))
        minus_reliable = ((residual > threshold)
                          & (np.ptp(current_entropy) > 0)
                          & (np.ptp(current_entropy) >= cfg["min_entropy_spread"]))
        if cfg["require_raw_direction"]:
            plus_reliable &= changes < 0
            minus_reliable &= changes > 0
        for branch, score, reliable in (("correction", plus_score, plus_reliable),
                                        ("regression", minus_score, minus_reliable)):
            cutoff = cfg[branch + "_threshold"]
            prepared[f"pure_entropy_{branch}_gate"][rows] = (
                np.clip((score - cutoff) / (1.0 - cutoff), 0.0, 1.0) * reliable)

    metrics = {
        "pure_entropy/unique_actions": float(len(unique)),
        "pure_entropy/padding_copies": float(size - len(unique)),
        "pure_entropy/observed_pairs": float(prepared["pure_entropy_pair_observed"][unique].sum()),
        "pure_entropy/eligible_pairs": float(prepared["pure_entropy_pair_eligible"][unique].sum()),
        "pure_entropy/reliable_pairs": float(prepared["pure_entropy_pair_reliable"][unique].sum()),
        "pure_entropy/calibration_groups": float(len(groups)),
        "pure_entropy/small_groups": float(small_groups),
        "pure_entropy/low_delta_spread_groups": float(low_delta_groups),
        "pure_entropy/invalid_actions": float((~data["valid"][unique]).sum()),
        "pure_entropy/truncated_actions": float(data["truncated"][unique].sum()),
        "pure_entropy/zero_response_actions": float((data["tokens"][unique] == 0).sum()),
        "pure_entropy/missing_entropy_actions": float((~np.isfinite(data["entropy"][unique])).sum()),
        "pure_entropy/missing_coverage_actions": float((~np.isfinite(data["coverage"][unique])).sum()),
        "pure_entropy/low_coverage_actions": float((np.isfinite(data["coverage"][unique])
                                                  & (data["coverage"][unique] < cfg["min_coverage"])).sum()),
        "pure_entropy/group_size_min": float(min(map(len, groups.values()), default=0)),
        "pure_entropy/group_size_max": float(max(map(len, groups.values()), default=0)),
        "pure_entropy/correction_threshold_unreachable_groups": float(unreachable_correction),
        "pure_entropy/regression_threshold_unreachable_groups": float(unreachable_regression),
    }
    observed_count = max(metrics["pure_entropy/observed_pairs"], 1)
    for branch in ("correction", "regression"):
        gate = prepared[f"pure_entropy_{branch}_gate"]
        metrics[f"pure_entropy/{branch}_gate_coverage"] = float(np.count_nonzero(gate[unique]) / observed_count)
        observed = unique[prepared["pure_entropy_pair_observed"][unique]]
        _summarize(metrics, f"pure_entropy/{branch}_gate", gate[observed])
    observed_scopes: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in unique[prepared["pure_entropy_pair_observed"][unique]]:
        observed_scopes[(data["role"][row], int(data["turn"][row]))].append(int(row))
    for (role, turn), rows in observed_scopes.items():
        scope = "pure_entropy/" + _scope_key(role, turn)
        metrics[scope + "/observed_pairs"] = float(len(rows))
        metrics[scope + "/eligible_pairs"] = float(prepared["pure_entropy_pair_eligible"][rows].sum())
        for branch in ("correction", "regression"):
            metrics[scope + f"/{branch}_gate_coverage"] = float(
                np.mean(prepared[f"pure_entropy_{branch}_gate"][rows] > 0))
            _summarize(metrics, scope + f"/{branch}_gate", prepared[f"pure_entropy_{branch}_gate"][rows])
    for key in ("delta", "drift", "residual", "residual_threshold", "correction_score", "regression_score"):
        _summarize(metrics, "pure_entropy/" + key, prepared["pure_entropy_" + key][unique])
    return {key: values[data["first"]].copy() for key, values in prepared.items()}, metrics


def _prepared_arrays(meta: Mapping[str, Sequence[Any]], data: dict) -> dict[str, np.ndarray]:
    fields = ("pair_eligible", "pair_reliable", "prev_turn", "correction_gate", "regression_gate")
    result = {}
    for key in fields:
        name = "pure_entropy_" + key
        if name not in meta:
            raise KeyError(f"prepare_sparse_entropy_credit must run before finalization: {name}")
        values = np.asarray(meta[name])
        if values.shape != (data["size"],):
            raise ValueError(f"{name} has an inconsistent shape")
        if key in ("pair_eligible", "pair_reliable"):
            values = np.asarray([_binary(x) for x in values], dtype=bool)
        elif key == "prev_turn":
            values = np.asarray([_integer(x, minimum=-1) for x in values], dtype=np.int64)
        else:
            values = values.astype(np.float64)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("prepared gate values must be finite and in [0,1]")
        if not np.array_equal(values, values[data["first"]], equal_nan=True):
            raise ValueError(f"conflicting padding copies of prepared statistic {name}")
        result[key] = values
    return result


def _calibrate_component(c1: np.ndarray, u: np.ndarray, weights: np.ndarray,
                         lower: float, upper: float) -> tuple[np.ndarray, float, float]:
    """Solve a bounded monotone scalar projection of coefficient mass."""
    target = float(np.dot(weights, c1))
    lograw = np.log(c1) + u
    left = float(np.min(lograw - math.log(upper))) - 1.0
    right = float(np.max(lograw - math.log(lower))) + 1.0
    for _ in range(90):
        middle = (left + right) * 0.5
        candidate = np.clip(np.exp(lograw - middle), lower, upper)
        if float(np.dot(weights, candidate)) > target:
            left = middle
        else:
            right = middle
    raw = np.exp(lograw - (left + right) * 0.5)
    result = np.clip(raw, lower, upper)
    clipped_fraction = float(np.mean((raw < lower) | (raw > upper)))
    error = abs(float(np.dot(weights, result)) - target) / max(abs(target), 1e-15)
    return result, clipped_fraction, error


def finalize_sparse_entropy_credit(
    non_tensor_batch: Mapping[str, Sequence[Any]], base_advantages: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Use actual advantage signs, then transfer within active chain components.

    Success requires both neighboring advantages to be positive; failure
    requires both to be negative. Zero, mixed, or outcome-disagreeing signs are
    neutral. Only nonzero selected edges create calibration components.
    c1 remains bit-for-bit unchanged outside those components and when eta=0.
    """
    cfg = _configuration(config)
    data = _read_metadata(non_tensor_batch)
    prepared = _prepared_arrays(non_tensor_batch, data)
    size, unique = data["size"], data["unique"]
    c1 = np.asarray(non_tensor_batch["entropy_credit_action_multiplier"], dtype=np.float64)
    advantages = np.asarray(base_advantages, dtype=np.float64)
    if c1.shape != (size,) or advantages.shape != (size,):
        raise ValueError("c1 and base_advantages must have one scalar per row")
    if not np.isfinite(c1).all() or not np.isfinite(advantages).all():
        raise ValueError("c1 and base_advantages must be finite")
    lower, upper = cfg["multiplier_min"], cfg["multiplier_max"]
    if np.any(c1 < lower) or np.any(c1 > upper):
        raise ValueError("baseline stage-one coefficients must lie within final bounds")
    for values in (c1, advantages):
        if not np.array_equal(values, values[data["first"]]):
            raise ValueError("conflicting padding copies of c1 or base advantages")

    selected = np.zeros(size, dtype=np.float64)
    transfer = np.zeros(size, dtype=np.float64)
    u = np.zeros(size, dtype=np.float64)
    component = np.full(size, -1, dtype=np.int64)
    parent: dict[int, int] = {}

    def find(row: int) -> int:
        parent.setdefault(row, row)
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    candidates = mismatches = active_edges = conflicts = 0
    active_by_scope: dict[str, int] = defaultdict(int)
    for row in unique:
        branch = "correction" if data["success"][row] else "regression"
        gate = float(prepared[branch + "_gate"][row])
        if gate == 0:
            continue
        candidates += 1
        role, turn = data["role"][row], int(data["turn"][row])
        prior = data["identities"].get((data["trajectory"][row], role, turn - 1))
        if (role not in cfg["roles"] or prior is None or prepared["prev_turn"][row] != turn - 1
                or not prepared["pair_eligible"][row] or not prepared["pair_reliable"][row]):
            raise ValueError("nonzero prepared gate has no eligible adjacent same-role action")
        aligned = (advantages[prior] > 0 and advantages[row] > 0) if data["success"][row] else (
            advantages[prior] < 0 and advantages[row] < 0)
        if not aligned:
            mismatches += 1
            continue
        amount = cfg["eta"] * gate
        if amount == 0:
            continue
        selected[row], transfer[row] = gate, amount
        u[prior] -= amount
        u[row] += amount
        parent[find(row)] = find(prior)
        active_edges += 1
        conflicts += int(c1[row] < c1[prior])
        active_by_scope[_scope_key(role, turn)] += 1

    members: dict[int, list[int]] = defaultdict(list)
    for row in parent:
        members[find(row)].append(row)
    ordered_components = sorted(members.values(), key=lambda rows: min(
        (data["trajectory"][row], data["role"][row], int(data["turn"][row])) for row in rows))
    final = c1.copy()
    clipped_actions = 0.0
    mass_errors = []
    for component_id, rows in enumerate(ordered_components):
        rows.sort(key=lambda row: int(data["turn"][row]))
        weights = (data["tokens"][rows].astype(np.float64)
                   if cfg["calibration_weight"] == "response_tokens" else np.ones(len(rows)))
        if np.any(weights <= 0):
            raise ValueError("active component must have positive calibration weights")
        result, clipped, error = _calibrate_component(c1[rows], u[rows], weights, lower, upper)
        final[rows] = result
        component[rows] = component_id
        clipped_actions += clipped * len(rows)
        mass_errors.append(error)
    final = final[data["first"]]
    active_unique = unique[component[unique] >= 0]
    metadata = {
        "pure_entropy_selected_gate": selected[data["first"]].copy(),
        "pure_entropy_edge_transfer": transfer[data["first"]].copy(),
        "pure_entropy_u": u[data["first"]].copy(),
        "pure_entropy_active_component": component[data["first"]].copy(),
        "pure_entropy_base_advantage": advantages.copy(),
        "pure_entropy_response_tokens": data["tokens"].copy(),
        "pure_entropy_truncated": data["truncated"].copy(),
    }
    metrics = {
        "pure_entropy/final_unique_actions": float(len(unique)),
        "pure_entropy/final_padding_copies": float(size - len(unique)),
        "pure_entropy/candidate_edges": float(candidates),
        "pure_entropy/sign_mismatch_edges": float(mismatches),
        "pure_entropy/sign_mismatch_fraction": mismatches / max(candidates, 1),
        "pure_entropy/active_edges": float(active_edges),
        "pure_entropy/active_components": float(len(ordered_components)),
        "pure_entropy/active_actions": float(len(active_unique)),
        "pure_entropy/active_action_fraction": len(active_unique) / max(len(unique), 1),
        "pure_entropy/firststage_conflict_fraction": conflicts / max(active_edges, 1),
        "pure_entropy/clipped_action_fraction": clipped_actions / max(len(active_unique), 1),
        "pure_entropy/bound_touch_fraction": float(np.mean(
            np.isclose(final[active_unique], lower, rtol=0, atol=1e-12)
            | np.isclose(final[active_unique], upper, rtol=0, atol=1e-12))) if len(active_unique) else 0.0,
        "pure_entropy/weighted_coefficient_mass_error_max": max(mass_errors, default=0.0),
        "pure_entropy/final_changed_actions": float(np.count_nonzero(final[unique] != c1[unique])),
        "pure_entropy/final_mean_abs_change": float(np.mean(np.abs(final[unique] - c1[unique]))) if len(unique) else 0.0,
    }
    observed_scopes = {
        _scope_key(data["role"][row], int(data["turn"][row])) for row in unique
        if data["role"][row] in cfg["roles"] and int(data["turn"][row]) > 0
        and (data["trajectory"][row], data["role"][row], int(data["turn"][row]) - 1) in data["identities"]
    }
    for scope in observed_scopes:
        metrics["pure_entropy/" + scope + "/active_edges"] = float(active_by_scope[scope])
    _summarize(metrics, "pure_entropy/active_u", u[active_unique])
    return {
        "final_multipliers": final,
        "trajectory_multipliers": final / c1,
        "metadata": metadata,
        "metrics": metrics,
    }
