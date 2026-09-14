"""Read-only adapters for historical trajectory files; never expose rewards as features."""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path


def input_files(pattern, base):
    path = Path(pattern)
    if not path.is_absolute():
        path = Path(base) / path
    if path.is_dir():
        return sorted(p.resolve() for p in path.rglob("*")
                      if p.is_file() and p.suffix.lower() in (".json", ".jsonl"))
    return sorted(Path(p).resolve() for p in glob.glob(str(path), recursive=True) if Path(p).is_file())


def json_documents(path):
    """Accept pretty-printed JSON or JSONL, rejecting incomplete pasted fragments."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        return
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON/JSONL; provide complete files, not truncated excerpts") from error
            if not isinstance(document, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield number, document
    else:
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a JSON object or JSONL objects")
        yield 1, document


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _boolean(value, name):
    if isinstance(value, (bool, int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be boolean or 0/1")


def baseline_action_rows(document, source, fallback_budget=None, *, deduplication=None):
    """Expand a global_step -> trajectories -> steps baseline export.

    Only original question, causal action text/entropy/validity, identities and
    historical budget survive. ``pass`` is carried separately as a BCE target.
    Advantages, accumulated reward and old log probabilities are never inputs.
    Consecutive identical Solver exports are collapsed before assigning logical
    indices. This repair is specific to the historical baseline export; it is
    never applied across a Verifier action or across trajectories.
    """
    trajectories = document.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("baseline_nested requires a nonempty trajectories list")
    if "num_trajectories" in document:
        declared = positive_integer(document["num_trajectories"], "num_trajectories")
        if declared != len(trajectories):
            raise ValueError(f"num_trajectories={declared}, but file contains {len(trajectories)}; export may be incomplete")
    policy_step = document.get("global_step", document.get("step", source.get("step")))
    if policy_step is None:
        raise ValueError("baseline_nested needs global_step or an explicit manifest step")
    _nonnegative_integer(policy_step, "global_step")
    root_budget = document.get("max_solver_turns", document.get("value_max_solver_turns"))
    configured_budget = source.get("max_solver_turns", source.get("defaults", {}).get("value_max_solver_turns", fallback_budget))
    seen = set()
    for trajectory in trajectories:
        if not isinstance(trajectory, dict):
            raise ValueError("each trajectory must be an object")
        uid, tid = trajectory.get("uid"), trajectory.get("traj_uid")
        if not isinstance(uid, str) or not uid or not isinstance(tid, str) or not tid:
            raise ValueError("baseline trajectory needs nonempty uid and traj_uid")
        if tid in seen:
            raise ValueError(f"duplicate baseline traj_uid in one global_step: {tid}")
        seen.add(tid)
        logged_budget = trajectory.get("max_solver_turns", trajectory.get("value_max_solver_turns", root_budget))
        if root_budget is not None and logged_budget != root_budget:
            raise ValueError("conflicting logged historical Solver budgets")
        if logged_budget is not None and configured_budget is not None and logged_budget != configured_budget:
            raise ValueError("logged Solver budget differs from explicitly configured historical budget")
        budget = logged_budget if logged_budget is not None else configured_budget
        if budget is None:
            raise ValueError("baseline export omits historical max_solver_turns; supply --max-solver-turns or manifest defaults")
        positive_integer(budget, "historical max_solver_turns")
        env = trajectory.get("env_kwargs") or {}
        question = (trajectory.get("value_question") or env.get("question")
                    or trajectory.get("anchor_obs") or source.get("question_map", {}).get(uid))
        if not isinstance(question, str) or not question.strip():
            raise ValueError("baseline needs original value_question/env_kwargs.question/anchor_obs or question_map; rendered question/raw_prompt is not a safe substitute")
        label = _boolean(trajectory.get("pass"), "trajectory pass")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps or any(not isinstance(a, dict) for a in steps):
            raise ValueError("baseline trajectory needs a nonempty steps list")
        indices = [_nonnegative_integer(a.get("step_idx"), "step_idx") for a in steps]
        if sorted(set(indices)) != list(range(len(set(indices)))):
            raise ValueError("step_idx must cover consecutive indices starting at zero")
        counts = {"solver": 0, "verifier": 0}
        retained, indexed_actions = [], {}
        previous_signature = None
        removed = 0
        for action in sorted(steps, key=lambda a: a["step_idx"]):
            role = str(action.get("agent_id", "")).strip().lower().removesuffix(" agent")
            if role not in counts:
                raise ValueError("baseline supports only Solver Agent and Verifier Agent")
            valid = _boolean(action.get("is_action_valid"), "is_action_valid")
            response = action.get("response")
            if not isinstance(response, str):
                raise ValueError("baseline action needs full response text")
            stats = action.get("top16_entropy", action.get("top_16_entropy")) or {}
            if not isinstance(stats, dict):
                raise ValueError("top16_entropy must be a statistics object")
            mean = stats.get("mean")
            if mean is not None:
                if isinstance(mean, bool) or not isinstance(mean, (float, int)) or not math.isfinite(mean) or not 0 <= mean <= 1:
                    raise ValueError("mean Top-16 entropy must be normalized to [0,1]")
            response_tokens = action.get("num_response_tokens")
            entropy_tokens = stats.get("num_tokens")
            if response_tokens is not None:
                _nonnegative_integer(response_tokens, "num_response_tokens")
            if entropy_tokens is not None:
                _nonnegative_integer(entropy_tokens, "top16_entropy.num_tokens")
            coverage = stats.get("coverage")
            if coverage is not None and (isinstance(coverage, bool) or not isinstance(coverage, (float, int))
                                         or not math.isfinite(coverage) or not 0 <= coverage <= 1):
                raise ValueError("entropy coverage must lie in [0,1]")
            if response_tokens is not None and entropy_tokens is not None:
                if entropy_tokens > response_tokens:
                    raise ValueError("entropy num_tokens exceeds response tokens")
                counted_coverage = entropy_tokens / response_tokens if response_tokens else 0.0
                coverage = counted_coverage if coverage is None else min(coverage, counted_coverage)
            truncated = _boolean(action.get("truncated", False), "truncated") or action.get("finish_reason") == "length"
            # Compare the values actually used by the value model, including
            # token counts before their fallback/coverage conversion. Logging
            # IDs, rewards and advantages may differ between duplicate writes.
            signature = (valid, mean, coverage, response_tokens, entropy_tokens, truncated)
            previous = retained[-1] if retained else None
            duplicate_solver = (role == "solver" and previous is not None
                                and previous["agent_id"] == "solver"
                                and previous["value_action_text"] == response)
            if duplicate_solver and signature != previous_signature:
                raise ValueError(
                    f"global_step={policy_step}, traj_uid={tid}, step_idx={action['step_idx']}: "
                    "duplicate Solver response has conflicting entropy/token/validity/truncation metadata")
            fingerprint = (role, response, signature)
            index = action["step_idx"]
            if index in indexed_actions and not (duplicate_solver and indexed_actions[index] == fingerprint):
                raise ValueError(f"traj_uid={tid}: step_idx={index} is reused by conflicting or non-Solver actions")
            indexed_actions[index] = fingerprint
            if duplicate_solver:
                removed += 1
                continue
            retained.append({
                "uid": uid, "traj_uid": tid, "step": policy_step,
                "run_id": trajectory.get("run_id", document.get("run_id", source.get("run_id"))),
                "value_question": question, "value_max_solver_turns": budget,
                "value_action_index": len(retained), "role_turn_index": counts[role],
                "value_action_text": response, "agent_id": role, "is_action_valid": valid,
                "pass": label, "top16_entropy_mean": mean,
                "top16_entropy": {"coverage": coverage},
                "pure_entropy_response_tokens": entropy_tokens if entropy_tokens is not None else response_tokens,
                "pure_entropy_truncated": truncated,
            })
            previous_signature = signature
            counts[role] += 1
        if deduplication is not None:
            for key, increment in (("input_actions", len(steps)), ("removed_solver_actions", removed),
                                   ("affected_trajectories", int(removed > 0))):
                deduplication[key] = deduplication.get(key, 0) + increment
        yield from retained
