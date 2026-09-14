"""Nested historical exports reach the actual CLI without reward/advantage leakage."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("baseline_pretrain_cli", ROOT / "examples/drmas_trainer/pretrain_credit_value.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def document(step=1, success=True, retry=False):
    actions = [("Solver Agent", "answer one", .1),
               ("Verifier Agent", "<verify>reject</verify>" if retry else "<verify>approve</verify>", .2)]
    if retry:
        actions.append(("Solver Agent", "answer two", .3))
    trajectory = {
        "uid": "question-group", "traj_uid": "trajectory", "pass": float(success),
        "question": "system/user TEMPLATE and stale teammate answers", "raw_prompt": [],
        "anchor_obs": r"Find n such that \\( n+1 \\) is prime.",
        "episode_length": 1, "total_token_level_score": 999,
        "per_agent_advantage": {"Solver Agent": {"advantage_sum": 0}},
        "steps": [{"step_idx": i, "turn_id": i, "agent_id": role, "response": response,
                   "num_response_tokens": 10, "is_action_valid": True,
                   "advantage_sum": 0, "old_log_prob_mean": -100,
                   "top16_entropy": {"mean": entropy, "std": .05, "p90": .5,
                                     "effective_support": 16 ** entropy, "num_tokens": 10}}
                  for i, (role, response, entropy) in enumerate(actions)]}
    return {"global_step": step, "num_trajectories": 1, "trajectories": [trajectory]}


def write(tmp_path, data=None, name="global_step_1.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document() if data is None else data, indent=2), encoding="utf-8")
    return path


def read(path, **kwargs):
    return CLI.load_records([str(path)], max_solver_turns=2, run_id="baseline-run", **kwargs)


def test_nested_json_uses_original_question_role_counters_and_real_binary_label(tmp_path):
    raw = document()
    records, info = read(write(tmp_path, raw))
    record = records[0]
    assert record["question"] == raw["trajectories"][0]["anchor_obs"]
    assert record["label"] == 1 and record["max_solver_turns"] == 2
    assert [a["role_turn_index"] for a in record["actions"]] == [0, 0]
    assert [a["entropy_mean"] for a in record["actions"]] == [.1, .2]
    assert all(a["entropy_coverage"] == 1 for a in record["actions"])
    assert info["successful_trajectories"] == 1
    assert info["historical_solver_budgets"] == {"2": 1}
    assert info["eligible_prefixes"] == {
        "solver": {"absolute_nonterminal": 1, "temporal_nonterminal": 0},
        "verifier": {"absolute_nonterminal": 0, "temporal_nonterminal": 0}}


def test_retry_final_solver_has_temporal_feature_but_no_nonterminal_temporal_data(tmp_path):
    records, info = read(write(tmp_path, document(success=False, retry=True)))
    assert [a["role_turn_index"] for a in records[0]["actions"]] == [0, 0, 1]
    features = CLI._utilities().prefix_entropy_features(records[0]["actions"])
    assert features["temporal_available"].tolist() == [False, False, False, True]
    assert info["eligible_prefixes"]["solver"]["temporal_nonterminal"] == 0
    assert info["eligible_prefixes"]["verifier"]["absolute_nonterminal"] == 1
    assert info["failed_trajectories"] == 1


def test_logged_advantages_rewards_old_log_probs_and_extra_entropy_stats_are_not_features(tmp_path):
    raw = document()
    first, _ = read(write(tmp_path, raw))
    raw["trajectories"][0]["per_agent_advantage"] = {"leak": 1e100}
    raw["trajectories"][0]["total_token_level_score"] = -1000
    for action in raw["trajectories"][0]["steps"]:
        action.update(advantage_sum=999, old_log_prob_mean=0, token_level_reward_sum=100)
        action["top16_entropy"].update(std=100, p90=100, effective_support=100)
    second, _ = read(write(tmp_path, raw))
    assert first == second


def test_directory_recursion_and_global_steps_namespace_duplicate_trajectory_ids(tmp_path):
    write(tmp_path, document(1))
    child = tmp_path / "next"
    child.mkdir()
    write(child, document(2), "global_step_2.jsonl")
    records, info = read(tmp_path)
    assert len(records) == 2 and len({r["traj_uid"] for r in records}) == 2
    assert info["file_count"] == info["run_step_count"] == 2
    assert info["unique_questions"] == 1


def test_nested_jsonl_and_manifest_historical_budget(tmp_path):
    data = tmp_path / "all.jsonl"
    data.write_text("\n".join(json.dumps(document(i)) for i in (1, 2)), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": [{"input": "all.jsonl", "run_id": "baseline",
                                                 "format": "baseline_nested", "max_solver_turns": 2}]}))
    records, info = CLI.load_records(manifest=manifest)
    assert len(records) == 2
    assert info["format_documents"]["baseline_nested"] == 2


def test_missing_historical_budget_is_not_inferred_from_used_turns(tmp_path):
    with pytest.raises(ValueError, match="historical max_solver_turns"):
        CLI.load_records([str(write(tmp_path))])


def test_future_three_round_budget_cannot_relabel_exhausted_two_round_trajectory(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        CLI.load_records([str(write(tmp_path, document(retry=True)))], max_solver_turns=3)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(num_trajectories=256), "incomplete"),
    (lambda d: d["trajectories"][0].pop("anchor_obs"), "original"),
    (lambda d: d["trajectories"][0].update(**{"pass": 2}), "pass"),
    (lambda d: d["trajectories"][0]["steps"][0].pop("is_action_valid"), "is_action_valid"),
    (lambda d: d["trajectories"][0]["steps"][1].update(step_idx=0), "step_idx"),
    (lambda d: d["trajectories"][0]["steps"][0]["top16_entropy"].update(mean=1.5), "normalized"),
    (lambda d: d["trajectories"][0]["steps"][0]["top16_entropy"].update(num_tokens=11), "exceeds"),
    (lambda d: d.update(max_solver_turns=3), "differs"),
])
def test_incomplete_or_conflicting_export_is_rejected(tmp_path, mutation, match):
    raw = document()
    mutation(raw)
    with pytest.raises(ValueError, match=match):
        read(write(tmp_path, raw))


def test_pasted_fragment_is_rejected_instead_of_silently_training_a_subset(tmp_path):
    path = write(tmp_path)
    path.write_text(path.read_text()[:-3], encoding="utf-8")
    with pytest.raises(ValueError, match="complete files"):
        read(path)


def test_low_coverage_and_length_finish_are_preserved(tmp_path):
    raw = document(retry=True)
    raw["trajectories"][0]["steps"][0]["top16_entropy"]["num_tokens"] = 5
    raw["trajectories"][0]["steps"][1]["finish_reason"] = "length"
    records, _ = read(write(tmp_path, raw))
    actions = records[0]["actions"]
    assert actions[0]["entropy_coverage"] == .5 and actions[1]["truncated"] is True
    features = CLI._utilities().prefix_entropy_features(actions)
    assert not features["absolute_available"][1:3].any()


def test_cli_validate_supports_baseline_only_without_loading_torch_or_encoder(tmp_path, capsys):
    data = write(tmp_path)
    output = tmp_path / "not-created.pt"
    assert CLI.main(["--input", str(data), "--max-solver-turns", "2", "--run-id", "b",
                     "--input-format", "baseline_nested", "--allow-absolute-only",
                     "--output", str(output), "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trajectories"] == 1 and report["eligible_prefixes"]["solver"]["temporal_nonterminal"] == 0
    assert not output.exists()


def duplicate_solver_writes(raw, action_indices, *, original_indices=False):
    """Emulate duplicate exports without changing the underlying interaction."""
    result = copy.deepcopy(raw)
    originals = result["trajectories"][0]["steps"]
    expanded = [copy.deepcopy(originals[index]) for index in action_indices]
    if not original_indices:
        for index, action in enumerate(expanded):
            action["step_idx"] = index
    result["trajectories"][0]["steps"] = expanded
    return result


@pytest.mark.parametrize("indices,retry", [
    ([0, 0, 1], False),                 # SSV -> SV with approve
    ([0, 1, 2, 2], True),              # SVSS -> SVS with reject
    ([0, 0, 1, 2, 2], True),           # SSVSS -> SVS with reject
    ([0, 0, 1, 2], True),              # SSVS -> SVS with reject
    ([0, 0, 0, 0, 1, 2, 2, 2], True), # Entire runs of duplicates collapse
])
@pytest.mark.parametrize("original_indices", [False, True])
def test_consecutive_solver_duplicates_restore_exact_records_and_entropy_features(
        tmp_path, indices, retry, original_indices):
    clean = document(success=not retry, retry=retry)
    expected, _ = read(write(tmp_path, clean))
    raw = duplicate_solver_writes(clean, indices, original_indices=original_indices)
    records, info = read(write(tmp_path, raw))
    assert records == expected
    assert [a["role_turn_index"] for a in records[0]["actions"]] == ([0, 0, 1] if retry else [0, 0])
    features = CLI._utilities().prefix_entropy_features(records[0]["actions"])
    expected_features = CLI._utilities().prefix_entropy_features(expected[0]["actions"])
    assert {key: value.tolist() for key, value in features.items()} == {
        key: value.tolist() for key, value in expected_features.items()}
    assert info["deduplication"] == {
        "input_actions": len(indices),
        "removed_solver_actions": len(indices) - len(expected[0]["actions"]),
        "affected_trajectories": 1,
    }
    assert info["actions"] == len(expected[0]["actions"])
    assert info["entropy_actions"] == len(expected[0]["actions"])
    assert info["trajectories"] == 1


def test_duplicate_solver_array_order_does_not_override_storage_indices(tmp_path):
    raw = duplicate_solver_writes(document(retry=True), [0, 0, 1, 2, 2])
    actions = raw["trajectories"][0]["steps"]
    expected, expected_info = read(write(tmp_path, raw))
    for permutation in itertools.permutations(actions):
        raw["trajectories"][0]["steps"] = list(permutation)
        actual, info = read(write(tmp_path, raw))
        assert actual == expected
        assert info["deduplication"] == expected_info["deduplication"]


def test_solver_dedup_reindexes_canonical_actions_without_mutating_source():
    raw = duplicate_solver_writes(document(retry=True), [0, 0, 1, 2, 2])
    raw["trajectories"][0]["steps"].reverse()
    snapshot = copy.deepcopy(raw)
    stats = {}
    rows = list(CLI._input_adapter().baseline_action_rows(raw, {}, 2, deduplication=stats))
    assert [row["value_action_index"] for row in rows] == [0, 1, 2]
    assert [row["role_turn_index"] for row in rows] == [0, 0, 1]
    assert raw == snapshot
    assert stats == {"input_actions": 5, "removed_solver_actions": 2, "affected_trajectories": 1}


def test_solver_dedup_ignores_logging_ids_rewards_and_unused_entropy_statistics(tmp_path):
    clean = document()
    expected, _ = read(write(tmp_path, clean))
    raw = duplicate_solver_writes(clean, [0, 0, 1])
    duplicate = raw["trajectories"][0]["steps"][1]
    duplicate.update(turn_id=99, action_id="second-disk-write", advantage_sum=-999,
                     old_log_prob_mean=-1, token_level_reward_sum=999)
    duplicate["top16_entropy"].update(std=99, p10=99, p50=99, p90=99, effective_support=99)
    actual, info = read(write(tmp_path, raw))
    assert actual == expected
    assert info["deduplication"]["removed_solver_actions"] == 1


@pytest.mark.parametrize("mutation", [
    lambda action: action["top16_entropy"].update(mean=.11),
    lambda action: action["top16_entropy"].pop("mean"),
    lambda action: action["top16_entropy"].update(coverage=.5),
    lambda action: action["top16_entropy"].update(num_tokens=9),
    lambda action: action.update(num_response_tokens=11),
    lambda action: action.pop("num_response_tokens"),
    lambda action: action.update(is_action_valid=False),
    lambda action: action.update(truncated=True),
    lambda action: action.update(finish_reason="length"),
])
def test_identical_solver_text_with_conflicting_value_metadata_is_rejected(tmp_path, mutation):
    raw = duplicate_solver_writes(document(), [0, 0, 1])
    mutation(raw["trajectories"][0]["steps"][1])
    with pytest.raises(ValueError, match="duplicate Solver"):
        read(write(tmp_path, raw))


@pytest.mark.parametrize("new_response", ["different answer", "answer one "])
def test_adjacent_solver_actions_must_have_exactly_equal_full_text(tmp_path, new_response):
    raw = duplicate_solver_writes(document(), [0, 0, 1])
    raw["trajectories"][0]["steps"][1]["response"] = new_response
    with pytest.raises(ValueError, match="ambiguous, incomplete or conflicting"):
        read(write(tmp_path, raw))


def test_nonconsecutive_identical_solver_action_is_a_real_second_turn(tmp_path):
    raw = document(retry=True)
    actions = raw["trajectories"][0]["steps"]
    actions[2] = copy.deepcopy(actions[0])
    actions[2]["step_idx"] = 2
    records, info = read(write(tmp_path, raw))
    assert [a["role"] for a in records[0]["actions"]] == ["solver", "verifier", "solver"]
    assert [a["role_turn_index"] for a in records[0]["actions"]] == [0, 0, 1]
    assert records[0]["actions"][0]["text"] == records[0]["actions"][2]["text"]
    assert info["deduplication"] == {
        "input_actions": 3, "removed_solver_actions": 0, "affected_trajectories": 0}


@pytest.mark.parametrize("original_indices", [False, True])
def test_identical_verifier_exports_are_not_silently_deduplicated(tmp_path, original_indices):
    raw = duplicate_solver_writes(document(), [0, 1, 1], original_indices=original_indices)
    with pytest.raises(ValueError, match="step_idx|ambiguous, incomplete or conflicting"):
        read(write(tmp_path, raw))


def test_reused_index_with_different_solver_text_is_rejected(tmp_path):
    raw = duplicate_solver_writes(document(), [0, 0, 1], original_indices=True)
    raw["trajectories"][0]["steps"][1]["response"] = "a different action"
    with pytest.raises(ValueError, match="step_idx"):
        read(write(tmp_path, raw))


def test_dedup_does_not_repair_missing_raw_storage_indices(tmp_path):
    raw = duplicate_solver_writes(document(), [0, 0, 1], original_indices=True)
    raw["trajectories"][0]["steps"][2]["step_idx"] = 2
    with pytest.raises(ValueError, match="step_idx"):
        read(write(tmp_path, raw))


def test_dedup_keeps_identical_actions_in_separate_trajectories_and_aggregates_stats(tmp_path):
    raw = duplicate_solver_writes(document(), [0, 0, 1])
    other = copy.deepcopy(raw["trajectories"][0])
    other["traj_uid"] = "another-trajectory"
    raw["trajectories"].append(other)
    raw["num_trajectories"] = 2
    write(tmp_path, raw)
    write(tmp_path, document(step=2), name="global_step_2.json")
    records, info = read(tmp_path)
    assert len(records) == 3 and all(len(record["actions"]) == 2 for record in records)
    assert info["deduplication"] == {
        "input_actions": 8, "removed_solver_actions": 2, "affected_trajectories": 2}
    assert info["actions"] == info["entropy_actions"] == 6


def test_cli_validate_prints_deduplication_report_without_creating_checkpoint(tmp_path, capsys):
    raw = duplicate_solver_writes(document(retry=True), [0, 0, 1, 2, 2])
    data = write(tmp_path, raw)
    output = tmp_path / "not-created.pt"
    assert CLI.main(["--input", str(data), "--max-solver-turns", "2", "--run-id", "b",
                     "--input-format", "baseline_nested", "--allow-absolute-only",
                     "--output", str(output), "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deduplication"] == {
        "input_actions": 5, "removed_solver_actions": 2, "affected_trajectories": 1}
    assert report["trajectories"] == 1
    assert report["actions"] == report["entropy_actions"] == 3
    assert not output.exists()
