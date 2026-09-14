import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pretrain_entropy_cli_test", Path(__file__).resolve().parents[2] / "examples/drmas_trainer/pretrain_credit_value.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trajectory(uid="t", budget=2):
    return {"traj_uid": uid, "question": "exact question", "max_solver_turns": budget, "label": 1,
            "actions": [{"role": "solver", "text": "s", "entropy_mean": .3},
                        {"role": "verifier", "text": "<verify>approve</verify>", "entropy_mean": .4}]}


def test_mixed_baseline_sup_budgets_and_entropy(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(trajectory(str(i), budget)) for i, budget in enumerate((2, 3))))
    records, info = module.load_records([str(path)])
    assert {r["max_solver_turns"] for r in records} == {2, 3}
    assert info["entropy_actions"] == 4


def test_manifest_maps_legacy_fields_without_guessing(tmp_path):
    rows = []
    for i, action in enumerate(trajectory()["actions"]):
        rows.append({"tid": "t", "uid": "question-id", "answer": action["text"], "role": action["role"],
                     "index": i, "role_turn_index": 0, "top16_entropy_mean": action["entropy_mean"],
                     "is_action_valid": True, "pass": True})
    data = tmp_path / "legacy.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": [{"input": "legacy.jsonl", "run_id": "baseline-1", "step": 250,
        "defaults": {"value_max_solver_turns": 2}, "question_map": {"question-id": "exact question"},
        "field_map": {"traj_uid": "tid", "agent_id": "role", "value_action_index": "index", "value_action_text": "answer"}}]}))
    records, _ = module.load_records(manifest=manifest)
    assert records[0]["question"] == "exact question"
    assert records[0]["actions"][0]["entropy_mean"] == .3


def test_missing_question_is_not_reconstructed_from_prompt(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"prompt": "some prompt containing a question", "response": "x"}))
    with pytest.raises(ValueError, match="missing canonical"):
        module.load_records([str(path)])


def test_duplicate_selection_rejected(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(trajectory()))
    with pytest.raises(ValueError, match="more than once"):
        module.load_records([str(path), str(path)])
