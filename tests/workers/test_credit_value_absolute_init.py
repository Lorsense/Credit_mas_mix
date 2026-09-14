"""Baseline two-Solver bootstrap, permission routing, and checkpoint regressions."""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("absolute_init_test_helpers", Path(__file__).with_name("test_credit_value.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def scorer(**kwargs):
    return helpers.scorer(allow_absolute_only_pretrain=True, **kwargs)


def short_record(uid="short", label=1):
    record = helpers.record(uid, label)
    record["max_solver_turns"] = 2
    record["actions"] = record["actions"][:3]
    return record


def diagnostic_rows(s, temporal_solver=False, terminal_temporal=False):
    """Real feature masks, one diagnostic prefix per independently labeled row."""
    rows = []
    for i in range(16):
        role = "solver" if i < 8 else "verifier"
        row = helpers.encoded(s, helpers.record(str(i), i % 2))
        index = 3 if role == "solver" else 2
        for key in ("features", "absolute", "temporal", "abs_mask", "temp_mask", "prefix_terminal"):
            row[key] = row[key][index:index + 1].clone()
        row["prefix_roles"] = [role]
        row["prefix_terminal"].fill_(terminal_temporal and role == "verifier")
        if role == "verifier":
            row["temporal"].zero_()
            row["temp_mask"].zero_()
            if terminal_temporal:
                row["temporal"][0, 11] = 1
                row["temp_mask"].fill_(1)
        elif not temporal_solver:
            row["temporal"].zero_()
            row["temp_mask"].zero_()
        row["question_key"] = f"question-{i}"
        rows.append(row)
    return rows


def scores_for(data, abs_accuracy, full_accuracy, semantic_accuracy=.5):
    labels = data["labels"].float()
    probability = lambda accuracy: torch.where(labels.bool(), accuracy, 1 - accuracy)
    return torch.stack((probability(semantic_accuracy), probability(abs_accuracy), probability(full_accuracy)), -1)


def fake_predictions(monkeypatch, s, candidate_abs=.8, candidate_full=.95,
                     deployed_abs=.9, deployed_full=.01):
    def predict(head, data, **kwargs):
        if head is s.candidate_head:
            return scores_for(data, candidate_abs, candidate_full)
        if head is s.deployed_head:
            return scores_for(data, deployed_abs, deployed_full)
        if head is s.control_head:
            return scores_for(data, .5, .5)
        return scores_for(data, .8, .8)
    monkeypatch.setattr(s, "_predict", predict)


def test_two_solver_budget_has_no_nonterminal_same_role_temporal_signal():
    s = scorer()
    row = helpers.encoded(s, short_record())
    assert row["prefix_roles"] == ["initial", "solver", "verifier", "solver"]
    assert row["prefix_terminal"].tolist() == [False, False, False, True]
    assert row["temp_mask"].tolist() == [0, 0, 0, 1]
    s._fit_scaler([row])
    flat = s._flatten([row])
    metrics = s._evaluate(s.candidate_head, [row], .5,
                          s._predict(s.control_head, flat, no_entropy=True),
                          s._predict(s.temporal_control_head, flat, no_temporal=True))
    assert metrics["temporal_prefixes"] == 0
    assert not s._bootstrap_temporal_quality(metrics, "solver")
    assert not s._bootstrap_temporal_quality(metrics, "verifier")


def test_absolute_pretrain_qualifies_entropy_without_training_terminal_temporal_head():
    s = scorer(head_hidden_dim=16, entropy_hidden_dim=16, learning_rate=.01,
               min_train_trajectories=16, min_val_trajectories=12,
               min_train_questions=8, min_val_questions=8, min_val_per_class=4,
               min_entropy_val_prefixes=8, validation_fraction=.3)
    before = {name: copy.deepcopy(getattr(s, name).temporal.state_dict())
              for name in ("candidate_head", "control_head", "temporal_control_head")}
    for i in range(128):
        row = helpers.encoded(s, short_record(str(i), i % 2))
        # Semantic-constant prefixes isolate whether entropy is actually learned.
        row["features"].zero_()
        row["absolute"][:, [0, 6]] = .2 + .6 * (i % 2)
        row["question_key"] = f"independent-question-{i}"
        s._pending[str(i)] = row
    metrics = s.pretrain(semantic_epochs=10, entropy_epochs=60)
    assert metrics["ready"] == 1
    assert metrics["pretraining_mode"] == "absolute"
    assert metrics["deployment_stage"] == "absolute"
    assert metrics["candidate_temporal_prefixes"] == 0
    assert metrics["candidate_absolute_gain"] > 0
    assert metrics["candidate_matched_control_gain"] > 0
    assert s.temporal_reliability == {"solver": 0.0, "verifier": 0.0}
    assert s.reliability["solver"] == 1
    for name, parameters in before.items():
        for key, parameter in parameters.items():
            torch.testing.assert_close(parameter, getattr(s, name).temporal.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("key", ["absolute_gain", "matched_control_gain", "abs_matched_control_gain", "auc", "abs_auc", "entropy_prefixes"])
def test_absolute_bootstrap_requires_predictive_entropy_gain_and_population(key):
    s = scorer(min_entropy_val_prefixes=2)
    metrics = {"auc": .9, "brier": .1, "prior_brier": .25, "entropy_prefixes": 8,
               "absolute_gain": .05, "matched_control_gain": .05, "abs_auc": .9,
               "abs_brier": .1, "abs_matched_control_gain": .05}
    assert s._bootstrap_quality_passes(metrics)
    assert not s._bootstrap_quality_passes({**metrics, key: 0})


def test_unqualified_temporal_weights_cannot_change_served_full_predictions():
    s = scorer()
    row = helpers.encoded(s)
    s._fit_scaler([row])
    s.ready = True
    before = s.prepare([helpers.record("before")])["values"]["before"]
    with torch.no_grad():
        for parameter in s.deployed_head.temporal.parameters():
            parameter.fill_(20)
    after = s.prepare([helpers.record("after")])["values"]["after"]
    assert before == after
    assert all(value["full"] == value["abs"] for value in after)
    s.temporal_reliability["solver"] = 1
    promoted = s.prepare([helpers.record("promoted")])["values"]["promoted"]
    assert promoted[3]["full"] != promoted[3]["abs"]
    assert promoted[4]["full"] == promoted[4]["abs"]  # V2 cannot borrow Solver permission.


def test_candidate_comparison_and_drift_ignore_unqualified_full_predictions(monkeypatch):
    s = scorer(min_entropy_val_prefixes=2)
    rows = diagnostic_rows(s)
    s.ready, s.version = True, 1
    s.reliability = {"solver": 1.0, "verifier": 1.0}
    fake_predictions(monkeypatch, s)
    for step in range(4):
        validation = [{**row, "traj_uid": f"{step}-{i}"} for i, row in enumerate(rows)]
        metrics = {}
        s._validate_deploy(rows, validation, metrics)
        assert metrics["status"] == "candidate_worse"
        assert metrics["candidate_brier"] == pytest.approx(.04)
        assert metrics["deployed_brier"] == pytest.approx(.01)
    assert s.ready and s.version == 1 and s.bad_windows == 0
    assert s.temporal_reliability == {"solver": 0.0, "verifier": 0.0}


def test_online_validation_promotes_temporal_only_for_qualified_nonterminal_role(monkeypatch):
    s = scorer(min_entropy_val_prefixes=2)
    rows = diagnostic_rows(s, temporal_solver=True, terminal_temporal=True)
    fake_predictions(monkeypatch, s)
    metrics = {}
    s._validate_deploy(rows, rows, metrics)
    assert s.ready and s.version == 1
    assert s.temporal_reliability == {"solver": 1.0, "verifier": 0.0}
    assert metrics["candidate_solver_temporal_prefixes"] == 8
    assert metrics["candidate_verifier_temporal_prefixes"] == 0
    assert metrics["deployment_stage"] == "hybrid"


def test_online_update_trains_all_branches_after_absolute_initialization(monkeypatch):
    s = scorer(min_train_trajectories=1, min_val_trajectories=1, min_train_questions=1,
               min_val_questions=1, min_val_per_class=1)
    rows = diagnostic_rows(s, temporal_solver=True)
    s._fit_scaler(rows)
    before = copy.deepcopy(s.candidate_head.temporal.state_dict())
    monkeypatch.setattr(s, "_ingest", lambda: (rows, rows))
    monkeypatch.setattr(s, "_validate_deploy", lambda *args: None)
    s.update()
    assert any(not torch.equal(value, s.candidate_head.temporal.state_dict()[key]) for key, value in before.items())


def test_absolute_checkpoint_initialization_and_resume_preserve_phase_without_runtime_flag(tmp_path):
    original = scorer()
    original._fit_scaler([helpers.encoded(original)])
    original.ready = original.pretrained = True
    original.version = 1
    original.reliability = {"solver": 1.0, "verifier": 1.0}
    path = tmp_path / "absolute.pt"
    original.save(path)
    fresh = helpers.scorer(learning_rate=.0001)
    result = fresh.load(path)
    assert result["deployment_stage"] == "absolute"
    assert result["pretraining_mode"] == "absolute"
    assert not fresh.config["allow_absolute_only_pretrain"]
    assert fresh.deployment_mode == "absolute_bootstrap"
    assert fresh.prepare([helpers.record()])["values"]["t"][3]["full"] == fresh.prepare([helpers.record()])["values"]["t"][3]["abs"]
    fresh.temporal_reliability["solver"] = 1.0
    fresh.save(path)
    resumed = helpers.scorer(learning_rate=.0001)
    assert resumed.load(path, resume=True)["deployment_stage"] == "hybrid"
    assert resumed.temporal_reliability == fresh.temporal_reliability
    assert resumed.deployment_mode == fresh.deployment_mode
    state = torch.load(path, weights_only=False)
    state["deployment_stage"] = "full"
    torch.save(state, path)
    with pytest.raises(ValueError, match="deployment stage"):
        resumed.load(path, resume=True)


def test_legacy_v2_full_checkpoint_remains_loadable(tmp_path):
    original = helpers.scorer()
    original._fit_scaler([helpers.encoded(original)])
    original.ready = original.pretrained = True
    path = tmp_path / "legacy.pt"
    original.save(path)
    state = torch.load(path, weights_only=False)
    for key in ("deployment_mode", "pretraining_mode", "deployment_stage"):
        state.pop(key)
    state["config"].pop("allow_absolute_only_pretrain")
    torch.save(state, path)
    restored = helpers.scorer()
    restored.load(path, resume=True)
    assert restored.deployment_mode == "full"
