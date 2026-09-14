import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("entropy_value_worker_test", Path(__file__).resolve().parents[2] / "verl/workers/credit_value.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.config = SimpleNamespace(hidden_size=8, model_type="qwen3", vocab_size=32)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=x.cumsum(1) / torch.arange(1, x.shape[1] + 1)[None, :, None])


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(char) % 32 for char in text]


class TinyScorer(module.PrefixValueScorer):
    def _load_backbone(self):
        with torch.random.fork_rng():
            torch.manual_seed(71)
            return Encoder(), Tokenizer()


def scorer(**kwargs):
    return TinyScorer({"model_path": "tiny-fixed", "head_hidden_dim": 8, "entropy_hidden_dim": 8,
                       "dropout": 0.0, "train_epochs": 1, **kwargs})


def record(uid="t", label=1):
    return {"traj_uid": uid, "question": "problem", "max_solver_turns": 3, "label": label,
            "actions": [{"role": "solver", "text": "one", "entropy_mean": .2},
                        {"role": "verifier", "text": "<verify>reject</verify>", "entropy_mean": .3},
                        {"role": "solver", "text": "two", "entropy_mean": .5},
                        {"role": "verifier", "text": "<verify>reject</verify>", "entropy_mean": .4},
                        {"role": "solver", "text": "three", "entropy_mean": .4}]}


def encoded(s, rec=None):
    r = rec or record()
    row, _ = s._encode(r)
    return {**row, "label": float(r["label"]), "question_key": "q", "traj_uid": r["traj_uid"]}


def test_all_prefixes_equal_weight_and_no_future_features():
    s = scorer()
    a = encoded(s)
    r = record("u", 0)
    r["actions"] = r["actions"][:2]
    b = encoded(s, r)
    flat = s._flatten([a, b])
    assert torch.all(flat["weights"] == 1)
    changed = record()
    changed["actions"][-1].update(text="a very different future", entropy_mean=.9)
    c = encoded(s, changed)
    for key in ("features", "absolute", "temporal"):
        torch.testing.assert_close(a[key][:-1], c[key][:-1])


def test_complete_early_prefixes_survive_overlong_tail():
    s = scorer(max_length=150)
    r = record()
    r["actions"][-1]["text"] = "x" * 1000
    row = encoded(s, r)
    assert 1 < len(row["features"]) < 6
    assert not row["prefix_terminal"][-1]


def test_frozen_semantic_entropy_stage_changes_only_entropy_heads():
    s = scorer()
    row = encoded(s)
    s._fit_scaler([row])
    encoder = copy.deepcopy(s.encoder.state_dict())
    before = copy.deepcopy(s.candidate_head.state_dict())
    s._train([row], 2, "entropy")
    for key, value in encoder.items():
        torch.testing.assert_close(value, s.encoder.state_dict()[key], rtol=0, atol=0)
    for key, value in before.items():
        if key.startswith("semantic."):
            torch.testing.assert_close(value, s.candidate_head.state_dict()[key], rtol=0, atol=0)
    assert any(not torch.equal(value, s.candidate_head.state_dict()[key])
               for key, value in before.items() if key.startswith("temporal."))
    assert all(p.grad is None for p in s.encoder.parameters())


def test_matched_control_is_invariant_to_entropy_values():
    s = scorer()
    row = encoded(s)
    s._fit_scaler([row])
    original = s._predict(s.control_head, row, no_entropy=True)
    changed = copy.deepcopy(row)
    changed["absolute"][:, :5] += 20
    changed["temporal"][:, :5] -= 20
    actual = s._predict(s.control_head, changed, no_entropy=True)
    torch.testing.assert_close(original, actual)


def test_prepare_does_not_fit_labels_or_alias_deployed():
    s = scorer()
    row = encoded(s)
    s._fit_scaler([row])
    s.ready = True
    before = copy.deepcopy(s.deployed_head.state_dict())
    first = s.prepare([record("one", 0)])["values"]["one"]
    second = s.prepare([record("two", 1)])["values"]["two"]
    assert first == second
    s._train([row], 1)
    for key, value in before.items():
        torch.testing.assert_close(value, s.deployed_head.state_dict()[key], rtol=0, atol=0)


def test_insufficient_online_data_does_not_disable_loaded_head():
    s = scorer()
    s.ready = True
    result = s.update()
    assert result["ready"] == 1
    assert result["disabled"] == 0


def test_checkpoint_identity_scaler_and_resume(tmp_path):
    s = scorer()
    row = encoded(s)
    s._fit_scaler([row])
    s.ready = s.pretrained = True
    s.version = 1
    s.prepare([record()])
    s.step = 9
    path = tmp_path / "value.pt"
    s.save(path)
    fresh = scorer(replay_max_trajectories=99)
    result = fresh.load(path)
    assert result["ready"]
    assert fresh.config["replay_max_trajectories"] == 99
    assert not fresh._pending and fresh.step == 0
    resumed = scorer()
    resumed.load(path, resume=True)
    assert resumed.step == 9 and len(resumed._pending) == 1
    state = torch.load(path, weights_only=False)
    state["scaler"]["temporal"]["names"].reverse()
    torch.save(state, path)
    with pytest.raises(ValueError, match="feature order"):
        fresh.load(path)


def test_quality_requires_actual_entropy_and_capacity_control_gain():
    s = scorer(min_entropy_val_prefixes=2)
    metrics = {"auc": .9, "brier": .1, "prior_brier": .25, "entropy_prefixes": 9,
               "temporal_prefixes": 9, "absolute_gain": .01, "temporal_gain": .01, "matched_control_gain": .01,
               "matched_temporal_gain": .01}
    assert s._quality_passes(metrics)
    for key in ("absolute_gain", "temporal_gain", "matched_control_gain", "matched_temporal_gain"):
        assert not s._quality_passes({**metrics, key: 0})


@pytest.mark.parametrize("setting,value", [("dropout", .2), ("max_length", 4096), ("scaler_clip", 4.0),
                                            ("learning_rate", .0002), ("train_epochs", 3), ("replay_max_trajectories", 4000)])
def test_resume_rejects_changed_training_or_scaling_configuration(tmp_path, setting, value):
    original = scorer()
    original._fit_scaler([encoded(original)])
    original.ready = original.pretrained = True
    path = tmp_path / "resume.pt"
    original.save(path)
    changed = scorer(**{setting: value})
    with pytest.raises(ValueError, match="mismatch|scaling"):
        changed.load(path, resume=True)


def test_initialization_preserves_runtime_training_settings(tmp_path):
    original = scorer()
    original._fit_scaler([encoded(original)])
    original.ready = original.pretrained = True
    path = tmp_path / "initial.pt"
    original.save(path)
    changed = scorer(learning_rate=.0002, train_epochs=3, replay_max_trajectories=4096)
    changed.load(path)
    assert changed.optimizer.param_groups[0]["lr"] == .0002
    assert changed.config["train_epochs"] == 3
    assert changed.config["replay_max_trajectories"] == 4096


def test_pretraining_learns_entropy_signal_and_qualifies_against_matched_control():
    """A semantic-constant dataset has both noisy absolute and extra temporal signal."""
    s = scorer(head_hidden_dim=16, entropy_hidden_dim=16, learning_rate=.01,
               min_train_trajectories=16, min_val_trajectories=12,
               min_train_questions=8, min_val_questions=8, min_val_per_class=4,
               min_entropy_val_prefixes=8, validation_fraction=.3)
    for i in range(128):
        label = i % 2
        noisy_absolute = label if i % 5 else 1 - label
        row = encoded(s, record(str(i), label))
        # These cached features isolate head learning from backbone quality.
        for key in ("features", "absolute", "temporal", "abs_mask", "temp_mask", "prefix_terminal"):
            row[key] = row[key][:1].clone()
        row["features"].zero_()
        row["absolute"][0] = torch.tensor([.2 + .4 * noisy_absolute, 1, 2, 1, 0, 1] * 2)
        row["temporal"][0] = torch.tensor([.3, .2 * (2 * label - 1), 0, .2 * (2 * label - 1), 1, 1] * 2)
        row["abs_mask"].fill_(True)
        row["temp_mask"].fill_(True)
        row["prefix_terminal"].fill_(False)
        row["prefix_roles"] = ["solver" if (i // 2) % 2 else "verifier"]
        row["question_key"] = f"independent-question-{i}"
        s._pending[str(i)] = row
    metrics = s.pretrain(semantic_epochs=10, entropy_epochs=60)
    assert metrics["ready"] == 1
    assert metrics["candidate_absolute_gain"] > 0
    assert metrics["candidate_temporal_gain"] > 0
    assert metrics["candidate_matched_control_gain"] > 0
    assert metrics["candidate_matched_temporal_gain"] > 0
    assert s.pretrained


def test_role_drift_revokes_stale_permission_without_disabling_healthy_role(monkeypatch):
    s = scorer(min_entropy_val_prefixes=2, miscalibration_patience=3)
    row = encoded(s)
    s._fit_scaler([row])
    s.ready, s.version = True, 1
    s.reliability = {"solver": 1.0, "verifier": 1.0}
    s.temporal_reliability = s.reliability.copy()
    monkeypatch.setattr(s, "_predict", lambda *a, **k: None)
    monkeypatch.setattr(s, "_quality_passes", lambda m: True)
    monkeypatch.setattr(s, "_role_quality", lambda m, role: role == "solver")
    monkeypatch.setattr(s, "_role_temporal_quality", lambda m, role: role == "solver")
    monkeypatch.setattr(s, "_evaluate", lambda head, *a, **k: {
        "brier": .3 if head is s.candidate_head else .2,
        "entropy_prefixes": 8, "temporal_prefixes": 8,
        "solver_prefixes": 4, "verifier_prefixes": 4,
        "solver_temporal_prefixes": 4, "verifier_temporal_prefixes": 4})
    for index in range(3):
        validation = {**row, "traj_uid": f"new-window-{index}"}
        s._validate_deploy([row], [validation], {})
        # Seeing the same validation data twice must not count as a new window.
        counters = s.role_bad_windows.copy()
        s._validate_deploy([row], [validation], {})
        assert s.role_bad_windows == counters
    assert s.ready and s.version == 1  # candidate is worse; old global head stays
    assert s.reliability == {"solver": 1.0, "verifier": 0.0}
    assert s.temporal_reliability == {"solver": 1.0, "verifier": 0.0}


def test_dropout_training_resumes_exactly_without_consuming_global_rng(tmp_path):
    original = scorer(dropout=.3)
    rows = [encoded(original, record("p", 1)), encoded(original, record("n", 0))]
    original._fit_scaler(rows)
    original._train(rows, 2)
    path = tmp_path / "dropout.pt"
    original.save(path)
    resumed = scorer(dropout=.3)
    resumed.load(path, resume=True)
    before_rng = torch.random.get_rng_state().clone()
    original._train(rows, 2)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    # Unrelated consumers may change the global RNG between runs.
    torch.rand(41)
    resumed._train(rows, 2)
    for key, value in original.candidate_head.state_dict().items():
        torch.testing.assert_close(value, resumed.candidate_head.state_dict()[key], rtol=0, atol=0)
    assert torch.equal(original._rng.get_state(), resumed._rng.get_state())


def test_temporal_matched_control_retains_absolute_but_ignores_temporal_values():
    s = scorer()
    row = encoded(s)
    s._fit_scaler([row])
    prediction = s._predict(s.temporal_control_head, row, no_temporal=True)
    changed = copy.deepcopy(row)
    changed["temporal"][:, :5] += 100
    torch.testing.assert_close(prediction, s._predict(s.temporal_control_head, changed, no_temporal=True))
    changed["absolute"][:, 0] += .2
    assert not torch.equal(prediction, s._predict(s.temporal_control_head, changed, no_temporal=True))
