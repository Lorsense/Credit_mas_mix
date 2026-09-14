"""Current adaptive trajectories are scored but never enter value replay."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
SPEC = importlib.util.spec_from_file_location("curriculum_value_worker", Path(__file__).parents[2] / "verl/workers/credit_value.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.config = SimpleNamespace(hidden_size=8, model_type="qwen3", vocab_size=32)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(char) % 32 for char in text]


class TinyScorer(MOD.PrefixValueScorer):
    def _load_backbone(self):
        return Encoder(), Tokenizer()


def record(uid, **extra):
    return {"traj_uid": uid, "question": "problem", "max_solver_turns": 2, "label": 1,
            "actions": [{"role": "solver", "text": "first", "entropy_mean": .2},
                        {"role": "verifier", "text": "reject", "entropy_mean": .3},
                        {"role": "solver", "text": "second", "entropy_mean": .4}], **extra}


def scorer():
    result = TinyScorer({"model_path": "tiny-fixed", "head_hidden_dim": 8, "entropy_hidden_dim": 8,
                         "dropout": 0.0, "train_epochs": 1})
    encoded, _ = result._encode(record("init"))
    result._fit_scaler([{**encoded, "label": 1.0}])
    result.ready = True
    return result


def test_score_only_flag_keeps_predictions_but_excludes_pending_training_rows():
    value = scorer()
    result = value.prepare([record("fresh", train_eligible=True), record("hard", train_eligible=False)])
    assert set(result["values"]) == {"fresh", "hard"}
    assert result["values"]["fresh"] == result["values"]["hard"]
    assert set(value._pending) == {"fresh"}
    assert result["metrics"]["score_only_trajectories"] == 1
    assert result["metrics"]["encoded_trajectories"] == 2


def test_unspecified_eligibility_retains_legacy_offline_training_behavior():
    value = scorer()
    result = value.prepare([record("baseline"), record("sup")])
    assert set(value._pending) == {"baseline", "sup"}
    assert set(result["values"]) == {"baseline", "sup"}
    assert result["metrics"]["score_only_trajectories"] == 0


def test_all_adaptive_batch_leaves_pending_empty_even_when_value_not_ready():
    value = scorer()
    value.ready = False
    result = value.prepare([record("learning", train_eligible=False)])
    assert not value._pending
    assert not result["values"]
    assert result["metrics"]["encoded_trajectories"] == 1


def test_non_boolean_eligibility_does_not_silently_admit_adaptive_data():
    value = scorer()
    result = value.prepare([record("bad", train_eligible="false")])
    assert result["metrics"]["skipped_invalid"] == 1
    assert not value._pending and not result["values"]
