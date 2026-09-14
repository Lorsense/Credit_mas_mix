import importlib.util
import json
from pathlib import Path
import pickle
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mix_question_curriculum", ROOT / "verl/utils/question_curriculum.py")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def make(size=200, **overrides):
    return MOD.QuestionCurriculum({"enabled": True, "seed": 7, **overrides},
                                  dataset_size=size, dataset_fingerprint="ordered-train-v1")


def test_cold_start_retains_base_batch():
    model = make()
    selected, tags, metrics = model.select(list(range(20)), 0)
    assert selected == list(range(20))
    assert tags.count("fresh") == 14
    assert tags.count("fresh_adaptive") == 6
    assert metrics["curriculum/replacement_fraction"] == 0


def test_fractional_hard_quota_eventually_retries_without_exceeding_batch_cap():
    model = make(retry_cooldown_steps=1, max_hard_retries=100, hard_ttl_steps=200)
    model.observe({i: [0, 0] if i == 30 else [1, 1] for i in range(30, 60)}, 0)
    counts = []
    for step in range(1, 101):
        _, tags, _ = model.select(list(range(30)), step)
        counts.append(tags.count("hard"))
    assert sum(counts) > 0  # target is 0.3 questions/step, not permanently zero
    assert max(counts) <= 1


def test_success_groups_are_downsampled_but_at_least_seventy_percent_retained():
    model = make()
    for step in range(3):
        model.observe({i: [1] * 8 for i in range(20)}, step)
    selected, tags, metrics = model.select(list(range(20)), 3)
    assert 14 <= len(set(selected) & set(range(20))) < 20
    assert tags.count("fresh_replacement") > 0
    assert tags.count("hard") == 0
    assert metrics["curriculum/replacement_fraction"] <= 0.3
    assert len(selected) == len(set(selected)) == 20


def test_overdue_mastered_questions_are_preserved_for_review():
    model = make(mastery_revisit_steps=5)
    for step in range(3):
        model.observe({i: [1, 1] for i in range(20)}, step)
    selected, tags, metrics = model.select(list(range(20)), 10)
    assert selected == list(range(20))
    assert metrics["curriculum/mastery_review_preserved"] == 6


def test_learning_pool_is_preferred_to_distant_failure_questions():
    model = make(stats_ema_beta=0.0)
    model.observe({i: [0, 1] for i in range(20, 40)}, 0)
    model.observe({i: [1, 1] for i in range(20)}, 1)
    selected, tags, metrics = model.select(list(range(20)), 2)
    assert tags.count("learning") == 6
    assert all(20 <= index < 40 for index, tag in zip(selected, tags) if tag == "learning")
    assert metrics["curriculum/hard_target_fraction"] == 0


def test_hard_retry_quota_depends_on_failure_rate_not_combined_zero_rate():
    high_success = make(stats_ema_beta=0.0, retry_cooldown_steps=1)
    high_failure = make(stats_ema_beta=0.0, retry_cooldown_steps=1)
    # Both batches have 100% zero-variance trajectory groups.
    high_success.observe({i: [0, 0] if i == 20 else [1, 1] for i in range(20, 40)}, 0)
    high_failure.observe({i: [0, 0] if i != 20 else [1, 1] for i in range(20, 40)}, 0)
    _, success_tags, sm = high_success.select(list(range(20)), 1)
    _, failure_tags, fm = high_failure.select(list(range(20)), 1)
    assert success_tags.count("hard") <= 1
    assert failure_tags.count("hard") == 3
    assert sm["curriculum/hard_target_fraction"] < fm["curriculum/hard_target_fraction"]
    assert fm["curriculum/replacement_fraction"] <= 0.3


def test_current_retry_outcomes_reclassify_and_remove_hard_entries():
    model = make(retry_cooldown_steps=1, stats_ema_beta=0.0)
    model.observe({i: [0, 0] for i in range(20, 30)}, 0)
    selected, tags, _ = model.select(list(range(20)), 1)
    retried = [index for index, tag in zip(selected, tags) if tag == "hard"]
    assert retried
    model.observe({retried[0]: [0, 1], retried[1]: [1, 1]}, 1)
    assert retried[0] not in model.hard
    assert retried[1] not in model.hard
    assert model.questions[retried[0]]["latest_class"] == "mixed"
    assert model.questions[retried[1]]["latest_class"] == "success"


def test_retry_cooldown_and_max_attempts_limit_repeated_negative_examples():
    model = make(retry_cooldown_steps=2, max_hard_retries=1, retire_cooldown_steps=10)
    model.observe({i: [0, 0] for i in range(20, 23)}, 0)
    _, early_tags, _ = model.select(list(range(20)), 1)
    assert "hard" not in early_tags
    selected, tags, _ = model.select(list(range(20)), 2)
    retried = [i for i, tag in zip(selected, tags) if tag == "hard"]
    assert len(retried) == 3
    assert not model.hard
    model.observe({i: [0, 0] for i in retried}, 2)
    assert not model.hard  # failed retry does not immediately re-enroll itself
    assert all(model.questions[i]["retired_until"] == 12 for i in retried)


def test_ttl_is_based_on_buffer_entry_not_reset_by_repeated_failures():
    model = make(hard_ttl_steps=3)
    model.observe({20: [0, 0]}, 0)
    model.observe({20: [0, 0]}, 2)
    assert model.hard[20]["entered_step"] == 0
    model.select(list(range(20)), 3)
    assert 20 not in model.hard
    assert model.questions[20]["retired_until"] > 3


def test_capacity_evicts_oldest_and_suppresses_immediate_reentry():
    model = make(hard_capacity=2)
    model.observe({20: [0, 0], 21: [0, 0]}, 0)
    model.observe({22: [0, 0]}, 1)
    assert set(model.hard) == {21, 22}
    assert model.questions[20]["retired_until"] > 1


def test_small_dataset_keeps_batch_when_no_distinct_candidates_exist():
    model = make(size=3)
    model.observe({0: [1, 1], 1: [0, 0], 2: [0, 1]}, 0)
    selected, tags, _ = model.select([0, 1, 2], 3)
    assert selected == [0, 1, 2]
    assert tags == ["fresh"] * 3


def test_no_duplicate_with_both_hard_and_learning_replacements():
    model = make(retry_cooldown_steps=1, stats_ema_beta=0.0)
    model.observe({i: [0, 1] for i in range(40, 60)}, 0)
    model.observe({i: [0, 0] if i < 30 else [1, 1] for i in range(20, 40)}, 1)
    selected, tags, _ = model.select(list(range(20)), 2)
    assert tags.count("hard") == 3
    assert tags.count("learning") == 3
    assert len(selected) == len(set(selected)) == 20
    assert tags.count("fresh") >= 14


def test_json_checkpoint_resume_reproduces_indices_sources_and_metrics():
    original = make(retry_cooldown_steps=1)
    original.observe({i: [0, 0] if i < 40 else [0, 1] for i in range(20, 60)}, 0)
    original.select(list(range(20)), 1)
    state = json.loads(json.dumps(original.state_dict()))
    resumed = make(retry_cooldown_steps=1)
    resumed.load_state_dict(state)
    assert original.select(list(range(20)), 2) == resumed.select(list(range(20)), 2)
    observations = {0: [1, 1], 20: [0, 1], 30: [0, 0]}
    assert original.observe(observations, 2) == resumed.observe(observations, 2)
    assert json.dumps(original.state_dict(), sort_keys=True) == json.dumps(resumed.state_dict(), sort_keys=True)


def test_checkpoint_only_contains_indices_and_statistics_never_response_payloads():
    model = make()
    model.observe({20: [0, 0], 21: [0, 1]}, 0)
    state = model.state_dict()
    assert state["hard"]["20"] == {"entered_step": 0, "attempts": 0, "last_retry_step": 0}
    encoded = json.dumps(state)
    for forbidden in ("response", "input_ids", "log_prob", "rollout", "prompt"):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="binary"):
        model.observe({20: {"response": "old model answer"}}, 1)


def test_changed_dataset_or_configuration_cannot_resume():
    model = make()
    model.observe({20: [0, 0]}, 0)
    state = model.state_dict()
    other_data = MOD.QuestionCurriculum(model.config, dataset_size=200, dataset_fingerprint="changed-order")
    with pytest.raises(ValueError, match="identity"):
        other_data.load_state_dict(state)
    with pytest.raises(ValueError, match="configuration"):
        make(seed=8).load_state_dict(state)


def test_invalid_observation_or_checkpoint_does_not_partially_mutate_state():
    model = make()
    before = model.state_dict()
    with pytest.raises(ValueError):
        model.observe({20: [0, 0], 21: [0.5, 1]}, 0)
    assert model.state_dict() == before
    model.observe({20: [0, 0]}, 0)
    before = model.state_dict()
    corrupt = json.loads(json.dumps(before))
    corrupt["questions"]["20"]["success_ema"] = float("nan")
    with pytest.raises(ValueError):
        model.load_state_dict(corrupt)
    assert model.state_dict() == before


def test_duplicate_rows_out_of_range_and_duplicate_step_are_rejected():
    model = make()
    with pytest.raises(ValueError, match="unique"):
        model.select([0, 0], 0)
    with pytest.raises(ValueError, match="out of bounds"):
        model.select([200], 0)
    model.select([0, 1], 0)
    with pytest.raises(ValueError, match="increasing"):
        model.select([0, 1], 0)
    model.observe({0: [1, 1]}, 0)
    with pytest.raises(ValueError, match="increasing"):
        model.observe({0: [1, 1]}, 0)


@pytest.mark.parametrize("config", [
    {"max_replacement_fraction": 0.31}, {"max_hard_fraction": 0.16},
    {"mastery_keep_probability": 0}, {"hard_capacity": True},
    {"stats_ema_beta": 1}, {"zero_rate_scale": 0},
    {"learning_max_success": 0.99}, {"unknown": True},
])
def test_invalid_governance_options_fail_early(config):
    with pytest.raises(ValueError):
        make(**config)


def test_disabled_preserves_original_selection_and_keeps_buffer_empty():
    model = make(enabled=False)
    model.observe({20: [0, 0]}, 0)
    selected, tags, _ = model.select(list(range(20)), 1)
    assert selected == list(range(20))
    assert tags == ["fresh"] * 20
    assert not model.questions and not model.hard


def test_representative_fresh_slots_are_chosen_independently_of_question_difficulty():
    easy = make(stats_ema_beta=0.0)
    hard = make(stats_ema_beta=0.0)
    # Same RNG seed, identical base batch, opposite mastery history: the fresh
    # representative positions must remain identical, even if replacements differ.
    for step in range(3):
        easy.observe({i: [1, 1] for i in range(20)}, step)
        hard.observe({i: [0, 0] for i in range(20)}, step)
    _, easy_tags, _ = easy.select(list(range(20)), 3)
    _, hard_tags, _ = hard.select(list(range(20)), 3)
    assert [i for i, tag in enumerate(easy_tags) if tag == "fresh"] == [
        i for i, tag in enumerate(hard_tags) if tag == "fresh"]
    assert easy_tags.count("fresh") == hard_tags.count("fresh") == 14


def test_wrapper_adds_row_index_without_mutating_source_and_collates_as_scalar():
    import numpy as np

    source = [{"input_ids": [1, 2], "reward_model": {"ground_truth": "3"}},
              {"input_ids": [4, 5], "reward_model": {"ground_truth": "6"}}]
    wrapped = MOD.IndexedQuestionDataset(source)
    items = [wrapped[0], wrapped[1]]
    assert len(wrapped) == 2
    assert all("curriculum_dataset_index" not in row for row in source)
    indices = np.array([item["curriculum_dataset_index"] for item in items], dtype=object)
    assert indices.shape == (2,)
    assert indices.tolist() == [0, 1]
    assert items[0] is not source[0]
    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored[1] == wrapped[1]
    with pytest.raises(IndexError):
        wrapped[2]
