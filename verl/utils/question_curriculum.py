"""Bounded question resampling using only current-policy terminal outcomes.

The buffer contains dataset row indices and statistics, never generated text,
tokens, rewards-to-replay, or old log probabilities. Call ``select`` before
loading/tokenizing the selected rows, then generate new Solver/Verifier
trajectories and call ``observe`` with one binary label per unique trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real
import random


DEFAULT_QUESTION_CURRICULUM_CONFIG = {
    "enabled": False,
    "seed": 0,
    "stats_ema_beta": 0.9,
    "max_replacement_fraction": 0.3,
    "max_hard_fraction": 0.15,
    "zero_rate_scale": 0.5,
    "mastery_threshold": 0.95,
    "mastery_streak": 3,
    "mastery_keep_probability": 0.2,
    "mastery_revisit_steps": 20,
    "learning_min_success": 0.2,
    "learning_max_success": 0.8,
    "hard_capacity": 1024,
    "hard_ttl_steps": 100,
    "retry_cooldown_steps": 3,
    "max_hard_retries": 4,
    "retire_cooldown_steps": 20,
}


def normalize_question_curriculum_config(config=None):
    cfg = dict(DEFAULT_QUESTION_CURRICULUM_CONFIG)
    if config is not None:
        supplied = dict(config)
        unknown = supplied.keys() - cfg.keys()
        if unknown:
            raise ValueError(f"Unknown question curriculum options: {sorted(unknown)}")
        cfg.update(supplied)
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("question curriculum enabled must be boolean")
    for key in ("seed", "mastery_streak", "mastery_revisit_steps", "hard_capacity",
                "hard_ttl_steps", "retry_cooldown_steps", "max_hard_retries",
                "retire_cooldown_steps"):
        value = cfg[key]
        minimum = 0 if key == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"question curriculum {key} must be an integer >= {minimum}")
        cfg[key] = int(value)
    for key in ("stats_ema_beta", "max_replacement_fraction", "max_hard_fraction",
                "zero_rate_scale", "mastery_threshold", "mastery_keep_probability",
                "learning_min_success", "learning_max_success"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"question curriculum {key} must be finite")
        cfg[key] = float(value)
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"question curriculum {key} must lie in [0, 1]")
    if cfg["stats_ema_beta"] >= 1 or cfg["zero_rate_scale"] <= 0:
        raise ValueError("EMA beta must be < 1 and zero_rate_scale must be positive")
    if not 0 < cfg["mastery_keep_probability"] <= 1:
        raise ValueError("mastery_keep_probability must be positive to retain review")
    if not cfg["learning_min_success"] < cfg["learning_max_success"] < cfg["mastery_threshold"]:
        raise ValueError("require learning_min_success < learning_max_success < mastery_threshold")
    if cfg["max_replacement_fraction"] > 0.3:
        raise ValueError("at least 70% of each base batch must be retained")
    if cfg["max_hard_fraction"] > min(0.15, cfg["max_replacement_fraction"]):
        raise ValueError("hard replacement must be <= 15% and <= total replacement")
    return cfg


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, (list, tuple)) else value


class IndexedQuestionDataset:
    """Add a stable row index without modifying the underlying training item."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = _integer(index, "dataset row index")
        if index >= len(self):
            raise IndexError(index)
        result = dict(self.dataset[index])
        result["curriculum_dataset_index"] = index
        return result

    def __getattr__(self, name):
        # Avoid delegating pickle protocol methods or recursing before unpickle
        # has restored the dataset attribute.
        if name == "dataset" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


class QuestionCurriculum:
    """A deterministic, checkpointable index-only hard/learning question buffer.

    ``dataset_fingerprint`` must identify the *ordered*, filtered training data
    and its prompt preprocessing configuration. A changed fingerprint or config
    makes resume fail rather than replaying indices against different questions.
    Selection randomly protects at least 70% of the original unique question
    batch before consulting mastery statistics. Only those independently
    protected rows have the ``fresh`` tag used for representative value fitting;
    unused adaptive slots have the ``fresh_adaptive`` tag. Hard
    retry quotas depend on all-failure rate; mastery/learning quotas depend on
    all-success rate. Neither is driven by their combined zero-variance rate.
    """

    STATE_VERSION = 1

    def __init__(self, config=None, *, dataset_size, dataset_fingerprint):
        self.config = normalize_question_curriculum_config(config)
        self.dataset_size = _integer(dataset_size, "dataset_size", 1)
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint.strip():
            raise ValueError("dataset_fingerprint must be a nonempty string")
        self.dataset_fingerprint = dataset_fingerprint
        self.rng = random.Random(self.config["seed"])
        self.questions = {}
        self.hard = {}
        self.success_group_rate = 0.0
        self.failure_group_rate = 0.0
        self.rate_observations = 0
        self.last_select_step = -1
        self.last_observe_step = -1

    def _index(self, index):
        index = _integer(index, "dataset row index")
        if index >= self.dataset_size:
            raise ValueError(f"dataset row index {index} is out of bounds")
        return index

    def _is_mastered(self, record):
        return (record["success_ema"] >= self.config["mastery_threshold"]
                and record["success_streak"] >= self.config["mastery_streak"])

    def _retire(self, index, step):
        self.hard.pop(index, None)
        self.questions[index]["retired_until"] = step + self.config["retire_cooldown_steps"]

    def _expire(self, step):
        for index, entry in list(self.hard.items()):
            if step - entry["entered_step"] >= self.config["hard_ttl_steps"]:
                self._retire(index, step)

    def _new_question(self, step):
        return {"observations": 0, "success_ema": 0.0, "success_streak": 0,
                "failure_streak": 0, "last_observed_step": step,
                "last_selected_step": -1, "retired_until": -1,
                "latest_class": "unobserved"}

    def observe(self, outcomes_by_question, step):
        """Update once per question from *deduplicated complete trajectories*.

        Values are nonempty sequences of binary terminal labels, not action
        rewards or virtual rewards. A single trajectory must appear only once;
        the trainer owns deduplication by trajectory uid before calling here.
        """
        step = _integer(step, "step")
        if step <= self.last_observe_step:
            raise ValueError("question curriculum observe requires increasing steps")
        if not isinstance(outcomes_by_question, Mapping):
            raise ValueError("outcomes_by_question must map dataset indices to binary labels")
        validated = []
        for index, labels in outcomes_by_question.items():
            index = self._index(index)
            if isinstance(labels, (str, bytes, Mapping)):
                raise ValueError("outcomes must contain only binary terminal labels")
            try:
                labels = list(labels)
            except TypeError as exc:
                raise ValueError("outcomes must be a sequence of binary terminal labels") from exc
            if not labels or any(not isinstance(x, Real) or not math.isfinite(x) or x not in (0, 1) for x in labels):
                raise ValueError("outcomes must be a nonempty sequence of binary terminal labels")
            validated.append((index, sum(float(x) for x in labels) / len(labels)))
        # Validate every observation before changing checkpoint state.
        self.last_observe_step = step
        if not self.config["enabled"]:
            return self._metrics()
        self._expire(step)
        if not validated:
            return self._metrics()
        successes, failures = 0, 0
        beta = self.config["stats_ema_beta"]
        for index, success in validated:
            record = self.questions.setdefault(index, self._new_question(step))
            record["success_ema"] = (beta * record["success_ema"] + (1 - beta) * success
                                     if record["observations"] else success)
            record["observations"] += 1
            record["last_observed_step"] = step
            record["success_streak"] = record["success_streak"] + 1 if success == 1 else 0
            record["failure_streak"] = record["failure_streak"] + 1 if success == 0 else 0
            record["latest_class"] = "success" if success == 1 else "failure" if success == 0 else "mixed"
            successes += success == 1
            failures += success == 0
            if success > 0:
                # A retry that now succeeds in even one rollout is no longer an
                # all-failure group. Its next selection can use the learning pool.
                self.hard.pop(index, None)
                record["retired_until"] = -1
            elif index not in self.hard and step >= record["retired_until"]:
                self.hard[index] = {"entered_step": step, "attempts": 0,
                                    "last_retry_step": step}
        for name, fraction in (("success_group_rate", successes / len(validated)),
                               ("failure_group_rate", failures / len(validated))):
            old = getattr(self, name)
            setattr(self, name, beta * old + (1 - beta) * fraction if self.rate_observations else fraction)
        self.rate_observations += 1
        while len(self.hard) > self.config["hard_capacity"]:
            oldest = min(self.hard, key=lambda i: (self.hard[i]["entered_step"], i))
            self._retire(oldest, step)
        return self._metrics()

    def select(self, batch_indices, step):
        """Replace bounded base-batch slots, returning indices/tags/metrics.

        The caller must fetch the returned rows from its current training dataset
        and perform a fresh rollout. A small dataset or exhausted buffer simply
        retains original rows. Duplicate base indices are rejected because they
        already violate the one-question-once-per-batch sampling contract.
        """
        step = _integer(step, "step")
        if step <= self.last_select_step:
            raise ValueError("question curriculum select requires increasing steps")
        indices = [self._index(i) for i in batch_indices]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("base batch must contain nonempty, unique dataset indices")
        self.last_select_step = step
        tags = ["fresh"] * len(indices)
        if not self.config["enabled"]:
            return indices, tags, {**self._metrics(), "curriculum/replacement_fraction": 0.0}
        self._expire(step)
        cfg = self.config
        size = len(indices)
        max_replace = math.floor(size * cfg["max_replacement_fraction"] + 1e-9)
        hard_fraction = cfg["max_hard_fraction"] * min(1.0, self.failure_group_rate / cfg["zero_rate_scale"])
        learning_fraction = (cfg["max_replacement_fraction"] - hard_fraction) * min(
            1.0, self.success_group_rate / cfg["zero_rate_scale"])
        # This random protection must precede every mastery-dependent decision.
        # Keeping only the leftovers of mastery filtering as "fresh" would bias
        # the value model's supposedly representative train/calibration rows.
        adaptive_positions = self.rng.sample(range(size), max_replace)
        # Fractional quotas must not starve a small hard buffer forever. Seeded
        # stochastic rounding preserves the target over time and the per-batch
        # hard/total ceilings; the existing checkpointed RNG makes it resumable.
        def rounded_quota(target):
            whole = math.floor(target + 1e-9)
            return whole + int(self.rng.random() < max(0.0, target - whole))
        hard_quota = min(max_replace, math.floor(size * cfg["max_hard_fraction"] + 1e-9),
                         rounded_quota(size * hard_fraction))
        learning_quota = min(max_replace - hard_quota, rounded_quota(size * learning_fraction))
        for position in adaptive_positions:
            tags[position] = "fresh_adaptive"
        protected, mastered, other = [], [], []
        for position in adaptive_positions:
            index = indices[position]
            record = self.questions.get(index)
            if record is not None and self._is_mastered(record):
                overdue = step - record["last_observed_step"] >= cfg["mastery_revisit_steps"]
                if overdue or self.rng.random() < cfg["mastery_keep_probability"]:
                    protected.append(position)
                else:
                    mastered.append(position)
            else:
                other.append(position)
        self.rng.shuffle(mastered)
        self.rng.shuffle(other)
        slots = mastered + other  # mastered questions are first to be replaced
        occupied = set(indices)  # never reinsert a removed base question this step
        hard_candidates = [i for i, entry in self.hard.items()
                           if i not in occupied
                           and step - entry["last_retry_step"] >= cfg["retry_cooldown_steps"]
                           and entry["attempts"] < cfg["max_hard_retries"]]
        # Prefer less often retried, older entries. Random tie-breaking prevents
        # fixed dataset-index ordering from starving equally eligible questions.
        self.rng.shuffle(hard_candidates)
        hard_candidates.sort(key=lambda i: (self.hard[i]["attempts"], self.hard[i]["last_retry_step"]))
        learning_candidates = [i for i, record in self.questions.items()
                               if i not in occupied and i not in self.hard
                               and record["latest_class"] != "failure"
                               and cfg["learning_min_success"] <= record["success_ema"] <= cfg["learning_max_success"]]
        self.rng.shuffle(learning_candidates)
        learning_candidates.sort(key=lambda i: abs(self.questions[i]["success_ema"] - 0.5))

        def replace(candidate, tag):
            if not slots:
                return False
            position = slots.pop(0)
            indices[position] = candidate
            tags[position] = tag
            occupied.add(candidate)
            return True

        for candidate in hard_candidates[:hard_quota]:
            if not replace(candidate, "hard"):
                break
            entry = self.hard[candidate]
            entry["attempts"] += 1
            entry["last_retry_step"] = step
            if entry["attempts"] >= cfg["max_hard_retries"]:
                self._retire(candidate, step)
        used_learning = 0
        for candidate in learning_candidates[:learning_quota]:
            if not replace(candidate, "learning"):
                break
            used_learning += 1
        # When most observed questions are already solved, replace a limited
        # number of them with uniformly drawn unmastered training questions.
        # Never do this solely because all-failure rate rose.
        fresh_quota = learning_quota - used_learning
        for _ in range(fresh_quota):
            if not slots:
                break
            candidate = self._draw_unmastered(occupied)
            if candidate is None:
                break
            replace(candidate, "fresh_replacement")
        for index in indices:
            if index in self.questions:
                self.questions[index]["last_selected_step"] = step
        counts = {tag: tags.count(tag) for tag in ("fresh", "fresh_adaptive", "hard", "learning", "fresh_replacement")}
        metrics = self._metrics()
        metrics.update({f"curriculum/{tag}_fraction": count / size for tag, count in counts.items()})
        replacements = counts["hard"] + counts["learning"] + counts["fresh_replacement"]
        metrics.update({"curriculum/replacement_fraction": replacements / size,
                        "curriculum/representative_fraction": counts["fresh"] / size,
                        "curriculum/hard_target_fraction": hard_fraction,
                        "curriculum/learning_target_fraction": learning_fraction,
                        "curriculum/mastery_review_preserved": float(len(protected))})
        return indices, tags, metrics

    def _draw_unmastered(self, occupied):
        # Bounded rejection sampling avoids scanning a huge dataset every step.
        for _ in range(min(128, self.dataset_size * 2)):
            index = self.rng.randrange(self.dataset_size)
            record = self.questions.get(index)
            if (index not in occupied and index not in self.hard
                    and (record is None or (record["latest_class"] != "failure" and not self._is_mastered(record)))):
                return index
        # Exact small-data fallback; retaining the base batch is always allowed.
        if self.dataset_size <= 4096:
            candidates = [i for i in range(self.dataset_size) if i not in occupied and i not in self.hard
                          and (i not in self.questions or (self.questions[i]["latest_class"] != "failure"
                                                          and not self._is_mastered(self.questions[i])))]
            if candidates:
                return self.rng.choice(candidates)
        return None

    def _metrics(self):
        return {"curriculum/success_group_rate_ema": self.success_group_rate,
                "curriculum/failure_group_rate_ema": self.failure_group_rate,
                "curriculum/hard_buffer_size": float(len(self.hard)),
                "curriculum/observed_questions": float(len(self.questions)),
                "curriculum/mastered_questions": float(sum(self._is_mastered(r) for r in self.questions.values()))}

    def state_dict(self):
        """Return JSON-serializable state containing no model output payload."""
        return {"version": self.STATE_VERSION, "config": dict(self.config),
                "dataset_size": self.dataset_size, "dataset_fingerprint": self.dataset_fingerprint,
                "rng": self.rng.getstate(),
                "questions": {str(i): deepcopy(r) for i, r in self.questions.items()},
                "hard": {str(i): deepcopy(r) for i, r in self.hard.items()},
                "success_group_rate": self.success_group_rate, "failure_group_rate": self.failure_group_rate,
                "rate_observations": self.rate_observations,
                "last_select_step": self.last_select_step, "last_observe_step": self.last_observe_step}

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported question curriculum checkpoint version")
        if state.get("config") != self.config:
            raise ValueError("question curriculum resume configuration mismatch")
        if (state.get("dataset_size") != self.dataset_size
                or state.get("dataset_fingerprint") != self.dataset_fingerprint):
            raise ValueError("question curriculum resume dataset identity mismatch")
        expected_question_keys = set(self._new_question(0))
        questions, hard = {}, {}
        for key, value in state["questions"].items():
            index = self._index(int(key))
            if set(value) != expected_question_keys:
                raise ValueError("invalid question metadata in curriculum checkpoint")
            for field in ("observations", "success_streak", "failure_streak"):
                _integer(value[field], field)
            for field in ("last_observed_step", "last_selected_step", "retired_until"):
                _integer(value[field], field, -1)
            if (not isinstance(value["success_ema"], Real) or not math.isfinite(value["success_ema"])
                    or not 0 <= value["success_ema"] <= 1
                    or value["latest_class"] not in ("unobserved", "success", "mixed", "failure")):
                raise ValueError("invalid success statistics in curriculum checkpoint")
            questions[index] = deepcopy(value)
        for key, value in state["hard"].items():
            index = self._index(int(key))
            if index not in questions or set(value) != {"entered_step", "attempts", "last_retry_step"}:
                raise ValueError("invalid hard buffer in curriculum checkpoint")
            for field in value:
                _integer(value[field], field)
            if (questions[index]["latest_class"] != "failure"
                    or value["attempts"] >= self.config["max_hard_retries"]):
                raise ValueError("invalid hard retry state in curriculum checkpoint")
            hard[index] = deepcopy(value)
        if len(hard) > self.config["hard_capacity"]:
            raise ValueError("hard buffer exceeds configured capacity")
        rates = [state["success_group_rate"], state["failure_group_rate"]]
        if any(not isinstance(x, Real) or not math.isfinite(x) or not 0 <= x <= 1 for x in rates) or sum(rates) > 1 + 1e-8:
            raise ValueError("invalid curriculum rate statistics")
        rate_observations = _integer(state["rate_observations"], "rate_observations")
        last_select = _integer(state["last_select_step"], "last_select_step", -1)
        last_observe = _integer(state["last_observe_step"], "last_observe_step", -1)
        rng = random.Random()
        rng.setstate(_tuple_tree(state["rng"]))
        # All validation precedes mutation so failed loads keep live state intact.
        self.questions, self.hard, self.rng = questions, hard, rng
        self.success_group_rate, self.failure_group_rate = map(float, rates)
        self.rate_observations = rate_observations
        self.last_select_step, self.last_observe_step = last_select, last_observe
