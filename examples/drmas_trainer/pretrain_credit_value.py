#!/usr/bin/env python3
"""Pretrain entropy-aware prefix value on one or more historical data sources.

Accepts complete canonical trajectory records or canonical action rows. Legacy
logs require an explicit manifest field_map/defaults/question_map; prompt text
is never truncated or guessed. Different real solver budgets may coexist.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def _utilities():
    path = Path(__file__).resolve().parents[2] / "verl/utils/value_credit.py"
    spec = importlib.util.spec_from_file_location("offline_value_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_adapter():
    path = Path(__file__).resolve().parents[2] / "verl/utils/offline_trajectories.py"
    spec = importlib.util.spec_from_file_location("offline_trajectory_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path):
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON/YAML mapping")
    if "algorithm" in config:
        config = config["algorithm"]["entropy_credit"]["value"]
    elif "entropy_credit" in config:
        config = config["entropy_credit"]["value"]
    return dict(config)


def _get_path(row, path):
    result = row
    for key in path.split("."):
        result = result[key]
    return result


def load_records(patterns=None, manifest=None, *, input_format="auto", max_solver_turns=None, run_id=None):
    """Return validated records and a read-only data provenance summary.

    Manifest format: {"sources":[{"input":"baseline/*.jsonl", "run_id":"b1",
    "step":250, "defaults":{"value_max_solver_turns":2},
    "field_map":{"value_question":"question", "value_action_text":"answer"},
    "question_map":{"dataset_question_id":"exact question text"}}]}.
    field_map maps canonical destination fields to dotted source field paths.
    """
    if input_format not in ("auto", "canonical", "baseline_nested"):
        raise ValueError("input_format must be auto, canonical, or baseline_nested")
    adapter = _input_adapter()
    if max_solver_turns is not None:
        adapter.positive_integer(max_solver_turns, "historical max_solver_turns")
    base = Path.cwd()
    sources = [{"input": pattern} for pattern in (patterns or [])]
    if manifest:
        manifest = Path(manifest).resolve()
        document = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or not isinstance(document.get("sources"), list):
            raise ValueError("manifest must contain a sources list")
        base = manifest.parent
        sources.extend(document["sources"])
    if not sources:
        raise ValueError("provide --input or --manifest")
    util = _utilities()
    rows, files, observed_steps = [], [], set()
    format_counts = {"canonical": 0, "baseline_nested": 0}
    deduplication = {"input_actions": 0, "removed_solver_actions": 0, "affected_trajectories": 0}
    for source in sources:
        source = dict(source)
        if run_id is not None:
            source.setdefault("run_id", run_id)
        selected_format = source.get("format", input_format)
        if selected_format not in ("auto", "canonical", "baseline_nested"):
            raise ValueError(f"unknown source format: {selected_format}")
        pattern = source["input"]
        matches = adapter.input_files(pattern, base)
        if not matches:
            raise ValueError(f"No files match {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path in files:
                raise ValueError(f"input file selected more than once: {path}")
            files.append(path)
            for line_number, document in adapter.json_documents(path):
                try:
                    nested = selected_format == "baseline_nested" or (selected_format == "auto" and "trajectories" in document)
                    originals = (adapter.baseline_action_rows(document, source, max_solver_turns,
                                                              deduplication=deduplication)
                                 if nested else [document])
                    format_counts["baseline_nested" if nested else "canonical"] += 1
                    for original in originals:
                        row = {**source.get("defaults", {}), **original}
                        for destination, origin in source.get("field_map", {}).items():
                            row[destination] = _get_path(original, origin)
                        run = str(row.get("run_id") or source.get("run_id") or path)
                        step = str(row.get("step", source.get("step", "unspecified")))
                        observed_steps.add((run, step))
                        namespace = hashlib.sha256((run + "\0" + step).encode()).hexdigest()[:20]
                        if isinstance(row.get("actions"), list):
                            question = row.get("question", source.get("question_map", {}).get(str(row.get("question_id"))))
                            budget = row.get("max_solver_turns", row.get("value_max_solver_turns", max_solver_turns))
                            for index, action in enumerate(row["actions"]):
                                rows.append({"traj_uid": namespace + ":" + str(row["traj_uid"]),
                                             "uid": row.get("uid", row.get("question_id", question)),
                                             "value_question": question, "value_max_solver_turns": budget,
                                             "value_action_index": index, "value_action_text": action["text"],
                                             "agent_id": action["role"], "role_turn_index": action.get("role_turn_index", index // 2),
                                             "is_action_valid": action.get("valid", True), "pass": row["label"],
                                             "top16_entropy_mean": action.get("entropy_mean", action.get("top16_entropy_mean")),
                                             "top16_entropy": {"coverage": action.get("entropy_coverage")},
                                             "pure_entropy_response_tokens": action.get("entropy_token_count"),
                                             "pure_entropy_truncated": action.get("truncated", False)})
                        else:
                            if not row.get("value_question"):
                                row["value_question"] = source.get("question_map", {}).get(str(row.get("uid")))
                            if max_solver_turns is not None:
                                row.setdefault("value_max_solver_turns", max_solver_turns)
                            required = util._RECORD_FIELDS + ("value_max_solver_turns",)
                            missing = [key for key in required if key not in row or row[key] is None]
                            if missing:
                                raise ValueError(f"missing canonical fields {missing}; supply explicit field_map/defaults/question_map")
                            row["traj_uid"] = namespace + ":" + str(row["traj_uid"])
                            rows.append(row)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("input contains no trajectories")
    fields = set(util._RECORD_FIELDS) | {"value_max_solver_turns", "top16_entropy_mean", "top16_entropy",
                                       "pure_entropy_response_tokens", "pure_entropy_truncated"}
    columns = {key: [row.get(key, False if key == "pure_entropy_truncated" else None) for row in rows] for key in fields}
    records, metrics = util.build_trajectory_records(columns, None)
    if metrics["value_credit/skipped_trajectories"]:
        raise ValueError(f"ambiguous, incomplete or conflicting historical trajectories: {metrics}")
    if not records:
        raise ValueError("no valid complete trajectories")
    entropy_actions = sum(action.get("entropy_mean") is not None for record in records for action in record["actions"])
    if not entropy_actions:
        raise ValueError("entropy-aware pretraining requires recorded action mean Top-16 entropy")
    prefixes = {role: {"absolute_nonterminal": 0, "temporal_nonterminal": 0} for role in ("solver", "verifier")}
    budgets = {}
    for record in records:
        budget = record["max_solver_turns"]
        budgets[str(budget)] = budgets.get(str(budget), 0) + 1
        features = util.prefix_entropy_features(record["actions"])
        for i, action in enumerate(record["actions"][:-1]):
            role = action["role"]
            offset = 0 if role == "solver" else 6
            prefixes[role]["absolute_nonterminal"] += int(features["absolute"][i + 1, offset + 5])
            prefixes[role]["temporal_nonterminal"] += int(features["temporal"][i + 1, offset + 5])
    successes = sum(record["label"] == 1 for record in records)
    return records, {"files": [str(p) for p in files], "file_count": len(files),
                     "format_documents": format_counts, "run_step_count": len(observed_steps),
                     "trajectories": len(records), "unique_questions": len({r["question_key"] for r in records}),
                     "successful_trajectories": successes, "failed_trajectories": len(records) - successes,
                     "historical_solver_budgets": budgets, "eligible_prefixes": prefixes,
                     "actions": metrics["value_credit/record_unique_actions"],
                     "entropy_actions": entropy_actions, "reconstruction": metrics,
                     "deduplication": deduplication}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--manifest")
    parser.add_argument("--input-format", choices=("auto", "canonical", "baseline_nested"), default="auto")
    parser.add_argument("--max-solver-turns", type=int, help="actual historical Solver budget; never use the future RL budget")
    parser.add_argument("--run-id", help="stable identity shared by all global_step files in one historical run")
    parser.add_argument("--allow-absolute-only", action="store_true", help="pretrain qualified absolute entropy; enable temporal control only after online qualification")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--encoder-model")
    parser.add_argument("--device")
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--entropy-epochs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.semantic_epochs, args.entropy_epochs, args.rounds, args.batch_size or 1) < 1:
        parser.error("epochs, rounds and batch size must be positive")
    try:
        records, provenance = load_records(args.input, args.manifest, input_format=args.input_format,
                                           max_solver_turns=args.max_solver_turns, run_id=args.run_id)
        print(json.dumps(provenance, ensure_ascii=False), flush=True)
        if args.validate_only:
            return 0
        config = load_config(args.config)
        if args.allow_absolute_only:
            config["allow_absolute_only_pretrain"] = True
        for arg, key in ((args.encoder_model, "model_path"), (args.device, "device"), (args.batch_size, "train_batch_size")):
            if arg is not None:
                config[key] = arg
        if not config.get("model_path"):
            raise ValueError("provide --encoder-model or a model_path config")
        # Explicitly retain every offline row; main training keeps its own
        # smaller runtime replay size when loading this initialization.
        config["replay_max_trajectories"] = max(len(records), int(config.get("replay_max_trajectories", 2048)))
        project = str(Path(__file__).resolve().parents[2])
        if project not in sys.path:
            sys.path.insert(0, project)
        worker_spec = importlib.util.spec_from_file_location("offline_prefix_value", Path(project) / "verl/workers/credit_value.py")
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        scorer = worker.PrefixValueScorer(config)
        prepared = scorer.prepare(records)
        if prepared["metrics"]["skipped_invalid"]:
            raise ValueError("encoder rejected historical records; inspect canonical data before training")
        metrics = scorer.pretrain(args.semantic_epochs, args.entropy_epochs, args.rounds)
        # An unqualified artifact remains inspectable, but main training rejects
        # it. The exit code prevents launch scripts treating file existence as ready.
        scorer.save(args.output)
        report = {"ready": scorer.ready, "preparation": prepared["metrics"], "training": metrics,
                  "provenance": provenance, "feature_schema": "causal-role-mean-top16-v2"}
        Path(args.output + ".report.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_safe(report), allow_nan=False), flush=True)
        return 0 if scorer.ready else 2
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
