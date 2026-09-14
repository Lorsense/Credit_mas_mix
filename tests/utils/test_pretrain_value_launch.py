"""Exercise the pretraining Bash entry point without loading any model."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/drmas_trainer/pretrain_value.sh"
SETTINGS = (
    "VALUE_INPUT", "VALUE_MANIFEST", "VALUE_INPUT_FORMAT", "VALUE_PRETRAIN_MODE", "HISTORICAL_MAX_SOLVER_TURNS",
    "VALUE_RUN_ID", "VALUE_ENCODER", "VALUE_OUTPUT", "VALUE_CONFIG", "VALUE_DEVICE",
    "SEMANTIC_EPOCHS", "ENTROPY_EPOCHS", "VALUE_ROUNDS", "VALUE_BATCH_SIZE", "PYTHON_BIN", "DRY_RUN",
)


def bash_path():
    if os.name == "nt" and Path("E:/Git/bin/bash.exe").is_file():
        return "E:/Git/bin/bash.exe"
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash unavailable")
    return executable


def launch(*args, **settings):
    env = {key: value for key, value in os.environ.items() if key not in SETTINGS}
    env.update(settings)
    return subprocess.run(
        [bash_path(), SCRIPT.as_posix(), *args], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=20,
    )


def argv(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip())


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def test_quoted_baseline_glob_and_all_editable_settings():
    args = argv(launch(
        "train", VALUE_INPUT="/data/baseline runs/*.json", VALUE_INPUT_FORMAT="baseline_nested",
        HISTORICAL_MAX_SOLVER_TURNS="4", VALUE_RUN_ID="baseline run 1",
        VALUE_ENCODER="/models/Qwen 3", VALUE_OUTPUT="/checkpoints/value init/value.pt",
        VALUE_CONFIG="/configs/value pretrain.yaml", VALUE_DEVICE="cuda:1", SEMANTIC_EPOCHS="7",
        ENTROPY_EPOCHS="12", VALUE_ROUNDS="2", VALUE_BATCH_SIZE="128", PYTHON_BIN="/tools/python 3",
        DRY_RUN="1",
    ))
    assert args[:2] == ["/tools/python 3", "examples/drmas_trainer/pretrain_credit_value.py"]
    for name, expected in {
        "--input": "/data/baseline runs/*.json", "--input-format": "baseline_nested",
        "--max-solver-turns": "4", "--run-id": "baseline run 1", "--encoder-model": "/models/Qwen 3",
        "--output": "/checkpoints/value init/value.pt", "--config": "/configs/value pretrain.yaml",
        "--device": "cuda:1", "--semantic-epochs": "7", "--entropy-epochs": "12", "--rounds": "2",
        "--batch-size": "128",
    }.items():
        assert option(args, name) == expected
    assert "--manifest" not in args and "--validate-only" not in args


@pytest.mark.parametrize("mode", [(), ("train",), ("validate",)])
def test_manifest_calls_and_optional_values(mode):
    args = argv(launch(*mode, VALUE_MANIFEST="/data/baseline + sup.json", DRY_RUN="1"))
    assert option(args, "--manifest") == "/data/baseline + sup.json"
    assert option(args, "--input-format") == "auto"
    assert option(args, "--semantic-epochs") == "5"
    assert option(args, "--entropy-epochs") == "10"
    assert option(args, "--rounds") == "1"
    assert option(args, "--max-solver-turns") == "2"
    assert "--allow-absolute-only" in args
    assert "--run-id" not in args and "--batch-size" not in args
    assert ("--validate-only" in args) == (mode == ("validate",))


def test_legacy_flags_without_mode_are_forwarded_unchanged():
    args = argv(launch("--validate-only", "--batch-size", "64",
                       VALUE_MANIFEST="/data/manifest.json", DRY_RUN="1"))
    assert args[-3:] == ["--validate-only", "--batch-size", "64"]


@pytest.mark.parametrize("settings,message", [
    ({}, "Set VALUE_INPUT"),
    ({"VALUE_INPUT": "/data/a", "VALUE_MANIFEST": "/data/b"}, "Set only one"),
    ({"VALUE_INPUT": "/data/a", "VALUE_INPUT_FORMAT": "unknown"}, "Invalid VALUE_INPUT_FORMAT"),
    ({"VALUE_INPUT": "/data/a", "VALUE_PRETRAIN_MODE": "unknown"}, "Invalid VALUE_PRETRAIN_MODE"),
])
def test_invalid_sources_fail_before_launch(settings, message):
    result = launch(DRY_RUN="1", **settings)
    assert result.returncode == 2
    assert message in result.stderr


def test_help_requires_no_data_and_unknown_mode_fails():
    result = launch("--help")
    assert result.returncode == 0
    assert "HISTORICAL_MAX_SOLVER_TURNS" in result.stdout
    result = launch("evaluate")
    assert result.returncode == 2 and "Unknown mode" in result.stderr


@pytest.mark.parametrize("pretrain_mode", ["absolute", "full"])
def test_absolute_and_full_qualification_modes(pretrain_mode):
    args = argv(launch("train", VALUE_INPUT="/data/baseline", VALUE_PRETRAIN_MODE=pretrain_mode,
                       HISTORICAL_MAX_SOLVER_TURNS="", DRY_RUN="1"))
    assert ("--allow-absolute-only" in args) == (pretrain_mode == "absolute")
    # Explicit empty budget defers to metadata, preserving mixed-budget manifests.
    assert "--max-solver-turns" not in args


@pytest.mark.parametrize("returncode", [0, 2, 7])
def test_exec_preserves_exact_argument_boundaries_and_exit_code(tmp_path, returncode):
    # A fake interpreter avoids imports, CUDA and MSYS native-argv conversion.
    fake = tmp_path / "fake python"
    fake.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\"\nexit " + str(returncode) + "\n",
        encoding="utf-8", newline="\n",
    )
    fake.chmod(0o755)
    result = launch(
        "validate", "--encoder-model", "literal $(echo unsafe) `echo unsafe`",
        VALUE_INPUT="/data/history with spaces/*.jsonl", HISTORICAL_MAX_SOLVER_TURNS="3",
        PYTHON_BIN=fake.as_posix(),
    )
    assert result.returncode == returncode, result.stderr
    args = result.stdout.split("\0")[:-1]
    assert args[0] == "examples/drmas_trainer/pretrain_credit_value.py"
    assert option(args, "--input") == "/data/history with spaces/*.jsonl"
    assert option(args, "--max-solver-turns") == "3"
    assert "--validate-only" in args
    assert args[-2:] == ["--encoder-model", "literal $(echo unsafe) `echo unsafe`"]


def test_shell_has_lf_line_endings_and_valid_syntax():
    assert b"\r" not in SCRIPT.read_bytes()
    result = subprocess.run([bash_path(), "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
