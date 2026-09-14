#!/usr/bin/env bash
set -euo pipefail

# Edit these values here, or set environment variables before running this script.
# Choose exactly one source. VALUE_INPUT accepts a file, directory, or quoted glob;
# a directory is searched recursively for .json/.jsonl files by the Python loader.
VALUE_INPUT=${VALUE_INPUT:-}
VALUE_MANIFEST=${VALUE_MANIFEST:-}
VALUE_INPUT_FORMAT=${VALUE_INPUT_FORMAT:-auto} # auto | canonical | baseline_nested
# The supplied baseline run used 2 Solver turns; it can initialize absolute-entropy
# supervision, then acquire temporal qualification online during 3-turn training.
VALUE_PRETRAIN_MODE=${VALUE_PRETRAIN_MODE:-absolute} # absolute | full
# Use the maximum Solver turns configured in the HISTORICAL rollout run, not the
# number of turns in one (possibly early-stopped) trajectory or the main-run budget.
# 2 is the user's confirmed historical setting. For another run change it, or set
# an explicit empty string if its input/manifest records the setting. Never guess.
HISTORICAL_MAX_SOLVER_TURNS=${HISTORICAL_MAX_SOLVER_TURNS-2}
VALUE_RUN_ID=${VALUE_RUN_ID:-} # Optional namespace when importing one historical run.

VALUE_ENCODER=${VALUE_ENCODER:-Qwen/Qwen3-4B}
VALUE_OUTPUT=${VALUE_OUTPUT:-} # Empty => <project>/checkpoints/value_init/prefix_value.pt
VALUE_CONFIG=${VALUE_CONFIG:-} # Empty => <project>/examples/drmas_trainer/value_pretrain.yaml
VALUE_DEVICE=${VALUE_DEVICE:-cuda:0}
SEMANTIC_EPOCHS=${SEMANTIC_EPOCHS:-5}
ENTROPY_EPOCHS=${ENTROPY_EPOCHS:-10}
VALUE_ROUNDS=${VALUE_ROUNDS:-1}
VALUE_BATCH_SIZE=${VALUE_BATCH_SIZE:-} # Empty => use the YAML setting.
PYTHON_BIN=${PYTHON_BIN:-python3}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'HELP'
Usage: bash examples/drmas_trainer/pretrain_value.sh [train|validate] [Python options...]

train (default): train and qualify a prefix value model; returns the Python exit code.
validate:       validate/reconstruct trajectories only; does not load the encoder.
--help:         show this message.

Set exactly one data source:
  VALUE_INPUT='/data/baseline/run1'       File, recursive directory, or quoted glob.
  VALUE_MANIFEST='/data/manifest.json'   Existing multi-source manifest format.

Other settings can be edited at the top of this script or set in the environment:
  VALUE_INPUT_FORMAT=auto               auto | canonical | baseline_nested
  VALUE_PRETRAIN_MODE=absolute          absolute | full; full requires temporal qualification.
  HISTORICAL_MAX_SOLVER_TURNS=2          Confirmed supplied baseline budget, not main-run 3.
  VALUE_RUN_ID=...                      Optional historical run namespace.
  VALUE_ENCODER=/models/Qwen3-4B
  VALUE_OUTPUT=/checkpoints/value_init/prefix_value.pt
  VALUE_CONFIG=examples/drmas_trainer/value_pretrain.yaml
  VALUE_DEVICE=cuda:0
  SEMANTIC_EPOCHS=5 ENTROPY_EPOCHS=10 VALUE_ROUNDS=1
  VALUE_BATCH_SIZE=256 PYTHON_BIN=python3 DRY_RUN=1

Examples (for a different historical run, set its actual budget):
  VALUE_INPUT='/data/baseline/run1' HISTORICAL_MAX_SOLVER_TURNS=2 \
    bash examples/drmas_trainer/pretrain_value.sh validate
  VALUE_MANIFEST='/data/manifest.json' DRY_RUN=1 \
    bash examples/drmas_trainer/pretrain_value.sh train

Additional Python options are forwarded unchanged. Existing calls without an
explicit mode (including --validate-only) remain supported. A quoted glob is
passed intact to the Python loader rather than expanded by Bash.
HELP
}

mode=train
case "${1:-}" in
  train|validate) mode=$1; shift ;;
  -h|--help|help) usage; exit 0 ;;
  ""|--*) ;;
  *) printf 'Unknown mode: %s. Expected train or validate.\n' "$1" >&2; exit 2 ;;
esac

if [[ -n "$VALUE_INPUT" && -n "$VALUE_MANIFEST" ]]; then
  printf 'Set only one of VALUE_INPUT and VALUE_MANIFEST.\n' >&2
  exit 2
fi
if [[ -z "$VALUE_INPUT" && -z "$VALUE_MANIFEST" ]]; then
  printf 'Set VALUE_INPUT to trajectory data, or VALUE_MANIFEST to a manifest JSON.\n' >&2
  exit 2
fi
case "$VALUE_INPUT_FORMAT" in
  auto|canonical|baseline_nested) ;;
  *) printf 'Invalid VALUE_INPUT_FORMAT: %s.\n' "$VALUE_INPUT_FORMAT" >&2; exit 2 ;;
esac
case "$VALUE_PRETRAIN_MODE" in
  absolute|full) ;;
  *) printf 'Invalid VALUE_PRETRAIN_MODE: %s. Expected absolute or full.\n' "$VALUE_PRETRAIN_MODE" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$SCRIPT_DIR/../.."
VALUE_OUTPUT=${VALUE_OUTPUT:-$PWD/checkpoints/value_init/prefix_value.pt}
VALUE_CONFIG=${VALUE_CONFIG:-$PWD/examples/drmas_trainer/value_pretrain.yaml}

command=("$PYTHON_BIN" examples/drmas_trainer/pretrain_credit_value.py)
if [[ -n "$VALUE_INPUT" ]]; then
  command+=(--input "$VALUE_INPUT")
else
  command+=(--manifest "$VALUE_MANIFEST")
fi
command+=(--input-format "$VALUE_INPUT_FORMAT" --output "$VALUE_OUTPUT"
  --config "$VALUE_CONFIG" --encoder-model "$VALUE_ENCODER" --device "$VALUE_DEVICE"
  --semantic-epochs "$SEMANTIC_EPOCHS" --entropy-epochs "$ENTROPY_EPOCHS"
  --rounds "$VALUE_ROUNDS")
[[ -z "$HISTORICAL_MAX_SOLVER_TURNS" ]] || command+=(--max-solver-turns "$HISTORICAL_MAX_SOLVER_TURNS")
[[ -z "$VALUE_RUN_ID" ]] || command+=(--run-id "$VALUE_RUN_ID")
[[ -z "$VALUE_BATCH_SIZE" ]] || command+=(--batch-size "$VALUE_BATCH_SIZE")
[[ "$VALUE_PRETRAIN_MODE" != absolute ]] || command+=(--allow-absolute-only)
[[ "$mode" != validate ]] || command+=(--validate-only)
command+=("$@")

if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
