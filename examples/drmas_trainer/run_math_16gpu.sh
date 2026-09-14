#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in train|eval|evaluation) ;; *) echo "Usage: $0 [train|eval] [Hydra overrides...]" >&2; exit 1;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$SCRIPT_DIR/../.."

# Sixteen physical GPUs TOTAL: shared 15-rank Solver/Verifier pool + one value GPU.
# NNODES=2 selects two hosts with eight GPUs each, leaving a 7+8 Actor pool.
NNODES=${NNODES:-1}
case "$NNODES" in
  1) PHYSICAL_GPUS_PER_NODE=16; AGENT_LAYOUT='[15]' ;;
  2) PHYSICAL_GPUS_PER_NODE=8; AGENT_LAYOUT='[7,8]'
     : "${RAY_ADDRESS:?For two nodes, start Ray on both hosts and set RAY_ADDRESS=auto or head:port}" ;;
  *) echo "This launcher supports one 16-GPU host or two 8-GPU hosts." >&2; exit 1 ;;
esac

SOLVER_MODEL=${SOLVER_MODEL:-Qwen/Qwen3-4B}
VERIFIER_MODEL=${VERIFIER_MODEL:-Qwen/Qwen3-4B}
VALUE_ENCODER=${VALUE_ENCODER:-Qwen/Qwen3-4B}
VALUE_CHECKPOINT=${VALUE_CHECKPOINT:-}
RESUME_FROM=${RESUME_FROM:-}
TRAIN_DATA=${TRAIN_DATA:-$HOME/data/drmas_math/train.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-30}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_SOLVER_TURNS=${MAX_SOLVER_TURNS:-3}
if (( TRAIN_BATCH_SIZE <= 0 || GROUP_SIZE <= 0 || TRAIN_BATCH_SIZE % 15 != 0 )); then
  echo 'TRAIN_BATCH_SIZE must be a positive multiple of 15; GROUP_SIZE must be positive.' >&2; exit 1
fi
RUN_NAME=${RUN_NAME:-mix_recovery_entropy_16gpu}
RUN_DIR=${RUN_DIR:-$PWD/checkpoints/$RUN_NAME}
LOGGERS=${LOGGERS:-'[console,wandb]'}
VAL_ONLY=False
RESUME_MODE=disable
INITIAL_CHECKPOINT=null
RESUME_CHECKPOINT=null
if [[ -n "$RESUME_FROM" ]]; then
  [[ -d "$RESUME_FROM" ]] || { echo "Missing checkpoint directory: $RESUME_FROM" >&2; exit 1; }
  RESUME_MODE=resume_path
  RESUME_CHECKPOINT="$RESUME_FROM"
fi
if [[ "$MODE" == train ]]; then
  VAL_DATA=${VAL_DATA:-$HOME/data/drmas_math/test_sampled.parquet}
  VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-110}
  VAL_GROUP_SIZE=${VAL_GROUP_SIZE:-1}
  if [[ -z "$RESUME_FROM" ]]; then
    : "${VALUE_CHECKPOINT:?Set VALUE_CHECKPOINT to an entropy-qualified offline prefix_value.pt}"
    [[ -f "$VALUE_CHECKPOINT" ]] || { echo "Missing VALUE_CHECKPOINT: $VALUE_CHECKPOINT" >&2; exit 1; }
    INITIAL_CHECKPOINT="$VALUE_CHECKPOINT"
  else
    [[ -f "$RESUME_FROM/prefix_value.pt" && -f "$RESUME_FROM/entropy_controller.json" ]] || {
      echo 'Resume needs prefix_value.pt and entropy_controller.json beside the Actor checkpoint.' >&2; exit 1;
    }
  fi
else
  VAL_ONLY=True
  VAL_DATA=${VAL_DATA:-$HOME/data/drmas_math/test.parquet}
  VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-60}
  VAL_GROUP_SIZE=${VAL_GROUP_SIZE:-16}
  : "${RESUME_FROM:?Set RESUME_FROM to the trained global_step_N directory for evaluation}"
fi
[[ -f "$TRAIN_DATA" && -f "$VAL_DATA" ]] || { echo 'TRAIN_DATA and VAL_DATA must be existing parquet files.' >&2; exit 1; }

args=(
  algorithm.adv_estimator=grpo algorithm.group_by_agent_id=True
  algorithm.advantage_recovery.enable=True
  algorithm.advantage_recovery.curriculum.enabled=True
  actor_rollout_ref.actor.advantage_recovery.enable=True
  algorithm.entropy_credit.enable=True algorithm.entropy_credit.sparse.enable=True
  algorithm.entropy_credit.value.enable=True algorithm.entropy_credit.control.enabled=True
  "algorithm.entropy_credit.value.model_path=$VALUE_ENCODER"
  "algorithm.entropy_credit.value.initial_checkpoint=$INITIAL_CHECKPOINT"
  algorithm.entropy_credit.value.device=cuda:0
  algorithm.entropy_credit.value.require_pretrained=True
  "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_DATA"
  "data.train_batch_size=$TRAIN_BATCH_SIZE" "data.val_batch_size=$VAL_BATCH_SIZE"
  data.max_prompt_length=8192 data.max_response_length=4096 data.filter_overlong_prompts=True
  +data.apply_chat_template_kwargs.enable_thinking=False data.truncation=middle data.return_raw_chat=True
  actor_rollout_ref.model.path=null actor_rollout_ref.actor.optim.lr=null
  '+agent.agent_specific_parameters.actor.optim.lr=[1e-6,1e-6]'
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True
  actor_rollout_ref.actor.ppo_mini_batch_size=240
  actor_rollout_ref.actor.ppo_mini_update_num=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null
  '+agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=[4,4]'
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.entropy_control.enabled=True actor_rollout_ref.actor.entropy_control.loss_coef=0.01
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=sglang
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
  env.env_name=math env.seed=0 "env.rollout.n=$GROUP_SIZE" "env.rollout.val_n=$VAL_GROUP_SIZE"
  'agent.agent_ids=["Solver Agent","Verifier Agent"]'
  "agent.model_ids=[\"$SOLVER_MODEL\",\"$VERIFIER_MODEL\"]" agent.model_sharing=False
  agent.orchestra_type=math "agent.orchestra.math.max_loop_num=$MAX_SOLVER_TURNS"
  trainer.critic_warmup=0 "trainer.logger=$LOGGERS" trainer.project_name=DrMAS_math_mix
  "trainer.experiment_name=$RUN_NAME" "trainer.default_local_dir=$RUN_DIR"
  "trainer.rollout_data_dir=$RUN_DIR/rollouts" "trainer.validation_data_dir=$RUN_DIR/validation"
  "trainer.n_gpus_per_node=$PHYSICAL_GPUS_PER_NODE" "trainer.nnodes=$NNODES"
  "trainer.agent_gpus_per_node=$AGENT_LAYOUT" trainer.save_freq=50 trainer.test_freq=10
  trainer.total_epochs=2 "trainer.val_only=$VAL_ONLY" trainer.val_before_train=True
  "trainer.resume_mode=$RESUME_MODE" "trainer.resume_from_path=$RESUME_CHECKPOINT"
)
# DRY_RUN performs file/layout checks and prints the exact command, without Ray.
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' python3 -m verl.trainer.main_ppo "${args[@]}" "$@"
  printf '\n'
else
  exec python3 -m verl.trainer.main_ppo "${args[@]}" "$@"
fi
