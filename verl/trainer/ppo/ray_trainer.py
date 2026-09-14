# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    compute_pass_at_k_and_avg_at_k,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.entropy_credit import compute_entropy_credit_multipliers, first_unique_action_indices
from verl.utils.sparse_entropy_credit import (
    prepare_sparse_entropy_credit,
    finalize_sparse_entropy_credit,
    validate_sparse_entropy_config,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.credit_resources import agent_world_size, validate_pool_layout
from verl.utils.value_credit import build_trajectory_records, attach_value_predictions
from verl.utils.entropy_control import EntropyController
from verl.utils.advantage_recovery import (
    logical_action_indices, recover_collapsed_advantages, validate_advantage_recovery_config,
)
from verl.utils.question_curriculum import QuestionCurriculum, IndexedQuestionDataset
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch, split_batch_by_wg_ids, combine_batches
from agent_system.agent.utils import build_wg_ids, normalize_agent_id, normalize_model_id

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        validate_pool_layout(ray.state.available_resources_per_node(), self.resource_pool_spec, cpus_per_gpu=1)


def value_scorer_metrics(phase: str, raw: dict) -> dict[str, float]:
    """Log the deployed scorer separately from candidate training diagnostics."""
    metrics = {
        f"value_model/{phase}/{key}": float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float, np.number)) and np.isfinite(value)
    }
    if "status" in raw:
        for status in ("insufficient_train", "insufficient_validation", "deployed",
                       "candidate_rejected", "candidate_worse", "insufficient_data_retained_deployment"):
            metrics[f"value_model/{phase}/status_{status}"] = float(raw["status"] == status)
    return metrics


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, group_by_agent_id=False, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # Determine grouping strategy based on configuration
    if group_by_agent_id:
        # Group by both uid and agent_id
        group_index = np.array([f"{uid}_{agent_id}" for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"])], dtype=object)
    else:
        # Use original uid grouping
        group_index = data.non_tensor_batch["uid"]
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_algos.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            group_by_agent_id=group_by_agent_id,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def compute_advantage_with_recovery(data: DataProto, recovery_config, **advantage_kwargs):
    """Compute real GRPO once per logical action, then restore only collapsed groups."""
    rewards = data.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
    unique, inverse = logical_action_indices(data.non_tensor_batch, rewards)
    unique_data = compute_advantage(data.select_idxs(unique), **advantage_kwargs)
    index = torch.as_tensor(inverse, device=unique_data.batch["advantages"].device)
    for key in ("advantages", "returns"):
        data.batch[key] = unique_data.batch[key][index].clone()
    # float32 mean/std can manufacture a nonzero advantage for identical
    # shaped rewards (e.g. eight 0.9 values). Correct only exact constants;
    # preserve every truly differing reward group and the real return tensor.
    groups = defaultdict(list)
    metadata = data.non_tensor_batch
    for row in unique:
        groups[(str(metadata["uid"][row]), str(metadata["agent_id"][row]))].append(int(row))
    constant_ordinals = np.zeros(len(unique), dtype=bool)
    ordinal = {int(row): position for position, row in enumerate(unique)}
    roundoff_groups = 0
    for rows in groups.values():
        if len(rows) > 1 and np.all(rewards[rows] == rewards[rows[0]]):
            roundoff_groups += int(bool(data.batch["advantages"][rows].ne(0).any()))
            constant_ordinals[[ordinal[row] for row in rows]] = True
    constant_rows = torch.as_tensor(constant_ordinals[inverse], device=index.device)
    data.batch["advantages"][constant_rows] = 0
    mask = data.batch["response_mask"]
    if advantage_kwargs.get("multi_turn", False):
        mask = data.batch["loss_mask"][:, -data.batch["responses"].shape[-1]:]
    lengths = mask.sum(-1)
    base = (data.batch["advantages"] * mask).sum(-1) / lengths.clamp_min(1)
    arrays, metrics = recover_collapsed_advantages(
        data.non_tensor_batch, rewards, base.detach().cpu().numpy(),
        lengths.detach().cpu().numpy(), recovery_config,
    )
    metrics["advantage_recovery/roundoff_constant_groups_corrected"] = float(roundoff_groups)
    # Preserve real returns and token rewards. The virtual sample has no row.
    recovered = torch.as_tensor(arrays["advantages"], dtype=data.batch["advantages"].dtype,
                                device=mask.device)
    changed = torch.as_tensor(np.isin(arrays["kind"], ("success", "failure")), device=mask.device)
    data.batch["advantages"][changed] = (recovered.unsqueeze(-1) * mask)[changed]
    data.batch["policy_action_weight"] = torch.as_tensor(
        arrays["policy_action_weight"], dtype=torch.float32, device=mask.device).detach()
    for key in ("kind", "virtual_advantage", "recovery_weight", "policy_action_weight"):
        data.non_tensor_batch["recovery_" + key] = arrays[key]
    data.non_tensor_batch["recovery_base_advantage"] = base.detach().cpu().numpy()
    return data, metrics


def prepare_entropy_credit(data: DataProto, entropy_credit_config) -> tuple[DataProto, dict[str, float]]:
    """Compute detached entropy-credit factors before any batch padding/copying.

    ``adjust_batch`` may randomly duplicate actions independently for each
    worker group.  Ranking must therefore happen on the original rollout set;
    the resulting non-tensor factors then follow each action through all later
    split, copy, concat, and reorder operations.
    """

    required = {
        "uid",
        "traj_uid",
        "agent_id",
        "role_turn_index",
        "pass",
        "top16_entropy",
        "top16_entropy_mean",
    }
    missing = required.difference(data.non_tensor_batch)
    if missing:
        raise KeyError(f"entropy credit is missing rollout fields: {sorted(missing)}")

    terminal_success = np.empty(len(data), dtype=bool)
    for row, value in enumerate(data.non_tensor_batch["pass"]):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pass[{row}] must be a binary terminal outcome, got {value!r}") from exc
        if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
            raise ValueError(f"pass[{row}] must be 0 or 1, got {value!r}")
        terminal_success[row] = bool(numeric_value)

    action_valid = np.asarray(
        data.non_tensor_batch.get("is_action_valid", np.ones(len(data), dtype=bool)),
        dtype=bool,
    )
    multipliers = compute_entropy_credit_multipliers(
        prompt_group_ids=data.non_tensor_batch["uid"],
        trajectory_ids=data.non_tensor_batch["traj_uid"],
        agent_ids=data.non_tensor_batch["agent_id"],
        role_turn_indices=data.non_tensor_batch["role_turn_index"],
        terminal_success=terminal_success,
        action_entropies=data.non_tensor_batch["top16_entropy_mean"],
        action_valid=action_valid,
        action_scale=float(entropy_credit_config.get("action_scale", 0.2)),
        # Sparse stage 2 replaces the legacy delta multiplier; stage 1 is unchanged.
        trajectory_scale=(
            0.0 if entropy_credit_config.get("sparse", {}).get("enable", False)
            else float(entropy_credit_config.get("trajectory_scale", 0.1))
        ),
        trajectory_deadzone=float(entropy_credit_config.get("trajectory_deadzone", 0.05)),
        multiplier_min=float(entropy_credit_config.get("multiplier_min", 0.8)),
        multiplier_max=float(entropy_credit_config.get("multiplier_max", 1.2)),
        final_multiplier_min=float(entropy_credit_config.get("final_multiplier_min", 0.8)),
        final_multiplier_max=float(entropy_credit_config.get("final_multiplier_max", 1.2)),
        relative_epsilon=float(entropy_credit_config.get("relative_epsilon", 1e-12)),
    )

    data.non_tensor_batch["entropy_credit_terminal_success"] = terminal_success
    data.non_tensor_batch["entropy_credit_action_multiplier"] = multipliers["action"]
    data.non_tensor_batch["entropy_credit_trajectory_multiplier"] = multipliers["trajectory"]
    data.non_tensor_batch["entropy_credit_final_multiplier"] = multipliers["final"]
    data.non_tensor_batch["entropy_credit_relative_change"] = multipliers["relative_change"]

    entropies = np.asarray(data.non_tensor_batch["top16_entropy_mean"], dtype=np.float64)
    entropy_valid = np.asarray(multipliers["entropy_valid"], dtype=bool)
    roles = np.asarray(data.non_tensor_batch["agent_id"], dtype=object)
    turns = np.asarray(data.non_tensor_batch["role_turn_index"], dtype=np.int64)
    entropy_stats = data.non_tensor_batch["top16_entropy"]
    raw_final = multipliers["action"] * multipliers["trajectory"]

    metrics: dict[str, float] = {}

    def add_scope_metrics(prefix: str, scope: np.ndarray) -> None:
        valid_scope = scope & action_valid
        entropy_scope = scope & entropy_valid
        valid_count = int(valid_scope.sum())
        metrics[f"{prefix}/action_count"] = float(scope.sum())
        metrics[f"{prefix}/valid_action_fraction"] = float(valid_count / max(int(scope.sum()), 1))
        metrics[f"{prefix}/entropy_action_coverage"] = float(entropy_scope.sum() / max(valid_count, 1))

        if entropy_scope.any():
            values = entropies[entropy_scope]
            metrics[f"{prefix}/top16_entropy_mean"] = float(values.mean())
            metrics[f"{prefix}/top16_entropy_std"] = float(values.std(ddof=0))
            metrics[f"{prefix}/top16_entropy_p10"] = float(np.quantile(values, 0.10))
            metrics[f"{prefix}/top16_entropy_p50"] = float(np.quantile(values, 0.50))
            metrics[f"{prefix}/top16_entropy_p90"] = float(np.quantile(values, 0.90))
            for outcome_name, outcome in (("success", True), ("failure", False)):
                outcome_scope = entropy_scope & (terminal_success == outcome)
                if outcome_scope.any():
                    metrics[f"{prefix}/{outcome_name}_top16_entropy_mean"] = float(
                        entropies[outcome_scope].mean()
                    )

            token_coverages = []
            effective_supports = []
            for row in np.flatnonzero(entropy_scope):
                stats = entropy_stats[row]
                if isinstance(stats, dict) and stats.get("coverage") is not None:
                    token_coverages.append(float(stats["coverage"]))
                if isinstance(stats, dict) and stats.get("effective_support") is not None:
                    effective_supports.append(float(stats["effective_support"]))
            if token_coverages:
                metrics[f"{prefix}/top16_token_coverage_mean"] = float(np.mean(token_coverages))
            if effective_supports:
                metrics[f"{prefix}/top16_effective_support_mean"] = float(np.mean(effective_supports))

        if valid_scope.any():
            for name in ("action", "trajectory", "final"):
                values = multipliers[name][valid_scope]
                metrics[f"{prefix}/{name}_multiplier_mean"] = float(values.mean())
                metrics[f"{prefix}/{name}_multiplier_min"] = float(values.min())
                metrics[f"{prefix}/{name}_multiplier_max"] = float(values.max())
            if not entropy_credit_config.get("sparse", {}).get("enable", False):
                metrics[f"{prefix}/final_clip_fraction"] = float(
                    np.mean(~np.isclose(raw_final[valid_scope], multipliers["final"][valid_scope]))
                )

        later_scope = valid_scope & (turns > 0)
        observed_scope = later_scope & np.isfinite(multipliers["relative_change"])
        metrics[f"{prefix}/trajectory_eligible_fraction"] = float(
            later_scope.sum() / max(valid_count, 1)
        )
        metrics[f"{prefix}/trajectory_observed_fraction"] = float(
            observed_scope.sum() / max(int(later_scope.sum()), 1)
        )
        if observed_scope.any():
            changes = multipliers["relative_change"][observed_scope]
            deadzone = float(entropy_credit_config.get("trajectory_deadzone", 0.05))
            metrics[f"{prefix}/entropy_rise_fraction"] = float(np.mean(changes > deadzone))
            metrics[f"{prefix}/entropy_fall_fraction"] = float(np.mean(changes < -deadzone))
            metrics[f"{prefix}/entropy_deadzone_fraction"] = float(np.mean(np.abs(changes) <= deadzone))
            metrics[f"{prefix}/trajectory_active_fraction"] = float(
                np.mean(~np.isclose(multipliers["trajectory"][observed_scope], 1.0))
            )

    add_scope_metrics("entropy_credit/global", np.ones(len(data), dtype=bool))
    for role in np.unique(roles):
        role_key = normalize_agent_id(str(role))
        add_scope_metrics(f"entropy_credit/{role_key}", roles == role)

    return data, metrics


def prepare_sparse_entropy_credit_for_batch(data: DataProto, sparse_config) -> tuple[DataProto, dict[str, float]]:
    """Freeze pair statistics on logical rollout actions before distributed padding."""
    response_width = data.batch["responses"].shape[-1]
    response_mask = (
        compute_response_mask(data) if response_width
        else data.batch["attention_mask"][:, :0]
    )
    response_tokens = response_mask.sum(-1).detach().cpu().numpy().astype(np.int64)
    data.non_tensor_batch["pure_entropy_response_tokens"] = response_tokens
    # No finish_reason is propagated by this rollout path. Full-width responses
    # are conservatively excluded, including those ending exactly at the limit.
    data.non_tensor_batch["pure_entropy_truncated"] = response_tokens >= response_width
    prepared, metrics = prepare_sparse_entropy_credit(data.non_tensor_batch, sparse_config)
    data.non_tensor_batch.update(prepared)
    return data, metrics


def finalize_sparse_entropy_credit_for_advantages(
    data: DataProto, sparse_config
) -> tuple[DataProto, dict[str, float]]:
    """Select gates with actual GRPO signs, then merge into the untouched stage 1."""
    advantages = data.batch["advantages"].detach().float()
    response_mask = data.batch["response_mask"].bool()
    if advantages.shape != response_mask.shape:
        raise ValueError("pure entropy credit requires advantages and response_mask with the same shape")
    token_counts = response_mask.sum(-1)
    # torch.where avoids contaminating a valid action with NaNs on masked pads.
    base_advantages = (
        torch.where(response_mask, advantages, torch.zeros_like(advantages)).sum(-1)
        / token_counts.clamp_min(1)
    ).cpu().numpy()
    result = finalize_sparse_entropy_credit(data.non_tensor_batch, base_advantages, sparse_config)
    data.non_tensor_batch.update(result["metadata"])
    data.non_tensor_batch["entropy_credit_trajectory_multiplier"] = result["trajectory_multipliers"]
    data.non_tensor_batch["entropy_credit_final_multiplier"] = result["final_multipliers"]
    metrics = dict(result["metrics"])

    # Report the factors actually used by PPO, once per logical action. These
    # overwrite the provisional stage-1-only summaries produced before GRPO.
    unique = first_unique_action_indices(
        data.non_tensor_batch["traj_uid"],
        data.non_tensor_batch["agent_id"],
        data.non_tensor_batch["role_turn_index"],
    )
    roles = np.asarray(data.non_tensor_batch["agent_id"], dtype=object)[unique]
    valid = np.asarray(
        data.non_tensor_batch.get("is_action_valid", np.ones(len(data), dtype=bool)), dtype=bool
    )[unique]
    lower = float(sparse_config.get("multiplier_min", 0.8))
    upper = float(sparse_config.get("multiplier_max", 1.2))
    scopes = [("global", np.ones(len(unique), dtype=bool))]
    scopes.extend((normalize_agent_id(str(role)), roles == role) for role in np.unique(roles))
    for scope_name, scope in scopes:
        rows = unique[scope & valid]
        prefix = f"entropy_credit/{scope_name}"
        for name in ("action", "trajectory", "final"):
            values = np.asarray(data.non_tensor_batch[f"entropy_credit_{name}_multiplier"])[rows]
            for statistic, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                metrics[f"{prefix}/{name}_multiplier_{statistic}"] = float(fn(values)) if len(values) else 1.0
        final = np.asarray(result["final_multipliers"])[rows]
        effective = np.asarray(result["trajectory_multipliers"])[rows]
        metrics[f"{prefix}/trajectory_active_fraction"] = (
            float(np.mean(np.abs(effective - 1.0) > 1e-12)) if len(rows) else 0.0
        )
        # Touching a bound differs from a clip operation or a normalization shift.
        metrics[f"{prefix}/final_bound_fraction"] = (
            float(np.mean(np.isclose(final, lower) | np.isclose(final, upper))) if len(rows) else 0.0
        )
    return data, metrics


def apply_entropy_credit_to_advantages(data: DataProto) -> DataProto:
    """Scale only the already-computed advantages by a detached action scalar."""

    if "advantages" not in data.batch:
        raise KeyError("base advantages must be computed before entropy credit is applied")
    if "entropy_credit_final_multiplier" not in data.non_tensor_batch:
        raise KeyError("entropy-credit factors must be prepared before batch adjustment")
    advantages = data.batch["advantages"]
    final_multiplier = torch.as_tensor(
        np.asarray(data.non_tensor_batch["entropy_credit_final_multiplier"], dtype=np.float64),
        device=advantages.device,
        dtype=advantages.dtype,
    ).detach()
    if final_multiplier.shape != (len(data),):
        raise ValueError(
            f"entropy-credit multiplier has shape {final_multiplier.shape}, expected ({len(data)},)"
        )
    data.batch["advantages"] = advantages * final_multiplier.unsqueeze(-1)
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizers,
        default_wg,
        wg_to_agents_mapping: dict[str, list[dict[str, str]]],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processors=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizers = tokenizers
        self.processors = processors
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        # if self.hybrid_engine:
        #     assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.multi_agent = self.config.agent.multi_agent
        assert self.multi_agent
        self.agent_ids = self.config.agent.agent_ids
        self.model_ids = self.config.agent.model_ids
        assert len(self.agent_ids) == len(self.model_ids), "agent_ids and model_ids must have the same length"

        self.model_sharing = self.config.agent.model_sharing
        self.wg_to_agents_mapping = wg_to_agents_mapping

        self.default_wg = default_wg

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_multi_agent_config()
        self._validate_entropy_credit_config()
        self.value_scorer = None
        self.value_scorer_ready = False
        self.value_scorer_reliability = 0.0
        self.value_scorer_temporal_reliability = {}
        self.entropy_controller = None
        self._validate_value_control_config()
        self.question_curriculum = None
        self._validate_advantage_recovery_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
    

    def _validate_advantage_recovery_config(self):
        cfg = self.config.algorithm.get("advantage_recovery", {})
        validate_advantage_recovery_config(cfg)
        self.advantage_recovery_enabled = bool(cfg.get("enable", False))
        actor_cfg = self.config.actor_rollout_ref.actor
        actors = [actor_cfg]
        actors.extend(agent["config_actor_rollout_ref"].actor
                      for agents in getattr(self, "wg_to_agents_mapping", {}).values() for agent in agents)
        for actor in actors:
            if self.advantage_recovery_enabled != bool(actor.get("advantage_recovery", {}).get("enable", False)):
                raise ValueError("Trainer and every Agent Actor advantage_recovery.enable must agree")
        if cfg.get("curriculum", {}).get("enabled", False) and not self.advantage_recovery_enabled:
            raise ValueError("question curriculum requires advantage recovery")
        if not self.advantage_recovery_enabled:
            return
        if (self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO
                or not self.config.algorithm.group_by_agent_id
                or not self.config.algorithm.norm_adv_by_std_in_grpo
                or self.config.algorithm.use_pf_ppo):
            raise ValueError("advantage recovery requires standard per-agent GRPO with std normalization")
        if self.config.agent.orchestra_type != "math" or self.config.algorithm.filter_groups.enable:
            raise ValueError("advantage recovery requires math rollouts with filter_groups disabled")
        for actor in actors:
            if (actor.loss_agg_mode != "token-mean" or actor.strategy not in ("fsdp", "fsdp2")
                    or actor.get("ulysses_sequence_parallel_size", 1) != 1):
                raise ValueError("advantage recovery requires FSDP token-mean loss and sequence parallel size 1")
        credit = self.config.algorithm.get("entropy_credit", {})
        if credit.get("enable", False):
            bound = max(float(credit.get("final_multiplier_max", 1.2)),
                        float(credit.get("sparse", {}).get("multiplier_max", 1.2)))
            if float(cfg.get("pure_factor_bound", 1.2)) < bound:
                raise ValueError("recovery pure_factor_bound must cover the final pure multiplier")

    def _select_curriculum_batch(self, batch_dict):
        if self.question_curriculum is None:
            return batch_dict, {}
        original = [int(i) for i in batch_dict["curriculum_dataset_index"]]
        selected, sources, metrics = self.question_curriculum.select(original, self.global_steps)
        if list(selected) != original:
            batch_dict = self.train_collate_fn([self.curriculum_dataset[i] for i in selected])
        batch_dict["curriculum_source"] = np.asarray(sources, dtype=object)
        return batch_dict, metrics

    def _observe_curriculum_batch(self, batch):
        if self.question_curriculum is None:
            return {}
        metadata = batch.non_tensor_batch
        outcomes = defaultdict(dict)
        trajectory_questions = {}
        for index, trajectory, label in zip(metadata["curriculum_dataset_index"],
                                            metadata["traj_uid"], metadata["pass"]):
            if label not in (0, 1):
                raise ValueError("curriculum terminal labels must be binary")
            if trajectory_questions.setdefault(str(trajectory), int(index)) != int(index):
                raise ValueError("inconsistent dataset index within one trajectory")
            previous = outcomes[int(index)].setdefault(str(trajectory), int(label))
            if previous != int(label):
                raise ValueError("inconsistent terminal labels within one trajectory")
        return self.question_curriculum.observe(
            {index: list(labels.values()) for index, labels in outcomes.items()}, self.global_steps)

    def _recovery_state(self):
        cfg = OmegaConf.to_container(self.config.algorithm.advantage_recovery, resolve=True)
        cfg.pop("allow_missing_resume", None)
        return {"version": 1, "config": cfg,
                "curriculum": self.question_curriculum.state_dict() if self.question_curriculum else None}

    def _validate_multi_agent_config(self):
        """Validate configuration for each agent after agent-specific parameters are applied."""
        from omegaconf import OmegaConf
        
        for wg_id, agents_configs in self.wg_to_agents_mapping.items():
            for ac in agents_configs:
                agent_id = ac.get("agent_id", "unknown")
                # Create a temporary config by merging agent-specific actor_rollout_ref with base config
                temp_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
                # Replace actor_rollout_ref with agent-specific config
                temp_config.actor_rollout_ref = ac["config_actor_rollout_ref"]
                # Validate this agent's config
                try:
                    self._validate_config(temp_config)
                except (ValueError, AssertionError) as e:
                    raise ValueError(
                        f"[Agent {agent_id} in WG {wg_id}] Configuration validation failed: {str(e)}"
                    ) from e

    def _validate_entropy_credit_config(self):
        """Fail fast when entropy credit is enabled with an incompatible path."""

        entropy_config = self.config.algorithm.get("entropy_credit", {})
        sparse_config = entropy_config.get("sparse", {})
        if sparse_config.get("enable", False):
            if not entropy_config.get("enable", False):
                raise ValueError("pure entropy credit requires algorithm.entropy_credit.enable=True")
            validate_sparse_entropy_config(sparse_config)
            if (float(sparse_config.get("multiplier_min", 0.8)),
                    float(sparse_config.get("multiplier_max", 1.2))) != (0.8, 1.2):
                raise ValueError("pure entropy credit final coefficients must stay within [0.8, 1.2]")
            if self.config.env.env_name != "math" or self.config.agent.orchestra_type != "math":
                raise ValueError("pure entropy credit currently requires the math environment and orchestra")
        if not entropy_config.get("enable", False):
            return
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("entropy credit currently requires algorithm.adv_estimator=grpo")
        if not self.config.algorithm.get("group_by_agent_id", False):
            raise ValueError("entropy credit requires algorithm.group_by_agent_id=True")

        for wg_id, agents_configs in self.wg_to_agents_mapping.items():
            for agent_config in agents_configs:
                rollout_config = agent_config["config_actor_rollout_ref"].rollout
                if rollout_config.name != "sglang":
                    raise ValueError(
                        f"entropy credit requires SGLang rollout, got {rollout_config.name!r} "
                        f"for worker group {wg_id!r}"
                    )
                if rollout_config.multi_turn.enable:
                    raise ValueError(
                        "entropy credit currently uses DrMAS orchestra turns and requires "
                        "actor_rollout_ref.rollout.multi_turn.enable=False; the SGLang "
                        f"request-level tool path is unsupported for worker group {wg_id!r}"
                    )
                if int(rollout_config.get("top_logprobs_num", 0) or 0) != 16:
                    raise ValueError(
                        "entropy credit requires actor_rollout_ref.rollout.top_logprobs_num=16 "
                        f"for every agent; mismatch in worker group {wg_id!r}"
                    )

        bounds = (
            float(entropy_config.get("multiplier_min", 0.8)),
            float(entropy_config.get("multiplier_max", 1.2)),
            float(entropy_config.get("final_multiplier_min", 0.8)),
            float(entropy_config.get("final_multiplier_max", 1.2)),
        )
        if bounds != (0.8, 1.2, 0.8, 1.2):
            raise ValueError(
                "both entropy-credit factors and their final product must be clipped to [0.8, 1.2]"
            )
        action_scale = float(entropy_config.get("action_scale", 0.2))
        trajectory_scale = float(entropy_config.get("trajectory_scale", 0.1))
        if not 0.0 <= action_scale <= 0.2:
            raise ValueError("entropy_credit.action_scale must be in [0, 0.2]")
        if not 0.0 <= trajectory_scale <= 0.2:
            raise ValueError("entropy_credit.trajectory_scale must be in [0, 0.2]")
        if float(entropy_config.get("trajectory_deadzone", 0.05)) < 0:
            raise ValueError("entropy_credit.trajectory_deadzone must be non-negative")
        if float(entropy_config.get("relative_epsilon", 1e-12)) <= 0:
            raise ValueError("entropy_credit.relative_epsilon must be positive")

    def _validate_value_control_config(self):
        """Value predicts outcomes; pure alone constructs advantage multipliers."""
        entropy = self.config.algorithm.get("entropy_credit", {})
        value = entropy.get("value", {})
        control = entropy.get("control", {})
        self.value_credit_enabled = bool(value.get("enable", False))
        self.entropy_control_enabled = bool(control.get("enabled", False))
        if self.entropy_control_enabled and not self.value_credit_enabled:
            raise ValueError("entropy control requires entropy_credit.value.enable=True")
        for agents in self.wg_to_agents_mapping.values():
            for agent in agents:
                actor = agent["config_actor_rollout_ref"].actor
                actor_enabled = bool(actor.get("entropy_control", {}).get("enabled", False))
                if actor_enabled != self.entropy_control_enabled:
                    raise ValueError("Actor and algorithm entropy_control.enabled must agree for every Agent")
                if actor_enabled and actor.strategy not in ("fsdp", "fsdp2"):
                    raise ValueError("value-gated entropy control currently requires the FSDP Actor")
                if actor_enabled and float(actor.entropy_coeff) != 0.0:
                    raise ValueError("Use the conditional entropy constraint with actor.entropy_coeff=0")
        if not self.value_credit_enabled:
            return
        if not entropy.get("enable", False) or not entropy.get("sparse", {}).get("enable", False):
            raise ValueError("mix value control requires both pure entropy-credit stages")
        if self.config.agent.orchestra_type != "math" or self.config.env.env_name != "math":
            raise ValueError("The current prefix record adapter supports the math Solver/Verifier orchestra")
        if set(self.agent_ids) != {"Solver Agent", "Verifier Agent"}:
            raise ValueError("mix prefix values require Solver Agent and Verifier Agent")
        if not value.get("model_path"):
            raise ValueError("value.model_path must identify the independent frozen encoder")
        if value.get("device", "cpu") not in ("cpu", "cuda:0"):
            raise ValueError("value.device must be cpu or cuda:0 (one dedicated Ray GPU)")
        if float(value.get("num_cpus", 1)) <= 0:
            raise ValueError("value.num_cpus must be positive")
        if self.config.trainer.default_hdfs_dir is not None:
            raise ValueError("Value/controller checkpoints require local or shared filesystem storage")
        if self.entropy_control_enabled:
            self.entropy_controller = EntropyController(OmegaConf.to_container(control, resolve=True))

    def _init_value_scorer(self):
        if not self.value_credit_enabled or self.config.trainer.get("val_only", False):
            return
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from verl.workers.credit_value import PrefixValueScorer

        value = OmegaConf.to_container(self.config.algorithm.entropy_credit.value, resolve=True)
        initial = value.get("initial_checkpoint")
        if initial:
            initial = os.path.abspath(os.path.expanduser(initial))
            if not os.path.isfile(initial):
                raise FileNotFoundError(f"Missing pretrained entropy-aware value checkpoint: {initial}")
        use_gpu = value.get("device", "cpu") == "cuda:0"
        node_id = ray.get_runtime_context().get_node_id()
        available = ray.state.available_resources_per_node()
        validate_pool_layout(available, self.resource_pool_manager.resource_pool_spec,
                             reserved_node=node_id if use_gpu else None, cpus_per_gpu=1,
                             reserved_cpus=float(value.get("num_cpus", 1)) if use_gpu else 0)
        cpu_need = float(value.get("num_cpus", 1))
        if available.get(node_id, {}).get("CPU", 0) < cpu_need:
            raise ValueError("The trainer node needs an additional CPU for the value scorer")
        remote_cls = ray.remote(PrefixValueScorer)
        self.value_scorer = remote_cls.options(
            num_cpus=cpu_need, num_gpus=int(use_gpu),
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(value)
        ray.get(self.value_scorer.prepare.remote([]), timeout=float(value.get("startup_timeout", 1800)))
        if initial:
            loaded = ray.get(self.value_scorer.load.remote(initial, resume=False))
            if value.get("require_pretrained", True) and not loaded.get("ready", False):
                raise ValueError("Initial value checkpoint has not passed entropy-aware readiness checks")

    def _prepare_value_credit(self, batch: DataProto) -> dict[str, float]:
        records, metrics = build_trajectory_records(
            batch.non_tensor_batch, int(self.config.agent.orchestra.math.max_loop_num)
        )
        if getattr(self, "question_curriculum", None) is not None:
            sources = {}
            for trajectory, source in zip(batch.non_tensor_batch["traj_uid"], batch.non_tensor_batch["curriculum_source"]):
                if sources.setdefault(trajectory, source) != source:
                    raise ValueError("inconsistent curriculum source within one trajectory")
            for record in records:
                # Score every current trajectory; learn/calibrate only on the
                # randomly protected representative slice of the base sampler.
                record["train_eligible"] = sources[record["traj_uid"]] == "fresh"
        scored = ray.get(self.value_scorer.prepare.remote(records))
        self.value_scorer_ready = bool(scored["ready"])
        self.value_scorer_reliability = scored.get("reliability", float(self.value_scorer_ready))
        self.value_scorer_temporal_reliability = scored.get("temporal_reliability", {})
        metrics.update(value_scorer_metrics("score", scored["metrics"]))
        batch.non_tensor_batch["value_credit_scorer_version"] = np.full(
            len(batch), int(scored["metrics"].get("version", 0)), dtype=np.int64
        )
        metrics.update(attach_value_predictions(batch.non_tensor_batch, scored["values"], scored["ready"]))
        return metrics

    def _prepare_entropy_control(self, batch: DataProto) -> dict[str, float]:
        if self.entropy_controller is None:
            return {}
        # Called once after all Agent old-log-prob outputs have been combined;
        # the per-row means followed the same reorder/padding as their actions.
        arrays, metrics = self.entropy_controller.prepare(
            batch.non_tensor_batch, batch.non_tensor_batch["action_full_entropy"],
            self.global_steps, self.value_scorer_ready, self.value_scorer_reliability,
            temporal_reliability=self.value_scorer_temporal_reliability,
        )
        for key, array in arrays.items():
            dtype = torch.bool if key == "entropy_control_valid" else torch.float32
            batch.batch[key] = torch.as_tensor(array, dtype=dtype,
                                               device=batch.batch["responses"].device).detach()
        batch.non_tensor_batch["entropy_control_brake"] = arrays["entropy_control_weight"]
        batch.non_tensor_batch["entropy_control_cap_value"] = arrays["entropy_control_cap"]
        batch.non_tensor_batch["entropy_control_valid_action"] = arrays["entropy_control_valid"]
        return metrics

    def _validate_config(self, config):
        # config = self.config
        # number of GPUs total
        n_gpus = agent_world_size(config.trainer)

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizers[self.default_wg], self.processors[self.default_wg])
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizers[self.default_wg], self.processors[self.default_wg])
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_collate_fn = collate_fn
        loader_dataset = self.train_dataset
        curriculum_cfg = self.config.algorithm.get("advantage_recovery", {}).get("curriculum", {})
        if curriculum_cfg.get("enabled", False) and not self.config.trainer.get("val_only", False):
            dataframe = getattr(self.train_dataset, "dataframe", None)
            fingerprint = getattr(dataframe, "_fingerprint", None)
            if not fingerprint:
                raise ValueError("question curriculum requires an ordered dataset fingerprint")
            data_cfg = OmegaConf.to_container(self.config.data, resolve=True)
            preprocessing_keys = ("prompt_key", "max_prompt_length", "truncation", "return_raw_chat",
                                  "filter_overlong_prompts", "apply_chat_template_kwargs", "chat_template_func")
            identity = json.dumps({"data": fingerprint,
                                   "preprocessing": {key: data_cfg.get(key) for key in preprocessing_keys},
                                   "tokenizer": getattr(self.tokenizers[self.default_wg], "name_or_path", None)},
                                  sort_keys=True, ensure_ascii=False)
            self.question_curriculum = QuestionCurriculum(
                curriculum_cfg, dataset_size=len(self.train_dataset),
                dataset_fingerprint=hashlib.sha256(identity.encode("utf-8")).hexdigest())
            self.curriculum_dataset = IndexedQuestionDataset(self.train_dataset)
            loader_dataset = self.curriculum_dataset
        self.train_dataloader = StatefulDataLoader(
            dataset=loader_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        rollout_metadata=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        if rollout_metadata:
            for key, values in rollout_metadata.items():
                if len(values) == n:
                    base_data[key] = values

        # ``adjust_batch`` may append exact row copies for distributed batch
        # divisibility.  Keep those copies in training, but remove them from the
        # trajectory export so every logical action is represented exactly once.
        action_identity_keys = ("traj_uid", "agent_id", "role_turn_index")
        if rollout_metadata and all(key in base_data for key in action_identity_keys):
            keep_indices = first_unique_action_indices(
                base_data["traj_uid"],
                base_data["agent_id"],
                base_data["role_turn_index"],
            )
            if len(keep_indices) != n:
                base_data = {
                    key: [values[int(row)] for row in keep_indices]
                    for key, values in base_data.items()
                }
                n = len(keep_indices)

        def json_safe(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        with open(filename, "w", encoding="utf-8") as f:
            for i in range(n):
                entry = {key: json_safe(values[i]) for key, values in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        uid_list = []  # Add uid collection for grouping
        task_pass_lst = []
        # success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # Add progress bar for validation
        val_progress_bar = tqdm(total=len(self.val_dataloader), desc="Validation Progress")

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                val_progress_bar.close()
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizers[self.default_wg].decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizers[self.default_wg].eos_token_id,
                "pad_token_id": self.tokenizers[self.default_wg].pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            wg_ids = test_output_gen_batch.non_tensor_batch["wg_id"]
            output_texts = [self.tokenizers[model].decode(ids, skip_special_tokens=True) for ids, model in zip(output_ids, wg_ids)]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            uid_list.append(test_output_gen_batch.non_tensor_batch['uid'])  # Collect uid for grouping
            task_pass_lst.append(test_output_gen_batch.non_tensor_batch['pass'])

            # Update validation progress bar
            val_progress_bar.update(1)

            # success rate
            # for k in test_batch.non_tensor_batch.keys():
            #     if 'success_rate' in k:
            #         if k not in success_rate_dict:
            #             success_rate_dict[k] = []
            #         success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
            #         # all success_rate should be the same
            #         for i in range(1, len(test_batch.non_tensor_batch[k])):
            #             assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        uids = np.concatenate(uid_list, axis=0)  # Group UIDs for task grouping
        task_pass = np.concatenate(task_pass_lst, axis=0)
        # success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # Get validation rollout n for pass@k and avg@k computation
        val_rollout_n = getattr(self.config.env.rollout, 'val_n', 1)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        unique_uids = uids[unique_idx]
        unique_task_pass = task_pass[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        # for k, v in success_rate.items():
        #     metric_dict[f'val/{k.replace("success_rate", "pass@1")}'] = v
        
        # Compute pass@k and avg@k for all data
        passk_avgk_metrics = compute_pass_at_k_and_avg_at_k(
            unique_uids=unique_uids,
            unique_task_pass=unique_task_pass,
            k=val_rollout_n
        )
        metric_dict.update({f'val/{k}': v for k, v in passk_avgk_metrics.items()})
        
        # Compute pass@k and avg@k per data source
        for data_source in set(unique_data_sources):
            data_source_mask = unique_data_sources == data_source
            if np.any(data_source_mask):
                ds_passk_avgk_metrics = compute_pass_at_k_and_avg_at_k(
                    unique_uids=unique_uids[data_source_mask],
                    unique_task_pass=unique_task_pass[data_source_mask],
                    k=val_rollout_n
                )
                metric_dict.update({f'val/{data_source}/{k}': v for k, v in ds_passk_avgk_metrics.items()})

        val_progress_bar.close()
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # Reserve the independent scorer GPU before creating the shared Agent
        # placement groups; otherwise sixteen Agent ranks can consume it.
        self._init_value_scorer()
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Set configurations for each model
        wg_configs = {}
        for wg_id, agents_list in self.wg_to_agents_mapping.items():
            wg_configs[wg_id] = agents_list[0]["config_actor_rollout_ref"]
            wg_configs[wg_id].model.path = agents_list[0]["model_id"]

        # create actor and rollout
        if self.hybrid_engine:
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                role_name = "actor_rollout/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=wg_configs[wg_id],
                    role=role_name,
                )
                self.resource_pool_to_cls[resource_pool][role_name] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                role_name = "ref/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                # we use the same config for all ref policies
                ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=wg_configs[wg_id], role=role_name)
                self.resource_pool_to_cls[resource_pool][role_name] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = {}
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                role_name = "ref/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                assert role_name in all_wg, f"Role {role_name} not found in all_wg"
                ref_policy_wg = all_wg[role_name]
                ref_policy_wg.init_model()
                self.ref_policy_wg[wg_id] = ref_policy_wg

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = {}
        for wg_id, agents_list in self.wg_to_agents_mapping.items():
            role_name = "actor_rollout/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
            assert role_name in all_wg, f"Role {role_name} not found in all_wg"
            actor_rollout_wg = all_wg[role_name]
            actor_rollout_wg.init_model()

            self.actor_rollout_wg[wg_id] = actor_rollout_wg

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")

        for wg_id in self.wg_to_agents_mapping.keys():
            actor_local_path = os.path.join(local_global_step_folder, f"actor/{wg_id}")

            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", f"actor/{wg_id}")

            remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
            if remove_previous_ckpt_in_save:
                print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
            max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
            max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

            self.actor_rollout_wg[wg_id].save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        if self.value_scorer is not None:
            ray.get(self.value_scorer.save.remote(os.path.abspath(os.path.join(local_global_step_folder, "prefix_value.pt"))))
        if self.entropy_controller is not None:
            control_path = os.path.join(local_global_step_folder, "entropy_controller.json")
            with open(control_path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(self.entropy_controller.state_dict(), handle, allow_nan=False)
            os.replace(control_path + ".tmp", control_path)

        if getattr(self, "advantage_recovery_enabled", False):
            recovery_path = os.path.join(local_global_step_folder, "advantage_recovery.json")
            with open(recovery_path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(self._recovery_state(), handle, allow_nan=False)
            os.replace(recovery_path + ".tmp", recovery_path)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        if self.value_scorer is not None:
            value_path = os.path.join(global_step_folder, "prefix_value.pt")
            allow_missing = self.config.algorithm.entropy_credit.value.get("allow_missing_resume", False)
            if os.path.isfile(value_path):
                ray.get(self.value_scorer.load.remote(value_path, resume=True))
            elif not allow_missing:
                raise FileNotFoundError(f"Missing resumed value checkpoint: {value_path}")
            if self.entropy_controller is not None:
                control_path = os.path.join(global_step_folder, "entropy_controller.json")
                if os.path.isfile(control_path):
                    with open(control_path, encoding="utf-8") as handle:
                        self.entropy_controller.load_state_dict(json.load(handle))
                elif not allow_missing:
                    raise FileNotFoundError(f"Missing resumed entropy calibration: {control_path}")

        if getattr(self, "advantage_recovery_enabled", False) and not self.config.trainer.get("val_only", False):
            recovery_path = os.path.join(global_step_folder, "advantage_recovery.json")
            if os.path.isfile(recovery_path):
                with open(recovery_path, encoding="utf-8") as handle:
                    recovery_state = json.load(handle)
                expected = self._recovery_state()
                if recovery_state.get("version") != expected["version"] or recovery_state.get("config") != expected["config"]:
                    raise ValueError("resumed advantage recovery configuration does not match")
                if self.question_curriculum is not None:
                    self.question_curriculum.load_state_dict(recovery_state["curriculum"])
            elif not self.config.algorithm.advantage_recovery.get("allow_missing_resume", False):
                raise FileNotFoundError(f"Missing resumed advantage recovery state: {recovery_path}")

        # load actor
        for wg_id in self.wg_to_agents_mapping.keys():
            actor_path = os.path.join(global_step_folder, f"actor/{wg_id}")
            self.actor_rollout_wg[wg_id].load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        critic_path = os.path.join(global_step_folder, "critic")
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, wg_id, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg[wg_id].world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.value_scorer is not None:
            initial = ray.get(self.value_scorer.prepare.remote([]))
            # A resumed deployed model may have been temporarily disabled by
            # online validation. Preserve that state and let it recover; only
            # a fresh run requires an already qualified initialization.
            if self.global_steps == 0 and self.config.algorithm.entropy_credit.value.get("require_pretrained", True) and not initial["ready"]:
                raise ValueError("Main training requires a ready entropy-aware prefix_value.pt; run offline pretraining first")
            logger.log(data={"value_model/ready_at_training_start": float(initial["ready"])}, step=self.global_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and (self.config.trainer.get("val_before_train", True) or self.config.trainer.get("val_only", False)):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch_dict, curriculum_metrics = self._select_curriculum_batch(batch_dict)
                metrics.update(curriculum_metrics)
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                non_tensor_batch_keys_to_pop.extend(
                    key for key in ("curriculum_dataset_index", "curriculum_source") if key in batch.non_tensor_batch)
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output
                    metrics.update(self._observe_curriculum_batch(batch))

                    # Compute entropy ranks on the unadjusted rollout set.  The
                    # factors are detached NumPy metadata and will follow their
                    # actions through per-WG padding/copying and batch balance.
                    entropy_credit_config = self.config.algorithm.get("entropy_credit", {})
                    entropy_credit_enabled = entropy_credit_config.get("enable", False)
                    sparse_entropy_config = entropy_credit_config.get("sparse", {})
                    sparse_entropy_enabled = sparse_entropy_config.get("enable", False)
                    if entropy_credit_enabled:
                        batch, entropy_credit_metrics = prepare_entropy_credit(
                            batch,
                            entropy_credit_config,
                        )
                        metrics.update(entropy_credit_metrics)
                        if sparse_entropy_enabled:
                            batch, sparse_metrics = prepare_sparse_entropy_credit_for_batch(
                                batch, sparse_entropy_config
                            )
                            metrics.update(sparse_metrics)

                    if self.value_scorer is not None:
                        with _timer("value_encode", timing_raw):
                            metrics.update(self._prepare_value_credit(batch))

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_algos.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    # multiagent_batch (Dict[str, DataProto]): Dictionary mapping unique_wg_ids to their respective DataProto batches
                    unique_wg_ids = list(self.wg_to_agents_mapping.keys())

                    multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)

                    for wg_id in multiagent_batch.keys():
                        sub_batch = multiagent_batch[wg_id]
                        
                        # Create agent-specific config for adjust_batch
                        agent_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
                        # Replace actor_rollout_ref with agent-specific config
                        agent_config.actor_rollout_ref = self.wg_to_agents_mapping[wg_id][0]["config_actor_rollout_ref"]
                        
                        sub_batch = adjust_batch(agent_config, sub_batch, wg_id=wg_id)
                        multiagent_batch[wg_id] = sub_batch
                    
                        sub_batch.batch["response_mask"] = compute_response_mask(sub_batch)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.config.trainer.balance_batch:
                            self._balance_batch(sub_batch, wg_id, metrics=metrics, logging_prefix= f"global_seqlen/{wg_id}")

                        # compute global_valid tokens
                        sub_batch.meta_info[f"{wg_id}/global_token_num"] = torch.sum(sub_batch.batch["attention_mask"], dim=-1).tolist()

                        multiagent_batch[wg_id] = sub_batch

                    batch: DataProto = combine_batches(multiagent_batch)

                    batch.meta_info[f"global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizers)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        for wg_id in multiagent_batch.keys():
                            sub_batch = multiagent_batch[wg_id]
                            assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                            old_log_prob = self.actor_rollout_wg[wg_id].compute_log_prob(sub_batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = sub_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {f"actor/{wg_id}/entropy_loss": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            if self.entropy_controller is not None:
                                action_entropy = (entropys.detach().float() * response_masks).sum(-1) / response_masks.sum(-1).clamp_min(1)
                                sub_batch.non_tensor_batch["action_full_entropy"] = action_entropy.cpu().numpy()
                            old_log_prob.batch.pop("entropys")
                            sub_batch = sub_batch.union(old_log_prob)

                            if "rollout_log_probs" in sub_batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = sub_batch.batch["rollout_log_probs"]
                                actor_old_log_probs = sub_batch.batch["old_log_probs"]
                                attention_mask = sub_batch.batch["attention_mask"]
                                responses = sub_batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        f"training/{wg_id}/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        f"training/{wg_id}/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        f"training/{wg_id}/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )
                            multiagent_batch[wg_id] = sub_batch

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg[wg_id].compute_ref_log_prob(sub_batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg[wg_id].compute_ref_log_prob(sub_batch)
                                sub_batch = sub_batch.union(ref_log_prob)

                                multiagent_batch[wg_id] = sub_batch

                    batch: DataProto = combine_batches(multiagent_batch)
                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        group_by_agent_id = self.config.algorithm.get("group_by_agent_id", False) # 

                        advantage_fn = compute_advantage
                        recovery_kwargs = {}
                        if self.advantage_recovery_enabled:
                            advantage_fn = compute_advantage_with_recovery
                            recovery_kwargs["recovery_config"] = self.config.algorithm.advantage_recovery
                        advantage_result = advantage_fn(
                            batch,
                            **recovery_kwargs,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            group_by_agent_id=group_by_agent_id,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )
                        if self.advantage_recovery_enabled:
                            batch, recovery_metrics = advantage_result
                            metrics.update(recovery_metrics)
                        else:
                            batch = advantage_result
                        if entropy_credit_enabled:
                            if sparse_entropy_enabled:
                                batch, sparse_metrics = finalize_sparse_entropy_credit_for_advantages(
                                    batch, sparse_entropy_config
                                )
                                metrics.update(sparse_metrics)
                            batch = apply_entropy_credit_to_advantages(batch)

                    # This control branch never changes advantages, rewards,
                    # pure's final factors, returns, or PPO clipping.
                    if self.entropy_controller is not None:
                        metrics.update(self._prepare_entropy_control(batch))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                        # update actor
                        with _timer("update_actor", timing_raw):
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                                sub_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg[wg_id].update_actor(sub_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info[f"{wg_id}/metrics"])
                                metrics.update(actor_output_metrics)
                                multiagent_batch[wg_id] = sub_batch

                        batch: DataProto = combine_batches(multiagent_batch)

                    # Current terminal outcomes train only the candidate AFTER
                    # both Actors used the frozen deployed values for this batch.
                    if self.value_scorer is not None:
                        with _timer("value_update", timing_raw):
                            metrics.update(value_scorer_metrics("update", ray.get(self.value_scorer.update.remote())))

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs, outputs = [], []
                            multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                sub_inputs = self.tokenizers[wg_id].batch_decode(sub_batch.batch["prompts"], skip_special_tokens=True)
                                sub_outputs = self.tokenizers[wg_id].batch_decode(sub_batch.batch["responses"], skip_special_tokens=True)
                                inputs += sub_inputs
                                outputs += sub_outputs
                                
                            batch: DataProto = combine_batches(multiagent_batch)

                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            entropy_metadata_keys = (
                                "uid",
                                "traj_uid",
                                "agent_id",
                                "role_turn_index",
                                "env_step",
                                "is_action_valid",
                                "pass",
                                "top16_entropy",
                                "top16_entropy_mean",
                                "entropy_credit_action_multiplier",
                                "entropy_credit_trajectory_multiplier",
                                "entropy_credit_final_multiplier",
                                "entropy_credit_relative_change",
                            )
                            entropy_metadata_keys += tuple(
                                key for key in batch.non_tensor_batch
                                if key.startswith(("pure_entropy_", "value_", "entropy_control_", "recovery_", "curriculum_"))
                                or key == "action_full_entropy"
                            )
                            rollout_metadata = {
                                key: batch.non_tensor_batch[key]
                                for key in entropy_metadata_keys
                                if key in batch.non_tensor_batch
                            }
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                rollout_metadata=rollout_metadata,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, unique_wg_ids=unique_wg_ids, group_n=self.config.env.rollout.n, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
