# mix：pure 信用分配与熵感知价值监督

本版本修改 `Credit_mas_mix`。`credit_mas_pure`、`credit_mas_sup` 作为参考项目保留。
实现范围为 Math 的 Solver/Verifier、GRPO、FSDP 和 SGLang。

用户当前选择先用 baseline 的 185344 条两轮轨迹初始化，再做三轮主训练。对应的直接 JSON 读取、可配置 Bash、绝对熵资格与在线时序启用见 [baseline 预训练说明](baseline_value_pretrain.md)。`pretrain_value.sh` 现默认 `VALUE_PRETRAIN_MODE=absolute`；下文涉及离线完整时序预训练的流程需要显式设置 `VALUE_PRETRAIN_MODE=full`。

新增零优势恢复与题目课程已接入默认训练入口，最新主训练顺序和参数见 [零优势恢复说明](advantage_recovery_mix.md)。下文的价值预训练、熵特征和 16 卡资源配置继续适用。

## 实际训练流程

1. 用 baseline + sup 的完整历史轨迹离线初始化价值模型，不需要 pure 历史轨迹。
2. 主训练仍使用 pure 的完整两阶段信用分配。第一阶段在题目、角色、成功/失败分组中对 action 熵排序；第二阶段采用同角色相邻 action 的稀疏轨迹信用转移，并依据真实优势符号做分量内校准。默认第二阶段只作用于 Solver。
3. 稀疏第二阶段产出的 final multiplier 已包含第一阶段；只乘一次。价值模型不增加 `mV`，不生成伪奖励，不改 PPO ratio、clip、原奖励或 returns。
4. 冻结的部署价值头为本轮前缀预测成功概率，生成独立的条件熵约束强度。
5. Solver 和 Verifier 完成本轮 Actor 更新之后，新轨迹才进入候选价值头训练。通过检查的候选头从下一轮开始用于控制。

**正式训练时价值模型继续更新。** 编码器始终冻结；小型语义、绝对熵、时序熵网络在线训练。在线默认学习率 `1e-4`、每轮 1 epoch，离线默认 `1e-3`。本轮样本不足时保留仍有效的已部署头，不因某个 pure 稀疏事件未触发而关闭价值监督。

已部署模型在连续 3 个不同、样本充分的在线留出窗口中失准时，会暂停相应权限；两个角色的基础/时序资格分别检查，避免整体指标掩盖其中一个角色的退化。恢复训练会保留这种暂时停用状态，继续学习候选头并等待重新合格。

## 两种熵的口径

历史数据和 pure 排序使用已有 `top16_entropy_mean`：每个 token 的 Top-16 概率在集合内重新归一化，计算熵并除以 `ln(16)`，最后按 action 内 token 平均，范围 `[0,1]`。沿用项目 `compute_action_topk_entropy` 的定义。baseline 的这些统计可以直接使用，无需 pure 轨迹，也不需要重新计算旧策略 logits。

只有 action 均值时，不构造不存在的 token 分位数、最大值或方差。可选 coverage/token count 缺失会保留缺失标记；明确 coverage 不足、无效或截断的 action 不提供可靠熵特征。不同历史导出若使用未归一化熵，须先按实际统计定义转换，不能直接混用。

Actor 的可微熵约束使用**当前 Actor 全词表熵，单位 nats**。旧的 Top-16 数值只作为价值输入和 pure 统计，不能对 Actor 反向传播，也不能与全词表 cap 比较。

## 价值网络如何学习熵

一个独立、冻结的因果 Qwen3 编码器读取原题和按时间排列的 action 文本。在每个完整 action 边界提取隐藏状态，并加入角色、已用轮数、剩余预算等当时可知的结构状态。

输出由三个残差 logit 组成：

```text
z_sem  = semantic(prefix_embedding, state)
z_abs  = z_sem + absolute(prefix_embedding, absolute_entropy)
z_full = z_abs + temporal(prefix_embedding, absolute_entropy, temporal_entropy)
V_sem, V_abs, V_full = sigmoid(z_sem, z_abs, z_full)
```

每个角色的绝对熵特征含最近完成 action 的熵均值、coverage、token 数、有效性、截断及可用标记。时序特征含同角色前一次均值、差分、此前差分均值、扣除历史漂移后的残差、历史长度及可用标记。所有特征只使用当前前缀；不输入最终成败、未来轮数、纯信用事件标签、同题其他轨迹未来行为。

离线先训练语义分支，然后冻结语义分支训练熵残差。三个输出都用真实终局二元标签做 BCE，缺少相应熵信息的分支通过 mask 跳过。前缀等权，不按最终轨迹长度的倒数加权。无效 action 的文本仍保留在后续前缀上下文中；过长尾部不能编码时，保留之前完整前缀。

这是多层条件网络，不是单一线性价值头。实际参数量写入报告；例如编码器 hidden size 为 2560、默认 head sizes 为 256/64 时，一个三分支价值模型有约 100 万可训练参数。除此之外还训练两个独立、同容量的对照头，共用同一个冻结编码器：

- 无熵对照：屏蔽全部熵数值，保留相同输入维度、结构和可用 mask。
- 无时序熵对照：保留绝对熵，屏蔽时序熵数值。

对照与候选头使用相同数据、阶段和 epoch 预算。部署须满足样本/题目数、成败类别覆盖、AUC、Brier 优于常数先验，以及熵相对两个同容量对照的增量收益。熵收益在非终局、熵可用前缀上检查，防止只利用容易识别的终局答案过关。按原题内容哈希划分价值训练/留出集合；同题不同 run/step/方法不会跨集合泄漏。

整体资格、各角色绝对熵资格、各角色时序熵资格分别记录。`V_full - V_abs` 的差异只用于预测性诊断，不能解释为熵造成成功率变化的因果效应。这里的 `V_abs` 是实际训练的分支；控制时不伪造未见过的“时序全零”输入去查询部署头。

## 条件熵约束

对 action 前后的前缀：

```text
D = V_full(after) - V_full(before)
Q = D - [V_abs(after) - V_abs(before)]
bad  = clip((-D - 0.02) / 0.1, 0, 1)
harm = clip((-Q - 0.01) / 0.1, 0, 1)   # 仅 Q 可用且该角色时序资格通过
weight = reliability * ramp * entropy_risk * bad * [0.25 + 0.75 * harm]
L_actor = L_pure_PPO + 0.01 * mean_valid_unique_action(weight.detach() * relu(H_current - cap))
```

缺少时序历史时 `harm=0`，但绝对熵及 D 可靠的首个 Solver/Verifier action 仍可获得基础 `0.25` 刹车。缺少可靠价值/绝对熵的 action 权重为零。`weight` 和 cap 均 detach；只有当前 Actor 的全词表熵参与求导，因此原 GRPO 优势为零时仍可能产生熵梯度。约束不包含对终局奖励的额外判断，也不扩大 pure 系数。

`cap` 默认使用第一个训练 rollout 中、每个角色/轮次的全词表 action 熵均值与方差校准：`mean + 0.2 + 2*std`，之后固定。每组至少 8 个不同 action；首批不足的组保持未校准，不能到策略已经高熵时再建立参考。控制器用该组后续 action 熵均值的快慢 EMA 和超 cap 程度构成风险，20 step 逐步启用。

**当前实现使用训练 rollout 统计，没有额外的固定开发前缀 probe。** 不同题目组成可能影响 EMA，因此单看风险升高不能确定策略失稳；还必须同时有负价值进展、预测资格和超 cap 的可微 hinge。也不能由当前实现保证一定超过 baseline。可以先观察首批校准 coverage 与实际控制覆盖，再确定阈值是否适合数据。

## 准备 baseline + sup 数据

最稳妥的输入是一行一条完整轨迹的 JSONL，例如：

```json
{"traj_uid":"b1-t1","question":"原始完整题目","max_solver_turns":3,"label":1,"actions":[{"role":"solver","text":"完整 Solver 输出","entropy_mean":0.21,"entropy_coverage":1.0,"entropy_token_count":200},{"role":"verifier","text":"<verify>approve</verify>","entropy_mean":0.18,"entropy_coverage":1.0,"entropy_token_count":40}]}
```

`label` 必须是真实终局成败。`max_solver_turns` 是生成该轨迹时的真实配置，不能把历史 2 轮改写为新训练的 3 轮。允许两种预算混合。Verifier approve、reject 和最终 Solver 的终止规则会检查，缺失中间 action 或截掉结尾的历史会拒绝。当前 mix 导出的逐 action JSONL 已包含规范字段，可直接输入。

清单 `value_manifest.json`：

```json
{
  "sources": [
    {"input": "/data/baseline/run1/*.jsonl", "run_id": "baseline-run1"},
    {"input": "/data/sup/run1/*.jsonl", "run_id": "sup-run1"}
  ]
}
```

旧的逐 action 日志可在每个 source 中添加 `field_map`（规范字段→原字段的点路径）、`defaults`（如真实 `value_max_solver_turns`）及 `question_map`（题目 uid→完整原题）。需要 `traj_uid, uid, agent_id, role_turn_index, is_action_valid, pass, value_question, value_action_text, value_action_index, value_max_solver_turns, top16_entropy_mean`。`value_action_index` 是跨角色全局顺序，从 0 开始。脚本不从可能截断的 prompt 猜测原题，不把缺失的有效性或标签猜成成功；先恢复这些数据再训练。

按 `run_id + step + traj_uid` 隔离轨迹身份，避免不同检查点复用轨迹 id 造成错误拼接。跨文件分片的同一运行要显式设置同一个 run_id；跨策略 step 的文件应有原始 step 字段，或在 manifest 中按实际 step 分组。不要把主任务验证/测试题加入价值训练数据。

先只检查数据，不加载模型：

```bash
python3 examples/drmas_trainer/pretrain_credit_value.py \
  --manifest /data/value_manifest.json --output /models/value_init/prefix_value.pt --validate-only
```

离线预训练只需一张 GPU；此时不启动 15 卡 Actor：

```bash
VALUE_MANIFEST=/data/value_manifest.json \
VALUE_ENCODER=Qwen/Qwen3-4B \
VALUE_OUTPUT=/models/value_init/prefix_value.pt \
bash examples/drmas_trainer/pretrain_value.sh
```

输出 `.pt` 及 `.pt.report.json`。报告 `ready=true` 且退出码为 0 才能初始化主训练；模型增益未过检查返回 2，并保留未合格文件供分析。数据或配置错误返回 1。不能只凭文件存在就启动。优先检查样本覆盖、分支和对照 Brier，不应简单把资格阈值改成全放行。编码器模型路径/revision、结构维度、特征 schema、缩放器参数必须与主训练一致。

## 16 卡主训练、恢复与评估

默认分配为 **15 张 GPU 的共享 Agent 资源池 + 1 张独立价值 GPU，总计 16 张**。Solver 与 Verifier 是两个不同模型，沿用框架的 colocated workers，共享这 15 个 rank；不是每个 Agent 独占 15 张。价值 GPU 用于一个冻结编码器及小头，不与 Actor FSDP 通信。TP=1、SP=1。

```bash
VALUE_CHECKPOINT=/models/value_init/prefix_value.pt \
VALUE_ENCODER=Qwen/Qwen3-4B \
TRAIN_DATA=/data/drmas_math/train.parquet \
VAL_DATA=/data/drmas_math/test_sampled.parquet \
bash examples/drmas_trainer/run_math.sh train
```

`run_math.sh` 转发到 `run_math_16gpu.sh`。默认一台 16 卡机器；`TRAIN_BATCH_SIZE=30`、每题 `GROUP_SIZE=8`、Solver 最多 3 次。启动 mini-batch 显式为 240，以通过 15 rank 的 Worker 初始化检查；实际每轮仍根据收集到的 action 数自适应。

双机各 8 卡时，使用相同代码和可访问的模型/数据/检查点路径。先按集群惯例启动 Ray，例如：

```bash
# 第一台
ray start --head --port=6379 --num-gpus=8
# 第二台；替换 HEAD_IP
ray start --address=HEAD_IP:6379 --num-gpus=8
# 在第一台运行
NNODES=2 RAY_ADDRESS=auto VALUE_CHECKPOINT=/models/value_init/prefix_value.pt \
bash examples/drmas_trainer/run_math.sh train
```

双机 Actor 布局为 `[7,8]`。资源预检会为价值 GPU 和 CPU 留位；较大 placement group 先完成分配，再分配较小 group，避免相互卡住。各节点还需足够 CPU：每个 Actor rank 占 1 个 Ray CPU，协调器和价值 worker 另外各占 1 个。可用 `DRY_RUN=1` 验证路径并打印完整启动参数，不启动 Ray。

默认每 50 step 保存每个 Actor 分片、`prefix_value.pt`、`entropy_controller.json` 和 dataloader 状态。价值文件包括部署/候选/对照头、优化器、缓存数据、缩放器、资格和 RNG。控制器文件包括固定 cap、EMA、ramp、最后 step。恢复要求这些文件齐全；不静默改用全新价值头。

```bash
RESUME_FROM=/checkpoints/mix/global_step_250 RUN_DIR=/checkpoints/mix \
bash examples/drmas_trainer/run_math.sh train

RESUME_FROM=/checkpoints/mix/global_step_250 RUN_DIR=/checkpoints/mix \
bash examples/drmas_trainer/run_math.sh eval
```

双机恢复/评估也设置 `NNODES=2 RAY_ADDRESS=auto`。评估不加载价值模型、不占价值 GPU，但仍保持 **15 个 Actor rank**，匹配训练时的分片数。编码器配置和双 Agent 模型配置沿用原训练；实际多机恢复要求共享检查点目录。真正的恢复会严格检查价值训练/缩放配置与控制器配置，避免悄悄改变学习过程。

## 日志、验证和消融

主训练日志分开记录 `value_model/score/*`、`value_model/update/*`；检查 `ready/version`、绝对熵及时间特征对照增益、两个角色 reliability、skipped/partial 轨迹。`entropy_control/*` 含 calibrated groups、prediction/temporal coverage、风险、cap、active fraction 和 mean weight。Actor 指标记录 hinge 及实际参与的 action 数。逐 action 导出还包括三分支预测、D/Q、冻结的刹车系数及全词表熵。

完整新增与 pure 回归检查可以在装好项目依赖的环境运行：

```bash
python3 -m pytest -q \
  tests/utils/test_entropy_credit.py tests/utils/test_sparse_entropy_credit.py \
  tests/utils/test_value_credit.py tests/workers/test_credit_value.py \
  tests/utils/test_pretrain_credit_value_cli.py tests/utils/test_math_value_metadata.py \
  tests/utils/test_entropy_control.py tests/utils/test_entropy_actor.py \
  tests/utils/test_credit_resources.py tests/utils/test_mix_16gpu_launch.py \
  tests/trainer/ppo/test_sparse_entropy_credit_integration.py \
  tests/trainer/ppo/test_mix_entropy_value_integration.py
```

测试使用实际小头优化、CPU autograd、生产 Actor/Trainer 方法的隔离调用及模拟 Ray 资源，不启动真实 FSDP/SGLang。合成数据上通过资格检查证明实现可学到熵信号，不代表 baseline+sup 实际数据已经通过。正式 16 卡运行的显存、吞吐和效果还需在训练服务器验证。

建议保持相同 15-rank 布局、题目批次和 Solver 轮数比较：完整 mix；关闭 `algorithm.entropy_credit.control.enabled` 与 `actor_rollout_ref.actor.entropy_control.enabled` 的 pure+价值诊断；进一步关闭 `algorithm.entropy_credit.value.enable` 的 pure。所有开关要同时匹配；比较使用相同验证集及随机种子。不要把“训练更慢退化”自动解释成“已找到最优更新方向”。
