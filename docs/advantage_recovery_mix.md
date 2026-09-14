# mix：零优势恢复与题目课程

本次修改只针对 `Credit_mas_mix`。默认 16 卡入口 `examples/drmas_trainer/run_math.sh` 已启用本方案；基础 YAML 默认关闭，便于旧实验复现。离线初始化仍使用 baseline + sup 真实轨迹，已有 Top-16 action 平均熵可以继续使用。

## 主训练流程

1. 从原训练集取一批题目。独立随机保护至少 70% 的题目，其余最多 30% 槽位可替换为学习区间题目或有限的困难题重试。
2. 对最终题目集合用当前 Solver/Verifier 重新生成完整轨迹组。默认每题 8 条轨迹，按唯一 `traj_uid` 记录真实终局成败。缓存从不提供旧 response。
3. 准备 pure 第一阶段熵排序和第二阶段需要的轨迹统计。部署中的价值头为全部当前轨迹预测前缀成功概率。
4. 计算真实训练奖励，包括已有格式/无效 action 惩罚。按 `(uid, agent_id, traj_uid, role_turn_index)` 去除训练补齐副本后计算标准 GRPO。
5. 只对**实际奖励无方差且原优势为零**的满成功/满失败组恢复优势。奖励有差异的组保留标准 GRPO；单条有效轨迹的组不恢复。
6. 将恢复后的优势交给 pure 第二阶段做轨迹信用分配，最终 pure multiplier 只乘一次。恢复权重不重复乘入 Actor loss。
7. Actor 用真实 token 的加权 PPO loss 更新，同时保留已有由价值模型调节强度的独立条件熵约束。
8. 两个 Actor 完成更新后，价值候选头继续学习本轮真实数据。只有随机保护部分的轨迹进入在线价值训练/校准；适应性采样的轨迹也会获得预测，但不进入价值 replay。合格候选头下一轮部署；编码器始终冻结。

## 两种恢复信号

| 实际训练组 | 处理 |
|---|---|
| 奖励有差异 | 保留标准 GRPO 优势 |
| 满成功且零优势 | 加一个虚拟低分 `0.5`，得到小幅正优势 |
| 满失败且零优势 | 加一个虚拟高分 `1.0`，得到小幅负优势，同时记录困难题索引 |
| 成败标签混合，但实际奖励恰好一致 | 不恢复，避免人为指定错误方向 |

虚拟值只参与归一化，不增加 response、token、终局标签或价值训练样本。设组内有效真实轨迹数为 G，归一化统计使用 G 个真实轨迹奖励和一个虚拟值，采用样本标准差（`ddof=1`）及 `epsilon=1e-6`。例如 G=8 时，`[1,...,1,0.5]` 的真实轨迹优势约为 `+1/3`，`[0,...,0,1]` 约为 `-1/3`。

先分别乘 `success_weight=0.2`、`failure_weight=0.1`，再执行预算限制。不得对恢复后的真实优势重新减去组均值，否则恢复信号会再次消失。若格式惩罚已经造成真实奖励差异，完全沿用 GRPO；若所有动作受到相同处罚仍然零优势，则按真实奖励常数归一化。虚拟值必须位于正确方向，否则跳过。

这相当于增加有界的成功巩固和失败抑制目标，无法在全部同分的 response 之间产生新的质量排序，也不等同于无偏 GRPO。

此外，float32 对整组 `0.9` 等相同数值计算均值/标准差时，可能因舍入生成伪负优势。恢复适配器会将奖励**完全相同**且含多个逻辑 action 的组的基础优势数值归零，再执行上述判定；不扩大到奖励略有差异的组，也不改真实 returns。

## 防止成功组占据更新

恢复组内，第 t 条轨迹的所有 action 使用相同的策略权重：

```text
w_t = T_group / (G * T_t)
```

`T_t` 是该角色在该轨迹中的有效 response token 总数，`T_group` 是整个组的总数。因此组内每条轨迹具有相同的加权 token 总量，组的总 token 权重保持不变。标准非恢复组的唯一 action 权重为 1，补齐副本为 0。

按 Solver/Verifier 分别限制：

```text
sum(recovered_success |A| * w * tokens) / all_unique_role_tokens <= 0.05 / 1.2
sum(recovered_failure |A| * w * tokens) / all_unique_role_tokens <= 0.03 / 1.2
```

预留的 `1.2` 覆盖 pure 最终 multiplier 上界，所以最终优势质量分别不超过每真实角色 token 的 `0.05` / `0.03`。预算包含本批标准组和恢复组的全部唯一有效 token，防止成功组数量增多时新增信号不受限地增长。它约束的是**优势加权质量**，不是实际梯度范数或模型参数变化量。默认每个角色每批一次 PPO 更新；若更改 minibatch 划分，预算仍是整个 rollout batch 的统计约束。

Actor 的 PPO 分母按整个 minibatch 的加权真实 token 计算，并跨训练 rank 归一化。单独由补齐副本组成的微批/rank 提供零策略梯度。已有熵约束保留独立的有效 action 分母，不会因 PPO 优势为零而失效。第一版仅支持 FSDP/FSDP2、`token-mean`、sequence parallel size 1、标准 per-agent GRPO；关闭 DAPO `filter_groups` 和 PF-PPO。

## 题目缓存与动态采样

缓存仅保存训练集行索引、成功率 EMA、最近类别、访问时间和重试次数。相同题目在一批中只出现一次，然后展开为完整的 G 条新轨迹。

- 原采样器中随机保护至少 70%，`curriculum_source=fresh`。这部分在任何难度选择之前确定，用于价值模型在线训练和校准。
- 最多 30% 槽位参与适应性选择；未被替换的候选槽位标记 `fresh_adaptive`，不冒充代表样本。
- 学习区间默认成功率 EMA 在 `[0.2,0.8]`，优先靠近 0.5。满成功率高时可增加学习区间或未掌握题目的尝试。
- 困难题重试上限为每批 15%，目标比例只随**满失败组率**变化，不随成功/失败合并后的零方差率一起上涨。
- 小于一道题的目标配额采用可复现的随机舍入，避免低失败率时缓存永久不被抽取；每批总替换和困难题上限仍严格取整限制。
- 掌握条件默认 EMA≥0.95 且连续 3 次满成功。适应性槽位中的掌握题仍以至少 0.2 的保留概率复习；距上次观测 20 step 以上的题目若被基础采样器抽到，会保留。本实现不保证每道题每 20 step 必然被抽到。
- Hard buffer 默认容量 1024、TTL 100 step、重试间隔至少 3 step、每次入池最多重试 4 次，退役冷却 20 step。重试出现任意成功轨迹即从 hard 池移出，并依据当前结果重新分类。

因此，原始奖励零方差率仍可能很高，但它与恢复后的零优势比例是两个不同指标。应同时检查验证成功率、恢复后信号占比及预算使用情况。

## 16 卡运行与恢复

继续使用 15 张共享 Agent GPU 加 1 张独立价值编码器 GPU。单机配置 `[15]`；两台各 8 卡配置 `[7,8]` 加价值 GPU，总计仍是 16 张。离线预训练步骤见 [原 mix 文档](entropy_value_mix.md)。

```bash
VALUE_CHECKPOINT=/absolute/path/prefix_value.pt \
TRAIN_DATA=/absolute/path/train.parquet \
VAL_DATA=/absolute/path/val.parquet \
bash examples/drmas_trainer/run_math.sh train
```

双机先启动两台主机的 Ray，再增加 `NNODES=2 RAY_ADDRESS=auto`。运行名默认 `mix_recovery_entropy_16gpu`。可先设置 `DRY_RUN=1` 检查最终命令与路径。

检查点新增 `global_step_N/advantage_recovery.json`，保存恢复配置、题目缓存、数据身份和采样 RNG。恢复时须与训练数据及恢复参数匹配。`prefix_value.pt`、`entropy_controller.json`、Actor 和 DataLoader 检查点仍一并保存。

```bash
RESUME_FROM=/absolute/path/global_step_100 \
bash examples/drmas_trainer/run_math.sh train
```

从**尚未包含新机制**的旧 mix 检查点开始本方案时，显式追加 `algorithm.advantage_recovery.allow_missing_resume=True`，只允许缺失的新恢复状态从空缓存开始；已有文件的配置/数据不匹配仍会拒绝。纯评估不加载题目课程状态、不进行恢复或重采样。

消融：

```bash
# 保留优势恢复，关闭题目重采样
bash examples/drmas_trainer/run_math.sh train algorithm.advantage_recovery.curriculum.enabled=False

# 回到此前的 pure + 价值熵控制组合；三个开关需要一起关闭
bash examples/drmas_trainer/run_math.sh train \
  algorithm.advantage_recovery.enable=False \
  algorithm.advantage_recovery.curriculum.enabled=False \
  actor_rollout_ref.actor.advantage_recovery.enable=False
```

消融更改配置后使用新的运行目录；严格断点恢复会拒绝中途静默改变算法。

## 观察指标与验证

`advantage_recovery/*_group_fraction` 区分满成功、满失败、实际奖励零方差；每角色记录 `raw_zero_advantage_token_fraction`、`remaining_zero_advantage_token_fraction`、正/负恢复预算缩放及 `*_post_pure_mass_bound`。`curriculum/*` 记录保护样本、替换比例和 hard 状态；价值 scorer 的 `score_only_trajectories` 说明多少适应性轨迹只评分。

rollout JSONL 新增 `recovery_*` 和 `curriculum_*` 元数据。原真实奖励与标签继续导出，虚拟奖励不会出现在真实样本列。

CPU 测试覆盖虚拟归一化、padding、变长多轮轨迹、PPO 梯度、15 rank 分母、题目缓存和确定性恢复，并回归已有 pure/价值/熵控制与启动配置。CPU 验证不能替代真实 16 卡训练；本地未启动 GPU 主训练，也没有据此宣称能够超过 baseline。
