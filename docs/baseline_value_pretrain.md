# 仅用 baseline 历史轨迹初始化价值网络

已确认的数据设置是 **724 个 global step × 32 道题 × 8 条轨迹 = 185,344 条完整轨迹**。每个 step 保存一个包含 `trajectories` 数组的 JSON 文件。历史 Solver 最多 2 轮，后续主训练最多 3 轮。

## 先分清哪些信息可以学习

| baseline 字段 | 用途 |
|---|---|
| `anchor_obs` | 完整原题；优先级在显式 `value_question`、`env_kwargs.question` 之后 |
| `question`、`raw_prompt` | 附件中是带系统/角色模板的 prompt，不直接当原题 |
| `uid`、`traj_uid`、`global_step` | 分组、去重和历史运行身份；不作为数值预测特征 |
| `steps[].step_idx` | 用于排序；原始编号从零连续，允许已确认的连续 Solver 重复记录复用编号 |
| `steps[].agent_id` | Solver/Verifier 角色 |
| `steps[].response` | 完整 action 文本，按因果前缀编码 |
| `steps[].top16_entropy.mean` | 已归一化的 action 平均 Top-16 熵 |
| `steps[].top16_entropy.num_tokens`、`num_response_tokens` | 熵样本数和覆盖率；不会重新除以 token 数 |
| `steps[].is_action_valid`、可选截断标志 | 熵特征可用性；无效 action 文本仍保留在后续上下文 |
| 轨迹级 `pass` | 真实 0/1 BCE 监督标签，不进入编码器输入 |
| `episode_reward`、各类 reward sum | 不作为价值输入，不替代 `pass` |
| `advantage_sum`、`advantage_token_mean`、`per_agent_advantage` | 不读取为特征，不据此筛掉零优势轨迹 |
| `old_log_prob_mean` | 不作为当前价值模型特征 |
| 熵 `std/p10/p50/p90/effective_support` | 原文件保留，当前特征 schema 不使用 |

附件的 `total_token_level_score=2.0` 包含多个 Agent action 的累计结果，不能用作成功概率标签；实际标签使用 `pass=1.0`。同理，优势为零不代表轨迹无法提供价值监督。

附件的 `turn_id` 分别为 Solver 0、Verifier 1，是全局顺序的表现，不可直接用作角色内部轮数。适配器按 `step_idx` 排序，完成下面的重复落盘清理后，再分别累计 Solver、Verifier 轮数，得到 `S0,V0,S1`。

## 连续 Solver 重复落盘

baseline JSON 适配器默认清理同一轨迹中、按 `step_idx` 排序后连续出现且 `response` 文本完全相同的 Solver 记录。保留第一份，不改写原始 JSON 文件；支持重复记录使用新的连续 `step_idx`，也支持复用原动作的 `step_idx`。

例如 `S1,S1,V1` 恢复为 `S1,V1`；`S1,V1,S2,S2`、`S1,S1,V1,S2,S2`、`S1,S1,V1,S2` 均恢复为 `S1,V1,S2`。连续三份及更多份同样只保留一份。去重之后重新生成连续的逻辑动作序号和角色轮数，再检查交替顺序、历史两轮预算和真实终止条件。

去重要求用于价值训练的熵均值、覆盖率、token 数、有效性和截断状态一致；相同文本但这些字段冲突时会报出轨迹 ID 和动作序号，不任意选择一份统计量。奖励、优势、旧策略概率及导出 ID 不参与去重判断。文本按原样比较，不去空白、不做近似匹配。

不会跨 Verifier 去重，即使 `S1` 与 `S2` 文本相同也保留两次真实动作；不会跨轨迹去重，也不会合并连续 Verifier。连续 Solver 文本不同或去重后仍不满足终止规则的轨迹会继续报错。此规则只适用于 `baseline_nested` 输入（含 `auto` 识别），不改变正式训练或 canonical 数据的动作处理。

熵直接使用 `mean`。示例数值满足 `effective_support = exp(mean * ln(16))`，与当前项目归一化 Top-16 熵一致；不再除以 `ln(16)`，不再求一次 token 平均。`anchor_obs` 与示例 prompt 内的原题逐字一致，不做反斜杠反转义。

## 为什么需要绝对熵初始化模式

两轮历史最长是 `S1 → V1 → S2`。S2 有同角色熵变化，但它已是终局；Verifier 没有第二次 action。当前时序资格检查要求**非终局**同角色时序样本，所以这些历史轨迹没有可用于该检查的时序样本。扩大数据量也不会补出第三轮。

默认 Bash 使用 `VALUE_PRETRAIN_MODE=absolute`：

1. 冻结价值编码器，先训练语义分支。
2. 冻结语义和时序分支，训练绝对熵分支及匹配对照。
3. 必须验证绝对熵在非终局前缀上优于语义分支和无熵同容量对照，并通过 AUC、Brier、题目/成败覆盖等检查；不是仅拟合一个成功率常数。
4. 初始化时两个角色的时序权限为零，实际提供给控制器的 `V_full` 回退为 `V_abs`，不使用未经验证的时序残差。
5. 三轮主训练中候选网络继续学习真实新数据。某角色的非终局时序分支通过资格检查后，才允许该角色使用时序残差。部署头/候选头比较与失准检查按实际启用的分支计算。

报告阶段为 `absolute`、`hybrid` 或 `full`；`hybrid` 表示只启用了部分角色的时序分支。`ready=true` 仍需结合各角色基础/时序权限解释。当前三阶段网络结构保持不变，新增的是训练阶段和部署资格管理。

如果以后提供了足够的三轮历史轨迹，可显式选择 `VALUE_PRETRAIN_MODE=full`，使用原先的完整时序资格要求。不要为通过检查而把两轮历史预算写成 3；主训练的 `MAX_SOLVER_TURNS=3` 是独立设置。

## Bash 的配置与命令

入口：`examples/drmas_trainer/pretrain_value.sh`。可以直接编辑该文件顶部变量，也可在调用时设置环境变量。以下均为 Linux/Bash 命令，路径替换为训练服务器上的真实路径。

```bash
cd /workspace/Credit_mas_mix

export VALUE_INPUT='/data/baseline/run1'
export VALUE_RUN_ID='baseline-run1'
export VALUE_INPUT_FORMAT=baseline_nested
export HISTORICAL_MAX_SOLVER_TURNS=2
export VALUE_PRETRAIN_MODE=absolute

export VALUE_ENCODER='/models/Qwen3-4B'
export VALUE_OUTPUT='/checkpoints/baseline_value_init/prefix_value.pt'
export VALUE_DEVICE=cuda:0
export SEMANTIC_EPOCHS=5
export ENTROPY_EPOCHS=10
export VALUE_ROUNDS=1
export VALUE_BATCH_SIZE=256
```

`VALUE_INPUT` 支持单文件、目录或带引号的 glob，例如 `'/data/baseline/run1/global_step_*.json'`。目录会递归读取 `.json/.jsonl`，因此应指向纯轨迹目录，不要把报告 JSON、配置 JSON 混放进去。模型编码会处理全部历史轨迹，目前是单 GPU 的离线任务，没有自动启动 16 卡数据并行。

先验证完整文件与样本组成，不加载编码器：

```bash
bash examples/drmas_trainer/pretrain_value.sh validate
```

报告会给出文件数、run/step 数、完整轨迹数、去重题目数、成功/失败数、历史预算以及每个角色的非终局绝对熵/时序熵样本数。若导出完整且没有重复/遗漏，应为 724 个文件、185344 条轨迹；独立题目数必须实际统计，不能简单当成 185344。

`deduplication` 另给出 baseline 输入的原始动作数 `input_actions`、移除的重复 Solver 动作数 `removed_solver_actions` 和受影响轨迹数 `affected_trajectories`。外层 `actions`、熵样本数和前缀资格统计均基于去重结果；同一完整轨迹内删除重复动作不会减少完整轨迹数。`validate` 和正式预训练共用这一处理，无需额外开关。

对于这批两轮数据，非终局时序样本数应为零，这与 absolute 模式一致。数据检查不等于价值模型资格检查。附件只粘贴了一条 trajectory 和未闭合的外层 JSON；它足以核对字段，但不能作为完整数据文件训练，生产加载器会拒绝截断 JSON 或声明轨迹数与实际数量不一致的文件。

打印准确 Python 命令，不执行：

```bash
DRY_RUN=1 bash examples/drmas_trainer/pretrain_value.sh train
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/drmas_trainer/pretrain_value.sh train
```

完成后查看：

```bash
python3 -m json.tool "${VALUE_OUTPUT}.report.json"
```

退出码 0 表示合格；2 表示模型未通过资格检查；1 表示数据/配置等错误。Bash 原样返回 Python 退出码。模型 `.pt` 存在不表示 ready。

若仍希望使用已有 manifest，请取消 `VALUE_INPUT`，设置 `VALUE_MANIFEST`。每个 source 可指定 `format: baseline_nested`、`run_id` 和历史预算。多种预算混合时可设置 `HISTORICAL_MAX_SOLVER_TURNS=''`，让每个 source 或文件显式提供真实预算。

## 接入现有 16 卡主训练

预训练合格后：

```bash
VALUE_CHECKPOINT='/checkpoints/baseline_value_init/prefix_value.pt' \
VALUE_ENCODER='/models/Qwen3-4B' \
TRAIN_DATA='/data/drmas_math/train.parquet' \
VAL_DATA='/data/drmas_math/test_sampled.parquet' \
MAX_SOLVER_TURNS=3 \
bash examples/drmas_trainer/run_math.sh train
```

主训练从 checkpoint 识别 absolute 初始化阶段，不需要手工把时序权限打开。编码器路径/结构须与离线模型一致。资源仍是 15 张共享 Agent GPU 加 1 张价值 GPU；已有 pure 信用分配、优势恢复、题目缓存和条件熵约束继续运行。

历史两轮到当前三轮存在策略/预算分布变化，预训练只是初始化。实际 baseline 数据是否学到足够熵增益、线上何时获得时序资格，必须看真实报告和后续留出验证，不能由 185344 条的总量推断。
