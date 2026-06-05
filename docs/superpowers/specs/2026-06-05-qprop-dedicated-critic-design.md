# Dedicated Q-Prop Critic for Pathwise SLDAC Design

日期：2026-06-05

## 背景

当前 `qprop_conservative` v1 已经能把 score-function gradient 与
pathwise control gradient 聚合起来，但它复用了 SLDAC 主 critic 作为
control variate 来源。这个版本适合验证组合公式，但不完全符合 Q-Prop
论文的关键结构：Q-Prop 使用一个 off-policy critic 来构造 control
variate，并用该 critic 的 pathwise gradient 补回被减掉的控制项。

本设计是 v2：新增一个专门的 `QPropCritic`。它只服务 Q-Prop 的
control variate 与 pathwise control term；现有 SLDAC 主 critic 继续服务
原来的 SLDAC 更新链路。

必须保持本项目口径：

- 问题是 infinite-horizon average reward / average cost CMDP。
- 不引入 discounted return。
- `func_value[0]` 是待最小化的 objective cost；`func_value[1:]` 是约束残差。
- 环境 `reward` 与内部 `objective_cost = -reward` 必须区分。

参考文献：

- Q-Prop: Sample-Efficient Policy Gradient with an Off-Policy Critic
  https://arxiv.org/abs/1611.02247

## 用户确认的范围

用户选择方案 1：

> 复用当前 SLDAC score signal，Q-Prop critic 只做 control variate。

因此 v2 不重做 score-function signal，不修改 SLDAC 主梯度估计口径。
这样可以把实验变量隔离为：

```text
同一个 SLDAC score signal
+ dedicated Q-Prop critic control variate
```

## 目标

1. 为 `SLDAC_Pathwise` 增加独立的 `QPropCritic`。
2. 让 `qprop_conservative` 的 Taylor control signal 与
   `Q(s, mu_theta(s))` 来自 `QPropCritic`，而不是 SLDAC 主 critic。
3. score signal 继续沿用当前 SLDAC 口径。
4. `QPropCritic` 使用 rolling replay 数据做 off-policy average-cost TD 更新。
5. 保留 v1 conservative gate：只有 control signal 与 score signal 正相关时，
   才启用 Q-Prop control variate。
6. 增加 diagnostics，能够区分：
   - gate 是否启用；
   - Q-Prop critic 训练是否稳定；
   - score gradient 与 pathwise gradient 的实际范数占比。

## 非目标

1. 不修改 `utils.update_policy()` 的二次规划 / 投影逻辑。
2. 不把现有 SLDAC 主 critic 与 `QPropCritic` 合并。
3. 不把 `gamma_pow_reward`、`gamma_pow_cost` 解释为 discounted factor。
4. 不改变 `stochastic_pathwise`、`deterministic_dpg` 的既有行为。
5. 不直接运行会覆盖正式 `.mat`、`.png`、checkpoint 的训练入口。

## 当前代码事实

涉及文件：

- `MIMO3/SLDAC_Pathwise.py`
- `CLQR2/SLDAC_Pathwise.py`
- `MIMO3/critic_opt.py`
- `CLQR2/critic_opt.py`
- `MIMO3/model.py`
- `CLQR2/model.py`
- `MIMO3/buffer.py`
- `CLQR2/buffer.py`

现有 `DataStorage` 已经维护 rolling transition：

```text
state
action
costs
next_state
```

`take_experiences()` 返回的是滚动窗口内的 transition array。由于 actor
在训练过程中持续变化，这个 rolling window 已经包含历史策略采样的数据；
对当前 actor 而言，它可作为 lightweight off-policy replay source。

现有主 critic 的 TD 目标是 average-cost differential TD：

```text
y_h = costs_h - func_value_h + Q_target_h(next_state, next_action)
```

v2 的 `QPropCritic` 也必须使用这个结构。

## 推荐架构

### 总体结构

```text
DataStorage rolling transitions
        │
        ├── SLDAC main critic
        │       └── 继续用于 SLDAC score signal 与原有更新
        │
        └── QPropCritic
                ├── off-policy average-cost TD update
                ├── Taylor control signal
                └── pathwise control loss
```

### 新增组件

新增两个局部模块，保持 MIMO3 与 CLQR2 当前目录内导入风格：

- `MIMO3/qprop_critic.py`
- `CLQR2/qprop_critic.py`

两个模块职责相同，但分别使用各自目录下的 model 类。

核心类：

```python
class QPropCritic:
    def __init__(
        self,
        example_name,
        state_dim,
        action_dim,
        constraint_dim,
        device,
        qprop_lr_scale=1.0,
        qprop_target_tau_reward=None,
        qprop_target_tau_cost=None,
    ):
        ...
```

更新接口必须显式，不依赖调用方的隐式约定：

```python
def update_from_replay(
    self,
    func_value,
    state_buffer,
    action_buffer,
    costs_buffer,
    next_state_buffer,
    actor,
    batch_size,
    update_steps,
    target_action_mode,
    tau_reward,
    tau_cost,
    rng,
):
    ...
```

职责：

1. 为每个 head 维护 online net、target net、optimizer。
2. 使用 replay batch 更新 online net。
3. soft update target net。
4. 提供 differentiable `head_value(head_idx, state, action, use_target=True)`。
5. 提供 `all_head_values(state, action, use_target=True)` 用于 diagnostics。
6. 提供 `flatten_parameters()`，便于未来 checkpoint / drift diagnostics。

### 网络结构

v2 不新建网络 architecture。`QPropCritic` 复用现有 critic 网络类：

MIMO3：

```text
Critic_net_MIMO
```

CLQR2：

```text
Critic_net_CLQR_0
Critic_net_CLQR_1
```

如果在 MIMO3 内部遇到非 MIMO example，则复用 `Critic_net_CLQR`。
如果在 CLQR2 内部遇到 MIMO example，则沿用 `Critic_net_MIMO`。

这样做可以把变量限制在“critic 是否 dedicated”，而不是同时改变函数逼近器。

head 数量必须显式校验：

```text
head_count = 1 + constraint_dim
```

- MIMO 预期 `constraint_dim == 4`，因此 `head_count == 5`。
- CLQR 预期 `constraint_dim == 1`，因此 `head_count == 2`。
- 如果 example name 与 `constraint_dim` 不匹配，应直接 `raise ValueError`，
  不能静默构造错误 head 数量。

## Replay 数据来源

v2 不新增环境数据采样器。`QPropCritic` 从 `DataStorage.take_experiences()`
返回的 rolling arrays 中采样：

```text
state_buffer
action_buffer
costs_buffer
next_state_buffer
```

默认策略：

- `qprop_replay_batch_size = grad_T`
- 如果 replay 可用条数少于 batch size，则使用全部可用 transition。
- 采样方式为均匀无放回；不引入 prioritized replay。
- 训练只在现有 SLDAC update timing 触发时执行。

MIMO3 与 CLQR2 的 `DataStorage` 构造签名不同，不能抽成共享导入。
实现时应在 `SLDAC_Pathwise.py` 内把当前 `take_experiences()` 结果传给
`QPropCritic.update_from_replay(...)`，让 `QPropCritic` 只接收 numpy arrays，
不直接依赖 `DataStorage` 类型。

## QPropCritic TD 目标

每个 head 单独更新。

输入：

```text
func_value: shape = (1 + constraint_dim,)
state_batch: shape = (B, state_dim)
action_batch: shape = (B, action_dim)
costs_batch: shape = (B, 1 + constraint_dim)
next_state_batch: shape = (B, state_dim)
actor: current GaussianPolicy
```

默认 target action 使用 actor mean：

```text
next_action = detach(actor.mean_action_tensor(next_state))
```

原因：

- Q-Prop 的 Taylor expansion 点是 `mu_theta(s)`。
- mean target 能减少 dedicated critic target 噪声。
- score signal 已经来自 SLDAC 主路径，不需要在 Q-Prop critic target 中
  再引入 stochastic score 估计。

average-cost TD target：

```text
y_h = costs_batch[:, h] - func_value[h]
      + QPropCritic_target_h(next_state_batch, next_action)
```

loss：

```text
smooth_l1_loss(QPropCritic_online_h(state_batch, action_batch), y_h)
```

注意：

- 这里没有 discounted factor。
- `qprop_target_tau_reward`、`qprop_target_tau_cost` 只表示 target network
  soft update 的插值系数。
- `SLDAC_Pathwise_main()` 中的默认解析规则为：

```text
qprop_target_tau_reward =
    args.qprop_target_tau_reward if present else gamma_pow_reward

qprop_target_tau_cost =
    args.qprop_target_tau_cost if present else gamma_pow_cost
```

- 这里复用的是主 critic 当前运行时的 target soft-update 系数。
- 即使变量来源名含有 `gamma`，在本项目中也只能解释为更新权重 /
  target-network 插值系数，不能解释为 discounted factor。

critic 训练阶段必须阻断 actor 梯度污染：

- `next_action` 必须 detached。
- `QPropCritic.update_from_replay(...)` 调用前后，actor parameters 的 `.grad`
  应保持不变。
- TD loss backward 只能更新 `QPropCritic` online nets。

`QPropCritic` 直接消费主线 `costs_batch`，不得对第 0 维再次取负。
当前主线已经把内部 objective cost 放入 `costs[:, 0]`；v2 不修改该构造。

## Actor gradient 数据流

### 仍由 SLDAC 主 critic 产生 score signal

`score_signal` 继续沿用 v1 实现：

```text
q_behavior_main = SLDAC_main_critic(s, a_behavior)
score_signal = preprocess(q_behavior_main, head_idx)
```

其中 preprocessing 继续保持当前 MIMO / CLQR 差异：

- MIMO：按 head 去均值。
- CLQR：objective head 标准差作为共享缩放。

### control signal 改由 QPropCritic 产生

对每个 head：

```text
mu = actor.mean_action_tensor(state)
qprop_q_mu = QPropCritic_target_h(state, mu)
action_grad = grad_a qprop_q_mu
control_signal = action_grad · (a_behavior - mu)
```

`control_signal` 在 score residual 中必须 detach。

### conservative gate

继续使用 batch-head 级别 gate：

```text
eta_h = 1 if cov(score_signal, control_signal) > 0 else 0
eta_h = 0 otherwise
```

这样当 dedicated critic 的 Taylor signal 与当前 score signal 不同向时，
Q-Prop 自动退化为原 score-function gradient。

### combined loss

```text
score_residual_loss =
    mean(log_prob(a_behavior | state)
         * detach(score_signal - eta_h * control_signal))

pathwise_control_loss =
    eta_h * mean(QPropCritic_target_h(state, mu))

combined_loss =
    score_residual_loss + pathwise_control_loss
```

然后沿用当前 `_extract_actor_gradient(...)` 生成 actor gradient vector。

## 参数冻结规则

actor gradient 抽取时：

- `QPropCritic` 参数不应累积 `.grad`。
- `QPropCritic_target_h(state, mu)` 必须保留对 `mu` 的 autograd path。
- 只能冻结 critic parameters 的 grad accumulation，不能把整个 forward 包进
  `torch.no_grad()`。
- 这条规则只适用于 actor gradient 阶段。critic TD update 阶段则必须 detach
  actor 产生的 target action。

推荐 helper：

```python
def _freeze_module_parameters(module):
    ...

def _restore_parameter_grad_states(states):
    ...
```

该 helper 可复用当前 `SLDAC_Pathwise.py` 里已有的 critic 参数保护思路。

## 配置项

新增配置项应集中在 `run_*_sldac_pathwise.py` 顶部，且支持传入
`SLDAC_Pathwise_main(args)`：

```python
# 是否为 qprop_conservative 启用 dedicated Q-Prop critic。
DEFAULT_USE_QPROP_DEDICATED_CRITIC = True

# Q-Prop critic 每次 SLDAC update 时的 TD 更新步数。
DEFAULT_QPROP_CRITIC_UPDATE_STEPS = 1

# Q-Prop critic replay batch 大小；None 表示使用 grad_T。
DEFAULT_QPROP_REPLAY_BATCH_SIZE = None

# Q-Prop critic target action；v2 默认使用 mean。
DEFAULT_QPROP_TARGET_ACTION_MODE = "mean"

# Q-Prop critic 学习率缩放；1.0 表示沿用主 critic 同阶学习率。
DEFAULT_QPROP_CRITIC_LR_SCALE = 1.0

# Q-Prop critic target net 的 objective head soft-update 系数；None 表示复用
# 当前运行时 gamma_pow_reward，但语义仍是 target 插值系数，不是折扣。
DEFAULT_QPROP_TARGET_TAU_REWARD = None

# Q-Prop critic target net 的 constraint head soft-update 系数；None 表示复用
# 当前运行时 gamma_pow_cost，但语义仍是 target 插值系数，不是折扣。
DEFAULT_QPROP_TARGET_TAU_COST = None
```

在 `SLDAC_Pathwise.py` 内部通过 `getattr(args, ..., default)` 读取，避免破坏
现有调用方。

## Diagnostics

现有 Q-Prop diagnostics 保留：

- `qprop_eta`
- `qprop_covariance`
- `score_grad_norm`
- `pathwise_grad_norm`
- `combined_grad_norm`
- `score_signal_mean`
- `score_signal_std`
- `control_signal_mean`
- `control_signal_std`

新增：

- `qprop_control_source`
  - 不作为逐步 numeric diagnostics 写入。
  - 在 metadata/config 中保存字符串：`"dedicated_critic"` 或 `"main_critic"`。
- `qprop_control_source_code`
  - 逐步 numeric diagnostics。
  - `1.0` 表示 dedicated critic，`0.0` 表示 main critic v1 ablation。
- `qprop_critic_loss`
  - shape = `(head_count,)`。
- `qprop_critic_td_error_mean`
  - shape = `(head_count,)`。
- `qprop_critic_target_mean`
  - shape = `(head_count,)`。
- `qprop_critic_pred_mean`
  - shape = `(head_count,)`。
- `qprop_replay_batch_size`
  - scalar。
- `qprop_target_action_mode`
  - 不作为逐步 numeric diagnostics 写入。
  - 在 metadata/config 中保存字符串，例如 `"mean"`。
- `qprop_target_action_mode_code`
  - 逐步 numeric diagnostics。
  - `0.0` 表示 `"mean"`；后续如加入 `"sample"`，使用 `1.0`。
- `qprop_pathwise_grad_ratio`
  - `pathwise_grad_norm / (score_grad_norm + pathwise_grad_norm + 1e-12)`。
- `qprop_score_grad_ratio`
  - `score_grad_norm / (score_grad_norm + pathwise_grad_norm + 1e-12)`。

这些字段用于回答“score gradient 与 pathwise gradient 实际占比是多少”。
注意它是梯度范数占比，不是公式中的凸组合权重。

## 模式兼容性

`POLICY_GRADIENT_MODES` 保持：

```python
POLICY_GRADIENT_MODES = (
    "stochastic_pathwise",
    "deterministic_dpg",
    "qprop_conservative",
)
```

新增 dedicated critic 不新增 mode 名。通过配置控制：

```text
policy_gradient_mode = "qprop_conservative"
use_qprop_dedicated_critic = True
```

如果 `use_qprop_dedicated_critic = False`，可保留 v1 行为作为 ablation：

```text
score signal: main critic
control signal: main critic
```

默认建议：

```text
use_qprop_dedicated_critic = True
```

## 测试策略

### Unit tests

新增：

- `MIMO3/test_qprop_critic.py`
- `CLQR2/test_qprop_critic.py`

覆盖：

1. `QPropCritic` 初始化 head 数量正确。
2. `update_from_replay(...)` 返回 finite loss diagnostics。
3. TD target 公式为 `cost - func_value + next_q`，不出现 discount multiplier。
4. `head_value(..., use_target=True)` 对 action 保持可微。
5. actor gradient 抽取时，`QPropCritic` 参数 `.grad` 不被累积。
6. `update_from_replay(...)` 不改变 actor parameters 的 `.grad`。
7. `costs_batch[:, 0]` 被直接作为 objective cost 使用，不做二次取负。

扩展：

- `MIMO3/test_sldac_pathwise.py`
- `CLQR2/test_sldac_pathwise.py`

覆盖：

1. `qprop_conservative` 可接收 dedicated critic。
2. dedicated critic 与 main critic 数值来源可区分。
3. diagnostics 中 `qprop_control_source == "dedicated_critic"`。
4. `qprop_pathwise_grad_ratio` 与 `qprop_score_grad_ratio` finite 且落在 `[0, 1]`。
5. `stochastic_pathwise`、`deterministic_dpg` 不要求构造 `QPropCritic`。
6. `use_qprop_dedicated_critic = False` 保留 v1 main-critic control source。
7. MIMO head count 为 5，CLQR head count 为 2；不匹配时抛出 `ValueError`。

### 静态与聚焦验证

使用 C 盘工作区默认 conda 环境：

```powershell
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m py_compile `
  "MIMO3\qprop_critic.py" `
  "CLQR2\qprop_critic.py" `
  "MIMO3\SLDAC_Pathwise.py" `
  "CLQR2\SLDAC_Pathwise.py"
```

```powershell
Push-Location "MIMO3"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest `
  "test_qprop_critic.py" "test_sldac_pathwise.py"
Pop-Location

Push-Location "CLQR2"
& "C:\Users\Admin\.conda\envs\torch_work\python.exe" -m unittest `
  "test_qprop_critic.py" "test_sldac_pathwise.py"
Pop-Location
```

### 仿真验证

实现完成后的 smoke / temporary simulation 必须写入 `Trash` 隔离目录。

不得直接运行默认 `run_mimo_sldac_pathwise.py` 或
`run_clqr_sldac_pathwise.py` 产生临时对比，因为默认入口可能写入正式
`outputs`、`.mat`、`.png` 或 checkpoint。

允许的方式只有两类：

1. 使用 direct harness 调用 `SLDAC_Pathwise_main(args)`，显式把
   `output_dir`、checkpoint 目录和图像路径都指向 `Trash`。
2. 或者先改入口支持临时 output root，并在命令中显式传入 `Trash` 路径。

默认先做：

```text
10 episode temporary comparison
```

再视结果做：

```text
100 episode temporary comparison
```

不得覆盖正式 `outputs`、checkpoint、历史 `.mat` 或 `.png`。

## 实现顺序建议

1. 写 RED tests：先要求 `QPropCritic` 存在、TD target 公式正确、action 可微。
2. 实现 `MIMO3/qprop_critic.py`。
3. 接入 `MIMO3/SLDAC_Pathwise.py`。
4. 镜像实现 `CLQR2/qprop_critic.py`。
5. 接入 `CLQR2/SLDAC_Pathwise.py`。
6. 扩展 diagnostics 保存与临时绘图脚本，使梯度范数占比可视化。
7. 只运行 py_compile、focused unittest、Trash 隔离仿真。

## 风险与缓解

### 风险 1：QPropCritic 早期不准

缓解：

- 保留 conservative gate。
- 增加 `qprop_min_critic_updates_before_gate` 可选配置，默认 0。
- diagnostics 记录 critic loss 与 TD error。

### 风险 2：target action 与主 critic 口径不同

缓解：

- v2 默认 `mean`，因为 Q-Prop control 在 `mu_theta(s)` 展开。
- 保留 `qprop_target_action_mode`，后续可增加 `"sample"` 做 ablation。

### 风险 3：梯度范数占比被误解为公式权重

缓解：

- 文档、日志、图标题统一写为 gradient norm ratio。
- 明确公式权重仍是 conservative gate `eta`，不是固定凸组合。

### 风险 4：重复 MIMO3 / CLQR2 代码

缓解：

- v2 先按目录内模块实现，避免改包结构和导入路径。
- 如果两个目录后续稳定，可再抽共享模块；不在本轮引入。

## 完成标准

1. `qprop_conservative` 默认使用 dedicated `QPropCritic` control source。
2. score signal 与 v1 SLDAC 口径一致。
3. `QPropCritic` 使用 average-cost TD target，代码和测试均不出现
   discounted target 解释。
4. `QPropCritic.update_from_replay(...)` 不污染 actor `.grad`。
5. target tau 来源明确，且只作为 target-network soft update 系数。
6. focused unittest 和 py_compile 通过。
7. diagnostics 能输出 `eta` 激活比例与 score/pathwise gradient norm ratio。
8. 临时仿真产物只写入 `Trash`。
