# Q-Prop Style Pathwise SLDAC Design

日期：2026-06-04

## 背景

用户希望参考 Q-Prop: Sample-Efficient Policy Gradient with an Off-Policy Critic，
修改 `Pathwise SLDAC`，使它聚合两种 policy gradient。

Q-Prop 的关键思想不是对两个梯度做普通线性加权，而是把
score-function policy gradient 与 critic pathwise gradient 组织成控制变量估计器：

```text
score-function residual gradient + critic pathwise control-variate gradient
```

论文原始表述基于 discounted return；本项目必须按
`infinite-horizon average reward / average cost CMDP` 理解。这里仅借用
Q-Prop 的控制变量结构，不引入 discounted objective，也不把
`gamma_pow_reward`、`gamma_pow_cost` 解释为折扣因子。

参考来源：

- arXiv: https://arxiv.org/abs/1611.02247
- OpenReview: https://openreview.net/forum?id=SJ3rcZcxl
- PDF: https://openreview.net/pdf?id=SJ3rcZcxl

## 当前代码事实

`MIMO3/SLDAC.py` 与 `CLQR2/SLDAC.py` 已有 score-function 估计路径：

```python
Q_hat_torch = critic.critic_value(state_batch_torch, action_batch_torch)
actor_loss = (Q_hat_torch[:, head] * log_prob).mean()
```

`MIMO3/SLDAC_Pathwise.py` 与 `CLQR2/SLDAC_Pathwise.py` 当前已有 pathwise 路径：

```python
action_for_grad = actor.sample_action_tensor(...)
head_value = _critic_head_value(critic, state_batch_torch, action_for_grad, head_idx)
head_objective = torch.mean(head_value)
head_objective.backward()
```

Pathwise 版本的 actor 梯度最终仍进入原 SLDAC 更新链：

```python
grad = (1 - alpha) * grad + alpha * grad_tilda
paras_bar = update_policy(func_value, grad, theta, tau_reward, tau_cost)
theta = (1 - beta) * theta + beta * paras_bar
```

因此最小侵入点是 `SLDAC_Pathwise.py` 的 actor 梯度估计函数，不应修改
`utils.update_policy()` 或 critic TD 更新主流程。

当前工作区已有未提交改动，包括：

- `MIMO3/SLDAC_Pathwise.py`
- `CLQR2/SLDAC_Pathwise.py`
- `MIMO3/run_mimo_sldac_pathwise.py`
- `CLQR2/run_clqr_sldac_pathwise.py`
- 对应 `test_sldac_pathwise.py`
- 若干 `outputs/checkpoints` 实验产物

实现时必须只叠加源码与测试改动，不回滚用户已有改动，不运行会覆盖正式实验产物的脚本。

## 目标

1. 在 Pathwise SLDAC 中新增一种 Q-Prop 风格的 actor 梯度估计模式。
2. 聚合 score-function 梯度与 critic pathwise 梯度，降低纯 score-function 方差，
   同时避免纯 pathwise 过度依赖 critic 准确性。
3. 保持现有 `stochastic_pathwise`、`deterministic_dpg` 行为可选，便于 ablation。
4. 保持 `objective cost` 口径：第 0 维为待最小化 objective cost，其余维为约束残差。
5. 增加 diagnostics，使后续实验能判断控制变量是否启用、梯度规模是否异常。

## 非目标

1. 不重写 critic TD 更新。
2. 不引入新的 replay buffer 或 off-policy 数据管线。
3. 不修改 `update_policy()` 的约束二次规划逻辑。
4. 不直接运行 `run_mimo_*`、`run_clqr_*` 训练入口生成正式结果。
5. 不改变现有 `.mat` / `.png` / checkpoint 文件命名与保存策略，除非用户另行确认。

## 方案比较

### 方案 A：普通加权和

形式：

```text
g = lambda * g_score + (1 - lambda) * g_pathwise
```

优点是实现最简单；缺点是没有 Q-Prop 的无偏校正结构。若 critic 有偏，
pathwise 分支可能直接把 actor 拉向错误方向；`lambda` 也会成为难调超参数。

结论：不作为主方案，可作为未来 ablation。

### 方案 B：标准 Q-Prop，始终启用控制变量

形式接近论文默认估计器：

```text
g = grad log pi(a|s) * (A_hat - A_bar) + grad_theta Q(s, mu_theta(s))
```

优点是最接近论文；缺点是当前 critic 早期可能较差，始终启用控制变量会放大错误
critic 的影响。论文也指出 conservative Q-Prop 更稳。

结论：不作为默认主方案。

### 方案 C：conservative Q-Prop 风格聚合

形式：

```text
eta in {0, 1}
g = grad log pi(a|s) * (score_signal - eta * control_signal)
    + eta * grad_theta Q(s, mu_theta(s))
```

当 score signal 与 Taylor control signal 正相关时启用控制变量，否则退化为
score-function 梯度。这个方案保守、可解释，适合先接入当前 Pathwise SLDAC。

结论：推荐作为 v1 实现。

## 推荐设计

### 配置

扩展两个 Pathwise 文件中的 `POLICY_GRADIENT_MODES`：

```python
POLICY_GRADIENT_MODES = (
    "stochastic_pathwise",
    "deterministic_dpg",
    "qprop_conservative",
)
```

运行入口默认值建议：

- `CLQR2/run_clqr_sldac_pathwise.py`：暂不强行改默认，保持现有默认以减少行为突变。
- `MIMO3/run_mimo_sldac_pathwise.py`：同样不强行改默认，用户确认实验方案后再切。

这样实现后可以通过顶部 Python 配置或 CLI 显式启用：

```text
policy_gradient_mode = "qprop_conservative"
```

### 梯度估计数据流

每次 `Q_update_index == Q_update_time` 时，对每个 head 单独计算梯度。

输入：

- `state_batch_torch`
- `action_batch_torch`
- `critic target_net{head}`
- 当前 actor 的 Gaussian policy
- `update_log_std`
- `normalize_actor_gradient`

`_compute_pathwise_gradients()` 的签名需要扩展为接收 `action_batch_torch`：

```python
def _compute_pathwise_gradients(
    actor,
    critic,
    state_batch_torch,
    action_batch_torch,
    constraint_dim,
    real_theta_dim,
    policy_gradient_mode,
    normalize_actor_gradient,
    update_log_std,
    return_diagnostics=False,
):
    ...
```

`SLDAC_Pathwise_main()` 中当前已经构造了：

```python
action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
```

实现时必须把它传入 `_compute_pathwise_gradients()`。既有测试中所有直接调用
`_compute_pathwise_gradients()` 的位置也需要同步传入测试用 `action_batch_torch`。

中间量：

1. `q_behavior = Q_h(s, a_behavior)`
   - 用当前 batch 中真实采样动作。
   - 作为 score-function 分支的 cost-like learning signal 来源。

2. `score_signal`
   - 复用现有 SLDAC 预处理口径。
   - MIMO：

```text
score_signal_h = q_behavior_h - mean(q_behavior_h)
```

   - CLQR：

```text
objective_std = std(q_behavior_0) + 1e-6
score_signal_0 = (q_behavior_0 - mean(q_behavior_0)) / objective_std
score_signal_h = (q_behavior_h - mean(q_behavior_h)) / objective_std, h >= 1
```

   - 输出必须 detach，不保留到 critic 或 actor 的 autograd graph。
   - 输出 shape 为 `(batch_size,)`，dtype 与当前 device 上 actor/critic tensor 一致。
   - v1 不改变这些历史行为，避免同时引入算法改造和基线语义漂移。

3. `mu = actor.mean_action_tensor(s)`
   - Q-Prop 的 Taylor 展开点使用策略均值。
   - 这与论文中围绕 `mu_theta(s)` 展开一致。

4. `control_signal`
   - 近似 Taylor 一阶项：

```text
control_signal = grad_a Q_h(s, a)|a=mu · (a_behavior - mu)
```

   - 在 score residual 中作为 detached scalar 使用，不让 residual 信号自身反传。
   - 对 batch 去均值后参与相关性判断。
   - v1 有意只把 Taylor 一阶项放入 score residual。`Q_h(s, mu)` 常数基线不进入
     `control_signal`，因为它对 `grad log pi(a|s)` 的期望梯度为 0，且当前
     score signal 预处理已经做了中心化 / 标准化。

5. `eta`
   - v1 使用 batch-head 级 conservative gate：

```text
eta_h = 1 if mean(score_signal_centered * control_signal_centered) > 0 else 0
```

   - 这样比逐样本 gate 更稳，也方便 diagnostics。
   - 后续若需要更贴近论文，可新增 per-sample gate 作为实验选项。

输出梯度：

```text
score_residual_loss =
    mean(log_prob(a_behavior | s) * (score_signal - eta_h * control_signal))

pathwise_control_loss =
    mean(eta_h * Q_h(s, mu))

combined_loss = score_residual_loss + pathwise_control_loss
```

对 `combined_loss.backward()` 后，沿用 `_extract_actor_gradient()` 抽取 actor 参数梯度。

### 建议 helper 拆分

为避免把 Q-Prop 逻辑堆进 `_compute_pathwise_gradients()`，实现时建议拆成
以下只服务于 `SLDAC_Pathwise.py` 的私有 helper：

1. `_preprocess_score_signal(example_name, q_values_np, head_idx)`
   - 输入 detached numpy / tensor 值。
   - 输出 detached tensor。
   - 负责复刻当前 `SLDAC.py` 中 MIMO 与 CLQR 的 score signal 预处理口径。
   - 输出 shape 固定为 `(batch_size,)`。
   - MIMO 对每个 head 单独去均值；CLQR 所有 head 共享 objective head 的标准差。

2. `_compute_taylor_control_signal(critic, state_batch_torch, action_batch_torch, mu_torch, head_idx)`
   - 使用 `target_net{head_idx}`。
   - 返回 detached control signal、`Q_h(s, mu)`、以及 action-gradient diagnostics。
   - 不改变 critic 参数的 `requires_grad` 恢复逻辑。

3. `_compute_conservative_qprop_eta(score_signal, control_signal)`
   - v1 返回 batch-head 标量 `0.0` 或 `1.0`。
   - 同时返回 covariance，写入 diagnostics。

4. `_compute_qprop_conservative_gradient(...)`
   - 组合 score residual loss 与 pathwise control loss。
   - 调用 `_extract_actor_gradient()` 生成最终梯度向量。
   - 返回 `grad_tilda_torch[head_idx]` 与该 head 的 diagnostics。

`_compute_pathwise_gradients()` 只负责模式分发：

```text
stochastic_pathwise -> 现有 reparameterized pathwise
deterministic_dpg   -> 现有 mean-action DPG
qprop_conservative  -> 新 Q-Prop conservative helper
```

### critic value 来源

Q-Prop 分支中 `q_behavior` 与 `Q_h(s, mu)` 应统一使用 target critic head，
即现有 `_critic_head_value()` 访问的 `target_net{head_idx}`。

不建议在 Q-Prop 分支直接调用 `critic.critic_value()` 作为唯一来源，因为该函数会
detach 到 numpy，并且 CLQR 的 `legacy_online_mode` 分支可能混用 `net1` 与
`target_net1`。实现时可以用 `critic.critic_value()` 的历史行为作为预处理参考，
但实际反传路径必须保留 tensor graph 到 actor action。

### 符号约定

本项目内部优化的是 cost-like objective：

- `func_value[0]`：待最小化 objective cost。
- `func_value[1:]`：约束残差。
- `reward_average_save` 等历史命名仍表示 objective cost 曲线。

因此 Q-Prop 分支中的 `Q_h`、`score_signal`、`control_signal` 都按 cost-like
梯度解释，不做 reward-maximization 方向翻转。

### log_std 处理

当前 Pathwise 代码已有 `update_log_std`：

- `True`：actor mean 网络与 `log_std` 都参与更新。
- `False`：只更新 mean 网络，`log_std` 梯度位置保持 0。

Q-Prop residual score 分支会通过 `log_prob` 产生 `log_std.grad`，但最终仍由
`_extract_actor_gradient(..., update_log_std=...)` 统一决定是否写入梯度向量。

### critic 参数保护

沿用现有 `_freeze_critic_head_parameters()` / `_restore_parameter_grad_states()`。
Q-Prop 模式中 critic 只作为 frozen control variate，不应积累 target critic 参数梯度。

### diagnostics

在现有 diagnostics 上增加 Q-Prop 字段：

- `score_grad_norm`
- `pathwise_grad_norm`
- `combined_grad_norm`
- `qprop_eta`
- `qprop_covariance`
- `score_signal_mean`
- `score_signal_std`
- `control_signal_mean`
- `control_signal_std`

非 Q-Prop 模式下字段填 0 数组，保证 `.mat` schema 稳定。数组形状约定：

```text
score_grad_norm      -> (head_count,)
pathwise_grad_norm   -> (head_count,)
combined_grad_norm   -> (head_count,)
qprop_eta            -> (head_count,)
qprop_covariance     -> (head_count,)
score_signal_mean    -> (head_count,)
score_signal_std     -> (head_count,)
control_signal_mean  -> (head_count,)
control_signal_std   -> (head_count,)
```

`_pack_pathwise_diagnostics()` 继续把每次更新的 head 级数组堆叠为：

```text
(num_actor_updates, head_count)
```

计算语义：

- `score_grad_norm`：仅在 `return_diagnostics=True` 且
  `policy_gradient_mode="qprop_conservative"` 时，通过单独反传
  `score_residual_loss` 得到 actor 梯度范数。反传前后必须清空 actor 梯度，
  不得污染最终更新梯度。
- `pathwise_grad_norm`：同样仅在 Q-Prop diagnostics 中，通过单独反传
  `pathwise_control_loss` 得到 actor 梯度范数。
- `combined_grad_norm`：最终实际用于更新的 `combined_loss` 梯度范数，必须与
  `grad_tilda_torch[head_idx]` 一致。
- diagnostics 关闭时，不做额外 component backward，只计算最终更新梯度。
- 空 history 时新增字段返回 shape 为 `(0, head_count)` 的 float64 数组。

## 代码修改范围

### MIMO3

- `MIMO3/SLDAC_Pathwise.py`
  - 扩展 `POLICY_GRADIENT_MODES`。
  - 新增 score signal 预处理 helper。
  - 新增 Q-Prop conservative 梯度 helper。
  - 扩展 diagnostics packing。

- `MIMO3/run_mimo_sldac_pathwise.py`
  - 仅确保 config / parser 可传入 `qprop_conservative`。
  - 不改变默认训练产物路径。

- `MIMO3/test_sldac_pathwise.py`
  - 增加 Q-Prop 梯度形状、有限值、diagnostics、critic 参数无污染测试。

### CLQR2

- `CLQR2/SLDAC_Pathwise.py`
  - 与 MIMO3 保持同构改动。

- `CLQR2/run_clqr_sldac_pathwise.py`
  - 同上。

- `CLQR2/test_sldac_pathwise.py`
  - 同上。

## 测试计划

只运行不会写正式实验产物的检查：

Python 环境按 `AGENTS.md` 规则选择：当前仓库在 C 盘，默认使用 conda 环境
`torch_work`；如果项目移到 D 盘，才默认使用 `torch`。

```powershell
& "<conda_base>\envs\torch_work\python.exe" -m py_compile `
  "MIMO3\SLDAC_Pathwise.py" `
  "CLQR2\SLDAC_Pathwise.py" `
  "MIMO3\run_mimo_sldac_pathwise.py" `
  "CLQR2\run_clqr_sldac_pathwise.py"
```

```powershell
Push-Location "MIMO3"
& "<conda_base>\envs\torch_work\python.exe" -m unittest "test_sldac_pathwise.py"
Pop-Location

Push-Location "CLQR2"
& "<conda_base>\envs\torch_work\python.exe" -m unittest "test_sldac_pathwise.py"
Pop-Location
```

不在实现验证中运行：

- `MIMO3/run_mimo_sldac_pathwise.py`
- `CLQR2/run_clqr_sldac_pathwise.py`
- `run_mimo_*`
- `run_clqr_*`
- plot/export 脚本

如后续需要正式实验，必须先明确输出路径、文件模式与覆盖风险，并获得用户确认。

## 风险与缓解

1. **critic 早期不准**
   - 使用 conservative gate，相关性非正时退回 score-function residual。

2. **梯度尺度突变**
   - 保留 `normalize_actor_gradient`。
   - diagnostics 记录 score/pathwise/combined 三类梯度范数。

3. **CLQR 与 MIMO 历史 signal 预处理不同**
   - v1 保持各自现有 SLDAC 口径，不把算法改造和基线重定义混在一起。

4. **log_std 更新语义不一致**
   - 统一交给 `update_log_std` 控制，不新增第二套开关。

5. **误解释 reward / objective**
   - spec 和注释均使用 objective cost 口径。
   - 不在实现中引入 reward maximization 符号翻转。

6. **实验产物覆盖**
   - 实现阶段只做编译和单元测试。
   - 不运行正式训练入口。

## 验收标准

1. `policy_gradient_mode="qprop_conservative"` 可通过配置解析。
2. MIMO3 与 CLQR2 的 Q-Prop 梯度输出 shape 均为：

```text
(1 + constraint_dim, real_theta_dim)
```

3. Q-Prop 模式输出梯度全部有限。
4. `update_log_std=False` 时，梯度向量的 `log_std` 段为 0。
5. Q-Prop 反传后 target critic 参数没有残留 `.grad`。
6. diagnostics schema 可保存为 `.mat` payload。
7. 只运行 `py_compile` 与 `unittest`，不产生或覆盖正式实验结果。
8. `qprop_eta=0` 路径退化为 score residual；`qprop_eta=1` 路径同时包含
   score residual 与 pathwise control。

## 后续可选扩展

1. 增加 `qprop_eta_granularity = "batch" | "sample"`。
2. 增加 `qprop_mode = "conservative" | "aggressive" | "adaptive"`。
3. 增加普通加权和 ablation。
4. 增加 isolated smoke run，输出到临时目录并在结束后移动到 `Trash`。
