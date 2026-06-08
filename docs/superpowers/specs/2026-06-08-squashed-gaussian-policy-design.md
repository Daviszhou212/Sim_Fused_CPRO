# Squashed Gaussian Policy Design for SLDAC and Fused-CPRO

## 背景

当前 `SLDAC`、`SLDAC_Pathwise` 与 `Fused_CPRO` 都复用
`model.py` 中的 `GaussianPolicy_MIMO` / `GaussianPolicy_CLQR`。
这些 actor 当前使用普通 Gaussian：

```text
raw action ~ Normal(mu, std)
log_prob = Normal(mu, std).log_prob(action)
```

MIMO actor 的网络均值虽然通过 `2.5 * sigmoid(...)` 被限制到正区间，
但采样动作仍来自无界 Gaussian。环境随后只对非正 MIMO 动作做下界保护，
这导致 `sample_action()`、环境实际动作和 `evaluate_action()` 的概率密度
不完全一致。

Fused-CPRO 还在 `Fused_CPRO.py` 内部维护自己的 mixture log-prob：

```text
log pi_mix(a|s) = logsumexp_n(log rho_n + log pi_n(a|s))
```

这部分是算法核心，不应因为引入 squashed Gaussian 而改变。

## 用户确认的范围

1. MIMO power 维保持有限区间：

```text
power_i in (ACTION_EPS, 2.5)
```

2. MIMO regularization factor 只要求正值，不强加 `2.5` 上界：

```text
reg_factor in (ACTION_EPS, +inf)
```

3. regularization factor 使用 softplus 变换：

```text
reg = softplus(raw_reg) + ACTION_EPS
```

4. CLQR 动作沿用当前 DK policy 的边界：

```text
action_i in (-1.5, 1.5)
```

5. 旧 SLDAC checkpoint 可以继续加载，但按新 squashed policy 重新解释参数；
   不承诺与旧 Gaussian 行为等价。正式实验应重新训练并产出新 checkpoint。

6. actor 类可以共享，但算法实现必须分离：

- `SLDAC.py` 保持 score-function policy-gradient 流程。
- `SLDAC_Pathwise.py` 保持 pathwise / Q-Prop 流程。
- `Fused_CPRO.py` 保持 mixture、rho 更新和 offline/online 混合流程。

7. Fused-CPRO 的 mixture 机制本身不受影响。只替换各 component policy 的
   `log pi_n(a|s)` 计算口径。

## 常量约定

新增独立动作下界常量：

```python
ACTION_EPS = 1e-6
```

`ACTION_EPS` 只用于 action transform 的开区间下界和 positive-only 维度偏移，
不得复用当前 `model.py` 中用于初始化的 `EPS = 0.003`，也不得复用
`Fused_CPRO.py` 中用于概率归一化保护的 `EPS = 1e-8`。这样可以保持 MIMO
环境当前的正值保护口径，不把最小 power / reg 从 `1e-6` 意外抬高到 `0.003`。

另外新增 inverse clamp 辅助常量：

```python
ACTION_INVERSE_EPS = 1e-6
```

它只用于把需要反变换的动作值移动到 open support 内部，避免 `logit(0)`、
`atanh(1)` 或 `log(expm1(0))`。

## 目标

1. 让 `sample_action()` 和 `evaluate_action()` 在同一个有界动作空间内一致。
2. 为 finite-bound 与 positive-only 动作维度补齐 log-prob Jacobian 修正。
3. 保持 `SLDAC`、`SLDAC_Pathwise`、`Fused_CPRO` 三条算法流程分离。
4. 保持 Fused-CPRO mixture 公式、rho 语义、simplex 更新、offline/online 数据混合不变。
5. 让 DK policy smoothing density 也落在同一 bounded action space 中，避免 mixture 中混入无界 density。
6. 明确旧 checkpoint 的兼容语义：参数可加载，行为按新分布解释。

## 非目标

1. 不重写 SLDAC / Fused-CPRO 的 actor update 或 critic update 公式。
2. 不合并 `SLDAC.py`、`SLDAC_Pathwise.py`、`Fused_CPRO.py` 的训练流程。
3. 不改变 `rho` 的含义、下界、投影或 CVX 更新。
4. 不改变 Fused-CPRO 的 old policy library 组成。
5. 不重新解释 `reward` 为 discounted reward；本项目仍按 average-cost CMDP 口径处理。
6. 不直接运行训练、绘图或会覆盖实验产物的脚本。

## 当前代码事实

### 共享 actor 类

`SLDAC.py`、`SLDAC_Pathwise.py` 和 `Fused_CPRO.py` 都使用
`GaussianPolicy_MIMO` / `GaussianPolicy_CLQR`。因此 actor 分布语义应该在
actor 类内部统一，否则不同算法路径会出现采样和 log-prob 不一致。

`MIMO3` 还存在 `GaussianPolicy_MultiCellMIMO_CTDE`。它当前只由
`MIMO3/SLDAC.py` 的 `MIMO_CTDE` 路径使用，Fused-CPRO 主线不使用 CTDE。

### Fused-CPRO 独立 log-prob

`Fused_CPRO.py` 目前没有直接调用 `actor.evaluate_action()`，而是有自己的
`_log_prob_batch()` 和 `_build_mixture_log_prob()`。引入 squashed Gaussian 后，
这部分必须同步到 actor 的新 log-prob 口径，但不能改变 mixture 公式。

### regularization factor

MIMO 动作最后一维是 regularization factor：

```text
[power_1, ..., power_UE, reg_factor]
```

CTDE MIMO 中每个 cell 都有一个 regularization factor：

```text
[cell0_power..., cell0_reg, cell1_power..., cell1_reg, ...]
```

环境当前只要求它为正，没有显式上界。DK policy 当前把它固定为 `0.25`。

## 推荐架构

### 总体边界

采用以下边界：

```text
共享 actor distribution
  - sample / mean / log_prob / inverse transform 统一在 actor 层处理

分离 algorithm update logic
  - SLDAC.py 不共享 Fused-CPRO 的更新代码
  - SLDAC_Pathwise.py 不共享 SLDAC.py 的梯度估计代码
  - Fused_CPRO.py 保留自己的 mixture/rho/offline-online 代码
```

这能保证 policy distribution 一致，同时不破坏三条算法实现的独立性。

### Actor 公共契约

保留现有 public methods：

```python
sample_action(state, use_mean=False)
sample_action_tensor(state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True)
evaluate_action(state_torch, action_torch)
mean_action_tensor(state_torch)
```

但语义调整为：

- `sample_action*()` 返回 transformed action。
- `mean_action_tensor()` 返回 transformed deterministic action。
- `evaluate_action()` 接收 transformed action，并计算 transformed density 的 log-prob。
- `log_std` 仍作为 actor 参数向量的一部分参与 SLDAC / Fused-CPRO 更新。

`GaussianPolicy_MultiCellMIMO_CTDE` 的本地执行接口也纳入同一契约：

```python
mean_cell_action_tensor(local_state_torch)
sample_cell_action(local_state, cell_index=0, use_mean=True)
```

这两个方法必须返回 transformed cell action。每个 cell block 内 power 使用
`(ACTION_EPS, 2.5)`，最后一维 regularization factor 使用
`softplus(raw) + ACTION_EPS`。

## Action Transform

### MIMO 单小区

动作维度：

```text
0 .. UE_num-1 : power
UE_num        : reg_factor
```

power 维使用 finite interval sigmoid transform：

```text
u_power ~ Normal(loc_power, std_power)
power = ACTION_EPS + (2.5 - ACTION_EPS) * sigmoid(u_power)
```

regularization factor 使用 positive softplus transform：

```text
u_reg ~ Normal(loc_reg, std_reg)
reg = softplus(u_reg) + ACTION_EPS
```

### MIMO CTDE

每个 cell 的 action block：

```text
[power_1, ..., power_K, reg_factor]
```

每个 block 内：

- power 维使用 `(ACTION_EPS, 2.5)` sigmoid transform。
- reg 维使用 `softplus(raw) + ACTION_EPS`。

### CLQR

所有维度使用 symmetric tanh transform：

```text
u ~ Normal(loc, std)
action = 1.5 * tanh(u)
```

## Log-Prob 修正

### 有限区间 sigmoid

```text
y = low + scale * sigmoid(u)
scale = high - low
z = (y - low) / scale
u = logit(z)
```

Jacobian：

```text
log |dy/du| = log(scale) + log_sigmoid(u) + log_sigmoid(-u)
```

因此：

```text
log pi_y(y|s) = log Normal(u; loc, std) - log |dy/du|
```

### 正值 softplus

```text
y = softplus(u) + ACTION_EPS
u = log(expm1(y - ACTION_EPS))
```

Jacobian：

```text
log |dy/du| = log sigmoid(u) = -softplus(-u)
```

因此：

```text
log pi_y(y|s) = log Normal(u; loc, std) - log sigmoid(u)
```

### CLQR tanh

```text
y = scale * tanh(u)
u = atanh(y / scale)
scale = 1.5
```

Jacobian：

```text
log |dy/du| = log(scale) + log(1 - tanh(u)^2)
```

实现时应使用稳定形式，避免 `u` 较大时 `log(1 - tanh(u)^2)` 下溢。

## 数值稳定规则

1. 所有 inverse transform 都只在 open support 内计算。
2. 对接近边界但仍来自合法 sampler 的 action，可用 `ACTION_INVERSE_EPS`
   做内部 clamp，避免 `logit(0)`、`logit(1)`、`atanh(1)`。
3. 对 actor component 而言，明确位于 support 外或精确落在 open-boundary
   上的 action 密度为 0，`evaluate_action()` 应返回 `-inf` mask，不能
   clamp 后照常给有限 log-prob。
4. DK smoothing density 是例外适配：DK deterministic sample 当前可能等于
   power 上界或 CLQR 裁剪边界。DK 的 `log_prob_batch()` 可以对
   `dk_mean_action` 和被评估 `action` 做内部 open-support clamp，用于获得
   有限 surrogate density；这个 clamp 不得回写到 buffer，也不得改变
   `sample_action()` 的 deterministic 输出。
5. `ACTION_EPS` 和 `ACTION_INVERSE_EPS` 应集中定义在 `model.py` /
   `Fused_CPRO.py` 顶部，并附中文注释说明用途。
6. softplus inverse 对很小的 `y - ACTION_EPS` 使用稳定实现，避免 `log(0)`。

## Fused-CPRO 设计约束

Fused-CPRO 的 mixture 机制保持不变：

```text
log pi_mix(a|s) = logsumexp_n(log rho_n + log pi_n(a|s))
```

不改以下逻辑：

- `rho` 表示 policy component 选择概率。
- `new_actor`、DK policy、旧 SLDAC policy 仍是不同 component。
- `rho_torch` 仍通过 mixture log-prob 得到梯度。
- `rho_lower_bounds`、simplex projection、CVX policy update 不变。
- `actor_xi` / `critic_xi` 与 offline/online 数据混合不变。
- old policy library 的构造顺序和 labels 不变。

需要改变的是 component density：

1. new actor:

```text
log pi_new(a|s) = transformed actor density under actor_new
```

Fused-CPRO 不能简单丢弃当前 `_log_prob_batch()` 的 `log_std_leaf` 设计。
推荐保留 `_log_prob_batch()` 作为 Fused-CPRO 专用 adapter：它可以复用 actor
内部的 transform / inverse / log-det helper，但仍必须返回：

```python
log_prob, log_std_leaf
```

这样 `_flatten_actor_grad()` 仍能拼接 `log_std_leaf.grad`，避免只保留
`rho_torch.grad` 而静默丢掉 actor `log_std` 更新。

2. Frozen SLDAC actor:

```text
log pi_old_sldac(a|s) = frozen_actor.evaluate_action(s, a)
```

Frozen actor 不需要 `log_std_leaf`，但其 log-prob 必须使用同一 transformed
density。

3. DK policy:

DK 执行时仍返回确定性均值动作，保持当前采样行为。
DK 的 smoothing density 改成 bounded/squashed density：

```text
raw_mean = inverse_transform(dk_mean_action)
raw_action = inverse_transform(action)
log pi_dk(a|s) = log Normal(raw_action; raw_mean, fixed_std) - log_det(action)
```

这样 mixture 中所有 component 都在同一 transformed action space 上给 density。
`raw_mean` 和 `raw_action` 的 inverse 输入都必须先进入 open support。DK
对外 sample 行为仍保持 deterministic mean，不因为 density clamp 而变化。

## 旧 Checkpoint 兼容

旧 `.pt` checkpoint 的网络权重和 `log_std` 可以继续加载。

加载后的解释方式：

- MIMO output layer 的线性输出作为 raw loc 使用。
- CLQR output layer 的线性输出作为 raw loc 使用。
- `log_std` 作为 raw Gaussian 的 standard deviation 参数使用。

这意味着旧 checkpoint 的行为会发生变化：

- 旧版 MIMO 的 `2.5 * sigmoid(linear_output)` 只约束均值。
- 新版 MIMO 的 `linear_output` 会作为 raw loc，再经过 action transform。
- 旧 checkpoint 可作为 warm start，但不能作为旧曲线的行为等价复现。

文档、日志或 checkpoint metadata 应标明这是 squashed policy 口径。

可执行策略：

1. 新保存的 checkpoint 必须写入 policy transform metadata，例如：

```python
"actor_distribution": "squashed_gaussian_v1"
"action_transform": {
    "mimo_power": ["sigmoid_interval", ACTION_EPS, 2.5],
    "mimo_reg": ["softplus_positive", ACTION_EPS],
    "clqr": ["tanh_interval", -1.5, 1.5],
}
```

2. 加载缺少上述 metadata 的旧 checkpoint 时，默认允许加载，但必须打印
   warning，并在返回/日志中标明：

```text
legacy checkpoint loaded under squashed_gaussian_v1; behavior is not equivalent
```

3. 本设计不在加载阶段拒绝旧 checkpoint，因为用户已确认“按新 squashed
   Gaussian 重新解释并使用 checkpoint 参数”。后续如果需要严格复现实验，
   应单独增加 `strict_policy_transform` 开关。

## 代码修改范围

### MIMO3

- `MIMO3/model.py`
  - 为 MIMO / MIMO_CTDE / CLQR actor 加入 transformed Gaussian 逻辑。
  - 将 MIMO 网络输出解释为 raw loc，不再在 network `forward()` 里直接返回 action-space mean。
  - `mean_action_tensor()` 负责返回 transformed mean action。
  - `evaluate_action()` 负责 inverse transform + log-det 修正。

- `MIMO3/Fused_CPRO.py`
  - `_log_prob_batch()` 改为使用 actor 的 transformed log-prob。
  - `_log_prob_batch()` 必须继续返回 `log_std_leaf` 或等价梯度通路。
  - `FrozenActorPolicy.log_prob_batch()` 改为使用 frozen actor 的 transformed log-prob。
  - `HeuristicGaussianPolicy.log_prob_batch()` 改为 bounded smoothing density。
  - `HeuristicGaussianPolicy.log_prob_batch()` 内部对 DK mean 和 action 做
    open-support clamp，但 `sample_action()` 输出不变。
  - `_mean_action()` 改为使用 actor 的 transformed mean action。
  - mixture/rho/offline-online 逻辑不改。

- `MIMO3/SLDAC.py`
  - 保持算法流程不变。
  - 继续调用 `actor.sample_action()` 与 `actor.evaluate_action()`。
  - 只因 actor 契约改变而获得 bounded action 与修正 log-prob。

- `MIMO3/SLDAC_Pathwise.py`
  - 保持 pathwise / Q-Prop 算法流程不变。
  - `sample_action_tensor(reparameterized=True)` 返回 transformed differentiable action。
  - `mean_action_tensor()` 返回 transformed deterministic action。

- `dk_policies.py`
  - 必须同步 bounded DK smoothing density，或明确删除 / 降级其中的
    `log_prob_batch()`。当前文件暴露 `HeuristicGaussianPolicy.log_prob_batch()`，
    因此本设计默认要求同步实现，避免展示文件与主实现继续给出不同概率口径。

### CLQR2

`CLQR2` 中的镜像文件需要同步相同改动：

- `CLQR2/model.py`
- `CLQR2/Fused_CPRO.py`
- `CLQR2/SLDAC.py`
- `CLQR2/SLDAC_Pathwise.py`
- 相关测试文件

### 受共享 actor 影响的其他算法

`ACPO.py`、`SCAOPO.py` 等如果复用 `GaussianPolicy_MIMO` / `GaussianPolicy_CLQR`，
也会继承新的 transformed policy 行为。当前任务的验证重点是 SLDAC、Pathwise
和 Fused-CPRO；其他算法不在本 spec 的语义验收范围内。

## 测试计划

### Unit tests

新增或更新 actor 分布测试：

1. MIMO power sample 全部在 `(ACTION_EPS, 2.5)`。
2. MIMO reg sample 全部大于 `ACTION_EPS`，且允许大于 `2.5`。
3. MIMO CTDE 每个 cell 的 power / reg 维符合各自 support。
4. CLQR sample 全部在 `(-1.5, 1.5)`。
5. `evaluate_action()` 对 `sample_action_tensor()` 产生的动作返回 finite log-prob。
6. `reparameterized=True` 时 transformed action 对 actor 参数可微。
7. `use_mean=True` 返回 transformed mean action，而不是 raw loc。
8. 旧 checkpoint 结构加载后参数 shape 不变。
9. `mean_cell_action_tensor()` 与 `sample_cell_action()` 返回 transformed cell action。
10. raw/action round-trip 在 MIMO power、MIMO reg、CLQR 三类 transform 上成立。
11. `evaluate_action()` 数值等于 `Normal(raw) - log_det` 的手工计算结果。
12. 边界附近 action 的 log-prob finite，support 外或精确 open-boundary 的 actor
    density 为 `-inf`。
13. softplus inverse 对接近 `ACTION_EPS` 的 reg value 不产生 NaN。

新增或更新 Fused-CPRO 测试：

1. `_build_mixture_log_prob()` 仍使用 `logsumexp(log rho + log pi_n)`。
2. `rho_torch.grad` 仍非空且 finite。
3. new actor component 使用 transformed log-prob。
4. new actor component 的 `log_std_leaf.grad` 仍非空且 finite。
5. frozen SLDAC component 使用 transformed log-prob。
6. DK component 的 sample 仍是 deterministic mean。
7. DK component 的 log-prob 在 bounded action space 内 finite；DK mean 或
   action 位于裁剪边界时也不产生 `inf` / `NaN`。
8. mixture 逻辑不改变 component 数量、rho lower bounds 或 labels。
9. `MIMO3` 当前没有 `test_fused_cpro_rho_bounds.py`；实现时应新增 MIMO3
   对应 mixture/rho 单元测试文件，或新增同等覆盖的 MIMO3 Fused-CPRO 测试。

更新 Pathwise / Q-Prop 测试：

1. `sample_action_tensor(reparameterized=True)` 输出在 support 内且可微。
2. `deterministic_dpg` 的 `mean_action_tensor()` 输出在 support 内。
3. Q-Prop control signal 使用 transformed `mu` 和 transformed behavior action。

### 静态验证

在不运行训练脚本的前提下执行：

```powershell
python -m py_compile MIMO3\model.py MIMO3\SLDAC.py MIMO3\SLDAC_Pathwise.py MIMO3\Fused_CPRO.py
python -m py_compile CLQR2\model.py CLQR2\SLDAC.py CLQR2\SLDAC_Pathwise.py CLQR2\Fused_CPRO.py
```

涉及 PyTorch 的测试按项目约定优先使用 D 盘仓库默认 conda 环境 `torch`。

### 聚焦测试

优先运行不会写正式实验产物的单元测试：

```powershell
python -m unittest MIMO3\test_sldac_pathwise.py
python -m unittest MIMO3\test_qprop_critic.py
python -m unittest MIMO3\test_ctde_mimo.py
python -m unittest CLQR2\test_sldac_pathwise.py
python -m unittest CLQR2\test_qprop_critic.py
```

如需新增测试，测试应只做函数级或 mock 验证，不写 `outputs` / `checkpoints`。

### 仿真验证

不在本 spec 阶段直接运行。若后续需要 smoke simulation，必须：

1. 先确认脚本不会覆盖正式 `*.mat`、`*.png`、`*.pdf`、checkpoint。
2. 输出到临时目录、临时文件名或隔离前缀。
3. 完成后把临时产物移入对应 `Trash` 文件夹。

## 风险与缓解

### 风险 1：Fused-CPRO mixture 被意外改变

缓解：

- 不改 `_build_mixture_log_prob()` 的数学结构。
- 添加测试断言 mixture 仍是 `logsumexp(log rho + component_log_prob)`。
- 只替换 component log-prob 来源。

### 风险 2：旧 checkpoint 行为变化

缓解：

- 明确旧 checkpoint 按新 squashed policy 解释。
- checkpoint metadata 记录 policy transform 版本。
- 缺少 transform metadata 的旧 checkpoint 加载时必须 warning。
- 正式实验重新训练 SLDAC checkpoint。

### 风险 3：reg_factor 太大导致环境数值变化

softplus 无上界，理论上允许大 reg。

缓解：

- 初始 raw loc 和 `log_std` 沿用现有参数，softplus 比 exp 更温和。
- 单元测试检查 finite action 和 finite log-prob。
- 后续实验若发现数值问题，再考虑 raw loc clamp 或 log_std 上界；本 spec 不先引入额外约束。

### 风险 4：重复 MIMO3 / CLQR2 修改漂移

缓解：

- 两边镜像文件同步修改。
- 测试同时覆盖 `MIMO3` 与 `CLQR2`。
- 数学 helper 可以在每个 `model.py` 内部局部封装，避免算法模块共享训练逻辑。

### 风险 5：支持域外 action 造成 NaN

缓解：

- actor component 对 support 外 action 返回 `-inf` mask，不能 clamp 成有限密度。
- DK smoothing density 对 deterministic 边界样本做内部 open-support clamp。
- 更新测试数据，避免继续用 unconstrained `torch.randn` 直接当 action。

### 风险 6：Fused-CPRO 丢失 `log_std` 更新

缓解：

- 保留 `_log_prob_batch()` 的 `log_std_leaf` 返回值或提供等价梯度通路。
- 测试断言 actor `log_std` 梯度仍存在。
- 不把 Fused-CPRO 的 component density 改成一个只返回 detached log-prob 的调用。

## 完成标准

1. `SLDAC`、`SLDAC_Pathwise`、`Fused_CPRO` 仍是分离文件和分离算法流程。
2. actor 的 sample、mean、log-prob 都使用同一 transformed action distribution。
3. MIMO power 在 `(ACTION_EPS, 2.5)`，reg_factor 在 `(ACTION_EPS, +inf)`。
4. CLQR action 在 `(-1.5, 1.5)`。
5. Fused-CPRO mixture 公式、rho 更新、offline/online 混合不变。
6. Fused-CPRO 保留 `_log_prob_batch()` 的 `log_std_leaf` 梯度通路。
7. DK policy 的执行动作保持 deterministic mean，log-prob 改为 bounded smoothing density。
8. 旧 checkpoint 可加载并 warning，新 checkpoint 写入 transform metadata。
9. 聚焦单元测试和 `py_compile` 通过。
10. 未运行会覆盖正式实验产物的训练、绘图或导出脚本。
