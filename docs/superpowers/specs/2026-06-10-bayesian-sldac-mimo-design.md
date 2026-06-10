# Bayesian Critic SLDAC for MIMO Design

日期：2026-06-10

## 背景

本设计用于在仓库根目录新增一个独立子项目，实现面向 MIMO 环境的
Bayesian critic SLDAC。新实现用于和现有 SLDAC 做仿真比较，不直接修改
`SLDAC_code/`、`MIMO/`、`CLQR/` 或 `MultiCell_MIMO/` 的主线代码。

项目口径仍然是 infinite-horizon average-cost CMDP。代码中的
`gamma_pow_reward`、`gamma_pow_cost` 只表示 SLDAC 平滑步长的幂次，不解释
为 discounted return 的折扣因子。环境返回的 `reward` 在算法内部继续作为第
0 维 objective cost，含义是待最小化，越低越好。

## 目标

1. 在根目录新增独立子文件夹 `Bayesian_SLDAC_MIMO/`。
2. 环境固定为 MIMO，算法参数、样本量、episode/run 配置默认和现有 MIMO
   SLDAC 保持一致，方便公平比较。
3. 保留 SLDAC 的 average-cost critic TD、`func_value` 平滑估计、
   score-function actor gradient 和 CSSCA surrogate 更新结构。
4. 将普通 critic 替换为 Bayesian critic ensemble，提供 `Q_mean` 与
   `Q_std`，并按风险修正生成 actor 更新使用的 `Q_hat`。
5. CSSCA 子问题默认使用拉格朗日 primal-dual / dual 方法求解，不依赖 MOSEK
   作为主路径。
6. 提供统一入口 `run_bayesian_sldac_mimo.py`，支持多 seed、多 run tag、统一
   输出目录与 checkpoint，方便和现有 SLDAC 曲线比较。
7. 所有输出、checkpoint、日志和临时 smoke 产物隔离在
   `Bayesian_SLDAC_MIMO/` 下，不写入现有正式实验目录。

## 非目标

1. 第一版不实现完整 Bayesian neural network、MCMC 或 variational posterior。
   先使用 bootstrap ensemble 作为可落地的不确定性估计。
2. 不改写 MIMO 环境定义，不重新定义 objective/cost 语义。
3. 不实现 model-based Bayesian RL，不学习 transition posterior。
4. 不修改 Fused-CPRO、PRCRL、Pathwise/Q-Prop 实现。
5. 不运行会覆盖正式实验产物的训练、绘图或导出脚本。

## 推荐目录

```text
Bayesian_SLDAC_MIMO/
├── __init__.py
├── artifact_paths.py
├── bayesian_critic.py
├── buffer.py
├── checkpoint.py
├── config.py
├── environment.py
├── lagrangian_cssca.py
├── model.py
├── run_bayesian_sldac_mimo.py
├── sldac.py
├── outputs/
├── checkpoints/
├── logs/
├── Trash/
└── tests/
    ├── test_bayesian_critic.py
    ├── test_config.py
    ├── test_lagrangian_cssca.py
    └── test_sldac_smoke.py
```

目录中的运行时代码应自洽。允许只读参考 `SLDAC_code/MIMO1/` 与现有
`MIMO/` 的实现，但不得从旧参考目录反向覆盖主线代码。

## 算法结构

### 主循环

`sldac.py` 保持 SLDAC 的时间尺度：

1. actor 根据当前 MIMO state 采样 action。
2. 环境推进一步，返回 observation、objective cost 和约束 cost。
3. buffer 存储 `(state, action, costs, next_state, aver_reward, aver_cost)`。
4. 当 `t > 2 * T` 且达到 update cadence 时，取最近窗口数据更新 critic。
5. 更新 `func_value`：

```text
func_value = (1 - alpha_t) * func_value + alpha_t * mean(costs_buffer)
```

6. 当 `Q_update_index == Q_update_time` 时，从 Bayesian critic 得到风险修正
   后的 `Q_hat`，计算 score-function actor gradient。
7. 使用拉格朗日 CSSCA solver 求得 `theta_bar`。
8. 平滑更新 actor 参数：

```text
theta = (1 - beta_t) * theta + beta_t * theta_bar
```

### Cost 向量

MIMO 的 cost 向量保持现有 SLDAC 口径：

```text
costs[0] = objective_cost
costs[k] = info["cost_k"] - constr_lim[k - 1],  k >= 1
```

第 0 维是 objective cost，越低越好。约束维满足 `J_k(theta) <= 0` 时可行。

## Bayesian Critic

第一版采用 bootstrap ensemble critic。每个 objective/constraint head 维护
`ensemble_size` 个 critic member：

```text
Q_h^1(s, a), Q_h^2(s, a), ..., Q_h^K(s, a)
```

每个 member 结构与普通 SLDAC critic 对齐，使用独立初始化和独立 optimizer。
训练时对每个 member 使用 bootstrap mask 或 bootstrap weight，使 ensemble
产生可用分歧。TD target 仍是 average-cost 形式：

```text
y_h = cost_h - func_value_h + stop_gradient(Qbar_h(next_state, next_action))
```

其中 `Qbar_h` 是对应 member 的 target / smoothed critic。该 target 不包含
discount factor。

推理时对每个 head 汇总：

```text
Q_mean_h = mean_j Q_h^j(s, a)
Q_std_h  = std_j  Q_h^j(s, a)
```

## 风险修正

actor 更新不直接使用 `Q_mean`，而使用风险修正后的 `Q_used`：

```text
objective head:   Q_used_0 = Q_mean_0 - beta_uncertainty * Q_std_0
constraint heads: Q_used_h = Q_mean_h + beta_uncertainty * Q_std_h, h >= 1
```

含义：

- objective cost 要最小化。减去不确定性表示对高不确定区域更乐观，鼓励探索。
- constraint head 越高越危险。加上不确定性表示约束更保守，避免不确定时误判
  为安全。
- `beta_uncertainty = 0` 时退化为普通 ensemble mean SLDAC。

风险修正只作用于 actor gradient 所用的 `Q_hat`，不修改环境 reward/cost，不改变
average-cost CMDP 口径。

## Actor 与参数向量

第一版 actor 沿用 MIMO SLDAC 的 Gaussian actor 结构。为了和现有结果可比，默认
保留相同 hidden dims、`log_std` 参数、参数 flatten/writeback 方式与采样口径。

实现上应封装：

```text
actor_to_vector(actor) -> theta
vector_to_actor(actor, theta)
flatten_actor_grad(actor, log_std_grad) -> grad_vector
```

这些接口避免在主循环中散落手工 flatten 逻辑，也方便后续替换 actor 分布。

## 拉格朗日 CSSCA Solver

`lagrangian_cssca.py` 负责求解 surrogate constrained update。问题形式保持
SLDAC/CSSCA 结构：

```text
minimize    f_0(theta_t) + g_0^T d + tau_0 ||d||^2
subject to  f_h(theta_t) + g_h^T d + tau_h ||d||^2 <= 0, h >= 1
where       d = theta - theta_t
```

第一版使用 dual / primal-dual 拉格朗日求解。推荐先实现 dual SLSQP 路径：

1. 先解 feasible surrogate，判断约束是否可行。
2. 若可行，解 objective surrogate。
3. 若 solver 失败或返回非有限参数，保守回退为 `theta_t`，并记录诊断状态。

该 solver 不需要 MOSEK。后续若要对照 CVX/MOSEK，可作为可选 debug 路径，而不作
默认依赖。

## 统一入口

`run_bayesian_sldac_mimo.py` 提供唯一推荐入口。顶部 Python 配置优先，CLI 只作
可选覆盖。默认配置与现有 MIMO SLDAC 对齐：

```text
EXAMPLE_NAME = "MIMO"
DEFAULT_SEEDS = (6, 7, 8, 9, 10)
DEFAULT_EPISODE = 100
DEFAULT_UPDATE_TIME_PER_EPISODE = 10
SLDAC_RUNS = [("b100_q1", "Bayesian SLDAC, batchsize=100, q=1", 500, 500, 100, 1)]
DEFAULT_ENSEMBLE_SIZE = 5
DEFAULT_BETA_UNCERTAINTY = 0.5
DEFAULT_CSSCA_SOLVER = "lagrangian"
DEFAULT_ACTOR_DISTRIBUTION = "squashed"
DEFAULT_CHECKPOINT_INTERVAL_EPISODES = 10
```

入口负责：

1. 构造 run args。
2. 调用 `sldac.SLDAC_main(args, "MIMO")`。
3. 保存 reward/objective cost 曲线、constraint cost 曲线、Bayesian diagnostics。
4. 保存 checkpoint 和日志。
5. 所有路径落在 `Bayesian_SLDAC_MIMO/outputs/`、`checkpoints/`、`logs/`。

## Diagnostics

每次运行至少记录：

- `q_mean_objective`
- `q_std_objective`
- `q_mean_constraints`
- `q_std_constraints`
- `beta_uncertainty`
- `cssca_status`
- `constraint_violation`
- `objective_avg`
- `ensemble_size`

这些诊断用于区分“Bayesian uncertainty 有效”和“只是随机种子波动”。

## 测试与验证

不直接运行正式训练作为第一验收。默认验证顺序：

1. `py_compile` 覆盖 `Bayesian_SLDAC_MIMO/`。
2. 单元测试：
   - Bayesian critic 输出 shape、mean/std、finite 检查。
   - `beta_uncertainty = 0` 时风险修正等价于 ensemble mean。
   - constraint risk correction 方向为加 `std`。
   - objective risk correction 方向为减 `std`。
   - 拉格朗日 CSSCA solver 在小规模解析问题中返回有限参数。
3. 最小 smoke test 使用很小 `MAX_STEPS` 和临时隔离输出，产物完成后移动到
   `Bayesian_SLDAC_MIMO/Trash/`。

正式仿真前必须明确输出路径，不覆盖现有正式实验产物。

## 风险与边界

1. Ensemble critic 会增加训练时间，`ensemble_size` 默认先取 5，后续可比较
   3/5/10。
2. `beta_uncertainty` 过大可能导致 objective 探索过强或 constraint 过保守，默认
   0.5，实验可比较 0、0.25、0.5、1.0。
3. MIMO 旧 Gaussian actor 依赖环境侧正值投影行为；如果后续要换 bounded actor，
   必须另开设计，不在本实现中混入。
4. CSSCA dual solver 与 CVX/MOSEK 数值路径不完全相同，因此要记录 solver 状态和
   step norm，便于排查。

## 用户确认

用户已确认先采用 Bayesian critic ensemble，并采用：

```text
objective head 乐观修正
constraint heads 保守修正
```

下一阶段进入实现计划，再按计划落地代码。
