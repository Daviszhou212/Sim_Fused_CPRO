# Bayesian Critic SLDAC for MIMO Design

日期：2026-06-10

## 背景

本设计用于在仓库根目录新增一个独立子项目，实现面向 MIMO 环境的
Bayesian critic SLDAC。新实现用于和 `SLDAC_code/MIMO1` 中的原始 SLDAC
做仿真比较，不直接修改 `SLDAC_code/`、`Squash/MIMO/`、`Squash/CLQR/` 或
`MultiCell_MIMO/` 的主线代码。

本项目口径仍然是 infinite-horizon average-cost CMDP。代码中的
`gamma_pow_reward`、`gamma_pow_cost` 只表示 SLDAC target critic 平滑步长的
幂次，不解释为 discounted return 的折扣因子。`SLDAC_code/MIMO1` 的环境字段
名仍叫 `reward`，但在 MIMO 中其值是 `np.sum(power)`，算法内部第 0 维应解释
为 objective cost，含义是待最小化，越低越好。

用户已明确要求：Bayesian 版的 SLDAC 算法与配置应对齐 `SLDAC_code/MIMO1`
的 SLDAC，而不是当前主线 `Squash/MIMO/` 目录中的后续实现。随机流、动作分布、有界动作
处理、默认 seed、CPU 设备、输出命名和 SLDAC 时间尺度都应按
`SLDAC_code/MIMO1` 复刻；Bayesian critic 和拉格朗日 CSSCA solver 是第一版的
主要差异。

## 目标

1. 在根目录新增独立子文件夹 `Bayesian_SLDAC_MIMO/`。
2. 环境固定为 MIMO，并复刻 `SLDAC_code/MIMO1/environment.py` 的随机流、
   channel 生成、queue 更新、arrival 分布和动作投影行为。
3. actor 默认复刻 `SLDAC_code/MIMO1/model.py` 的 Gaussian actor：均值经
   `2.5 * sigmoid` 限制，但采样仍是 plain `Normal(mu, std)`，不使用当前主线
   `Squash/MIMO/` 的 squashed transformed Gaussian。
4. 算法参数、样本量、episode/run 配置默认对齐 `SLDAC_code/MIMO1/MIMO_main.py`
   和 `SLDAC_code/MIMO1/SLDAC.py`。
5. 保留 SLDAC 的 average-cost critic TD、`func_value` 平滑估计、
   score-function actor gradient 和 CSSCA surrogate 更新结构。
6. 将普通 critic 替换为 Bayesian critic ensemble，提供 `Q_mean` 与
   `Q_std`，并按风险修正生成 actor 更新使用的 `Q_hat`。
7. CSSCA 子问题默认使用拉格朗日 primal-dual / dual 方法求解，不依赖 MOSEK
   作为主路径；如果后续需要与原始 MOSEK 结果做 solver A/B，可单独配置。
8. 提供统一入口 `run_bayesian_sldac_mimo.py`，支持按旧版 run tag 运行并统一
   隔离输出，方便和 `SLDAC_code/MIMO1` 产物比较。

## 非目标

1. 第一版不实现完整 Bayesian neural network、MCMC 或 variational posterior。
   先使用 bootstrap ensemble 作为可落地的不确定性估计。
2. 不改写 MIMO 环境定义，不重新定义 objective/cost 语义。
3. 不引入当前主线 `Squash/MIMO/` 的 squashed actor、checkpoint schema、multi-seed runner
   或 artifact migration 逻辑。
4. 不实现 model-based Bayesian RL，不学习 transition posterior。
5. 不修改 Fused-CPRO、PRCRL、Pathwise/Q-Prop 实现。
6. 不运行会覆盖正式实验产物的训练、绘图或导出脚本。

## 推荐目录

```text
Bayesian_SLDAC_MIMO/
├── __init__.py
├── artifact_paths.py
├── bayesian_critic.py
├── buffer.py
├── config.py
├── environment.py
├── lagrangian_cssca.py
├── model.py
├── run_bayesian_sldac_mimo.py
├── sldac.py
├── outputs/
├── logs/
├── Trash/
└── tests/
    ├── test_bayesian_critic.py
    ├── test_config.py
    ├── test_environment_random_stream.py
    ├── test_lagrangian_cssca.py
    ├── test_model_legacy_action.py
    └── test_sldac_smoke.py
```

第一版不需要 checkpoint 目录作为核心目标，因为 `SLDAC_code/MIMO1` 原始入口只
保存 `.mat` 曲线，不保存 actor/critic checkpoint。若实现中为了调试新增 checkpoint，
必须默认关闭，并且只能写入 `Bayesian_SLDAC_MIMO/Trash/` 或显式隔离目录。

## SLDAC_code/MIMO1 兼容基准

### 随机流

兼容目标：

```text
SLDAC_main:
  seed = 0
  np.random.seed(seed)
  torch.manual_seed(seed)
  device = "cpu"

Environment_MIMO.__init__:
  self.seed = seed
  self.seed_step = seed
  np.random.seed(seed)
  PathGain_dB ~ Uniform(-10, 10)
  alpha_power_group ~ Exponential(1)
  AoD ~ Laplacian(mu=0, angular_spread=5)

Environment_MIMO.reset:
  np.random.seed(self.seed)

Environment_MIMO.step:
  np.random.seed(self.seed_step)
  self.seed_step += 1
  arrival A_d ~ Uniform(0, 2)
```

Bayesian 版不得默认改成多 seed runner。可以在入口提供可选 seed 覆盖用于实验扩展，
但默认必须是 `seed=0`，并且 smoke test 不能依赖多 seed 行为。

### 动作空间与有界动作处理

`SLDAC_code/MIMO1` 的 MIMO actor 不是 squashed distribution：

```text
mu = 2.5 * sigmoid(net(state))
action ~ Normal(mu, exp(log_std))
log_prob = Normal(mu, exp(log_std)).log_prob(action).sum(dim=1)
```

这意味着均值在 `(0, 2.5)`，但采样 action 本身仍是无界 Gaussian。环境只做下界投影：

```text
action[action <= 0] = 1e-6
power = action[0:UE_num]
reg_factor = action[UE_num]
```

环境没有对 power 或 reg_factor 做上界截断。因此 Bayesian 版第一版必须保留这个
“bounded mean + unbounded Gaussian sample + environment lower projection”的旧语义，
不能使用当前主线 `Squash/MIMO/` 的 `squashed_gaussian_v1`、tanh/sigmoid Jacobian 修正、
transform metadata 或 bounded log-prob。

### MIMO 环境参数

默认环境：

```text
Nt = 8
UE_num = 4
user_per_group = 2
Np = 4
noise_power = 1e-6
Dmax = 5
constr_lim = [1.2, 1.2, 1.2, 1.2]
state_dim = 2 * UE_num * Nt + UE_num
action_dim = UE_num + 1
constraint_dim = UE_num
```

Cost 向量：

```text
costs[0] = reward                 # MIMO 中实际是 sum(power)
costs[k] = info["cost_k"] - constr_lim[k - 1],  k >= 1
aver_reward = reward
aver_cost = info["cost"] / constraint_dim
```

## 默认配置

Bayesian 版默认配置对齐 `SLDAC_code/MIMO1/MIMO_main.py`：

```text
example_name = "MIMO"
seed = 0
device = "cpu"
alpha_pow = 0.6
beta_pow = 0.7
eta_pow = 0.01
gamma_pow_reward = 0.3
gamma_pow_cost = 0.3
tau_reward = 1
tau_cost = 1
T = 500
grad_T = 500
num_new_data = 100
window = 10000
episode = 60
update_time_per_episode = 10
num_update_time = episode * update_time_per_episode
Q_update_time = 1
MAX_STEPS = 2 * T + num_update_time * num_new_data
```

统一入口默认启用旧版 SLDAC 的四个 run 配置，输出文件名沿用旧版命名，但落在新目录：

```text
SLDAC_RUNS = [
  ("b100_q10", "Bayesian SLDAC, batchsize=100, q=10", 500, 500, 100, 10),
  ("b100_q1",  "Bayesian SLDAC, batchsize=100, q=1",  500, 500, 100, 1),
  ("b100_q5",  "Bayesian SLDAC, batchsize=100, q=5",  500, 500, 100, 5),
  ("b500_q10", "Bayesian SLDAC, source b500_q10 setting", 50, 100, 100, 10),
]
```

注意：`SLDAC_code/MIMO1/MIMO_main.py` 中 `b500_q10` 的打印标签与实际参数不完全
一致，代码实际使用 `T=50`、`grad_T=2*T=100`、`num_new_data=100`、
`Q_update_time=10`。Bayesian 版以代码实际参数为准，同时在日志中记录这一点。

Bayesian 专属默认参数：

```text
ensemble_size = 5
beta_uncertainty = 0.5
bootstrap_mask_prob = 0.8
cssca_solver = "lagrangian"
```

## 算法结构

### 主循环

`sldac.py` 保持 `SLDAC_code/MIMO1/SLDAC.py` 的时间尺度：

1. actor 根据当前 MIMO state 采样 action。
2. 环境推进一步，返回 observation、objective cost 和约束 cost。
3. buffer 存储 `(state, action, costs, next_state, aver_reward, aver_cost)`。
4. 当 `t > 2 * T` 且达到 update cadence 时，取最近窗口数据更新 critic。
5. 更新 `func_value`：

```text
func_value = (1 - alpha_t) * func_value + alpha_t * mean(costs_buffer)
```

6. 当 `Q_update_index == Q_update_time` 时，从 Bayesian critic 得到风险修正后的
   `Q_used`。
7. 对 `Q_used` 按旧版 SLDAC 口径逐 head 标准化后进入 actor gradient：

```text
Q_hat[:, 0] = (Q_used[:, 0] - mean(Q_used[:, 0])) / (std(Q_used[:, 0]) + 1e-6)
Q_hat[:, h] = (Q_used[:, h] - mean(Q_used[:, h])) / (std(Q_used[:, 0]) + 1e-6), h >= 1
```

这里保留旧版代码对约束 head 也使用 objective head 标准差归一化的行为。

8. 计算 score-function actor gradient：

```text
actor_loss_h = mean(Q_hat[:, h] * log_prob(action | state))
```

9. 使用拉格朗日 CSSCA solver 求得 `theta_bar`。
10. 平滑更新 actor 参数：

```text
theta = (1 - beta_t) * theta + beta_t * theta_bar
```

### 参数向量

参数 flatten/writeback 复刻旧版：

```text
theta = flatten(actor.net.parameters()) + actor.log_std
real_theta_dim = theta_dim + action_dim
```

`actor.log_std` 作为 `torch.Tensor` 直接参与参数向量。第一版为了对齐旧实现，不默认加入
当前主线的 log_std clamp；若后续发现数值失稳，应作为显式实验变体记录。

## Bayesian Critic

第一版采用 bootstrap ensemble critic。每个 objective/constraint head 维护
`ensemble_size` 个 critic member：

```text
Q_h^1(s, a), Q_h^2(s, a), ..., Q_h^K(s, a)
```

每个 member 的网络结构对齐 `SLDAC_code/MIMO1/model.py` 中的 `Critic_net_MIMO`：

```text
state branch:  state_dim -> 256 -> 128
action branch: action_dim -> 128
joint branch:  concat -> 128 -> 1
output:        10 * tanh(0.001 * fc3(x))
```

训练时 TD target 仍是 average-cost 形式：

```text
y_h = cost_h - func_value_h + stop_gradient(Qbar_h(next_state, next_action))
```

其中 `Qbar_h` 是对应 member 的 target / smoothed critic。该 target 不包含
discount factor。每个 member 使用独立初始化、独立 optimizer 和 bootstrap mask/weight。

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
- constraint head 越高越危险。加上不确定性表示约束更保守，避免不确定时误判为安全。
- `beta_uncertainty = 0` 时退化为 ordinary ensemble mean SLDAC，是实现中的必要
  A/A 对照。

风险修正只作用于 actor gradient 所用的 `Q_used/Q_hat`，不修改环境 reward/cost，
不改变 average-cost CMDP 口径。它会给普通 SLDAC actor gradient 加入探索/保守偏置，
因此不能把 `beta_uncertainty > 0` 的版本直接解释为原始 SLDAC 收敛理论的等价实现。

## 拉格朗日 CSSCA Solver

`lagrangian_cssca.py` 负责求解 surrogate constrained update。问题形式保持
SLDAC/CSSCA 结构：

```text
minimize    f_0(theta_t) + g_0^T d + tau_0 ||d||^2
subject to  f_h(theta_t) + g_h^T d + tau_h ||d||^2 <= 0, h >= 1
where       d = theta - theta_t
```

`SLDAC_code/MIMO1/utils.py` 使用 CVXPY + MOSEK 求解 `_feasible_update` 和
`_objective_update`。Bayesian 版默认不依赖 MOSEK，而使用 dual / primal-dual
拉格朗日求解：

1. 先解 feasible surrogate，判断约束是否可行。
2. 若可行，解 objective surrogate。
3. 若 solver 失败或返回非有限参数，保守回退为 `theta_t`，并记录诊断状态。

这意味着 solver 是与 `SLDAC_code/MIMO1` 的显式差异。实现与报告必须把
“Bayesian critic 差异”和“CSSCA solver 差异”分开记录；必要时运行
`beta_uncertainty=0`、`ensemble_size=1/mean` 的工程对照。

## 统一入口

`run_bayesian_sldac_mimo.py` 提供唯一推荐入口。默认执行旧版 SLDAC 四个 run 配置，
但不写回 `SLDAC_code/MIMO1` 的固定 `.mat`、`.png`、`.pdf` 文件。

入口语义：

1. 顶部 Python 配置是正式仿真的主配置来源。
2. CLI 可覆盖字段只用于 smoke/debug，必须显式列出。
3. `--smoke` 必须强制使用小 `T/grad_T/num_new_data/Q_update_time/MAX_STEPS`，并强制输出到
   `Bayesian_SLDAC_MIMO/Trash/<unique-run>/`。
4. 默认正式输出写入：

```text
Bayesian_SLDAC_MIMO/outputs/<run_tag>/
Bayesian_SLDAC_MIMO/logs/<run_tag>/
```

入口负责：

1. 构造 run args。
2. 调用 `sldac.SLDAC_main(args, "MIMO")`。
3. 保存 `Bayesian_SLDAC_reward_<tag>.mat` 和 `Bayesian_SLDAC_cost_<tag>.mat`。
4. 保存 diagnostics `.mat` 或 `.json`。
5. 记录 seed、随机流模式、actor distribution、risk correction、solver 状态。

## Diagnostics

每次运行至少记录：

- `seed`
- `random_stream = "sldac_code_mimo1_compatible"`
- `actor_distribution = "legacy_bounded_mean_plain_gaussian"`
- `q_mean_objective`
- `q_std_objective`
- `q_mean_constraints`
- `q_std_constraints`
- `beta_uncertainty`
- `cssca_status`
- `cssca_step_norm`
- `constraint_violation`
- `objective_avg`
- `ensemble_size`

这些诊断用于区分“Bayesian uncertainty 有效”“随机流变化”“动作分布变化”和
“solver 变化”。

## 测试与验证

不直接运行正式训练作为第一验收。默认验证顺序：

1. `py_compile` 覆盖 `Bayesian_SLDAC_MIMO/`。
2. 单元测试：
   - `test_config_matches_sldac_code_mimo1_defaults`：断言 seed、device、T、grad_T、
     num_new_data、window、episode、update cadence、alpha/beta/eta/gamma/tau、run tuple
     与 `SLDAC_code/MIMO1` 对齐。
   - `test_environment_random_stream_matches_sldac_code`：同 seed、同 action 下，
     reset/step 的关键随机流、queue 更新和 shape 与旧环境一致。
   - `test_legacy_action_semantics`：断言 actor 均值在 `(0, 2.5)`，采样仍可能小于 0
     或大于 2.5，log_prob 使用 plain Gaussian，无 transform Jacobian。
   - `test_environment_action_projection`：断言 `action <= 0` 被投影到 `1e-6`，正值
     action 不做上界截断。
   - `test_average_cost_td_and_cost_vector_semantics`：断言 TD target 是
     `cost - func_value + Qbar(next)`，没有 discount factor；断言第 0 维 objective
     cost 和约束残差方向与旧 SLDAC 一致。
   - `test_bayesian_critic`：检查 ensemble 输出 shape、mean/std、finite。
   - `test_risk_correction`：`beta_uncertainty = 0` 等价于 ensemble mean，objective
     方向减 `std`，constraint 方向加 `std`。
   - `test_q_hat_normalization_matches_sldac_code`：风险修正后仍按旧版 SLDAC 的
     `Q_hat` 标准化方式进入 actor gradient。
   - `test_lagrangian_cssca_no_cvx_fallback_on_default_path`：默认 solver 不静默退到
     CVX/MOSEK，小规模可行/不可行问题返回有限 `theta_bar` 或保守回退 `theta_t`。
   - `test_artifact_paths_are_isolated`：所有输出路径都在 `Bayesian_SLDAC_MIMO/` 内，
     不落到 `SLDAC_code/MIMO1`、`Squash/MIMO/outputs` 或仓库根 `checkpoints/SLDAC`。
3. 最小 smoke test 只能使用 `--smoke` 或测试内构造的小配置，输出写入
   `Bayesian_SLDAC_MIMO/Trash/<unique-run>/`，完成后不得长留正式目录。

正式仿真前必须明确输出路径，不覆盖现有正式实验产物。

## 风险与边界

1. Ensemble critic 会增加训练时间，`ensemble_size` 默认先取 5，后续可比较
   3/5/10。
2. `beta_uncertainty` 过大可能导致 objective 探索过强或 constraint 过保守，默认
   0.5，实验可比较 0、0.25、0.5、1.0。
3. 为对齐 `SLDAC_code/MIMO1`，第一版保留旧 Gaussian actor 的无界采样与环境投影；
   如果后续要换 bounded/squashed actor，必须另开设计，不在本实现中混入。
4. CSSCA dual solver 与原始 CVX/MOSEK 数值路径不完全相同，因此要记录 solver 状态、
   step norm 和 fallback 行为。
5. 旧版 `MIMO_main.py` 会读写固定 `.mat`、`.pdf`、`.png` 文件；Bayesian 版不得复刻
   这种写回当前目录的行为，只复刻算法配置和命名语义。

## 用户确认

用户已确认：

```text
1. 先采用 Bayesian critic ensemble。
2. objective head 使用乐观修正。
3. constraint heads 使用保守修正。
4. SLDAC 算法和配置对齐 SLDAC_code/MIMO1，而不是当前 Squash/MIMO/。
5. 随机流、有界动作/动作投影等细节也按 SLDAC_code/MIMO1 对齐。
```

下一阶段进入实现计划，再按计划落地代码。
