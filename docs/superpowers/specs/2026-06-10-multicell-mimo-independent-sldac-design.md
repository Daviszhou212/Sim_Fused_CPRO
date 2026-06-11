# Multi-Cell MIMO Independent SLDAC Design

日期：2026-06-10

## 背景

用户希望阅读 `CMARL_revised_CN.tex` 后，结合 SLDAC 论文源码，构建一个新的多小区场景。新场景不应继续嵌入现有 `Squash/MIMO/` 主线，而应放在新的子文件夹中，形成完整独立实现。

当前仓库已有两类相关材料：

- `Squash/MIMO/SLDAC.py`、`Squash/MIMO/critic_opt.py`、`Squash/MIMO/utils.py` 是 SLDAC 论文源码风格实现的当前主线版本。
- 旧 `MIMO` 内嵌多小区 prototype 已移除；多小区 CTDE / tree critic 场景统一落在 `MultiCell_MIMO/`。
- `CMARL_revised_CN.tex` 将问题修正为 multi-agent infinite-horizon average-cost CMDP，并要求 critic 使用 SLDAC 的 average-cost differential Q 半梯度 TD 更新，actor 使用 CSSCA surrogate。

本设计文档只定义新场景的工程边界与算法映射，不修改源码、不运行训练、不生成实验产物。

## 核心口径

1. 新场景仍是 `infinite-horizon average-cost CMDP`，不是 discounted MDP/CMDP。
2. 代码中的 `gamma_pow_reward`、`gamma_pow_cost` 只表示 SLDAC 中 critic 平滑步长的幂次，不解释为 discount factor。
3. 环境接口可以沿用历史 `reward` 字段名，但算法内部第 0 维必须解释为待最小化的 `objective_cost`，即越低越好。
4. 约束统一使用 shifted cost：

```text
C_0' = C_0
C_k' = C_k - c_k,  k >= 1
```

因此 SLDAC/CSSCA 中的约束函数值满足 `J_k(theta) <= 0`。

## 目标

1. 新建独立目录 `MultiCell_MIMO/`，包含环境、actor、critic、buffer、CSSCA、配置、seed/device、checkpoint、SLDAC 主循环、运行入口与单元测试。
2. v1 先实现一个可验证的多小区 SLDAC baseline：多小区无线环境、分散执行 actor、集中训练 critic、SLDAC average-cost TD 与 CSSCA actor update。
3. v2 在同一独立目录内加入 `CMARL_revised_CN.tex` 的树状可微消息传递 critic，逐步替换 v1 的集中式 critic。
4. 所有输出、checkpoint、临时 smoke 产物必须隔离在 `MultiCell_MIMO/outputs/`、`MultiCell_MIMO/checkpoints/` 或临时目录中，不写入现有 `Squash/MIMO/outputs`、`result/` 或历史实验目录。
5. 顶部配置优先：运行入口的可调参数集中放在 `.py` 文件顶部，CLI 只作为可选覆盖，不作为主要配置方式。

## 非目标

1. 不在第一阶段实现 Fused-CPRO、PRCRL、HRL、DK policy reuse。
2. 不直接把现有 `Squash/MIMO/SLDAC.py` 拷贝后机械改名作为长期结构；可以参考其算法逻辑，但新目录应有清晰模块边界。
3. 不修改现有 `Squash/MIMO/`、`Squash/CLQR/`、`result/` 的训练入口和实验产物。
4. 不运行 `run_mimo_*`、`run_multicell_*` 这类可能落盘训练产物的脚本，除非输出已明确隔离并得到用户确认。
5. 不在 v1 承诺完全分布式训练通信；v1 是工程 baseline，v2 才加入树状消息 critic。

## 推荐目录

```text
MultiCell_MIMO/
├── __init__.py
├── artifact_paths.py
├── buffer.py
├── checkpoint.py
├── config.py
├── critic.py
├── cssca.py
├── environment.py
├── model.py
├── run_multicell_sldac.py
├── seed_utils.py
├── sldac.py
├── tree_critic.py
├── outputs/
├── checkpoints/
├── Trash/
└── tests/
    ├── test_checkpoint.py
    ├── test_config.py
    ├── test_environment.py
    ├── test_model.py
    ├── test_critic.py
    └── test_sldac_smoke.py
```

`tree_critic.py` 在 v1 可以只保留设计占位或最小接口，不在主流程启用。实现计划阶段再决定是否立即创建空文件；若没有可执行代码需求，可以先不创建，避免无价值文件。

`config.py` 负责顶部默认值、CLI 可选覆盖与受保护字段；`seed_utils.py`
负责 NumPy/Torch seed、device 解析和可复现实验设置；`checkpoint.py`
负责 schema、CPU `state_dict`、临时 checkpoint 路径和禁用保存开关。新目录不得
从 `Squash/MIMO/` 隐式导入这些运行时工具。

## 场景定义

### 智能体

每个小区是一个智能体。设小区数为 `N_cell`，每小区用户数为 `K_user`，基站天线数为 `N_t`。

```text
agent i = cell i
local action a_i = [power_i,1, ..., power_i,K, reg_i]
joint action a = concat(a_1, ..., a_N)
```

### 状态

v1 使用集中训练、分散执行的状态拆分：

- 全局状态：所有直达链路、跨小区干扰链路、所有用户队列。
- 本地状态：本小区服务用户的直达 CSI、本小区用户队列；可选加入邻区摘要，但 v1 默认不加入，保持执行阶段完全本地。

全局状态用于 critic 和环境动力学，本地状态用于每个 cell actor。

v2 tree critic 要求全局状态可写成局部状态块的组合 `s=(s_1,...,s_N)`。
因此跨小区干扰链路必须有明确归属：默认放入接收小区的 `s_i`，即 cell
`i` 的本地 critic 输入包含“所有发射小区到本小区用户”的信道；分散执行
actor 仍只读取直达 CSI 和本小区队列。若后续改为发送小区归属或邻区摘要，
必须同步修改 `tree_critic.py` 的本地消息构造，保证 `Phi_phi` 只由各节点
`(s_i,a_i)` 消息构造。

### 动作空间

每个小区动作 block：

```text
[power_1, ..., power_K, reg_factor]
```

建议沿用当前 squashed Gaussian 约束口径：

- `power_j in (ACTION_EPS, POWER_MAX)`
- `reg_factor in (ACTION_EPS, +inf)`

这样 `sample_action()` 与 `evaluate_action()` 的 support 一致，避免环境侧再用裁剪弥补无界 Gaussian。

### 目标与约束

环境每步返回：

- `objective_cost = sum_{cell,user} power[cell,user]`
- `constraint_cost_user = queue_delay[cell,user]`
- `shifted_cost_user = queue_delay[cell,user] - constraint_limit`

SLDAC buffer 中保存：

```text
costs[0] = objective_cost
costs[1:] = per_user_delay - constraint_limit
```

v1 默认把每个 `(cell,user)` 的 delay constraint 视为一个独立约束 head：
`k=(cell,user)`，`C_k` 为该用户 delay，`c_k=constraint_limit`，所以
`C_k'=delay_k-c_k`。若后续改为小区级或全局聚合约束，则必须先汇聚对应
`C_k^{tot}`，再只减一次全局阈值 `c_k`，避免重复减阈值。

展示曲线中历史字段可仍叫 `reward_average`，但图注和文档必须解释为 objective cost。

## v1 算法设计：独立 SLDAC Baseline

### 主循环

`sldac.py` 参考现有 SLDAC 源码的时间尺度结构：

1. actor 按本地状态采样每个小区动作，拼接为 joint action。
2. 环境推进一步，得到全局 next state、objective cost、每用户约束代价。
3. buffer 保存 `(state, action, costs, next_state)`。
4. 每轮 critic update 使用 average-cost Bellman target。论文中把 bootstrap 写成
   online critic 的位置按笔误处理；实现统一使用 SLDAC 源码兼容口径：

```text
source_compatible:
    y_k = C_k' - hat_J_k + stop_gradient(Qbar_k(next_state, next_action))

e_k = Q_k(state, action) - stop_gradient(y_k)
```

`source_compatible` 是唯一支持的 critic target mode，用来贴近现有 SLDAC 源码中
`target_net.forward(...).detach()` 与 `soft_update(target_net, net, gamma)`
的行为。该 target 不含 discount factor；任何实验报告只需记录该模式以便追溯。

5. `hat_J` 使用 SLDAC 的递推平均：

```text
hat_J = (1 - alpha_t) * hat_J + alpha_t * mean(costs)
```

6. actor 梯度使用平滑 critic：

```text
g_{k,i} = mean[ Qbar_k(s, a) * grad log pi_i(a_i | s_i) ]
```

7. CSSCA 根据 `hat_J` 与所有 actor 参数分块梯度求 objective update；若不可行，则求 feasible update。
8. actor 参数按 SLDAC 形式平滑更新：

```text
theta = (1 - beta_t) * theta + beta_t * theta_bar
```

### Critic

v1 critic 是集中式多头 DNN：

```text
Q_k([global_state, joint_action])
```

每个 cost head 独立维护在线网络与平滑网络。在线网络用于半梯度 TD
学习；平滑网络按 SLDAC 的递推平均维护，并固定参与 TD target。它不是
discounted DQN target network，也不引入折扣因子。

### Actor

v1 actor 使用“共享本地 actor 网络”作为默认实现：

```text
pi_theta(a_i | s_i),  all cells share theta
```

共享 actor 是 v1 的参数绑定工程 baseline：动作采样仍满足
`pi(a|s)=prod_i pi_theta(a_i|s_i)` 的条件独立分解，但参数不再是 `.tex`
中完全独立的 `theta_i` 分块。单步 joint log-prob 定义为：

```text
log pi(a|s) = sum_i log pi_theta(a_i | s_i)
```

共享参数可以降低首版调试成本，并适配同构小区。共享 actor 下，梯度对共享参数
按 cell 维度求平均后进入 CSSCA；若后续需要严格复现 `.tex` 的分块 CSSCA，
则应扩展为 per-cell actor 参数列表：

```text
theta = [theta_1, ..., theta_N]
```

`theta` flatten 顺序固定为 `actor.net.parameters()` 后接 `log_std`。
v1 默认 `log_std` 为每个 cell action block 共享的长度 `K_user + 1` 向量；
若实现为 joint action 全维 `N_cell * (K_user + 1)`，必须在 `config.py` 与
`checkpoint.py` schema 中记录。设计上 `model.py` 应保留清晰接口，使共享
actor 与 per-cell actor 可以替换。

### CSSCA

`cssca.py` 独立实现 objective / feasible update。v1 输入为 actor 参数向量与各 head 梯度：

```text
func_value: shape (1 + K,)
grad:       shape (1 + K, theta_dim)
theta:      shape (theta_dim,)
```

默认 solver 固定为 `lagrangian_dual`，即沿用当前代码中通过拉格朗日对偶
问题求解 CSSCA objective / feasible update 的路线，不强制依赖 CVXPY 作为
主求解器。CVXPY 只作为 fallback 或数值对照。fallback 顺序为：

```text
lagrangian_dual -> cvxpy solver candidates -> fail-fast
```

拉格朗日法需要显式返回分支信息，例如 `objective` / `feasible`、对偶变量、
surrogate 值和 solver status，便于后续诊断 CSSCA 是否因为约束不可行而进入
feasible update。任何 solver 返回 `None`、非 finite 参数、shape 不匹配时立即抛错，不进入
actor restore。单测必须覆盖 objective 分支、feasible 分支、fallback 分支和
solver 失败分支。Windows/MOSEK/CVXPY 环境差异不能只靠打印 warning 吞掉。

v2 再将 CSSCA 子问题改写为 `.tex` 中的树状残差汇聚版本。

## v2 算法设计：树状可微 Critic

v2 引入 `tree_critic.py`，使 critic 结构对应 `.tex`：

```text
Q_k(s, a; omega_k, phi) = head_k(Phi_phi(s, a))
```

每个 cell 构造本地消息：

```text
m_i = psi_phi_i(s_i, a_i)
```

树上行聚合得到全局 bottleneck 表示，下行广播可用于本地 actor 梯度或分布式 CSSCA 内循环。

v2 必须维护完整平滑 critic：

```text
bar_omega_k = (1 - gamma_t) * bar_omega_k + gamma_t * omega_k
bar_phi     = (1 - gamma_t) * bar_phi     + gamma_t * phi
```

如果工程上只平均 head 而不平均 encoder，必须在文档和代码注释中标为近似实现，不当作理论等价版本。

## 测试策略

### v1 必须有的测试

1. `test_config.py`
   - 顶部默认配置能构建完整 runtime config。
   - CLI 覆盖只影响允许覆盖的字段，受保护字段保持 Python 顶部配置优先。
   - `critic_target_mode`、`actor_parameterization`、`log_std_mode` 只接受显式枚举值。

2. `test_checkpoint.py`
   - checkpoint schema 包含 seed、device、actor 参数化、`log_std_mode`、`critic_target_mode`、state/action/cost 维度。
   - CPU `state_dict` 保存不依赖当前 device。
   - `save_final_checkpoint=0` 时不写 `.pt`。

3. `test_environment.py`
   - reset/step shape 正确。
   - action 维度错误时报错。
   - `info["cost_i"]` 覆盖所有用户。
   - 队列、速率、objective cost 均为 finite。

4. `test_model.py`
   - actor global sampling 和 per-cell local sampling shape 正确。
   - power/reg action 落在 support 内。
   - `evaluate_action()` 对 transformed action 返回 finite log-prob。
   - shared actor 下 joint log-prob 等于各 cell log-prob 之和。

5. `test_critic.py`
   - critic 支持 `1 + N_cell * K_user` 个 head。
   - average-cost TD target 不含 discount factor。
   - 仅 `source_compatible` target 模式合法；`tex_strict` 必须报错。
   - 平滑 critic 更新只按 `gamma_t` 做参数递推平均。

6. `test_cssca.py`
   - objective update、feasible update、`lagrangian_dual` fallback、cvxpy fallback、solver 失败分支都有覆盖。
   - solver 返回 `None`、非 finite 或 shape mismatch 时 fail-fast。

7. `test_sldac_smoke.py`
   - 使用极小步数、禁用正式 checkpoint 保存。
   - 不调用正式入口脚本，只直接调用内存内主循环或测试专用函数。
   - 必须显式设置 `output_root=tempfile.TemporaryDirectory()`、`checkpoint_root=tempfile.TemporaryDirectory()`、`save_final_checkpoint=0`。
   - 只验证主循环返回曲线 shape 和 finite。
   - 若产生 `.mat`、`.png`、`.pdf`、`.pt`，测试结束前统一移动到 `MultiCell_MIMO/Trash` 或删除临时目录。

### 禁止的验证方式

不直接运行正式训练配置；不在测试中调用 `run_multicell_sldac.py`；不生成覆盖式
`.mat`、`.png`、`.pdf`、checkpoint 到 `Squash/MIMO/outputs`、`result/`、仓库根
`checkpoints/` 或任何历史路径。

## 迁移与复用边界

可以参考现有 SLDAC 源码的以下逻辑：

- `func_value` / `hat_J` 递推平均。
- differential Q TD target：`cost - func_value + next_Qbar`，固定使用平滑
  target critic；论文中 online bootstrap 写法按笔误处理。
- `critic_value()` 使用平滑 critic 估计 actor 梯度。
- CSSCA objective / feasible update。
- actor 参数 flatten / restore。
- seed/device/config/checkpoint 的边界设计。

不直接复用现有模块作为 runtime dependency。新目录应能独立导入和测试，避免后续修改 `Squash/MIMO/` 时影响新场景。

## 风险与开放问题

1. `critic_target_mode` 只保留 `source_compatible`。论文中 online bootstrap 写法按笔误处理，不再作为实验分支。
2. 共享 actor 与 per-cell actor 的理论口径不同。v1 默认共享参数用于同构场景；若论文表述需要完全分块的 `theta_i`，实现计划中应改为 per-cell actor。
3. 跨小区干扰链路的局部归属会影响 tree critic 是否严格对应 `s=(s_1,...,s_N)`。v2 实现前必须锁定接收小区归属、发送小区归属或邻区摘要中的一种。
4. 多小区干扰模型可能导致队列快速饱和。v1 需要保守设置 `arrival_upper`、`POWER_MAX`、`constraint_limit` 和跨小区 path gain。
5. v1 centralized critic 与 `.tex` 的树状 critic 不完全一致。因此实验结论只能说是 independent SLDAC baseline，不能声称已经实现树状分布式 critic。
6. CSSCA 在高维 actor 参数下可能求解慢。v1 测试使用极小网络；正式实验前再决定是否需要调优 `lagrangian_dual` 或启用 CVXPY fallback。

## 验收标准

1. 新增 `MultiCell_MIMO/` 独立目录，不修改现有 `Squash/MIMO/` 算法主线。
2. 单元测试覆盖配置、checkpoint、环境、actor、critic、CSSCA、SLDAC smoke。
3. smoke test 不写正式实验产物。
4. 文档和代码注释都保持 average-cost / objective-cost 口径。
5. 后续实现计划明确区分 v1 baseline 与 v2 tree critic。
6. 任何实验输出都记录 `critic_target_mode`、`actor_parameterization`、`log_std_mode` 与 seed。
