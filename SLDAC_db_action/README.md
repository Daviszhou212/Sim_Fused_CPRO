# SLDAC db-action MIMO comparison

本目录是基于 `SLDAC_code/MIMO1` 复制出的独立对比实现。

## 边界

- `SLDAC_code/MIMO1` 不回写、不覆盖。
- `SLDAC_db_action/MIMO1` 只改变 MIMO 环境接口动作域。
- 训练循环、critic、buffer、seed 初始化、环境随机数推进顺序保持原始结构。
- CSSCA policy update 默认使用 Lagrangian dual solver，不依赖 MOSEK。

## 动作口径

原始 MIMO 动作：

```text
action[:UE_num] = power
action[UE_num] = regularization factor
```

本目录 MIMO 动作：

```text
action[:UE_num] = 10 * log10(power / noise_power)
action[UE_num] = regularization factor
```

为避免改变 SLDAC 的 policy/critic 架构，actor、critic、buffer 内部仍使用旧版
power 坐标。进入环境前才转换成 dB action：

```text
snr_db = 10 * log10(max(power, noise_power) / noise_power)
```

环境执行时再转回真实功率：

```text
power = noise_power * 10 ** (snr_db / 10)
```

因此 `reward_average`、`objective_cost_real_power` 等输出仍然是真实功率和
`sum(power)`，不是 dB 值。

## 入口

单跑 dB-action SLDAC：

```powershell
& "D:\anaconda3\envs\torch\python.exe" "D:\Sim_Fused_CPRO\SLDAC_db_action\MIMO1\MIMO_main.py"
```

同配置对比 `legacy_power + dual` 与 `snr_db + dual`：

```powershell
& "D:\anaconda3\envs\torch\python.exe" "D:\Sim_Fused_CPRO\SLDAC_db_action\run_mimo_comparison.py"
```

`run_mimo_comparison.py` 顶部可直接配置：

```python
RUN_SELECTION = "new_only"  # "both" / "new_only" / "old_only"
PRINT_INTERVAL = 1          # 每隔多少个记录 episode 打印一次指标；0 表示关闭
```

`MIMO1/model.py` 顶部保留 legacy actor mean 上界：

```python
LEGACY_POWER_ACTION_MAX = 2.5  # 旧版 MIMO actor 均值上界：2.5 * sigmoid(raw)。
```

dB-action 链路不再对执行功率施加 `2.5` 上界；只把 raw power 中的非正值映射到
`1e-6`，这与 legacy SLDAC 的“均值有上界、采样/执行动作无上界、非正值折到
`1e-6`”口径一致。

输出默认写入：

```text
SLDAC_db_action/outputs/<timestamp>/
```

## 当前诊断记录（2026-06-11）

本目录的 dB-action 实现曾暴露出一个重要问题：它不仅改变了环境接口动作域，
还让“环境执行动作”和“学习器记录动作”出现了语义分裂。

旧版 power-action 链路是：

```text
raw power action -> env.step 原地裁剪 -> buffer 保存裁剪后的 action
```

dB-action 链路是：

```text
raw power action -> copy 后转 SNR-dB -> env 转回真实 power -> buffer 保存 raw power action
```

修复前，两版都可能通过环境下限执行 `1e-6` 真实功率，但 critic 和
score-function policy gradient 看到的 action 不一致。旧版 buffer 中保存的是
环境原地裁剪后的 `1e-6`；dB-action 版 buffer 中保存的仍可能是 actor 采样出的
非正 raw power。

当前修复口径：`Environment_MIMO.step()` 在 `info["executed_power_action"]`
中返回环境实际执行后的真实 power 坐标动作，`SLDAC.py` 和 `SCAOPO.py`
写入 buffer 时使用该动作，确保 `(state, action, cost, next_state)` 的 action
与产生 cost / next_state 的环境执行动作一致。

当前对齐 legacy 的动作上界口径：`2.5` 只约束 actor mean，不约束环境执行
power；dB-action 转换只把非正 raw power 折到 `1e-6`，正的小功率保持原值。

### 已确认现象

- 100 个记录 episode 的可比输出中，旧版 `legacy_power` 最后 10 点平均真实功率
  约为 `1.34148`，dB-action 版约为 `2.59103`；dB-action 版多用约
  `1.24955` 总功率，但平均 delay 只低约 `0.02515`。
- 分段统计显示，从第 `11-20` 个记录段开始，dB-action 版就稳定多用约
  `1.1-1.3` 总功率。这更像动作域/环境耦合导致的功率基线差异，而不是末期噪声。
- 内存内 monkeypatch 探针显示，旧版第 `11-20` 段 raw 非正 power 比例约
  `29.54%`，且这些动作在环境和 buffer 中都变成 `1e-6`；dB-action 版同段
  环境下限比例约 `23.23%`，但 buffer 仍保存 raw 非正 power。

### 环境层原因

- MIMO rate 公式为 `power[k] * gain / (sum(power * interference_gain) + noise)`。
  当多个用户功率共同缩放时，信号项和干扰项一起缩放；只要不完全进入噪声主导区，
  rate 对整体功率尺度很不敏感。
- 固定动作探针中，每用户 power 从 `1e-4` 增加到 `0.65` 或 `2.5`，平均 rate
  基本都在 `1.0` 附近，平均 delay 变化很小。这说明当前场景大部分时间是干扰受限，
  不是功率越大 rate 就线性变好。
- objective cost 是即时 `sum(power)`，而 delay/cost 反馈滞后一拍并且队列 `D`
  被截断在 `[0, 5]`。因此，压低功率能立刻降低 objective，但低功率带来的
  约束坏处不够尖锐。

### 后续最小验证

已加入 `MIMO1/test_db_action_buffer_semantics.py`，覆盖 dB-action 的 buffer
动作必须等于环境实际执行后的真实 power action，并且执行 power 不再被额外截到
`2.5`。

如需继续量化这次修复对曲线的影响，建议再做两个隔离 ablation，且不要覆盖正式实验产物：

1. 修复后的 dB-action 版：复跑隔离输出，再观察是否接近旧版低功率。
2. 旧版 power-action 临时探针：避免 `env.step()` 原地修改 actor 采样动作，让 buffer 保存
   raw action，再观察是否失去低功率优势。

这两个实验可以直接验证：功率差异的主因是不是 buffer 动作语义，而不是 dB 单位本身。
