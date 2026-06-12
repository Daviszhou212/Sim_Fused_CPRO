from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SLDAC_MIMO_DIR = ROOT_DIR / "SLDAC_code" / "MIMO1"
if str(SLDAC_MIMO_DIR) not in sys.path:
    sys.path.insert(0, str(SLDAC_MIMO_DIR))

from environment import Environment_MIMO  # noqa: E402


MIMO_NT = 8
MIMO_UE_NUM = 4
MIMO_STATE_DIM = 2 * MIMO_UE_NUM * MIMO_NT + MIMO_UE_NUM
MIMO_ACTION_DIM = MIMO_UE_NUM + 1

# 仿真窗口配置：默认对齐 SLDAC_code/MIMO1 的 100 个统计点，每点约 1000 步。
DEFAULT_EPISODE_COUNT = 100
DEFAULT_STEPS_PER_EPISODE = 1000
DEFAULT_SEED = 0
LOG_INTERVAL_EPISODES = 10

# 输出配置：使用带时间戳的新目录，避免覆盖历史实验产物。
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class HeuristicConfig:
    # 每用户最低发射功率，避免完全不服务导致队列长时间堆积。
    min_power_per_user: float = 0.02
    # 每用户动作上界，沿用 SLDAC_code/MIMO1 高斯均值的 [0, 2.5] 支撑口径。
    max_power_per_user: float = 2.5
    # 系统总功率上界；heuristic 会在该预算内按队列压力分配功率。
    max_total_power: float = 3.0
    # 空队列时的总功率，提供低成本探测与基础服务能力。
    idle_total_power: float = 0.2
    # 队列优先级下限，避免某个用户被完全饿死。
    queue_priority_floor: float = 0.05
    # MIMO 环境中的队列截断上界，用于把队列压力归一化。
    delay_max: float = 5.0
    # 压力响应指数；小于 1 时中等队列压力会更早增加功率。
    pressure_exponent: float = 0.7
    # RZF 正则因子，非学习基线中保持固定。
    reg_factor: float = 1.0


@dataclass(frozen=True)
class RolloutResult:
    objective_cost: np.ndarray
    avg_delay_per_user: np.ndarray


DEFAULT_CONFIG = HeuristicConfig()


def _extract_delay(state: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float64).reshape(-1)
    if state_arr.size < MIMO_UE_NUM:
        raise ValueError("state must contain at least {0} delay entries".format(MIMO_UE_NUM))
    return np.maximum(state_arr[-MIMO_UE_NUM:], 0.0)


def queue_aware_action(state: np.ndarray, config: HeuristicConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Return a deterministic non-learning MIMO action from current queue delays."""
    delay = _extract_delay(state)
    max_delay = float(np.max(delay)) if delay.size else 0.0
    pressure = np.clip(max_delay / max(float(config.delay_max), 1e-12), 0.0, 1.0)
    total_power = float(config.idle_total_power) + (
        float(config.max_total_power) - float(config.idle_total_power)
    ) * (pressure ** float(config.pressure_exponent))
    total_power = float(np.clip(total_power, MIMO_UE_NUM * config.min_power_per_user, config.max_total_power))

    priority = delay + float(config.queue_priority_floor)
    priority_sum = float(np.sum(priority))
    if priority_sum <= 0.0:
        priority = np.ones(MIMO_UE_NUM, dtype=np.float64)
        priority_sum = float(MIMO_UE_NUM)

    guaranteed_power = MIMO_UE_NUM * float(config.min_power_per_user)
    allocatable_power = max(total_power - guaranteed_power, 0.0)
    power = float(config.min_power_per_user) + allocatable_power * priority / priority_sum
    power = np.clip(power, float(config.min_power_per_user), float(config.max_power_per_user))

    action = np.empty(MIMO_ACTION_DIM, dtype=np.float64)
    action[:MIMO_UE_NUM] = power
    action[MIMO_UE_NUM] = float(config.reg_factor)
    return action


def run_heuristic_rollout(
    episode_count: int = DEFAULT_EPISODE_COUNT,
    steps_per_episode: int = DEFAULT_STEPS_PER_EPISODE,
    seed: int = DEFAULT_SEED,
    config: HeuristicConfig = DEFAULT_CONFIG,
    log_interval_episodes: int = LOG_INTERVAL_EPISODES,
    verbose: bool = True,
) -> RolloutResult:
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be positive")

    env = Environment_MIMO(seed=int(seed), Nt=MIMO_NT, UE_num=MIMO_UE_NUM)
    observation = env.reset()
    objective_cost = np.zeros(int(episode_count), dtype=np.float64)
    avg_delay_per_user = np.zeros(int(episode_count), dtype=np.float64)

    for episode_index in range(int(episode_count)):
        objective_sum = 0.0
        delay_sum = 0.0
        for _ in range(int(steps_per_episode)):
            action = queue_aware_action(observation, config)
            observation, reward, _done, info = env.step(action)
            objective_sum += float(reward)
            delay_sum += float(info.get("cost", 0.0)) / float(MIMO_UE_NUM)

        objective_cost[episode_index] = objective_sum / float(steps_per_episode)
        avg_delay_per_user[episode_index] = delay_sum / float(steps_per_episode)

        current_episode = episode_index + 1
        should_log = (
            current_episode == 1
            or current_episode == int(episode_count)
            or current_episode % max(int(log_interval_episodes), 1) == 0
        )
        if verbose and should_log:
            print(
                "HEURISTIC_MIMO_EPISODE {0}/{1}: objective_cost={2:.6f}, avg_delay_per_user={3:.6f}".format(
                    current_episode,
                    int(episode_count),
                    float(objective_cost[episode_index]),
                    float(avg_delay_per_user[episode_index]),
                ),
                flush=True,
            )

    return RolloutResult(objective_cost=objective_cost, avg_delay_per_user=avg_delay_per_user)


def write_outputs(
    result: RolloutResult,
    config: HeuristicConfig = DEFAULT_CONFIG,
    episode_count: int = DEFAULT_EPISODE_COUNT,
    steps_per_episode: int = DEFAULT_STEPS_PER_EPISODE,
    seed: int = DEFAULT_SEED,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "mimo_heuristic_seed{0}_ep{1}_{2}".format(int(seed), int(episode_count), timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "mimo_heuristic_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "objective_cost_sum_power", "avg_delay_per_user"])
        for index, (objective, delay) in enumerate(zip(result.objective_cost, result.avg_delay_per_user), start=1):
            writer.writerow([index, float(objective), float(delay)])

    tail_count = min(10, int(result.objective_cost.size))
    summary = {
        "source_environment": str(SLDAC_MIMO_DIR),
        "policy": "queue_aware_heuristic_action",
        "seed": int(seed),
        "episode_count": int(episode_count),
        "steps_per_episode": int(steps_per_episode),
        "num_steps": int(episode_count) * int(steps_per_episode),
        "config": asdict(config),
        "num_points": int(result.objective_cost.size),
        "final_objective_cost_sum_power": float(result.objective_cost[-1]),
        "final_avg_delay_per_user": float(result.avg_delay_per_user[-1]),
        "mean_last_{0}_objective_cost_sum_power".format(tail_count): float(np.mean(result.objective_cost[-tail_count:])),
        "mean_last_{0}_avg_delay_per_user".format(tail_count): float(np.mean(result.avg_delay_per_user[-tail_count:])),
        "files": {
            "csv": str(csv_path),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return output_dir


def main() -> None:
    result = run_heuristic_rollout(
        episode_count=DEFAULT_EPISODE_COUNT,
        steps_per_episode=DEFAULT_STEPS_PER_EPISODE,
        seed=DEFAULT_SEED,
        config=DEFAULT_CONFIG,
        log_interval_episodes=LOG_INTERVAL_EPISODES,
        verbose=True,
    )
    output_dir = write_outputs(
        result,
        config=DEFAULT_CONFIG,
        episode_count=DEFAULT_EPISODE_COUNT,
        steps_per_episode=DEFAULT_STEPS_PER_EPISODE,
        seed=DEFAULT_SEED,
        output_root=OUTPUT_ROOT,
    )
    print("outputs: {0}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
