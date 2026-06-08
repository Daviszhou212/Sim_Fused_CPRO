"""隔离比较 CLQR1 环境不同 seed 与 seed0 的随机趋势相似性。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLQR_DIR = SCRIPT_DIR.parents[1]
if str(CLQR_DIR) not in sys.path:
    sys.path.insert(0, str(CLQR_DIR))

from environment import Environment_CLQR


# 目标参考 seed：所有候选 seed 都与该 seed 的环境轨迹做对比。
TARGET_SEED = 0
# 候选 seed 搜索范围：默认扫描 0-100（含两端）。
SEED_START = 0
SEED_END = 100
# CLQR 固定场景维度：与当前主线实现保持一致。
STATE_DIM = 15
ACTION_DIM = 4
# 轨迹长度：步数越大，越能看出随机趋势，但运行时间也会增加。
HORIZON = 250

# 固定动作轨迹配置：所有 seed 共用同一组 action，隔离环境随机性影响。
FIXED_ACTION_SEED = 20260401
ACTION_STD = 0.35
ACTION_CLIP = 1.50

# 相似度权重：state 更能反映整体系统演化，因此默认权重最高。
STATE_WEIGHT = 0.60
REWARD_WEIGHT = 0.20
COST_WEIGHT = 0.20
# 控制台展示的候选条数：不含参考 seed 自己。
TOP_K_TO_PRINT = 10

# 默认输出目录：放到 Trash，避免污染正式实验目录。
OUTPUT_DIR = CLQR_DIR / "Trash" / "seed_similarity"
OUTPUT_CSV_NAME = "clqr_seed_similarity_rank.csv"

EPS = 1e-12


def build_fixed_action_trajectory(horizon: int, action_dim: int) -> np.ndarray:
    """生成所有 seed 共用的固定 action 序列。"""
    rng = np.random.default_rng(FIXED_ACTION_SEED)
    time_index = np.arange(horizon, dtype=np.float64).reshape(-1, 1)
    phase = np.linspace(0.20, 1.10, action_dim, dtype=np.float64).reshape(1, -1)
    harmonic = 0.25 * np.sin(0.08 * time_index + phase)
    harmonic += 0.15 * np.cos(0.03 * time_index * (1.0 + phase))
    noise = rng.normal(loc=0.0, scale=ACTION_STD, size=(horizon, action_dim))
    actions = harmonic + noise
    return np.clip(actions, -ACTION_CLIP, ACTION_CLIP)


def rollout_environment(seed: int, actions: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """在不经过训练入口的前提下，直接采样单个 seed 的环境轨迹。"""
    env = Environment_CLQR(seed=int(seed), state_dim=STATE_DIM, action_dim=ACTION_DIM)
    initial_state = np.asarray(env.reset(), dtype=np.float64).reshape(1, -1)

    states = [initial_state.reshape(-1)]
    rewards = []
    costs = []

    for action in actions:
        next_state, reward, _done, info = env.step(np.asarray(action, dtype=np.float64))
        states.append(np.asarray(next_state, dtype=np.float64).reshape(-1))
        rewards.append(float(reward))
        costs.append(float(info["cost"]))

    return {
        "seed": int(seed),
        "states": np.vstack(states),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "costs": np.asarray(costs, dtype=np.float64),
    }


def safe_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """计算稳健 Pearson 相关系数，并处理近似常量序列。"""
    left = np.asarray(lhs, dtype=np.float64).reshape(-1)
    right = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("shape mismatch in correlation: {0} vs {1}".format(left.shape, right.shape))

    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))

    if left_norm <= EPS and right_norm <= EPS:
        return 1.0 if np.allclose(left, right, atol=1e-10, rtol=1e-10) else 0.0
    if left_norm <= EPS or right_norm <= EPS:
        return 0.0

    corr = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))
    return float(np.clip(corr, -1.0, 1.0))


def normalize_weights() -> tuple[float, float, float]:
    """归一化权重，避免后续修改配置时忘记让权重和为 1。"""
    weights = np.asarray([STATE_WEIGHT, REWARD_WEIGHT, COST_WEIGHT], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("similarity weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= EPS:
        raise ValueError("at least one similarity weight must be positive.")
    normalized = weights / total
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def corr_to_score(corr_value: float) -> float:
    """将 [-1, 1] 的相关系数映射到 [0, 100]。"""
    return 50.0 * (float(corr_value) + 1.0)


def compute_similarity(
    reference_rollout: dict[str, np.ndarray | float | int],
    candidate_rollout: dict[str, np.ndarray | float | int],
    weights: tuple[float, float, float],
) -> dict[str, float | int]:
    """计算单个候选 seed 相对参考 seed 的相似度分数。"""
    state_weight, reward_weight, cost_weight = weights
    state_corr = safe_corr(reference_rollout["states"], candidate_rollout["states"])
    reward_corr = safe_corr(reference_rollout["rewards"], candidate_rollout["rewards"])
    cost_corr = safe_corr(reference_rollout["costs"], candidate_rollout["costs"])

    weighted_corr = (
        state_weight * state_corr
        + reward_weight * reward_corr
        + cost_weight * cost_corr
    )

    return {
        "seed": int(candidate_rollout["seed"]),
        "state_corr": state_corr,
        "reward_corr": reward_corr,
        "cost_corr": cost_corr,
        "weighted_corr": weighted_corr,
        "score_0_100": corr_to_score(weighted_corr),
    }


def assign_rank(results: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    """按分数降序、seed 升序生成稳定排名。"""
    ranked = sorted(
        results,
        key=lambda item: (-float(item["score_0_100"]), int(item["seed"])),
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def save_results_csv(results: list[dict[str, float | int]], output_path: Path) -> None:
    """将完整排名保存为 CSV，便于后续筛选和复用。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "seed",
        "score_0_100",
        "weighted_corr",
        "state_corr",
        "reward_corr",
        "cost_corr",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "rank": int(item["rank"]),
                    "seed": int(item["seed"]),
                    "score_0_100": "{0:.6f}".format(float(item["score_0_100"])),
                    "weighted_corr": "{0:.6f}".format(float(item["weighted_corr"])),
                    "state_corr": "{0:.6f}".format(float(item["state_corr"])),
                    "reward_corr": "{0:.6f}".format(float(item["reward_corr"])),
                    "cost_corr": "{0:.6f}".format(float(item["cost_corr"])),
                }
            )


def print_summary(results: list[dict[str, float | int]], output_path: Path) -> None:
    """输出控制台摘要，重点展示除参考 seed 外最相似的候选。"""
    print("CLQR1 环境 seed 相似性扫描完成")
    print("reference seed:", int(TARGET_SEED))
    print("scan range: [{0}, {1}]".format(int(SEED_START), int(SEED_END)))
    print("horizon:", int(HORIZON))
    print("output csv:", str(output_path))
    print("")

    target_row = next(item for item in results if int(item["seed"]) == int(TARGET_SEED))
    print(
        "self-check -> rank={0}, seed={1}, score={2:.4f}".format(
            int(target_row["rank"]),
            int(target_row["seed"]),
            float(target_row["score_0_100"]),
        )
    )
    print("")
    print("Top similar seeds excluding reference:")
    print("rank\tseed\tscore\tstate_corr\treward_corr\tcost_corr")

    shown = 0
    for item in results:
        if int(item["seed"]) == int(TARGET_SEED):
            continue
        print(
            "{0}\t{1}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}".format(
                int(item["rank"]),
                int(item["seed"]),
                float(item["score_0_100"]),
                float(item["state_corr"]),
                float(item["reward_corr"]),
                float(item["cost_corr"]),
            )
        )
        shown += 1
        if shown >= int(TOP_K_TO_PRINT):
            break


def main() -> None:
    candidate_seeds = list(range(int(SEED_START), int(SEED_END) + 1))
    if int(TARGET_SEED) not in candidate_seeds:
        raise ValueError("TARGET_SEED must stay within [SEED_START, SEED_END].")

    weights = normalize_weights()
    actions = build_fixed_action_trajectory(HORIZON, ACTION_DIM)
    rollout_cache = {
        int(seed): rollout_environment(int(seed), actions)
        for seed in candidate_seeds
    }
    reference_rollout = rollout_cache[int(TARGET_SEED)]

    similarity_rows = [
        compute_similarity(reference_rollout, rollout_cache[int(seed)], weights)
        for seed in candidate_seeds
    ]
    ranked_rows = assign_rank(similarity_rows)

    output_path = OUTPUT_DIR / OUTPUT_CSV_NAME
    save_results_csv(ranked_rows, output_path)
    print_summary(ranked_rows, output_path)


if __name__ == "__main__":
    main()
