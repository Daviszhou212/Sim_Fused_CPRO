from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat

from SLDAC import SLDAC_main


# 仿真参数：默认对齐 SLDAC_code/MIMO1 的 MIMO 主入口，可直接改这里。
T = 500
NUM_NEW_DATA = 100
WINDOW = 10000
GRAD_T = T
EPISODE = 60
UPDATE_TIME_PER_EPISODE = 10
NUM_UPDATE_TIME = EPISODE * UPDATE_TIME_PER_EPISODE
Q_UPDATE_TIME = 1
MAX_STEPS = 2 * T + NUM_UPDATE_TIME * NUM_NEW_DATA

# 步长与 surrogate 参数：沿用 SLDAC_code/MIMO1 默认值。
ALPHA_POW = 0.6
BETA_POW = 0.7
ETA_POW = 0.01
GAMMA_POW_REWARD = 0.3
GAMMA_POW_COST = 0.3
TAU_REWARD = 1
TAU_COST = 1

# 打印间隔：每隔多少个记录 episode 打印一次指标；设为 0 可关闭。
PRINT_INTERVAL = 1


def build_args():
	return Namespace(
		T=T,
		grad_T=GRAD_T,
		window=WINDOW,
		num_new_data=NUM_NEW_DATA,
		episode=EPISODE,
		update_time_per_episode=UPDATE_TIME_PER_EPISODE,
		num_update_time=NUM_UPDATE_TIME,
		Q_update_time=Q_UPDATE_TIME,
		MAX_STEPS=MAX_STEPS,
		alpha_pow=ALPHA_POW,
		beta_pow=BETA_POW,
		eta_pow=ETA_POW,
		gamma_pow_reward=GAMMA_POW_REWARD,
		gamma_pow_cost=GAMMA_POW_COST,
		tau_reward=TAU_REWARD,
		tau_cost=TAU_COST,
		print_interval=PRINT_INTERVAL,
	)


def main():
	out_dir = Path(__file__).resolve().parents[1] / "outputs" / datetime.now().strftime("mimo_db_action_%Y%m%d_%H%M%S")
	out_dir.mkdir(parents=True, exist_ok=False)
	reward_average, cost_average = SLDAC_main(build_args(), "MIMO")
	reward_average = np.asarray(reward_average, dtype=np.float64)
	cost_average = np.asarray(cost_average, dtype=np.float64)
	savemat(
		out_dir / "SLDAC_db_action_mimo.mat",
		{
			"objective_cost_real_power": reward_average,
			"avg_delay_per_user": cost_average,
		},
	)
	np.savetxt(
		out_dir / "SLDAC_db_action_mimo.csv",
		np.column_stack((np.arange(1, reward_average.size + 1), reward_average, cost_average)),
		delimiter=",",
		header="episode,objective_cost_real_power,avg_delay_per_user",
		comments="",
	)
	print("output_dir:", out_dir)


if __name__ == "__main__":
	main()
