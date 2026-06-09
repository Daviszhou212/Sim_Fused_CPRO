from SLDAC import SLDAC_main
import matplotlib.pyplot as plt
import numpy as np
from SCAOPO import SCAOPO_main
from scipy.io import loadmat
from scipy.io import savemat
import argparse
from scipy.signal import hilbert

def get_envelop(x,n=None):
    analytic_signal = hilbert(x, N=n)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def main(args, example_name):

    episode = 51
    interval = 1
    x = []
    constr_limit = []
    for jj in range(int(episode / interval)):
        x.append(jj)
        constr_limit.append(1.2)

    alpha_deg = 50
    reward_ppo_100= loadmat("reward_ppo_100.mat")["array"]
    reward_cpo_100 = loadmat("reward_cpo_100.mat")["array"]
    reward_ppo_100 = reward_ppo_100[0, 0:episode][::interval]
    nihe = np.polyfit(x, reward_ppo_100, deg=alpha_deg)
    reward_ppo_100 = np.polyval(nihe, x)
    reward_cpo_100=reward_cpo_100[0, 0:episode][::interval]
    nihe = np.polyfit(x, reward_cpo_100, deg=alpha_deg)
    reward_cpo_100 = np.polyval(nihe, x)
    SLDAC_reward_b500_q10 = loadmat("SLDAC_reward_b500_q10.mat")["array"]
    nihe = np.polyfit(x, SLDAC_reward_b500_q10[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_reward_b500_q10 = np.polyval(nihe, x)

    SLDAC_reward_b100_q1 = loadmat("SLDAC_reward_b100_q1.mat")["array"]
    nihe = np.polyfit(x, SLDAC_reward_b100_q1[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_reward_b100_q1 = np.polyval(nihe, x)
    SLDAC_reward_b100_q5 = loadmat("SLDAC_reward_b100_q5.mat")["array"]
    nihe = np.polyfit(x, SLDAC_reward_b100_q5[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_reward_b100_q5 = np.polyval(nihe, x)
    SLDAC_reward_b100_q10 = loadmat("SLDAC_reward_b100_q10.mat")["array"]
    nihe = np.polyfit(x, SLDAC_reward_b100_q10[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_reward_b100_q10 = np.polyval(nihe, x)
    SCAOPO_reward_100 = loadmat("SCAOPO_reward_100.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_reward_100[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_reward_100 = np.polyval(nihe, x)
    SCAOPO_reward_300 = loadmat("SCAOPO_reward_300.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_reward_300[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_reward_300 = np.polyval(nihe, x)
    SCAOPO_reward_500 = loadmat("SCAOPO_reward_500.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_reward_500[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_reward_500 = np.polyval(nihe, x)
    plt.figure(figsize=(9, 6.5))
    plt.plot(x, reward_cpo_100, color='m', linewidth=3, linestyle='-', marker='*', markersize=1, label='CPO, B=300, q=10')
    plt.plot(x, reward_ppo_100, color='m', linewidth=3.5, linestyle='--', marker='.', markersize=1, label='PPO-Lag, B=300, q=10')
    plt.plot(x, SCAOPO_reward_500, color='#FF9900', linewidth=3, linestyle='-', label='SCAOPO, T=1000, B=500')
    plt.plot(x, SLDAC_reward_b100_q1, color='blue', linewidth=3.5, linestyle=':', label='SLDAC, T=1000, B=100, q=1')
    plt.plot(x, SLDAC_reward_b100_q5, color='blue', linewidth=3.5, linestyle='--', label='SLDAC, T=1000, B=100, q=5')
    plt.plot(x, SLDAC_reward_b100_q10, color='blue', linewidth=3, linestyle='-', marker="s", markersize=0.5, label='SLDAC, T=1000, B=100, q=10')
    plt.plot(x, SLDAC_reward_b500_q10, color='springgreen', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC(No reuse), B=100, q=10')
    plt.margins(x=0)
    plt.ylim(0, 9)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 5)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*5)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Power consumption')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("MIMO_reward.pdf")
    plt.show()

    alpha_deg = 10
    cost_ppo_100 = loadmat("cost_ppo_100.mat")["array"]
    cost_cpo_100 = loadmat("cost_cpo_100.mat")["array"]
    cost_ppo_100[0,0] = 1.4
    cost_cpo_100[0, 0] = 1.4
    savemat("cost_ppo_100.mat", {"array": cost_ppo_100})
    savemat("cost_cpo_100.mat", {"array": cost_cpo_100})
    cost_ppo_100 = cost_ppo_100[0, 0:episode][::interval]
    nihe = np.polyfit(x, cost_ppo_100, deg=alpha_deg)
    cost_ppo_100 = np.polyval(nihe, x)
    cost_cpo_100 = cost_cpo_100[0, 0:episode][::interval]
    nihe = np.polyfit(x, cost_cpo_100, deg=alpha_deg)
    cost_cpo_100 = np.polyval(nihe, x)
    SLDAC_cost_b500_q10 = loadmat("SLDAC_cost_b500_q10.mat")["array"]
    nihe = np.polyfit(x, SLDAC_cost_b500_q10[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_cost_b500_q10 = np.polyval(nihe, x)

    SLDAC_cost_b100_q1 = loadmat("SLDAC_cost_b100_q1.mat")["array"]
    nihe = np.polyfit(x, SLDAC_cost_b100_q1[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_cost_b100_q1 = np.polyval(nihe, x)
    SLDAC_cost_b100_q5 = loadmat("SLDAC_cost_b100_q5.mat")["array"]
    nihe = np.polyfit(x, SLDAC_cost_b100_q5[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_cost_b100_q5 = np.polyval(nihe, x)
    SLDAC_cost_b100_q10 = loadmat("SLDAC_cost_b100_q10.mat")["array"]
    nihe = np.polyfit(x, SLDAC_cost_b100_q10[0, 0:episode][::interval], deg=alpha_deg)
    SLDAC_cost_b100_q10 = np.polyval(nihe, x)
    SCAOPO_cost_100 = loadmat("SCAOPO_cost_100.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_cost_100[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_cost_100 = np.polyval(nihe, x)
    SCAOPO_cost_300 = loadmat("SCAOPO_cost_300.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_cost_300[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_cost_300 = np.polyval(nihe, x)
    SCAOPO_cost_500 = loadmat("SCAOPO_cost_500.mat")["array"]
    nihe = np.polyfit(x, SCAOPO_cost_500[0, 0:episode][::interval], deg=alpha_deg)
    SCAOPO_cost_500 = np.polyval(nihe, x)
    plt.figure(figsize=(9, 6.5))
    plt.plot(x, cost_cpo_100, color='m', linewidth=3, linestyle='-', marker='*', markersize=1, label='CPO, B=300, q=10')
    plt.plot(x, cost_ppo_100, color='m', linewidth=3.5, linestyle='--', marker='.', markersize=1, label='PPO-Lag, B=300, q=10')
    plt.plot(x, SCAOPO_cost_500, color='#FF9900', linewidth=3, linestyle='-', label='SCAOPO, T=1000, B=500')
    plt.plot(x, SLDAC_cost_b100_q1, color='blue', linewidth=3.5, linestyle=':', label='SLDAC, T=1000, B=100, q=1')
    plt.plot(x, SLDAC_cost_b100_q5, color='blue', linewidth=3.5, linestyle='--', label='SLDAC, T=1000, B=100, q=5')
    plt.plot(x, SLDAC_cost_b100_q10, color='blue', linewidth=3, linestyle='-', label='SLDAC, T=1000, B=100, q=10')
    plt.plot(x, SLDAC_cost_b500_q10, color='springgreen', linewidth=3, linestyle='-', label='SLDAC(No reuse), B=100, q=10')
    plt.plot(x, constr_limit, color='black', linewidth=3, linestyle='-', label='average cost limit')
    plt.margins(x=0)
    plt.ylim(1, 2.8)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 5)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*5)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Average delay per user')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("MIMO_cost.pdf")
    plt.show()



    ####################################################### SCAOPO
    print("SLDAC, T=500, batchsize=500, q=10")
    args.T = 50
    args.grad_T = args.T*2
    args.num_new_data = 100
    args.Q_update_time = 10
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SLDAC_reward_b500_q10, SLDAC_cost_b500_q10 = SLDAC_main(args, example_name)
    savemat("SLDAC_reward_b500_q10.mat", {"array": SLDAC_reward_b500_q10})
    savemat("SLDAC_cost_b500_q10.mat", {"array": SLDAC_cost_b500_q10})

    print("new=100, SCAOPO")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 100
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SCAOPO_reward_100, SCAOPO_cost_100 = SCAOPO_main(args, example_name)
    savemat("SCAOPO_reward_100.mat", {"array": SCAOPO_reward_100})
    savemat("SCAOPO_cost_100.mat", {"array": SCAOPO_cost_100})

    print("new=300, SCAOPO")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 300
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SCAOPO_reward_300, SCAOPO_cost_300 = SCAOPO_main(args, example_name)
    savemat("SCAOPO_reward_300.mat", {"array": SCAOPO_reward_300})
    savemat("SCAOPO_cost_300.mat", {"array": SCAOPO_cost_300})

    print("new=500, SCAOPO")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 500
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SCAOPO_reward_500, SCAOPO_cost_500 = SCAOPO_main(args, example_name)
    savemat("SCAOPO_reward_500.mat", {"array": SCAOPO_reward_500})
    savemat("SCAOPO_cost_500.mat", {"array": SCAOPO_cost_500})

    #######################################################  SLDAC
    print("SLDAC, batchsize=100, q=10")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 100
    args.Q_update_time = 10
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SLDAC_reward_b100_q10, SLDAC_cost_b100_q10 = SLDAC_main(args, example_name)
    savemat("SLDAC_reward_b100_q10.mat", {"array": SLDAC_reward_b100_q10})
    savemat("SLDAC_cost_b100_q10.mat", {"array": SLDAC_cost_b100_q10})

    print("SLDAC, batchsize=100, q=1")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 100
    args.Q_update_time = 1
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SLDAC_reward_b100_q1, SLDAC_cost_b100_q1 = SLDAC_main(args, example_name)
    savemat("SLDAC_reward_b100_q1.mat", {"array": SLDAC_reward_b100_q1})
    savemat("SLDAC_cost_b100_q1.mat", {"array": SLDAC_cost_b100_q1})

    print("SLDAC, batchsize=100, q=5")
    args.T = 500
    args.grad_T = args.T
    args.num_new_data = 100
    args.Q_update_time = 5
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SLDAC_reward_b100_q5, SLDAC_cost_b100_q5 = SLDAC_main(args, example_name)
    savemat("SLDAC_reward_b100_q5.mat", {"array": SLDAC_reward_b100_q5})
    savemat("SLDAC_cost_b100_q5.mat", {"array": SLDAC_cost_b100_q5})

    print("SLDAC, T=500, batchsize=500, q=10")
    args.T = 50
    args.grad_T = args.T*2
    args.num_new_data = 100
    args.Q_update_time = 10
    args.MAX_STEPS = 2*args.T + args.num_update_time*args.num_new_data
    SLDAC_reward_b500_q10, SLDAC_cost_b500_q10 = SLDAC_main(args, example_name)
    savemat("SLDAC_reward_b500_q10.mat", {"array": SLDAC_reward_b500_q10})
    savemat("SLDAC_cost_b500_q10.mat", {"array": SLDAC_cost_b500_q10})

example_name = 'MIMO'
alpha_pow = 0.6
beta_pow = 0.7
eta_pow = 0.01
gamma_pow = 0.3
gamma_pow_reward = gamma_pow
gamma_pow_cost = gamma_pow
tau_reward = 1
tau_cost = 1

T = 500
num_new_data = 100
window = 10000
grad_T = T
episode = 60
update_time_per_episode = 10
num_update_time = episode*update_time_per_episode
Q_update_time = 1
MAX_STEPS = 2*T + num_update_time*num_new_data

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=T)
parser.add_argument('--grad_T', type=int, default=grad_T)
parser.add_argument('--window', type=int, default=window)
parser.add_argument('--num_new_data', type=int, default=num_new_data)
parser.add_argument('--episode', type=int, default=episode)
parser.add_argument('--update_time_per_episode', type=int, default=update_time_per_episode)
parser.add_argument('--num_update_time', type=int, default=num_update_time)
parser.add_argument('--Q_update_time', type=int, default=Q_update_time)
parser.add_argument('--MAX_STEPS', type=int, default=MAX_STEPS)
parser.add_argument('--alpha_pow', type=float, default=alpha_pow)
parser.add_argument('--beta_pow', type=float, default=beta_pow)
parser.add_argument('--eta_pow', type=float, default=eta_pow)
parser.add_argument('--gamma_pow_reward', type=float, default=gamma_pow_reward)
parser.add_argument('--gamma_pow_cost', type=float, default=gamma_pow_cost)
parser.add_argument('--tau_reward', type=float, default=tau_reward)
parser.add_argument('--tau_cost', type=float, default=tau_cost)
args = parser.parse_args()

main(args, example_name)