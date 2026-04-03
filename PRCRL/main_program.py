import torch
import numpy as np
import os
import random
from modules import Environment_MIMO
from modules import Actornn
from modules import seed_everything
from modules import Environment_CLQR
from modules import GaussianPolicy
from modules import BetaPolicy
from modules import DataStorage
from utility_functions import update_policy
from modules import MLP_Gaussian
from modules import MLP_Beta
from modules import projection
from modules import projection1
from modules import projection2
from actor_opt import Actor
import matplotlib.pyplot as plt

import math

def main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow):

    seed=0
    seed_everything(0)
    a = torch.rand(3, 3)
    # device = "cuda:0"
    device = "cpu"
    num_train = int(1e6)
    q = 1
    phi = 1  # probability of using policy reuse
    v = 0.95  # parameter to decrease phi

    Nt, UE_num = 5, 10  # The number of antennas and users.
    state_dim = 2 * UE_num * Nt + 3 * (4 + 5 + 6 + 7 + 7) * 2  # state_dim = 2 * UE_num * Nt + UE_num
    # state_dim =  3 * (4 + 5 + 6 + 7) * 2
    # state_dim = 2 * UE_num * Nt + 3 * UE_num
    action_dim = UE_num  #
    ranpolicy_num = 2

    omega = np.zeros(ranpolicy_num)
    number_chosen = np.zeros(ranpolicy_num)
    reward_chosen = np.zeros(ranpolicy_num)
    delta_tauu = 0.05
    tauu = 0

    prob = [0.5,0.5]
    # prob=[0, 1, 0]
    # prob = [0.99, 0.01]
    prob_tensor = torch.tensor(prob, dtype=torch.float, requires_grad=True)    #用前面计算得到的prob初始化


    #SCAOPO

    env = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
    env_base = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
    env_random = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
    constraint_dim = 0  # constraint_dim = UE_num constr_lim = np.array([1.0, 1.4, 1.0, 1.4])

    fc1_dim, fc2_dim = 128, 128
    actor = GaussianPolicy(state_dim, fc1_dim, fc2_dim, action_dim, device, T)
    # actor.net = actor_train.target_net0   #用训练后的网络初始化
    # model = MLP_Gaussian(state_dim, fc1_dim, fc2_dim, action_dim, device)
    # model.load_state_dict(torch.load(os.path.join(example_name+ '/saved_model', 'Actor_train')))
    # model.eval()
    # actor.net = model
    # actor = BetaPolicy(state_dim, fc1_dim, fc2_dim, action_dim, device, T)

    loaded_actor = GaussianPolicy(state_dim, fc1_dim, fc2_dim, action_dim , device, T)
    model = MLP_Gaussian(state_dim, fc1_dim, fc2_dim,  action_dim, device)
    model.load_state_dict(torch.load(os.path.join(example_name, 'ActorSCAOPO2222_ue5')))    #用别的场景下的训练收敛后网络初始化
    model.eval()
    loaded_actor.net = model

    # loaded_actor = Actornn(state_dim, action_dim)
    # loaded_actor.load_state_dict(torch.load(os.path.join(example_name, 'ActorSAC30507090_0.3')))  # 用别的场景下的训练收敛后网络初始化
    # loaded_actor.eval()
    # loaded_actor.net = model

    loaded_actor1 = GaussianPolicy(state_dim, fc1_dim, fc2_dim, action_dim, device, T)
    model = MLP_Gaussian(state_dim, fc1_dim, fc2_dim, action_dim, device)
    model.load_state_dict(torch.load(os.path.join(example_name, 'ActorSCAOPO30507090_ue5')))  # 用别的场景下的训练收敛后网络初始化
    model.eval()
    loaded_actor1.net = model



    tau_reward = tau
    tau_cost = tau
    interaction_steps = 0
    num_steps = int(2e6)
    chkpt_dir = example_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    chkpt_dir_data = chkpt_dir + '/saved_data'
    if not os.path.exists(chkpt_dir_data):
        os.makedirs(chkpt_dir_data)

    chkpt_dir_model = chkpt_dir + '/saved_model'
    if not os.path.exists(chkpt_dir_model):
        os.makedirs(chkpt_dir_model)

    # Initialization
    theta_dim = 0
    for para in actor.net.parameters():
        theta_dim += para.numel()
    # real_theta_dim = theta_dim + action_dim  # the dimension of the policy parameter.
    real_theta_dim = theta_dim + ranpolicy_num # use this when using the Beta policy
    # real_theta_dim = theta_dim
    paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)
    ind = 0
    for para in actor.net.parameters():
        tmp = para.numel()
        paras_torch[ind: ind + tmp] = para.data.view(-1)
        ind = ind + tmp
    # paras_torch[ind:] = actor.log_std  # comment（注释） this when using the Beta policy
    paras_torch[ind:] = prob_tensor
    func_value = np.zeros(constraint_dim + 1)
    grad = np.zeros((constraint_dim + 1, real_theta_dim))

    # Training
    buffer = DataStorage(T, num_new_data, state_dim, action_dim, constraint_dim)
    t_update = 0  # the number of updating policy
    model_saved_count = 0  # the number of saved model
    reward_rate_all = []  # all objective cost values
    reward_rate_all_base = []
    reward_rate_all_random = []
    cost_rate_all = []  # all constraint cost values
    reward_sum = 0
    reward_sum_base = 0
    reward_sum_random = 0
    cost_sum_total = 0
    observation = env.reset()
    observation_base = env_base.reset()
    observation_random = env_random.reset()
    for step in range(num_steps):
        # generate new data (sample one step of the env)
        if (step) % 1000 == 0:
            observation = env.reset()
            observation_base = env_base.reset()
            observation_random = env_random.reset()
        step1 = (step ) % 1000
        state = observation
        sigma_e = 0.45
        estimate = sigma_e * (np.random.randn(10, 5) + 1j * np.random.randn(10, 5)) / np.sqrt(2)
        e_real = np.real(estimate)
        e_real = e_real.reshape(-1)
        e_imag = np.imag(estimate)
        e_imag = e_imag.reshape(-1)
        estimate = np.hstack((e_real, e_imag))
        for yi in range(100):
            state[yi] = state[yi] + estimate[yi]
        # p_reuse = np.random.choice([0, 1], p=[1 - phi, phi])
        # if p_reuse == 1:
        #     action = loaded_actor.sample_action(state)
        # else:
        #     action = actor.sample_action(state)
        # phi = phi * v
        action1 = actor.sample_action(state)
        # action2 = loaded_actor.sample_action(state)
        # action2 = loaded_actor1.sample_action(state)
        action3 = loaded_actor.sample_action(state)
        # Dmax = [4, 5, 6, 7, 4, 5, 6, 7]
        # start = np.zeros(UE_num + 1)
        # for k in range(UE_num):
        #     start[k + 1] = np.sum(Dmax[0: k + 1])
        # D = state[2 * UE_num * Nt : 2 * UE_num * Nt + (4 + 5 + 6 + 7) * 2]
        # D_sum = np.zeros(UE_num)
        # for k in range(UE_num):
        #     D_sum[k] = np.sum(D[int(start[k]):int(start[k]) + Dmax[k]])
        #
        # if np.sum(D_sum) == 0:
        #     action3 = np.ones(UE_num) / UE_num
        # else:
        #     action3 = D_sum
        #     action3 = action3 / np.sum(action3)

        index = np.random.uniform(0, 10000)
        policy_index = 0
        if index <= 10000 * prob[0]:  #choose action1
            action = action1
        else:
            action = action3
            policy_index = 1
        # if step  <= 7000:
        #     if index <= 10000 * prob[0]:  # choose action1
        #         action = action1
        #     else:
        #         action = action3
        #         policy_index = 1
        # else:
        #     action = action3
        #     policy_index = 1

        # if index <= 10000 * prob[0]:  #choose action1
        #     action = action1
        # elif index <= 10000 * (prob[0] + prob[1]):
        #     action = action2
        #     policy_index = 1
        # else:
        #     action = action3
        #     policy_index = 2

        # action = action2





        observation, reward, done = env.step(action, step1)  # reward is the objective cost in the paper.
        reward_base, a, b= env_base.step1(step1)
        reward_random = env_random.step2(step1)

        # observation是state；reward是power的和；
        costs = np.zeros(constraint_dim + 1)
        costs[0] = reward
        #for k in range(1, constraint_dim + 1):
           # costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])
        buffer.store_experiences(state, action, costs, policy_index)
        interaction_steps += 1
        if step == 5e5:
            reward_sum = 0
            reward_sum_base = 0
            interaction_steps = interaction_steps - 5e5
        reward_sum += reward
        reward_sum_base += reward_base
        reward_sum_random += reward_random
        #cost_sum_total += info.get('cost', 0)

        # print results in the run
        if (step + 1) % 1000 == 0:
            reward_rate_all.append(reward_sum / interaction_steps )
            #cost_rate_all.append(cost_sum_total / interaction_steps)
            print('step: %d, reward_rate = %.3f' % (step, reward_sum / interaction_steps ,
                                                                          ))
            np.save(chkpt_dir_data + '/reward_rate_all.npy', np.array(reward_rate_all))
            #np.save(chkpt_dir_data + '/cost_rate_all.npy', np.array(cost_rate_all))

            reward_rate_all_base.append(reward_sum_base / interaction_steps)
            print('step: %d, reward_rate_base = %.3f' % (step, reward_sum_base / interaction_steps,
                                                    ))
            # np.save(chkpt_dir_data + '/reward_rate_all_base.npy', np.array(reward_rate_all_base))
            #
            reward_rate_all_random.append(reward_sum_random / interaction_steps)
            print('step: %d, reward_rate_random = %.3f' % (step, reward_sum_random / interaction_steps
                                                         ))

        # update the policy
        if ((interaction_steps - 1) % num_new_data == 0) and (buffer.n_entries == 2 * T):
            # estimate the function value.
            t_update += 1
            alpha = 1 / (t_update ** alpha_pow)
            beta = 1 / (t_update ** beta_pow)
            state_batch, action_batch, costs_batch, policyindex_batch = buffer.take_experiences()
            func_value_tilda = np.mean(costs_batch, axis=0)
            func_value = (1 - alpha) * func_value + alpha * func_value_tilda

            # estimate the Q-value
            Q_hat = np.zeros((T, 1 + constraint_dim))
            for _ in range(1, T + 1):
                costs_tmp = costs_batch[_: _ + T]
                Q_hat[_ - 1] = np.sum(costs_tmp, axis=0) - T * func_value

            Q_hat[:, 0] = (Q_hat[:, 0] - np.mean(Q_hat[:, 0])) / (np.std(Q_hat[:, 0]) + 1e-6)
            for _ in range(1, 1 + constraint_dim):
                Q_hat[:, _] = Q_hat[:, _] - np.mean(Q_hat[:, _])

            Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=device)

            # estimate the gradient
            state_batch_torch = torch.tensor(state_batch[1:T + 1], dtype=torch.float, device=device)
            action_batch_torch = torch.tensor(action_batch[1:T + 1], dtype=torch.float, device=device)
            grad_tilda_torch = torch.zeros((1 + constraint_dim, real_theta_dim), dtype=torch.float,
                                           device=device)
            for _ in range(1 + constraint_dim):
                # calculate the gradient

                actor.zero_grad()   #梯度归0
                # prob_tensor.grad.zero_()
                log_prob0 = actor.evaluate_action(state_batch_torch, action_batch_torch)
                # log_prob1 = loaded_actor1.evaluate_action(state_batch_torch, action_batch_torch)
                # log_prob2 = loaded_actor1.evaluate_action(state_batch_torch, action_batch_torch)
                prob2 = torch.zeros((T, 1), dtype=torch.float, device=device)
                for tt in range(T):
                    if policyindex_batch[tt] == 1:
                        prob2[tt] = 1
                actor_loss = (Q_hat_torch[:, _] * torch.log(prob_tensor[0]*torch.exp(log_prob0) + prob_tensor[1] * prob2
                                                            )).mean()
                # actor_loss = (Q_hat_torch[:, _] * torch.log(
                #     prob_tensor[0] * torch.exp(log_prob0) + prob_tensor[1] * torch.exp(log_prob1))
                #     ).mean()
                # actor_loss = (Q_hat_torch[:, _] * torch.log(
                #     prob_tensor[0] * torch.exp(log_prob0)
                #     + prob_tensor[1] * torch.exp(log_prob1)+prob_tensor[2] * torch.exp(log_prob2))).mean()


                # actor_loss = (Q_hat_torch[:, _] * torch.log(
                #     prob_tensor[0] * torch.exp(log_prob0) + prob_tensor[1] * torch.exp(log_prob1)
                #     )).mean()
                # log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
                # actor_loss = (Q_hat_torch[:, _] * log_prob).mean()
                actor_loss.backward()
                grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=device)
                ind = 0
                for para in actor.net.parameters():
                    tmp = para.numel()
                    grad_tmp[ind: ind + tmp] = para.grad.view(-1)
                    ind = ind + tmp
                # grad_tmp[ind:] = actor.log_std.grad  # comment this when using the Beta policy
                grad_tmp[ind:] = prob_tensor.grad
                grad_tilda_torch[_] = grad_tmp

            grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()

            # update the policy parameter
            paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(),
                                      tau_reward=tau_reward, tau_cost=tau_cost)
            paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
            paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
            ind = 0
            for para in actor.net.parameters():
                tmp = para.numel()
                para.data = paras_torch[ind: ind + tmp].view(para.shape)
                ind = ind + tmp
            # actor.log_std = paras_torch[ind:]  # comment this when using the Beta policy

            #将表示概率的参数投影到满足条件的凸集上
            # print('step: %d' % (step))
            # if (step == 1500):
            #     wer=0
            prob = projection1(paras_torch[ind:])
            prob_tensor = torch.tensor(prob, dtype=torch.float, requires_grad=True)

        # save model
        if (step + 1) % 10000 == 0:
            checkpoint_file = os.path.join(chkpt_dir_model, 'Actor' + str(model_saved_count))
            torch.save(actor.net.state_dict(), checkpoint_file)
            model_saved_count += 1

    # plot results
    epoc = np.linspace(0, len(reward_rate_all) - 1, len(reward_rate_all))
    reward_rate_all = np.array(reward_rate_all)
    reward_rate_all_base = np.array(reward_rate_all_base)
    #cost_limit = np.ones(epoc.shape[0]) * constr_lim.mean()
    #cost_rate_all = np.array(cost_rate_all) / constraint_dim
    if 'MIMO' in example_name:
        name_split = example_name.split('_')
        plt.figure()
        plt.plot(epoc, reward_rate_all, label=r'$T = 1500, new = 1000$')
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        #plt.ylabel('Power consumption (W)', fontsize=12, fontweight='roman')
        plt.ylabel('Reward', fontsize=12, fontweight='roman')
        plt.title('MIMO using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()

        #plt.figure()
        #plt.plot(epoc, cost_rate_all, label=r'$T = 1500, new = 1000$')
        #plt.plot(epoc, cost_limit, 'k:', linewidth=1.5)
        #plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        #plt.ylabel('Average delay per user (ms)', fontsize=12, fontweight='roman')
        #plt.title('MIMO using the ' + name_split[1] + ' policy')
        #plt.legend(loc=1)
        #plt.show()
    else:
        name_split = example_name.split('_')
        plt.figure()
        plt.plot(epoc, reward_rate_all, label=r'$T = 1500, new = 1000$')
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Objective cost', fontsize=12, fontweight='roman')
        plt.title('CLQR using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()

        plt.figure()
        plt.plot(epoc, cost_rate_all, label=r'$T = 1500, new = 1000$')
        plt.plot(epoc, cost_limit, 'k:', linewidth=1.5)
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Constraint cost', fontsize=12, fontweight='roman')
        plt.title('CLQR using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()


if __name__ == "__main__":
    example_name = 'MIMO_Gaussian'
    # T = 1500  # 2T is the number of stored experiences
    T = 500
    # T = 250
    num_new_data = 100  # the number of newly added experiences at each update.
    # num_new_data = 300
    #alpha_pow, beta_pow = 0.6, 0.8  # the powers of the decreasing sequences of step sizes alpha and beta.
    alpha_pow, beta_pow = 0.6, 0.7
    tau = 1.0  # the regularization constant in the surrogate function.
    main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'MIMO_Beta'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.9
    # tau = 0.35
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'CLQR_Gaussian'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.9
    # tau = 10.0
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'CLQR_Beta'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.8
    # tau = 10.0
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)
