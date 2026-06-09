import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import cvxpy as cp

Amin = 38
Amax = 42
# lambdak = [20, 40, 60, 80, 20, 40, 60, 80]
# lambdak = [22, 42, 62, 82, 82, 22, 42, 62, 82, 82]
# lambdak = [9, 10, 11, 12, 9, 10, 11, 12]
# lambdak = [34, 36, 38, 40, 34, 36, 38, 40]
# lambdak = [60, 65, 70, 75, 60, 65, 70, 75]
# lambdak = [50, 60, 70, 80, 50, 60, 70, 80]
lambdak = [30, 50, 70, 90, 90, 30, 50, 70, 90, 90]
# lambdak =  [22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
p_a = 0.3
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def correlation(H1, H2):
    a1 = np.linalg.norm(H1, 2)
    a2 = np.linalg.norm(H2, 2)
    B = H1.conjugate().T @ H2
    c = np.abs(B / (a1 * a2))
    return c

def projection(paras):
    # paras[paras < 0] = 1e-6
    n = paras.shape[0]
    paras_cvx = cp.Variable(shape=(n,))
    paras_t =  paras.detach().cpu().numpy()
    # paras_t1 = paras.detach().numpy()
    obj = cp.sum_squares(paras_cvx - paras_t)
    constr = []
    constr += [paras_cvx <= 1]
    constr += [paras_cvx >= 0]
    constr += [sum(paras_cvx) == 1]

    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.MOSEK)
    # prob.solve(solver=cp.OSQP)
    # prob.solve(solver=cp.MOSEK)

    paras_p = paras_cvx.value
    return paras_p

def projection1(paras):
    # paras[paras < 0] = 1e-6
    n = paras.shape[0]
    paras_t =  paras.detach().cpu().numpy()
    paras_p = np.zeros(n)
    if ((paras_t[0]-paras_t[1]+1)/2) > 1:
        paras_p[0] = 1
    elif ((paras_t[0]-paras_t[1]+1)/2) > 0:
        paras_p[0] = (paras_t[0]-paras_t[1]+1)/2
    else:
        paras_p[0] = 0

    if ((paras_t[1]-paras_t[0]+1)/2) > 1:
        paras_p[1] = 1
    elif ((paras_t[1]-paras_t[0]+1)/2) > 0:
        paras_p[1] = (paras_t[1]-paras_t[0]+1)/2
    else:
        paras_p[1] = 0
    return paras_p
def projection2(paras):
    n = paras.shape[0]
    paras_t = paras.detach().cpu().numpy()
    paras_p = np.zeros(n)
    # paras_ps = np.zeros(2, 7)
    min_distance = 0
    a = [1, 0, 0]
    min_diatance = sum((paras_t - a) ** 2)
    paras_p = a
    a = [0, 1, 0]
    if sum((paras_t - a) ** 2) <= min_diatance:
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    a = [0, 0, 1]
    if sum((paras_t - a) ** 2) <= min_diatance:
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    a = [0, paras_t[1]-(paras_t[1]+paras_t[2]-1)/2, paras_t[2]-(paras_t[1]+paras_t[2]-1)/2]
    if (sum((paras_t - a) ** 2) <= min_diatance) & (a[1] >= 0) & (a[1] <= 1) & (a[2] >= 0) & (a[2] <= 1) :
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    a = [paras_t[0] - (paras_t[0] + paras_t[2] - 1) / 2, 0, paras_t[2] - (paras_t[0] + paras_t[2] - 1) / 2]
    if (sum((paras_t - a) ** 2) <= min_diatance) & (a[0] >= 0) & (a[0] <= 1) & (a[2] >= 0) & (a[2] <= 1) :
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    a = [paras_t[0] - (paras_t[0] + paras_t[1] - 1) / 2, paras_t[1] - (paras_t[0] + paras_t[1] - 1) / 2, 0]
    if (sum((paras_t - a) ** 2) <= min_diatance) & (a[0] >= 0) & (a[0] <= 1) & (a[1] >= 0) & (a[1] <= 1):
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    a = [paras_t[0] - (paras_t[0] + paras_t[1] + paras_t[2] - 1) / 3, paras_t[1] - (paras_t[0] + paras_t[1] + paras_t[2] - 1) / 3, paras_t[2] - (paras_t[0] + paras_t[1] + paras_t[2] - 1) / 3]
    if (sum((paras_t - a) ** 2) <= min_diatance) & (a[0] >= 0) & (a[0] <= 1) & (a[1] >= 0) & (a[1] <= 1) & (a[2] >= 0) & (a[2] <= 1):
        min_diatance = sum((paras_t - a) ** 2)
        paras_p = a
    return paras_p
def MU_greedy(UE_num, H, weight, power, Q):
    rate = 0
    UEset1 = list()  #已选用户集
    UEset2 = list(range(UE_num))   #剩余用户集
    # calWSR([0,1], H, weight, power, Q)
    for l in range(UE_num):
        maxrate = 0
        for i in range(len(UEset2)):
            UEset = UEset1 + [UEset2[i]]
            # UEset.append(UEset2[i])
            # UEset.remove([])
            WSR = calWSR(UEset, H, weight, power, Q)
            if WSR >= maxrate:
                maxrate = WSR
                user = UEset2[i]
        if maxrate >= rate:
            rate = maxrate
            UEset1 = UEset1 + [user]
            UEset2.remove(user)
        else:
            break
        if len(UEset2) == 0:
            break

    return rate,UEset1

def calWSR(UEset, H, weight, power, Q):
    H_C = np.zeros((len(UEset), H.shape[1])) + 1j * np.zeros((len(UEset), H.shape[1]))
    for i in range(len(UEset)):
        H_C[i,:] = H[UEset[i],:]
    try:
        V = H_C.conjugate().T @ np.linalg.inv(H_C @ H_C.conjugate().T)
    except:
        V = H_C.conjugate().T @ np.linalg.pinv(H_C @ H_C.conjugate().T)
    WSR = 0
    powerk = power/len(UEset)
    for i in range(len(UEset)):
        hv_tilda = H_C[i, :] @ V[:, i]
        module_squ = np.abs(hv_tilda) ** 2
        dominator = 1e-6
        for j in range(len(UEset)):
            dominator = dominator + np.abs(H_C[i, :] @ V[:, j]) ** 2
        dominator = dominator - module_squ
        WSR = WSR + weight[UEset[i]] * min(np.log2(1 + module_squ / dominator), Q[UEset[i]])
        # WSR = WSR + weight[UEset[i]] * np.log2(1 + module_squ / dominator)
    return WSR

def calRate(UEset, H, weight, UEnum, power):
    H_C = np.zeros((len(UEset), H.shape[1])) + 1j * np.zeros((len(UEset), H.shape[1]))
    for i in range(len(UEset)):
        H_C[i,:] = H[UEset[i],:]
    try:
        V = H_C.conjugate().T @ np.linalg.inv(H_C @ H_C.conjugate().T)
    except:
        V = H_C.conjugate().T @ np.linalg.pinv(H_C @ H_C.conjugate().T)
    rate = np.zeros((UEnum, 1))
    powerk = power/len(UEset)
    for i in range(len(UEset)):
        hv_tilda = H_C[i, :] @ V[:, i]
        module_squ = np.abs(hv_tilda) ** 2
        dominator = 1e-6
        for j in range(len(UEset)):
            dominator = dominator + np.abs(H_C[i, :] @ V[:, j]) ** 2
        dominator = dominator - module_squ
        rate[UEset[i]] = np.log2(1 + module_squ / dominator)
    return rate

class Actornn(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actornn, self).__init__()
        seed_everything(0)
        self.fc1 = nn.Linear(state_dim, 128)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)
        self.fc2 = nn.Linear(128, 128)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)
        self.fc3 = nn.Linear(128, action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        mu = self.fc3(x)
        mu = 2.5 * torch.sigmoid(mu)
        sigma = torch.exp(-0.5 * torch.ones(8, dtype=torch.float, device='cpu'))
        return mu, sigma

    def sample(self, state, epsilon=1e-6):
        mean, std = self.forward(state)
        # std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        # action = action * self.action_bound
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, 0

class GaussianPolicy(nn.Module):
    """The class to realize the Gaussian policy.Lipschitz
    The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device, T):   # fc1和fc2分别表示两个隐藏层的宽度
        super(GaussianPolicy, self).__init__()
        self.net = MLP_Gaussian(state_dim, fc1_dim, fc2_dim, action_dim, device)
        self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
        self.action_dim = action_dim
        self.T = T
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        raise NotImplementedError

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        mu = self.net(state_torch)
        # self.log_std.requires_grad = True
        self.std_eval = torch.exp(self.log_std)
        self.std_eval = self.std_eval.view(1, -1).repeat(self.T, 1)
        gaussian_ = torch.distributions.normal.Normal(mu, self.std_eval)
        log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

        return log_prob_action

    def sample_action(self, state):
        self.net.eval()
        # self.log_std.requires_grad = False
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            mu = self.net(state_torch)
            self.std_sample = torch.exp(self.log_std)
            gaussian_ = torch.distributions.normal.Normal(mu, self.std_sample)
            action = gaussian_.sample()

        return action.detach().cpu().numpy()


class MLP_Gaussian(nn.Module):
    """The neural network used to approximate the Gaussian policy"""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
        super(MLP_Gaussian, self).__init__()
        self.input_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
        seed_everything(0)

        # a = torch.rand(3, 3)

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)

        self.fc3 = nn.Linear(self.fc2_dim, self.action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        mu = self.fc3(x)
        mu = 2.5 * torch.sigmoid(mu)

    # def forward(self, state):  # use this when simulating the CLQR.
    #     x = self.fc1(state)
    #     x = torch.tanh(x)
    #     x = self.fc2(x)
    #     x = torch.tanh(x)
    #     mu = self.fc3(x)

        return mu


class BetaPolicy(nn.Module):
    """The class to realize the Beta policy.
    The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device, T):
        super(BetaPolicy, self).__init__()
        self.net = MLP_Beta(state_dim, fc1_dim, fc2_dim, 2 * action_dim, device)
        self.action_dim = action_dim
        self.h = 2.5 * torch.ones(self.action_dim, dtype=torch.float, device=device)
        # self.h = 2 * torch.ones(self.action_dim, dtype=torch.float, device=device)  # use this when simulating CLQR.
        self.T = T
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        raise NotImplementedError

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        a_b_values = self.net(state_torch)
        a = a_b_values[:, 0:self.action_dim]
        b = a_b_values[:, self.action_dim:]
        beta_ = torch.distributions.beta.Beta(a, b)
        h = self.h.view(1, -1).repeat(self.T, 1)
        log_prob_action = beta_.log_prob(action_torch / h).sum(dim=1)
        # log_prob_action = beta_.log_prob((action_torch + h) / (2 * h)).sum(dim=1)  # use this when simulating CLQR.

        return log_prob_action

    def sample_action(self, state):
        self.net.eval()
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            a_b_values = self.net(state_torch)
            a = a_b_values[0:self.action_dim]
            b = a_b_values[self.action_dim:]
            beta_ = torch.distributions.beta.Beta(a, b)
            action = beta_.sample() * self.h
            # action = beta_.sample() * (2 * self.h) - self.h  # use this when simulating CLQR.

        return action.detach().cpu().numpy()


class MLP_Beta(nn.Module):
    """The neural network used to approximate the Beta policy"""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
        super(MLP_Beta, self).__init__()
        self.input_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)

        self.fc3 = nn.Linear(self.fc2_dim, self.action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        a_b_values = self.fc3(x)
        a_b_values = F.softplus(a_b_values)

    # def forward(self, state):  # use this when simulating CLQR.
    #     x = self.fc1(state)
    #     x = torch.tanh(x)
    #     x = self.fc2(x)
    #     x = torch.tanh(x)
    #     a_b_values = self.fc3(x)
    #     a_b_values = F.softplus(a_b_values) + 1

        return a_b_values


class DataStorage(object):
    """The class to realize the dynamic storage of data samples"""
    def __init__(self, T, num_new_data, state_dim, action_dim, constraint_dim):
        self.T = T
        self.num_new_data = num_new_data
        self.count = 0
        self.state_memory = np.zeros((2 * self.T, state_dim))
        self.action_memory = np.zeros((2 * self.T, action_dim))
        self.cost_memory = np.zeros((2 * self.T, 1+constraint_dim))
        self.index_memory = np.zeros((2 * self.T, 1))
        self.n_entries = 0
        self.state_memory_tmp = np.zeros((num_new_data, state_dim))
        self.action_memory_tmp = np.zeros((num_new_data, action_dim))
        self.cost_memory_tmp = np.zeros((num_new_data, 1 + constraint_dim))
        self.index_memory_tmp = np.zeros((num_new_data, 1))

    def store_experiences(self, state, action, costs, policy_index):
        if self.count < 2 * self.T:
            self.state_memory[self.count] = state
            self.action_memory[self.count] = action
            self.cost_memory[self.count] = costs
            self.index_memory[self.count] = policy_index
            self.count += 1
        else:
            ind = self.count % self.num_new_data
            self.state_memory_tmp[ind] = state
            self.action_memory_tmp[ind] = action
            self.cost_memory_tmp[ind] = costs
            self.index_memory_tmp[ind] = policy_index
            if ind == self.num_new_data-1:
                self.state_memory[0: 2 * self.T - self.num_new_data] = self.state_memory[self.num_new_data:]
                self.state_memory[2 * self.T - self.num_new_data:] = self.state_memory_tmp
                self.action_memory[0: 2 * self.T - self.num_new_data] = self.action_memory[self.num_new_data:]
                self.action_memory[2 * self.T - self.num_new_data:] = self.action_memory_tmp
                self.cost_memory[0: 2 * self.T - self.num_new_data] = self.cost_memory[self.num_new_data:]
                self.cost_memory[2 * self.T - self.num_new_data:] = self.cost_memory_tmp
                self.index_memory[0: 2 * self.T - self.num_new_data] = self.index_memory[self.num_new_data:]
                self.index_memory[2 * self.T - self.num_new_data:] = self.index_memory_tmp
            self.count += 1

        if self.n_entries < 2 * self.T:
            self.n_entries += 1

    def take_experiences(self):

        return self.state_memory, self.action_memory, self.cost_memory, self.index_memory


class Environment_MIMO(object):
    """The environment class of the MIMO power allocation.
    For conciseness, we adopt the 'delay' Q/mu in the simulation."""
    def __init__(self, seed, Nt, UE_num): # 调用的时候需要输入这三个参数
        super(Environment_MIMO, self).__init__()  # 意思是先执行父类
        self.seed = seed
        self.seed_step = seed
        self.Nt = Nt
        self.UE_num = UE_num
        self.user_per_group = 1
        self.group_num = int(UE_num / self.user_per_group)
        self.state_dim = 2 * UE_num * Nt + 3 * (4 + 5 + 6 + 7 + 7) * 2
        # self.state_dim = 3 * (4 + 5 + 6 + 7) * 2
        self.action_dim = UE_num
        self.Np = 4
        self.power = 12

        np.random.seed(seed)
        PathGain_dB = np.random.uniform(-10, 10, self.group_num) # uniform 均匀分布
        self.PathGain = 10 ** (PathGain_dB / 10)
        alpha_power_group = np.zeros((self.group_num, self.Np))
        for group in range(self.group_num):
            tmp = np.random.exponential(scale=1, size=self.Np)     #指数分布，f(x)=exp(-x)
            alpha_power_group[group] = (tmp * self.PathGain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))

        array_reponse_group = np.zeros((self.group_num * self.Nt, self.Np)) + \
                              1j * np.zeros((self.group_num * self.Nt, self.Np))
        for group in range(self.group_num):
            A_tmp = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
            for i in range(self.Np):
                AoD = self.laprnd(mu=0, angular_spread=5)   #均值为0，角度扩展为5的拉普拉斯分布
                # AoD = self.laprnd(mu=0, angular_spread=1)
                # AoD = self.laprnd(mu = 0, angular_spread = 5 + i)
                A_tmp[:, i] = np.exp(1j * np.pi * np.sin(AoD) * np.arange(0, self.Nt))
            array_reponse_group[group * self.Nt: (group+1) * self.Nt] = A_tmp

        # for group in range(4):
        #     A_tmp = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
        #     A_tmp1 = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
        #     for i in range(self.Np):
        #         AoD = self.laprnd(mu=0, angular_spread=5)  # 均值为0，角度扩展为5的拉普拉斯分布
        #         AoD1 = AoD + 0.2
        #         # AoD = self.laprnd(mu=0, angular_spread=1)
        #         # AoD = self.laprnd(mu = 0, angular_spread = 5 + i)
        #         A_tmp[:, i] = np.exp(1j * np.pi * np.sin(AoD) * np.arange(0, self.Nt))
        #         A_tmp1[:, i] = np.exp(1j * np.pi * np.sin(AoD1) * np.arange(0, self.Nt))
        #     array_reponse_group[group * self.Nt: (group + 1) * self.Nt] = A_tmp
        #     array_reponse_group[(group+4) * self.Nt: ((group+4) + 1) * self.Nt] = A_tmp1


        self.array_response = np.tile(array_reponse_group, (self.user_per_group, 1))

        self.H_g = np.zeros((self.group_num, Nt)) + 1j * np.zeros((self.group_num, Nt))
        self.H = np.zeros((UE_num, Nt)) + 1j * np.zeros((UE_num, Nt))
        self.Dmax = np.array([4, 5, 6, 7, 7, 4, 5, 6, 7, 7])
        # self.Dmax = np.array([4, 5, 6, 7])
        self.D = np.zeros(np.sum(self.Dmax))    # 包长度
        self.Dbar = np.zeros(np.sum(self.Dmax))  # 包原长度
        self.Dlay = np.zeros(np.sum(self.Dmax))  # 包时延
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6

        self.packet_sum = 0
        self.packettrans_sum = 0

        #self.index = np.zeros(UE_num)




    def reset(self):
        # Reset the environment and return the initial state.
        np.random.seed(self.seed)
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.D = np.zeros(np.sum(self.Dmax))
        self.Dbar = np.zeros(np.sum(self.Dmax))
        self.Dlay = np.zeros(np.sum(self.Dmax))
        self.seed_step = 0
        self.packet_sum = 0
        self.packettrans_sum = 0
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        # self.state = np.hstack((np.zeros(np.sum(self.Dmax)), np.zeros(np.sum(self.Dmax)), np.zeros(np.sum(self.Dmax))))
        self.state = np.hstack((h_real, h_imag, np.zeros(np.sum(self.Dmax)), np.zeros(np.sum(self.Dmax)), np.zeros(np.sum(self.Dmax))))

        return self.state

    def step(self, action, step):
        # action contains weights ( power allocation and regularization factor).
        # return the next_state, reward, done = False, info.

        # if (step + 1) % 3000 == 0:
        #     self.packet_sum = 0
        #     self.packettrans_sum = 0
        np.random.seed(self.seed_step)
        #print(self.H[0,:])
        self.seed_step += 1
        action = action.reshape(-1)
        action[action <= 0] = 1e-6
        #power = action[0: self.UE_num] # python 数组序号是从0开始的
        #reg_factor = action[self.UE_num]
        weight = action[0: self.UE_num]
        weight = weight / np.sum(weight)
        D_me = np.zeros(self.UE_num)
        Dbar_me = np.zeros(self.UE_num)
        Dlay_me = np.zeros(self.UE_num)


        #reward = np.sum(power)
        #costs = self.D
        #info = {'cost_' + str(i): costs[i - 1] for i in range(1, self.UE_num + 1)}
        #info['cost'] = np.sum(costs)

        #MU scheduling
        # print('step: %d' % (step ))
        # if step == 2302:
        #     w=0
        start = np.zeros(self.UE_num + 1)
        for k in range(self.UE_num):
            start[k + 1] = np.sum(self.Dmax[0: k + 1])

        D_sum = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            D_sum[k] = np.sum(self.D[int(start[k]):int(start[k]) + self.Dmax[k]])

        WSR, UEset = MU_greedy(self.UE_num, self.H, weight, self.power, D_sum)
        r_dr = calRate(UEset, self.H, weight, self.UE_num, self.power)

        # userindex = np.argmax(r_d)
        #
        # r_dr = np.zeros(self.UE_num)
        # r_dr[userindex] = r_d[userindex] / weight[userindex]

        A_d = np.zeros(self.UE_num)
        # start = np.array([0, 4, 9, 15])
        # lambdak = np.zeros(self.UE_num)
        # for k in range(self.UE_num):
        #     lambdak[k] = np.random.uniform(Amin, Amax)



        if step > 0:
            for k in range(np.sum(self.Dmax)):
                if self.Dlay[k] > 0:
                    self.Dlay[k] = self.Dlay[k] + 1

        for k in range(self.UE_num):
            flag = np.random.choice([0, 1], p=[1 - p_a, p_a])
            if flag == 1:
                self.packet_sum = self.packet_sum + 1
                # A_d[k] = np.random.uniform(Amin, Amax)
                A_d[k] = np.random.poisson(lambdak[k])
                for m in range(self.Dmax[k]):
                    if (step-m) % self.Dmax[k] == 0:
                        self.Dlay[int(start[k]+m)] = 1
                        self.Dbar[int(start[k]+m)] = A_d[k]

        A_d_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            for m in range(self.Dmax[k]):
                if (step - m) % self.Dmax[k] == 0:
                    A_d_segment[int(start[k]+m)] = A_d[k]

        #A_d = np.random.uniform(0, 2, self.UE_num)    #到达速率


        r_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            my_list = np.argsort(self.Dlay[int(start[k]):int(start[k+1])])
            thresh = np.zeros(self.Dmax[k] + 2)
            thresh [self.Dmax[k] + 1] = 1e6
            for j in range(self.Dmax[k]):
                thresh_temp = 0
                for m in range(j+1):
                    thresh_temp = thresh_temp + self.D[my_list[self.Dmax[k]-m-1]+int(start[k])]
                thresh [j+1] = thresh_temp
            for l in range(self.Dmax[k]):
                if (r_dr[k] > thresh[l]) & (r_dr[k] <= thresh[l+1]):
                    temp = 0
                    for n in range(l):
                        r_segment[my_list[self.Dmax[k]-n-1]+int(start[k])] = self.D[my_list[self.Dmax[k]-n-1]+int(start[k])]
                        temp = temp + self.D[my_list[self.Dmax[k]-n-1]+int(start[k])]
                    r_segment[my_list[self.Dmax[k] - l - 1]+int(start[k])] = r_dr[k] - temp
            if r_dr[k] > thresh[self.Dmax[k]]:
                for n in range(self.Dmax[k]):
                    r_segment[my_list[self.Dmax[k] - n - 1]+int(start[k])] = self.D[my_list[self.Dmax[k] - n - 1]+int(start[k])]



        self.D = self.D + A_d_segment - r_segment
        self.D[self.D <= 0] = 0.0
        #self.D[self.D >= self.Dmax] = self.Dmax
        for k in range(self.UE_num):
            for i in range(self.Dmax[k]):
                if self.Dlay[i + int(start[k])] >= self.Dmax[k]:
                    self.D[i + int(start[k])] = 0
                    self.Dbar[i + int(start[k])] = 0
                    self.Dlay[i + int(start[k])] = 0
                    # self.packetdrop_sum = self.packetdrop_sum + 1

        for k in range(self.UE_num):
            my_list = np.argsort(self.Dlay[int(start[k]):int(start[k])+self.Dmax[k]])
            D_me[k] = self.D[my_list[self.Dmax[k]-1]+int(start[k])]
            Dbar_me[k] = self.Dbar[my_list[self.Dmax[k]-1]+int(start[k])]
            Dlay_me[k] = self.Dlay[my_list[self.Dmax[k]-1]+int(start[k])]


        rewardk = np.zeros(np.sum(self.Dmax))
        for k in range(np.sum(self.Dmax)):
            if self.D[k] <= 0 and r_segment[k] > 0:
                rewardk[k] = self.Dbar[k]
                self.packettrans_sum = self.packettrans_sum + 1
        reward = - np.sum(rewardk)

        for k in range(np.sum(self.Dmax)):
            if self.D[k] == 0:
                self.Dbar[k] = 0

        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        # self.state = np.hstack((self.D, self.Dbar, self.Dlay))
        self.state = np.hstack((h_real, h_imag, self.D, self.Dbar, self.Dlay))
        d = False

        return self.state, reward, d

    def step1(self, step):
        # return the reward
        # if (step + 1) % 3000 == 0:
        #     self.packet_sum = 0
        #     self.packettrans_sum = 0

        np.random.seed(self.seed_step)
        #print(self.H[0,:])
        self.seed_step += 1

        start = np.zeros(self.UE_num + 1)
        for k in range(self.UE_num):
            start[k + 1] = np.sum(self.Dmax[0: k + 1])

        D_sum = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            D_sum[k] = np.sum(self.D[int(start[k]):int(start[k])+self.Dmax[k]])

        if np.sum(D_sum) ==0:
            weight = np.ones(self.UE_num) / self.UE_num
        else:
            weight = D_sum
            weight = weight / np.sum(weight)
        weight[weight <= 0] = 1e-6

        #MU scheduling
        # print('step: %d' % (step))
        # if step==2302:
        #     o=1
        WSR, UEset = MU_greedy(self.UE_num, self.H, weight, self.power, D_sum)
        r_dr = calRate(UEset, self.H, weight, self.UE_num, self.power)

        # userindex = np.argmax(r_d)
        #
        # r_dr = np.zeros(self.UE_num)
        # r_dr[userindex] = r_d[userindex] / weight[userindex]

        A_d = np.zeros(self.UE_num)
        # start = np.array([0, 4, 9, 15])
        # lambdak = np.zeros(self.UE_num)
        # for k in range(self.UE_num):
        #     lambdak[k] = np.random.uniform(Amin, Amax)

        if step > 0:
            for k in range(np.sum(self.Dmax)):
                if self.Dlay[k] > 0:
                    self.Dlay[k] = self.Dlay[k] + 1

        for k in range(self.UE_num):
            flag = np.random.choice([0, 1], p=[1 - p_a, p_a])
            if flag == 1:
                self.packet_sum = self.packet_sum + 1
                # A_d[k] = np.random.uniform(Amin, Amax)
                A_d[k] = np.random.poisson(lambdak[k])
                for m in range(self.Dmax[k]):
                    if (step - m) % self.Dmax[k] == 0:
                        self.Dlay[int(start[k] + m)] = 1
                        self.Dbar[int(start[k] + m)] = A_d[k]

        A_d_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            for m in range(self.Dmax[k]):
                if (step - m) % self.Dmax[k] == 0:
                    A_d_segment[int(start[k] + m)] = A_d[k]

        # A_d = np.random.uniform(0, 2, self.UE_num)    #到达速率

        r_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            my_list = np.argsort(self.Dlay[int(start[k]):int(start[k + 1])])
            thresh = np.zeros(self.Dmax[k] + 2)
            thresh[self.Dmax[k] + 1] = 1e6
            for j in range(self.Dmax[k]):
                thresh_temp = 0
                for m in range(j + 1):
                    thresh_temp = thresh_temp + self.D[my_list[self.Dmax[k] - m - 1] + int(start[k])]
                thresh[j + 1] = thresh_temp
            for l in range(self.Dmax[k]):
                if (r_dr[k] > thresh[l]) & (r_dr[k] <= thresh[l + 1]):
                    temp = 0
                    for n in range(l):
                        r_segment[my_list[self.Dmax[k] - n - 1] + int(start[k])] = self.D[
                            my_list[self.Dmax[k] - n - 1] + int(start[k])]
                        temp = temp + self.D[my_list[self.Dmax[k] - n - 1] + int(start[k])]
                    r_segment[my_list[self.Dmax[k] - l - 1] + int(start[k])] = r_dr[k] - temp
            if r_dr[k] > thresh[self.Dmax[k]]:
                for n in range(self.Dmax[k]):
                    r_segment[my_list[self.Dmax[k] - n - 1] + int(start[k])] = self.D[
                        my_list[self.Dmax[k] - n - 1] + int(start[k])]

        self.D = self.D + A_d_segment - r_segment
        self.D[self.D <= 0] = 0.0
        # self.D[self.D >= self.Dmax] = self.Dmax

        for k in range(self.UE_num):
            for i in range(self.Dmax[k]):
                if self.Dlay[i + int(start[k])] >= self.Dmax[k]:
                    self.D[i + int(start[k])] = 0
                    self.Dbar[i + int(start[k])] = 0
                    self.Dlay[i + int(start[k])] = 0
                    # self.packetdrop_sum = self.packetdrop_sum + 1


        rewardk = np.zeros(np.sum(self.Dmax))
        for k in range(np.sum(self.Dmax)):
            if self.D[k] <= 0 and r_segment[k] > 0:
                rewardk[k] = self.Dbar[k]
                self.packettrans_sum = self.packettrans_sum + 1
        reward = - np.sum(rewardk)

        for k in range(np.sum(self.Dmax)):
            if self.D[k] == 0:
                self.Dbar[k] = 0


        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D, self.Dbar, self.Dlay))


        return reward, weight, self.state


    def step2(self, step):
        # if (step + 1) % 3000 == 0:
        #     self.packet_sum = 0
        #     self.packettrans_sum = 0

        # return the reward
        np.random.seed(self.seed_step)
        #print(self.H[0,:])
        self.seed_step += 1

        weight = np.random.ranf((self.UE_num))
        weight[weight <= 0] = 1e-6
        weight = weight / np.sum(weight)

        start = np.zeros(self.UE_num + 1)
        for k in range(self.UE_num):
            start[k + 1] = np.sum(self.Dmax[0: k + 1])

        D_sum = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            D_sum[k] = np.sum(self.D[int(start[k]):int(start[k]) + self.Dmax[k]])
        #MU scheduling
        WSR, UEset = MU_greedy(self.UE_num, self.H, weight, self.power,D_sum)
        r_dr = calRate(UEset, self.H, weight, self.UE_num, self.power)

        A_d = np.zeros(self.UE_num)
        # start = np.array([0, 4, 9, 15])
        # lambdak = np.zeros(self.UE_num)
        # for k in range(self.UE_num):
            # lambdak[k] = np.random.uniform(Amin, Amax)

        start = np.zeros(self.UE_num + 1)
        for k in range(self.UE_num):
            start[k + 1] = np.sum(self.Dmax[0: k + 1])

        if step > 0:
            for k in range(np.sum(self.Dmax)):
                if self.Dlay[k] > 0:
                    self.Dlay[k] = self.Dlay[k] + 1

        for k in range(self.UE_num):
            flag = np.random.choice([0, 1], p=[1 - p_a, p_a])
            if flag == 1:
                self.packet_sum = self.packet_sum + 1
                # A_d[k] = np.random.uniform(Amin, Amax)
                A_d[k] = np.random.poisson(lambdak[k])
                for m in range(self.Dmax[k]):
                    if (step - m) % self.Dmax[k] == 0:
                        self.Dlay[int(start[k] + m)] = 1
                        self.Dbar[int(start[k] + m)] = A_d[k]

        A_d_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            for m in range(self.Dmax[k]):
                if (step - m) % self.Dmax[k] == 0:
                    A_d_segment[int(start[k] + m)] = A_d[k]

        # A_d = np.random.uniform(0, 2, self.UE_num)    #到达速率

        r_segment = np.zeros(np.sum(self.Dmax))
        for k in range(self.UE_num):
            my_list = np.argsort(self.Dlay[int(start[k]):int(start[k + 1])])
            thresh = np.zeros(self.Dmax[k] + 2)
            thresh[self.Dmax[k] + 1] = 1e6
            for j in range(self.Dmax[k]):
                thresh_temp = 0
                for m in range(j + 1):
                    thresh_temp = thresh_temp + self.D[my_list[self.Dmax[k] - m - 1] + int(start[k])]
                thresh[j + 1] = thresh_temp
            for l in range(self.Dmax[k]):
                if (r_dr[k] > thresh[l]) & (r_dr[k] <= thresh[l + 1]):
                    temp = 0
                    for n in range(l):
                        r_segment[my_list[self.Dmax[k] - n - 1] + int(start[k])] = self.D[
                            my_list[self.Dmax[k] - n - 1] + int(start[k])]
                        temp = temp + self.D[my_list[self.Dmax[k] - n - 1] + int(start[k])]
                    r_segment[my_list[self.Dmax[k] - l - 1] + int(start[k])] = r_dr[k] - temp
            if r_dr[k] > thresh[self.Dmax[k]]:
                for n in range(self.Dmax[k]):
                    r_segment[my_list[self.Dmax[k] - n - 1] + int(start[k])] = self.D[
                        my_list[self.Dmax[k] - n - 1] + int(start[k])]

        self.D = self.D + A_d_segment - r_segment
        self.D[self.D <= 0] = 0.0
        # self.D[self.D >= self.Dmax] = self.Dmax
        for k in range(self.UE_num):
            for i in range(self.Dmax[k]):
                if self.Dlay[i + int(start[k])] >= self.Dmax[k]:
                    self.D[i + int(start[k])] = 0
                    self.Dbar[i + int(start[k])] = 0
                    self.Dlay[i + int(start[k])] = 0
                    # self.packetdrop_sum = self.packetdrop_sum + 1


        rewardk = np.zeros(np.sum(self.Dmax))
        for k in range(np.sum(self.Dmax)):
            if self.D[k] <= 0 and r_segment[k] > 0:
                rewardk[k] = self.Dbar[k]
                self.packettrans_sum = self.packettrans_sum + 1
        reward = - np.sum(rewardk)


        for k in range(np.sum(self.Dmax)):
            if self.D[k] == 0:
                self.Dbar[k] = 0

        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D, self.Dbar, self.Dlay))


        return reward


    def laprnd(self, mu, angular_spread):
        # generate random number of Laplacian distribution.
        b = angular_spread / np.sqrt(2)
        a = np.random.rand(1) - 0.5
        x = mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

        return x


class Environment_CLQR(object):
    """The environment class of the CLQR."""
    def __init__(self, seed, state_dim, action_dim):
        super(Environment_CLQR, self).__init__()
        self.seed = seed
        self.seed_step = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.s = np.zeros(state_dim)
        self.A = np.zeros((state_dim, state_dim))
        self.B = np.zeros((state_dim, action_dim))
        self.Q1 = np.zeros((state_dim, state_dim))
        self.R1 = np.zeros((action_dim, action_dim))
        self.Q2 = np.zeros((state_dim, state_dim))
        self.R2 = np.zeros((action_dim, action_dim))
        self.noise_mu = 1
        self.noise_std = 0.9

    def reset(self):
        # Reset the environment and return the initial state.
        np.random.seed(self.seed)
        self.A = np.random.randn(self.state_dim, self.state_dim)
        self.A = (self.A + self.A.T) / 30
        self.B = np.random.randn(self.state_dim, self.action_dim) / 3
        eig_values = np.random.rand(self.state_dim)
        S = np.diag(eig_values)
        U = self.generate_ortho_mat(dim=self.state_dim)
        self.Q1 = U @ S @ (U.T)
        E1 = np.random.randn(self.action_dim, self.action_dim)
        self.R1 = E1 @ (E1.T)
        np.random.seed(self.seed + 1996)
        C2 = np.random.exponential(1/3, size=(self.state_dim, self.state_dim))
        self.Q2 = C2 @ (C2.T)
        eig_values = np.random.rand(self.action_dim)
        S = np.diag(eig_values)
        U = self.generate_ortho_mat(dim=self.action_dim)
        self.R2 = U @ S @ (U.T)
        self.R2 = self.R2 @ (self.R2.T)

        self.s = np.random.randn(self.state_dim)

        return self.s

    def step(self, a):
        # return the next_state, reward, done = False, info.
        np.random.seed(self.seed_step)
        self.seed_step += 1
        a = a.reshape(-1)
        r = self.s.T @ self.Q1 @ self.s + a.T @ self.R1 @ a
        c = self.s.T @ self.Q2 @ self.s + a.T @ self.R2 @ a
        d = False
        info = {'cost': c}
        self.s = self.A @ self.s + self.B @ a + (self.noise_mu + self.noise_std * np.random.randn(self.state_dim))

        return self.s, r, d, info

    def generate_ortho_mat(self, dim):
        # generate orthogonal matrix
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat = np.eye(dim)
            mat[n - 1:, n - 1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H
