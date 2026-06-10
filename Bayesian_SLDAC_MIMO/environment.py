import numpy as np


def project_mimo_action(action):
    projected = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    projected[projected <= 0] = 1e-6
    return projected


class Environment_MIMO(object):
    """Legacy MIMO power-allocation environment copied from SLDAC_code/MIMO1."""

    def __init__(self, seed, Nt, UE_num):
        super(Environment_MIMO, self).__init__()
        self.seed = int(seed)
        self.seed_step = int(seed)
        self.Nt = int(Nt)
        self.UE_num = int(UE_num)
        self.user_per_group = 2
        self.group_num = int(UE_num / self.user_per_group)
        self.state_dim = 2 * self.UE_num * self.Nt + self.UE_num
        self.action_dim = self.UE_num + 1
        self.Np = 4

        np.random.seed(self.seed)
        path_gain_db = np.random.uniform(-10, 10, self.group_num)
        self.PathGain = 10 ** (path_gain_db / 10)
        alpha_power_group = np.zeros((self.group_num, self.Np))
        for group in range(self.group_num):
            tmp = np.random.exponential(scale=1, size=self.Np)
            alpha_power_group[group] = (tmp * self.PathGain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))

        array_response_group = np.zeros((self.group_num * self.Nt, self.Np), dtype=np.complex128)
        for group in range(self.group_num):
            a_tmp = np.zeros((self.Nt, self.Np), dtype=np.complex128)
            for idx in range(self.Np):
                aod = self.laprnd(mu=0, angular_spread=5)
                a_tmp[:, idx] = np.exp(1j * np.pi * np.sin(aod) * np.arange(0, self.Nt))
            array_response_group[group * self.Nt : (group + 1) * self.Nt] = a_tmp
        self.array_response = np.tile(array_response_group, (self.user_per_group, 1))

        self.H_g = np.zeros((self.group_num, self.Nt), dtype=np.complex128)
        self.H = np.zeros((self.UE_num, self.Nt), dtype=np.complex128)
        self.D = np.zeros(self.UE_num)
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6
        self.Dmax = 5

    def reset(self):
        np.random.seed(self.seed)
        for group in range(self.group_num):
            alpha_power_g = self.alpha_power[group]
            a_g = self.array_response[group * self.Nt : (group + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + 1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[group] = a_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.D = np.zeros(self.UE_num)
        self.state = np.hstack((np.real(self.H).reshape(-1), np.imag(self.H).reshape(-1), self.D))
        return self.state

    def step(self, action):
        np.random.seed(self.seed_step)
        self.seed_step += 1
        action = project_mimo_action(action)
        power = action[0 : self.UE_num]
        reg_factor = action[self.UE_num]

        reward = np.sum(power)
        costs = self.D
        info = {"cost_" + str(idx): costs[idx - 1] for idx in range(1, self.UE_num + 1)}
        info["cost"] = np.sum(costs)

        try:
            v_mat = self.H.conjugate().T @ np.linalg.inv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))
        except Exception:
            v_mat = self.H.conjugate().T @ np.linalg.pinv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))

        norm_vector = np.zeros(self.UE_num)
        for user_idx in range(self.UE_num):
            norm_vector[user_idx] = 1 / (np.linalg.norm(v_mat[:, user_idx]) + 1e-7)
        v_tilda = v_mat @ np.diag(norm_vector)

        hv_tilda = self.H @ v_tilda
        r_d = np.zeros(self.UE_num)
        for user_idx in range(self.UE_num):
            module_squ = np.abs(hv_tilda[user_idx]) ** 2
            numerator = power[user_idx] * module_squ[user_idx]
            module_squ[user_idx] = 0
            denominator = np.sum(power * module_squ) + self.noise_power
            r_d[user_idx] = np.log2(1 + numerator / denominator)

        a_d = np.random.uniform(0, 2, self.UE_num)
        self.D = self.D + a_d - r_d
        self.D[self.D <= 0] = 0.0
        self.D[self.D >= self.Dmax] = self.Dmax

        for group in range(self.group_num):
            alpha_power_g = self.alpha_power[group]
            a_g = self.array_response[group * self.Nt : (group + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + 1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[group] = a_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.state = np.hstack((np.real(self.H).reshape(-1), np.imag(self.H).reshape(-1), self.D))
        return self.state, reward, False, info

    def laprnd(self, mu, angular_spread):
        b = angular_spread / np.sqrt(2)
        a = np.random.rand(1) - 0.5
        return mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

