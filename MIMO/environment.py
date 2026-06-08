import numpy as np

# 多小区 CTDE MIMO 的默认信道配置；直达链路强、跨小区链路弱，用于控制初版场景难度。
MULTICELL_DIRECT_PATH_GAIN_DB_RANGE = (-10.0, 10.0)
MULTICELL_CROSS_PATH_GAIN_DB_RANGE = (-30.0, -10.0)
# 每条链路的多径数量；越大信道更丰富，但环境 step 计算更慢。
MULTICELL_PATH_COUNT = 4
# 每个时隙到达队列的均匀分布上界；与原单小区 MIMO 场景保持一致。
MULTICELL_ARRIVAL_UPPER = 2.0
# 队列延迟裁剪上界；用于避免状态无界增长。
MULTICELL_DMAX = 5.0
# 动作中的最小功率/正则项，避免 SINR 和 RZF 计算数值退化。
MULTICELL_ACTION_EPS = 1e-6

class Environment_MIMO(object):
    """The environment class of the MIMO power allocation.
    For conciseness, we adopt the 'delay' Q/mu in the simulation."""
    def __init__(self, seed, Nt, UE_num):
        super(Environment_MIMO, self).__init__()
        self.seed = seed
        self.Nt = Nt
        self.UE_num = UE_num
        self.user_per_group = 2
        self.group_num = int(UE_num / self.user_per_group)
        self.state_dim = 2 * UE_num * Nt + UE_num
        self.action_dim = UE_num + 1
        self.Np = 4
        self.rng = self._make_rng(self.seed)

        init_rng = self._make_rng(self.seed)
        PathGain_dB = init_rng.uniform(-10, 10, self.group_num)
        self.PathGain = 10 ** (PathGain_dB / 10)
        alpha_power_group = np.zeros((self.group_num, self.Np))
        for group in range(self.group_num):
            tmp = init_rng.exponential(scale=1, size=self.Np)
            alpha_power_group[group] = (tmp * self.PathGain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))

        array_reponse_group = np.zeros((self.group_num * self.Nt, self.Np)) + 1j * np.zeros((self.group_num * self.Nt, self.Np))
        for group in range(self.group_num):
            A_tmp = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
            for i in range(self.Np):
                AoD = self.laprnd(mu=0, angular_spread=5, rng=init_rng)
                A_tmp[:, i] = np.exp(1j * np.pi * np.sin(AoD) * np.arange(0, self.Nt))
            array_reponse_group[group * self.Nt: (group+1) * self.Nt] = A_tmp
        self.array_response = np.tile(array_reponse_group, (self.user_per_group, 1))

        self.H_g = np.zeros((self.group_num, Nt)) + 1j * np.zeros((self.group_num, Nt))
        self.H = np.zeros((UE_num, Nt)) + 1j * np.zeros((UE_num, Nt))
        self.D = np.zeros(UE_num)
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6
        self.Dmax = 5

    def _make_rng(self, seed):
        return np.random.RandomState(int(seed))

    def reset(self):
        # Reset the environment and return the initial state.
        self.rng = self._make_rng(self.seed)
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = (
                np.sqrt(alpha_power_g / 2) * self.rng.randn(self.Np)
                + 1j * np.sqrt(alpha_power_g / 2) * self.rng.randn(self.Np)
            )
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.D = np.zeros(self.UE_num)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D))
        return self.state

    def step(self, action):
        # action contains power allocation and regularization factor.
        # return the next_state, reward, done = False, info.
        action = action.reshape(-1)
        action[action <= 0] = 1e-6
        power = action[0: self.UE_num]
        reg_factor = action[self.UE_num]

        reward = np.sum(power)
        costs = self.D
        info = {'cost_' + str(i): costs[i - 1] for i in range(1, self.UE_num + 1)}
        info['cost'] = np.sum(costs)

        try:
            V = self.H.conjugate().T @ np.linalg.inv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))
        except:
            V = self.H.conjugate().T @ np.linalg.pinv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))

        norm_vector = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            norm_vector[k] = 1 / (np.linalg.norm(V[:, k]) + 1e-7)
        V_tilda = V @ np.diag(norm_vector)

        hv_tilda = self.H @ V_tilda
        r_d = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            module_squ = np.abs(hv_tilda[k]) ** 2
            numerator = power[k] * module_squ[k]
            module_squ[k] = 0
            dominator = np.sum(power * module_squ) + self.noise_power
            r_d[k] = np.log2(1 + numerator / dominator)
        A_d = self.rng.uniform(0, 2, self.UE_num)
        self.D = self.D + A_d - r_d
        self.D[self.D <= 0] = 0.0
        self.D[self.D >= self.Dmax] = self.Dmax
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = (
                np.sqrt(alpha_power_g / 2) * self.rng.randn(self.Np)
                + 1j * np.sqrt(alpha_power_g / 2) * self.rng.randn(self.Np)
            )
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D))
        d = False

        return self.state, reward, d, info

    def laprnd(self, mu, angular_spread, rng=None):
        # generate random number of Laplacian distribution.
        if rng is None:
            rng = self.rng
        b = angular_spread / np.sqrt(2)
        a = rng.rand(1) - 0.5
        x = mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

        return x



class Environment_MultiCellMIMO_CTDE(object):
    """Multi-cell MIMO power allocation environment for CTDE SLDAC.

    State is centralized: all complex channels and all delay queues.
    Action is decentralized by layout: each cell owns K powers plus one
    regularization factor, concatenated as [cell0, cell1, ...].
    """

    def __init__(
        self,
        seed,
        Nt,
        cell_num,
        user_per_cell,
        direct_path_gain_db_range=MULTICELL_DIRECT_PATH_GAIN_DB_RANGE,
        cross_path_gain_db_range=MULTICELL_CROSS_PATH_GAIN_DB_RANGE,
        path_count=MULTICELL_PATH_COUNT,
        arrival_upper=MULTICELL_ARRIVAL_UPPER,
        dmax=MULTICELL_DMAX,
    ):
        super(Environment_MultiCellMIMO_CTDE, self).__init__()
        self.seed = int(seed)
        self.Nt = int(Nt)
        self.cell_num = int(cell_num)
        self.num_cells = self.cell_num
        self.user_per_cell = int(user_per_cell)
        self.users_per_cell = self.user_per_cell
        if self.Nt <= 0 or self.cell_num <= 0 or self.user_per_cell <= 0:
            raise ValueError("Nt, cell_num and user_per_cell must be positive integers.")

        self.total_user_num = self.cell_num * self.user_per_cell
        self.UE_num = self.total_user_num
        self.Np = int(path_count)
        if self.Np <= 0:
            raise ValueError("path_count must be positive.")

        self.channel_size = self.cell_num * self.user_per_cell * self.cell_num * self.Nt
        self.state_dim = 2 * self.channel_size + self.total_user_num
        self.local_state_dim = 2 * self.user_per_cell * self.Nt + self.user_per_cell
        self.cell_action_dim = self.user_per_cell + 1
        self.action_dim = self.cell_num * self.cell_action_dim
        self.constraint_dim = self.total_user_num
        self.noise_power = 1e-6
        self.arrival_upper = float(arrival_upper)
        self.Dmax = float(dmax)
        self.action_eps = MULTICELL_ACTION_EPS

        self.rng = self._make_rng(self.seed)
        init_rng = self._make_rng(self.seed)
        self.PathGain = self._build_path_gain(
            init_rng,
            direct_path_gain_db_range,
            cross_path_gain_db_range,
        )
        self.alpha_power = self._build_alpha_power(init_rng)
        self.array_response = self._build_array_response(init_rng)

        self.H = np.zeros(
            (self.cell_num, self.user_per_cell, self.cell_num, self.Nt),
            dtype=np.complex128,
        )
        self.D = np.zeros((self.cell_num, self.user_per_cell), dtype=np.float64)
        self.state = np.zeros(self.state_dim, dtype=np.float64)

    def _make_rng(self, seed):
        return np.random.RandomState(int(seed))

    def _build_path_gain(self, rng, direct_range, cross_range):
        path_gain_db = np.zeros((self.cell_num, self.user_per_cell, self.cell_num), dtype=np.float64)
        direct_low, direct_high = direct_range
        cross_low, cross_high = cross_range
        for rx_cell in range(self.cell_num):
            for user in range(self.user_per_cell):
                for tx_cell in range(self.cell_num):
                    if rx_cell == tx_cell:
                        path_gain_db[rx_cell, user, tx_cell] = rng.uniform(direct_low, direct_high)
                    else:
                        path_gain_db[rx_cell, user, tx_cell] = rng.uniform(cross_low, cross_high)
        return 10 ** (path_gain_db / 10.0)

    def _build_alpha_power(self, rng):
        alpha_power = np.zeros(
            (self.cell_num, self.user_per_cell, self.cell_num, self.Np),
            dtype=np.float64,
        )
        for rx_cell in range(self.cell_num):
            for user in range(self.user_per_cell):
                for tx_cell in range(self.cell_num):
                    tmp = rng.exponential(scale=1.0, size=self.Np)
                    alpha_power[rx_cell, user, tx_cell] = (
                        tmp * self.PathGain[rx_cell, user, tx_cell]
                    ) / np.sum(tmp)
        return alpha_power

    def _build_array_response(self, rng):
        array_response = np.zeros(
            (self.cell_num, self.user_per_cell, self.cell_num, self.Nt, self.Np),
            dtype=np.complex128,
        )
        for rx_cell in range(self.cell_num):
            for user in range(self.user_per_cell):
                for tx_cell in range(self.cell_num):
                    for path_idx in range(self.Np):
                        aod = self.laprnd(mu=0, angular_spread=5, rng=rng)
                        array_response[rx_cell, user, tx_cell, :, path_idx] = np.exp(
                            1j * np.pi * np.sin(aod) * np.arange(0, self.Nt)
                        )
        return array_response

    def _refresh_channels(self):
        for rx_cell in range(self.cell_num):
            for user in range(self.user_per_cell):
                for tx_cell in range(self.cell_num):
                    alpha_power = self.alpha_power[rx_cell, user, tx_cell]
                    array_response = self.array_response[rx_cell, user, tx_cell]
                    alpha = (
                        np.sqrt(alpha_power / 2.0) * self.rng.randn(self.Np)
                        + 1j * np.sqrt(alpha_power / 2.0) * self.rng.randn(self.Np)
                    )
                    self.H[rx_cell, user, tx_cell] = array_response @ alpha

    def _compose_state(self):
        h_real = np.real(self.H).reshape(-1)
        h_imag = np.imag(self.H).reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D.reshape(-1))).astype(np.float64, copy=False)
        return self.state

    def reset(self):
        self.rng = self._make_rng(self.seed)
        self._refresh_channels()
        self.D = np.zeros((self.cell_num, self.user_per_cell), dtype=np.float64)
        return self._compose_state()

    def _cell_beamformer(self, tx_cell, power, reg_factor):
        _ = power
        h_direct = self.H[tx_cell, :, tx_cell, :]
        eye = np.eye(self.user_per_cell)
        try:
            v = h_direct.conjugate().T @ np.linalg.inv(
                h_direct @ h_direct.conjugate().T + reg_factor * eye
            )
        except np.linalg.LinAlgError:
            v = h_direct.conjugate().T @ np.linalg.pinv(
                h_direct @ h_direct.conjugate().T + reg_factor * eye
            )
        norm_vector = np.zeros(self.user_per_cell, dtype=np.float64)
        for user in range(self.user_per_cell):
            norm_vector[user] = 1.0 / (np.linalg.norm(v[:, user]) + 1e-7)
        return v @ np.diag(norm_vector)

    def _decode_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(
                "action_dim mismatch: expected {0}, got {1}".format(self.action_dim, action.size)
            )
        action = np.maximum(action, self.action_eps)
        cell_action = action.reshape(self.cell_num, self.cell_action_dim)
        power = cell_action[:, : self.user_per_cell]
        reg_factor = np.maximum(cell_action[:, self.user_per_cell], self.action_eps)
        return power, reg_factor

    def _compute_rates(self, power, reg_factor):
        beamformers = np.zeros(
            (self.cell_num, self.Nt, self.user_per_cell),
            dtype=np.complex128,
        )
        for tx_cell in range(self.cell_num):
            beamformers[tx_cell] = self._cell_beamformer(tx_cell, power[tx_cell], reg_factor[tx_cell])

        rates = np.zeros((self.cell_num, self.user_per_cell), dtype=np.float64)
        for rx_cell in range(self.cell_num):
            for user in range(self.user_per_cell):
                desired_channel = self.H[rx_cell, user, rx_cell]
                desired_gain = np.abs(desired_channel @ beamformers[rx_cell, :, user]) ** 2
                numerator = power[rx_cell, user] * desired_gain

                interference = 0.0
                for tx_cell in range(self.cell_num):
                    link_channel = self.H[rx_cell, user, tx_cell]
                    gains = np.abs(link_channel @ beamformers[tx_cell]) ** 2
                    for stream in range(self.user_per_cell):
                        if tx_cell == rx_cell and stream == user:
                            continue
                        interference += power[tx_cell, stream] * gains[stream]
                rates[rx_cell, user] = np.log2(1.0 + numerator / (interference + self.noise_power))
        return rates

    def step(self, action):
        power, reg_factor = self._decode_action(action)

        # 环境接口沿用历史命名：reward 实际作为算法内部待最小化的 objective cost。
        reward = float(np.sum(power))
        costs = self.D.reshape(-1).copy()
        info = {"cost_" + str(i): costs[i - 1] for i in range(1, self.total_user_num + 1)}
        info["cost"] = float(np.sum(costs))
        info["cell_cost"] = np.sum(self.D, axis=1).astype(np.float64, copy=False)

        rates = self._compute_rates(power, reg_factor)
        arrivals = self.rng.uniform(0.0, self.arrival_upper, size=(self.cell_num, self.user_per_cell))
        self.D = np.clip(self.D + arrivals - rates, 0.0, self.Dmax)
        self._refresh_channels()
        next_state = self._compose_state()
        done = False
        return next_state, reward, done, info

    def local_observation(self, cell_index):
        cell = int(cell_index)
        if cell < 0 or cell >= self.cell_num:
            raise ValueError("cell_index out of range: {0}".format(cell_index))
        h_direct = self.H[cell, :, cell, :]
        return np.hstack(
            (
                np.real(h_direct).reshape(-1),
                np.imag(h_direct).reshape(-1),
                self.D[cell].reshape(-1),
            )
        ).astype(np.float64, copy=False)

    def laprnd(self, mu, angular_spread, rng=None):
        if rng is None:
            rng = self.rng
        b = angular_spread / np.sqrt(2)
        a = rng.rand() - 0.5
        return float(mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a)))




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
