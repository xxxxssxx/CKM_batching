from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import deque
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CKM sampler
# ============================================================
@dataclass
class CKM:
    """
    CKM backed by a real channel gain map.

    gain_map stores mu_dB(q) in dB, with values typically in [-250, -50].
    Building pixels are assumed to be mapped to the minimum value (or below
    a chosen threshold), and local building ratio is used to determine the
    fluctuation strength through CV^2(q) = sigma^2(q) / mu(q)^2.
    """
    gain_map: np.ndarray                              # shape [H, W], values in dB
    xlim: Tuple[float, float] = (0.0, 400.0)
    ylim: Tuple[float, float] = (0.0, 400.0)
    
    # 额外增益（发射接收端阵列以及波束成形）
    additional_gain: float = 33 # in dB

    # building / valid-region control
    building_db_threshold: float = -150.0            # <= threshold treated as building
    sample_only_non_building: bool = True

    # local neighborhood for complexity estimation
    window_size: int = 21

    # CV^2(q) = sigma^2(q) / mu(q)^2
    cv2_min: float = 0.05
    cv2_max: float = 0.50

    # numerical floor for mu in linear scale
    mu_floor_linear: float = 1e-30

    # cached masks / maps
    building_mask: np.ndarray = field(init=False)
    valid_mask: np.ndarray = field(init=False)
    cv2_map: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.gain_map = np.asarray(self.gain_map, dtype=np.float64)
        if self.gain_map.ndim != 2:
            raise ValueError("gain_map must be a 2D array.")

        self.H, self.W = self.gain_map.shape

        self.building_mask = self.gain_map <= self.building_db_threshold
        self.valid_mask = ~self.building_mask

        if self.sample_only_non_building and not np.any(self.valid_mask):
            raise ValueError("No valid non-building pixels available for user sampling.")
        if self.sample_only_non_building:
            self.valid_indices = np.argwhere(self.valid_mask)
        else:
            self.valid_indices = None

        self.cv2_map = self._build_cv2_map()

    # ------------------------------------------------------------
    # coordinate conversion
    # ------------------------------------------------------------
    def _xy_to_rc(self, q: Tuple[float, float]) -> Tuple[int, int]:
        """
        Map physical coordinate q=(x,y) to image index (row, col).

        Assumption:
          - x increases from left to right
          - y increases from bottom to top in physical coordinates
          - image row index increases from top to bottom
        """
        x, y = q
        x0, x1 = self.xlim
        y0, y1 = self.ylim

        x_norm = (x - x0) / max(x1 - x0, 1e-12)
        y_norm = (y - y0) / max(y1 - y0, 1e-12)

        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        col = int(round(x_norm * (self.W - 1)))
        row = int(round((1.0 - y_norm) * (self.H - 1)))  # flip y-axis

        row = int(np.clip(row, 0, self.H - 1))
        col = int(np.clip(col, 0, self.W - 1))
        return row, col

    def _rc_to_xy(self, row: int, col: int) -> Tuple[float, float]:
        x0, x1 = self.xlim
        y0, y1 = self.ylim

        x = x0 + (col / max(self.W - 1, 1)) * (x1 - x0)
        y = y0 + (1.0 - row / max(self.H - 1, 1)) * (y1 - y0)
        return float(x), float(y)

    # ------------------------------------------------------------
    # local complexity -> CV^2 map
    # ------------------------------------------------------------
    def _build_cv2_map(self) -> np.ndarray:
        """
        Build CV^2(q) map based on local building ratio.
        """
        pad = self.window_size // 2
        bld = self.building_mask.astype(np.float64)
        padded = np.pad(bld, ((pad, pad), (pad, pad)), mode="edge")

        cv2_map = np.zeros_like(self.gain_map, dtype=np.float64)

        for r in range(self.H):
            for c in range(self.W):
                local_ratio = np.sum(
                    padded[r:r + 2 * pad + 1, c:c + 2 * pad + 1]
                ) / ((2 * pad + 1) ** 2)
                cv2_map[r, c] = self.cv2_min + (self.cv2_max - self.cv2_min) * local_ratio
        return cv2_map

    # ------------------------------------------------------------
    # public API
    # ------------------------------------------------------------
    def lookup(self, q: Tuple[float, float]) -> Tuple[float, float]:
        """
        Return (mu, sigma2) in linear scale.
        """
        row, col = self._xy_to_rc(q)

        mu_db = float(self.gain_map[row, col])
        mu_db = mu_db + self.additional_gain  # 加上波束赋型等增益
        mu = 10.0 ** (mu_db / 10.0)
        mu = max(mu, self.mu_floor_linear)

        cv2 = float(self.cv2_map[row, col])
        sigma2 = cv2 * (mu ** 2)
        return float(mu), float(sigma2)

    def sample_user(self, rng: np.random.Generator) -> Tuple[Tuple[float, float], float, float]:
        """
        Sample a user location and return (q, mu, sigma2).
        """
        if self.sample_only_non_building:
            idx = int(rng.integers(0, len(self.valid_indices)))
            row, col = self.valid_indices[idx]
            q = self._rc_to_xy(int(row), int(col))
        else:
            row = int(rng.integers(0, self.H))
            col = int(rng.integers(0, self.W))
            q = self._rc_to_xy(row, col)

        mu, sigma2 = self.lookup(q)
        return q, mu, sigma2
    
    def visualize_maps(self) -> None:
        """
        Visualize gain_map, cv2_map, and sigma2_map.
        - gain_map: mu_dB(q), in dB
        - cv2_map: CV^2(q) = sigma^2(q) / mu(q)^2
        - sigma2_map: variance in linear scale; displayed in dB
        """

        gain_map = self.gain_map.astype(np.float64)
        cv2_map = self.cv2_map.astype(np.float64)

        mu_map = 10.0 ** (gain_map / 10.0)
        mu_map = np.maximum(mu_map, self.mu_floor_linear)
        sigma2_map = cv2_map * (mu_map ** 2)

        sigma2_map_dB = 10 * np.log10(np.maximum(sigma2_map, 1e-300))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(gain_map, cmap="gray", vmin=np.min(gain_map), vmax=np.max(gain_map))
        axes[0].set_title("gain_map_dB")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="dB")

        im1 = axes[1].imshow(cv2_map, cmap="viridis", vmin=self.cv2_min, vmax=self.cv2_max)
        axes[1].set_title("cv2_map")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="CV^2")

        im2 = axes[2].imshow(sigma2_map_dB, cmap="magma")
        axes[2].set_title("sigma2_map_dB")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="log10(linear)")

        plt.tight_layout()
        plt.show()
        

# ============================================================
# Prefill latency / power models
# ============================================================
@dataclass
class PrefillModel:
    """
    Prefill model from the note.

    Latency:
        tau_pf(B, L) ~= beta0 + beta1 * B * L + beta2 * B * L^2
    If prompt length L is fixed, this reduces to
        tau_pf(B) ~= beta0 + gamma1 * B,
    where gamma1 = beta1 * L + beta2 * L^2.

    GPU power:
        P_pf(B) = P_min + DeltaP / (1 + exp(-k1 * (log2(B) - k2)))
    with P_pf(0) = 0 by convention because no batch is launched.
    """
    beta0: float = 0.010
    gamma1: float = 0.004
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    N_token: Optional[float] = None  # token数

    p_min: float = 80.0
    delta_p: float = 140.0
    k1: float = 3.0
    k2: float = 1.5

    def tau(self, batch_size: int) -> float:
        B = int(batch_size)
        if B <= 0:
            return 0.0
        if self.beta1 is not None and self.beta2 is not None and self.N_token is not None:
            L = float(self.N_token)
            return float(self.beta0 + self.beta1 * B * L + self.beta2 * B * (L ** 2))
        return float(self.beta0 + self.gamma1 * B)

    def power(self, batch_size: int) -> float:
        B = int(batch_size)
        if B <= 0:
            return 0.0
        z = math.log2(max(B, 1))
        return float(self.p_min + self.delta_p / (1.0 + math.exp(-self.k1 * (z - self.k2))))

    def energy(self, batch_size: int) -> float:
        B = int(batch_size)
        if B <= 0:
            return 0.0
        return float(self.power(B) * self.tau(B))


# ============================================================
# Internal records
# ============================================================
@dataclass
class Request:
    req_id: int
    q: Tuple[float, float]
    mu: float
    sigma2: float
    t_arr: float

    # uplink-side lifecycle
    m: int = 0
    gamma_acc: float = 0.0
    e_ul: float = 0.0  # 该请求的上行传输的总能耗
    t_ul: Optional[float] = None
    t_cl: Optional[float] = None

    # cloud-side lifecycle
    t_grab: Optional[float] = None
    t_fin: Optional[float] = None
    batch_size: Optional[int] = None
    e_pf_share: float = 0.0  # 一个batch的prefill能量分摊到一条请求上的能量

    # currently running uplink round
    current_power: float = 0.0
    current_gain: float = 0.0
    ul_start_t: Optional[float] = None
    ul_end_t: Optional[float] = None
    # 当请求在传输队列中躺着，in_ul_flight和need_tx_start可以都是False。不会出现同时为True。
    in_ul_flight: bool = False  # 正在上传
    need_tx_start: bool = False  # 需要开启上传（不区分是新传输还是重传）

    # bookkeeping flags
    completed_ul: bool = False
    completed_pf: bool = False  # 是否完成prefilling
    counted_pf_completion: bool = False  # 完成prefilling后是否结算（计算指标）当前建模下，这两个量的值总相同


@dataclass
class BatchRecord:
    '''
    batch 信息
    '''
    batch_id: int
    gpu_id: int
    request_ids: List[int]
    batch_size: int
    start_t: float
    end_t: float
    tau_pf: float
    energy: float
    counted_completion: bool = False  # 该batch是否已结算


@dataclass
class GPUState:
    '''
    GPU 状态
    '''
    gpu_id: int
    current_batch_id: Optional[int] = None
    is_busy: bool = False
    rem_time: float = 0.0


# ============================================================
# Environment
# ============================================================
@dataclass
class CKMBatchingEnv:
    # horizon and time discretization
    T: float = 5.0
    delta_t: float = 0.01

    # traffic / radio / compute resources
    lam: float = 30.0               # requests / second
    lam_amp: float = 0.0            # A
    lam_omega: float = 0.0          # omega
    lam_phase: float = 0.0          # optional phase
    K: int = 2                      # maximum parallel uplink users
    N: int = 1                      # number of GPUs
    Bmax: int = 4                   # maximum batch size per GPU decision

    # uplink physical parameters
    L: float = 4096.0               # coded block length, bits
    R: float = 2.0e5                # transmit rate, bit/s (fixed in this note)
    N0: float = 1e-20               # noise PSD, W/Hz
    Bw: float = 1e6                 # Hz
    p_max: float = 1.0              # W

    # SLA and reward
    tau_SLA: float = 0.15
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    # models
    ckm: CKM | None = None 
    prefill_model: PrefillModel = field(default_factory=PrefillModel)

    # observation truncation / padding for PPO compatibility
    q_ul_obs_cap: int = 4
    q_cl_obs_cap: int = 8

    # random seed
    seed_value: int = 0

    def __post_init__(self) -> None:
        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive.")
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.K <= 0 or self.N <= 0 or self.Bmax <= 0:
            raise ValueError("K, N, and Bmax must be positive integers.")
        if self.R <= 0 or self.L <= 0 or self.N0 <= 0 or self.Bw <= 0:
            raise ValueError("L, R, N0, and Bw must be positive.")
        if self.lam < 0:
            raise ValueError("lam must be non-negative.")
        if self.lam_amp > self.lam:
            raise ValueError("lam_amp must be smaller than lam.")
        if self.p_max <= 0:
            raise ValueError("p_max must be positive.")
        assert self.ckm is not None, "CKM must be passed in"

        
        self.Nstep = int(round(self.T / self.delta_t))
        self.act_dim = self.K + self.N
        self.act_low = np.zeros(self.act_dim, dtype=np.float32)
        self.act_high = np.concatenate(
            [
                np.full(self.K, self.p_max, dtype=np.float32),
                np.full(self.N, self.Bmax, dtype=np.float32),
            ]
        )
        self.obs_dim = self._compute_obs_dim()
        self.reset(seed=self.seed_value)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # 不传seed的时候，只重置 episode 状态，不用seed重置 RNG。这样新 episode 会继续从当前seed对应的随机流往后采样，因此不会重复。
        # 传seed的时候，将环境随机流重置到seed对应的确定起点。
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

        self.step_idx = 0
        self.t = 0.0
        self.req_counter = 0
        self.batch_counter = 0

        self.requests: Dict[int, Request] = {}  # 这个字典存储系统运行周期内所有出现过的请求
        self.batches: Dict[int, BatchRecord] = {}  # 这个字典存储系统运行周期内所有出现过的 batch
        self.gpus: List[GPUState] = [GPUState(gpu_id=n) for n in range(self.N)]

        # Qul和Qcl两个队列都只放请求id，字典Aul放{请求id：请求}
        self.Qul: deque[int] = deque()
        self.Aul: Dict[int, Request] = {}
        self.Qcl: List[int] = []

        self.finished_request_ids: List[int] = []
        self.total_gpu_energy = 0.0
        self.total_ul_energy = 0.0
        self.total_started_batches = 0
        self.total_arrivals = 0

        self.current_events: Dict[str, Any] = {
            "Ut": [],
            "Ct": [],
            "Gt": [],
            "Eul_t": 0.0,
            "Epf_t": 0.0,
            "NSLA_t": 0,
        }
        self.tx_start_reqs: List[int] = []
        self.idle_gpu_ids: List[int] = []

        self._advance_predecision()
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        
        # a_t
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.act_dim:
            raise ValueError(f"action dimension mismatch: got {action.size}, expected {self.act_dim}")

        # reward uses events observed at the current decision epoch + newly started batches
        # 这俩在_advance_predecision系统演进的时候计算过了
        Eul_t = float(self.current_events["Eul_t"])
        NSLA_t = int(self.current_events["NSLA_t"])

        p_action = np.clip(action[: self.K], 0.0, self.p_max)
        b_action = np.clip(action[self.K :], 0.0, float(self.Bmax))

        # 9. actions for uplink rounds that must be started at current t
        active_sorted = [req for _, req in sorted(self.Aul.items(), key=lambda kv: (kv[1].t_arr, kv[0]))]
        launched_uplink: List[Dict[str, Any]] = []
        for idx, req in enumerate(active_sorted):
            if not req.need_tx_start:
                continue
            p_i = float(p_action[idx])
            self._start_uplink_round(req, p_i)
            launched_uplink.append(
                {
                    "req_id": req.req_id,
                    "m": req.m,
                    "power": p_i,
                    "ul_end_t": req.ul_end_t,
                    "slot_idx": idx,
                }
            )
        
        # 10. start new cloud batches on idle GPUs 计算奖励里prefill能耗的部分
        launched_batches: List[Dict[str, Any]] = []
        Epf_t = 0.0

        feasible_batch_alloc = self._project_idle_gpu_batch_sizes(b_action)

        for gpu_id in self.idle_gpu_ids:
            Bn = feasible_batch_alloc[gpu_id]
            if Bn <= 0:
                continue
            batch_info = self._start_batch(gpu_id=gpu_id, batch_size=Bn)
            launched_batches.append(batch_info)
            Epf_t += float(batch_info["energy"])
            
        self.current_events["Gt"] = [x["gpu_id"] for x in launched_batches]
        self.current_events["Epf_t"] = float(Epf_t)

        # r_t
        reward = float(self.alpha * NSLA_t - self.beta * Eul_t - self.gamma * Epf_t)

        info = {
            "t": float(self.t),
            "step_idx": int(self.step_idx),
            "reward_terms": {
                "NSLA_t": int(NSLA_t),
                "Eul_t": float(Eul_t),
                "Epf_t": float(Epf_t),
            },
            "decision_entities": {
                "tx_start_req_ids": list(self.tx_start_reqs),
                "idle_gpu_ids": list(self.idle_gpu_ids),
            },
            "launched_uplink": launched_uplink,
            "launched_batches": launched_batches,
            "queue_lengths": {
                "Qul": len(self.Qul),
                "Aul": len(self.Aul),
                "Qcl": len(self.Qcl),
            },
            "events_at_t": {
                "Ut": list(self.current_events["Ut"]),
                "Ct": list(self.current_events["Ct"]),
                "Gt": list(self.current_events["Gt"]),
            },
        }

        self.step_idx += 1
        done = bool(self.step_idx >= self.Nstep)
        if done:
            self.t = min(self.step_idx * self.delta_t, self.T)
            final_info = self.get_metrics()
            info["episode_metrics"] = final_info
            return self._obs(), reward, True, info

        self.t = self.step_idx * self.delta_t
        # s_t -> s_{t+1}，系统演化
        self._advance_predecision()
        return self._obs(), reward, False, info

    def _project_idle_gpu_batch_sizes(
        self,
        b_action: np.ndarray,
    ) -> Dict[int, int]:
        """
        For current idle GPUs, convert raw batch actions into a feasible integer
        batch-size allocation under the joint constraint:
            sum_n B_n <= len(self.Qcl)

        Rule:
        1) Round + clip each idle GPU proposal into [0, Bmax];
        2) If total proposal <= |Qcl|, keep them unchanged;
        3) Otherwise, scale them proportionally to |Qcl| and use
        floor + largest remainder to get integer allocations.
        """
        idle_ids = list(self.idle_gpu_ids)
        alloc: Dict[int, int] = {gpu_id: 0 for gpu_id in idle_ids}

        if len(idle_ids) == 0:
            # 如果没有空闲的GPU，那就不用处理，因为反正不会被用到
            return alloc

        Q = len(self.Qcl)
        if Q <= 0:
            return alloc

        # round + per-GPU clip
        raw = np.array(
            [
                max(0, min(int(np.rint(float(b_action[gpu_id]))), self.Bmax))  # agent输出的batch size动作是本来就是整数，但这里还是int一下
                for gpu_id in idle_ids
            ],
            dtype=np.int64,
        )  # b_action四舍五入得到raw

        S = int(raw.sum())
        if S <= Q:
            # 如果总数没到|Qcl|，就不变
            for gpu_id, bn in zip(idle_ids, raw.tolist()):
                alloc[gpu_id] = int(bn)
            return alloc

        # proportional projection onto total budget Q
        x = raw.astype(np.float64) * (float(Q) / float(S))  # 各决策缩放后的值（非整数）
        base = np.floor(x).astype(np.int64)  # 各决策缩放后的值向下取整
        remainder = x - base  # 各决策缩放后的值的小数部分（舍入误差）

        # distribute remaining slots by largest remainder
        R = int(Q - base.sum())  # 各决策值向下取整后，还剩多少请求没分配 R < 闲置GPU个数，且R是整数，因为Q和base.sum都是整数
        if R > 0:
            order = np.argsort(-remainder)  # 按舍入误差降序排序。谁舍入误差大，就先给谁分配。
            for idx in order[:R]:
                base[idx] += 1

        for gpu_id, bn in zip(idle_ids, base.tolist()):
            alloc[gpu_id] = int(bn)

        return alloc

    def _served_request_arrays(self) -> Dict[str, np.ndarray]:
        served = [self.requests[rid] for rid in self.finished_request_ids]

        if len(served) == 0:
            empty = np.array([], dtype=np.float64)
            return {
                "ttft": empty,
                "tx_side_time": empty,       # t_cl - t_arr
                "compute_side_time": empty,  # t_fin - t_cl
                "eul": empty,
                "epf_share": empty,
            }

        ttft = np.array([req.t_fin - req.t_arr for req in served], dtype=np.float64)
        tx_side_time = np.array([req.t_cl - req.t_arr for req in served], dtype=np.float64)
        compute_side_time = np.array([req.t_fin - req.t_cl for req in served], dtype=np.float64)
        eul = np.array([req.e_ul for req in served], dtype=np.float64)
        epf_share = np.array([req.e_pf_share for req in served], dtype=np.float64)

        return {
            "ttft": ttft,
            "tx_side_time": tx_side_time,
            "compute_side_time": compute_side_time,
            "eul": eul,
            "epf_share": epf_share,
        }

    @staticmethod
    def _tail_mean_by_ttft(
        ttft: np.ndarray,
        value: np.ndarray,
        tail_ratio: float,
    ) -> float:
        n = ttft.size
        if n == 0:
            return 0.0

        k = max(1, int(np.ceil(tail_ratio * n)))
        idx = np.argpartition(ttft, n - k)[-k:]   # TTFT 最大的 top-k 请求
        return float(np.mean(value[idx]))

    def get_metrics(self) -> Dict[str, Any]:
        
        # 各个请求的指标
        arr = self._served_request_arrays()
        ttfts = arr["ttft"]
        tx_side_time = arr["tx_side_time"]  # 传输侧花的时间，包括等待和传输
        compute_side_time = arr["compute_side_time"]  # 计算侧花的时间，包括等待和prefill
        eul = arr["eul"]
        epf_share = arr["epf_share"]

        Nsrv = int(ttfts.size)
        T_run = self.step_idx * self.delta_t

        if Nsrv > 0:
            
            eta_eff = float(np.sum(ttfts <= self.tau_SLA) / max(T_run, 1e-12))
            
            ttft_avg = float(np.mean(ttfts))
            ttft_p95 = float(np.quantile(ttfts, 0.95))
            ttft_p99 = float(np.quantile(ttfts, 0.99))
            ttft_tail5_avg_tx_time = self._tail_mean_by_ttft(ttfts, tx_side_time, 0.05)
            ttft_tail5_avg_compute_time = self._tail_mean_by_ttft(ttfts, compute_side_time, 0.05)
            ttft_tail1_avg_tx_time = self._tail_mean_by_ttft(ttfts, tx_side_time, 0.01)
            ttft_tail1_avg_compute_time = self._tail_mean_by_ttft(ttfts, compute_side_time, 0.01)
            eul_avg = float(np.mean(eul))
            epf_avg = float(np.mean(epf_share))
        else:
            ttfts = np.array([], dtype=np.float64)
            eta_eff = 0.0
            ttft_avg = 0.0
            ttft_p95 = 0.0
            ttft_p99 = 0.0
            ttft_tail5_avg_tx_time = 0.0
            ttft_tail5_avg_compute_time = 0.0
            ttft_tail1_avg_tx_time = 0.0
            ttft_tail1_avg_compute_time = 0.0
            eul_avg = 0.0
            epf_avg = 0.0

        return {
            "T_run": float(T_run),
            "N_arrivals": int(self.total_arrivals),
            "Nsrv": int(Nsrv),
            "throughput": float(Nsrv / max(T_run, 1e-12)),
            "effective_throughput": float(eta_eff),
            "avg_ttft": float(ttft_avg),
            "ttft_p95": float(ttft_p95),
            "ttft_p99": float(ttft_p99),
            "ttft_tail5_avg_tx_time": float(ttft_tail5_avg_tx_time),
            "ttft_tail5_avg_compute_time": float(ttft_tail5_avg_compute_time),
            "ttft_tail1_avg_tx_time": float(ttft_tail1_avg_tx_time),
            "ttft_tail1_avg_compute_time": float(ttft_tail1_avg_compute_time),
            "avg_ul_energy": float(eul_avg),
            "avg_gpu_energy": float(epf_avg),
            "total_ul_energy": float(self.total_ul_energy),
            "total_gpu_energy": float(self.total_gpu_energy),
            "total_batches": int(self.total_started_batches),
            "n_pending_qul": int(len(self.Qul)),
            "n_active_ul": int(len(self.Aul)),
            "n_pending_qcl": int(len(self.Qcl)),
        }

    def parse_obs(self, obs: np.ndarray) -> Dict[str, Any]:
        '''
        obs是一个数值向量，parse_obs()把它变得阅读友好。
        '''
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = 0
        global_part = {
            "period_indicator": float(obs[idx]),
            "len_Qul": float(obs[idx + 1]),
            "len_Aul": float(obs[idx + 2]),
            "len_Qcl": float(obs[idx + 3]),
        }
        idx += 4

        active_ul = []
        for _ in range(self.K):
            item = {
                "present": float(obs[idx]),
                "need_tx_start": float(obs[idx + 1]),
                "log10_mu": float(obs[idx + 2]),
                "log10_sigma2": float(obs[idx + 3]),
                "m": float(obs[idx + 4]),
                "gamma_acc": float(obs[idx + 5]),
                "wait": float(obs[idx + 6]),
                "tau_ul_rem": float(obs[idx + 7]),
            }
            active_ul.append(item)
            idx += 8

        q_ul_waits = obs[idx : idx + self.q_ul_obs_cap].astype(np.float32)
        idx += self.q_ul_obs_cap
        q_cl_waits = obs[idx : idx + self.q_cl_obs_cap].astype(np.float32)
        idx += self.q_cl_obs_cap
        
        gpu = []
        for _ in range(self.N):
            item = {
                "is_busy": float(obs[idx]),
                "gpu_rem": float(obs[idx+1]),
            }
            gpu.append(item)
            idx += 2

        return {
            "global": global_part,
            "active_ul": active_ul,
            "q_ul_waits": q_ul_waits,
            "q_cl_waits": q_cl_waits,
            "gpu": gpu,
        }

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def extract_action_masks(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从原始 obs 中提取当前时刻哪些动作维度是有效的。
        power_mask[k] = 1 表示第 k 个 uplink 槽位当前需要启动传输；
        batch_mask[n] = 1 表示第 n 个 GPU 当前空闲，可以启动 batch。
        """
        parsed = self.parse_obs(obs)

        power_mask = np.asarray(
            [slot["present"] * slot["need_tx_start"] for slot in parsed["active_ul"]],
            dtype=np.float32,
        )

        batch_mask = np.asarray(
            [1.0 - slot["is_busy"] for slot in parsed["gpu"]],
            dtype=np.float32,
        )

        return power_mask, batch_mask
    
    def _lambda_t(self) -> float:
        # 到达强度随时间波动波动
        lam_t = self.lam + self.lam_amp * math.sin(self.lam_omega * self.t + self.lam_phase)
        return lam_t
    
    def _compute_obs_dim(self) -> int:
        return 4 + 8 * self.K + self.q_ul_obs_cap + self.q_cl_obs_cap + 2 * self.N

    def _advance_predecision(self) -> None:
        """
        系统演进
        Build the current decision state S_t.

        This follows the pseudo-code order:
          1) new arrivals at current t;
          2-5) process completions whose exact end times fall in the past interval;
          6) fill idle uplink resources;
          7) construct current state.

        Exact event times are tracked continuously, but decisions still happen only
        on the discrete grid t = l * delta_t.
        """
        Ut: List[int] = []
        Ct: List[int] = []
        Gt: List[int] = []

        # 1. new arrivals at current t
        n_arrivals = int(self.rng.poisson(self._lambda_t() * self.delta_t))
        for _ in range(n_arrivals):
            self.req_counter += 1  # 所有请求的唯一编号
            self.total_arrivals += 1  # 成功进入Qul的请求总数 这两个量在当前系统设定下，值相同，但语义不同
            q, mu, sigma2 = self.ckm.sample_user(self.rng)
            req = Request(req_id=self.req_counter, q=q, mu=mu, sigma2=sigma2, t_arr=self.t)
            self.requests[req.req_id] = req
            self.Qul.append(req.req_id)

        # 2-3. uplink completions whose end time is <= current t
        completed_ul_req_ids: List[int] = []  # 在当前系统建模下，所有请求最终都会完成 
        for req_id, req in list(self.Aul.items()):
            if req.in_ul_flight and req.ul_end_t is not None and req.ul_end_t <= self.t + 1e-12:
                req.in_ul_flight = False
                round_energy = req.current_power * (self.L / self.R)
                req.e_ul += round_energy
                self.total_ul_energy += round_energy
                Ut.append(req_id)

                gamma_inc = req.current_power * req.current_gain / (self.N0 * self.Bw)
                req.gamma_acc += gamma_inc

                if self.Bw * math.log2(1.0 + req.gamma_acc) >= self.R:
                    req.t_cl = req.ul_end_t
                    completed_ul_req_ids.append(req_id)
                else:
                    req.need_tx_start = True  # 需要再上传

        for req_id in completed_ul_req_ids:
            req = self.requests[req_id]
            req.completed_ul = True
            self.Aul.pop(req_id, None)
            self._insert_into_qcl_sorted(req_id)

        # 4-5. GPU completions whose batch end time is <= current t
        for gpu in self.gpus:
            if gpu.current_batch_id is None:
                # 继续闲置
                continue
            batch = self.batches[gpu.current_batch_id]
            if (not batch.counted_completion) and batch.end_t <= self.t + 1e-12:
                # 完成一个batch的prefill
                batch.counted_completion = True
                Ct.extend(batch.request_ids)
                for req_id in batch.request_ids:
                    req = self.requests[req_id]
                    req.completed_pf = True
                    req.counted_pf_completion = True
                    self.finished_request_ids.append(req_id)
                gpu.current_batch_id = None
                gpu.is_busy = False
                gpu.rem_time = 0.0
            else:
                # 继续运行
                gpu.rem_time = max(batch.end_t - self.t, 0.0)  # gpu运行时长可直接由当前时刻和结束时刻直接推算，没用到rem_time这个量

        # 6. fill idle uplink resources
        while len(self.Aul) < self.K and len(self.Qul) > 0:
            req_id = self.Qul.popleft()
            req = self.requests[req_id]
            if req.t_ul is None:
                # 在当前系统下，不会出现req.t_ul不是None的情况。如果有把请求从Aul踢回Qul的机制，可能会出现这个情况。
                req.t_ul = self.t
            req.need_tx_start = True  # 需要开启上传
            self.Aul[req_id] = req

        # 7. decision entities and reward terms at current t
        self.tx_start_reqs = [
            req_id
            for req_id, req in sorted(self.Aul.items(), key=lambda kv: (kv[1].t_arr, kv[0]))
            if req.need_tx_start  # 需要上传的请求的集合（上面新加入Aul的请求和需要再上传的请求）
        ]  # 当前时刻在 Aul 中、并且 need_tx_start=True 的那些请求，按到达时间排序后的 ID 列表。
        self.idle_gpu_ids = [gpu.gpu_id for gpu in self.gpus if gpu.current_batch_id is None]

        self.current_events = {
            "Ut": Ut,  # 本步内完成一轮 uplink 传输的请求集合
            "Ct": Ct,  # 本步内完成 prefill 的请求集合
            "Gt": Gt,  # 本步内启动新 batch 的 GPU 集合
            # 奖励似乎提前计算了，不过没影响。因为和Ut、Ct有关的奖励计算受系统演化影响，而不是决策影响。
            "Eul_t": sum(self.requests[rid].current_power * (self.L / self.R) for rid in Ut),
            "Epf_t": 0.0,
            "NSLA_t": sum(
                1
                for rid in Ct
                if self.requests[rid].t_fin is not None
                and (self.requests[rid].t_fin - self.requests[rid].t_arr) <= self.tau_SLA
            ),
        }

    def _sample_lognormal_gain(self, mu: float, sigma2: float) -> float:
        '''
        给定信道功率增益在线性域中的均值方差，按对数高斯分布采样
        '''
        mu = max(float(mu), 1e-30)
        sigma2 = max(float(sigma2), 0.0)
        b2 = math.log(1.0 + sigma2 / max(mu ** 2, 1e-60))
        a = math.log(mu) - 0.5 * b2
        return float(self.rng.lognormal(mean=a, sigma=math.sqrt(b2)))

    def _start_uplink_round(self, req: Request, power: float) -> None:
        '''
        将请求 req 以 power 的传输功率开启一轮传输（可能是新传输或重传）
        '''
        req.m += 1
        req.current_power = float(power)
        req.current_gain = self._sample_lognormal_gain(req.mu, req.sigma2)
        req.ul_start_t = self.t
        req.ul_end_t = self.t + self.L / self.R
        req.in_ul_flight = True
        req.need_tx_start = False  # 开启一轮传输后，就不再需要“开启传输”，而是已经“正在传输”。

    def _start_batch(self, gpu_id: int, batch_size: int) -> Dict[str, Any]:
        '''
        在第 gpu_id 个GPU上启动大小为 batch_size 的批次 
        '''
        if batch_size <= 0:
            raise ValueError("batch_size must be positive in _start_batch().")
        gpu = self.gpus[gpu_id]
        if gpu.current_batch_id is not None:
            raise RuntimeError("Trying to launch a batch on a busy GPU.")

        self.batch_counter += 1  # 所有batch的唯一编号。
        self.total_started_batches += 1  # 成功启动的batch总数。这两个量在当前系统设定下，值相同，但语义不同
        tau_pf = self.prefill_model.tau(batch_size)
        energy = self.prefill_model.energy(batch_size)
        end_t = self.t + tau_pf

        req_ids: List[int] = []  # 这个batch里的请求的id
        for _ in range(batch_size):
            # 云端计算队列的队首取请求
            req_id = self.Qcl.pop(0)
            req = self.requests[req_id]
            req.t_grab = self.t
            req.t_fin = end_t
            req.batch_size = batch_size
            req.e_pf_share += energy / batch_size
            req_ids.append(req_id)

        batch = BatchRecord(
            batch_id=self.batch_counter,
            gpu_id=gpu_id,
            request_ids=req_ids,
            batch_size=batch_size,
            start_t=self.t,
            end_t=end_t,
            tau_pf=tau_pf,
            energy=energy,
        )
        
        self.batches[batch.batch_id] = batch
        gpu.current_batch_id = batch.batch_id
        gpu.rem_time = tau_pf
        gpu.is_busy = True
        self.total_gpu_energy += energy

        return {
            "gpu_id": gpu_id,
            "batch_id": batch.batch_id,
            "req_ids": req_ids,
            "batch_size": batch_size,
            "tau_pf": float(tau_pf),
            "energy": float(energy),
            "end_t": float(end_t),
        }

    def _head_waits(self, req_ids: Sequence[int], cap: int) -> List[float]:
        '''
        返回队首cap个请求的系统等待时长
        '''
        out: List[float] = []
        for req_id in list(req_ids)[:cap]:
            req = self.requests[req_id]
            out.append(float(self.t - req.t_arr))
        while len(out) < cap:
            out.append(0.0)
        return out

    def _obs(self) -> np.ndarray:
        '''
        返回总状态 S_t
        '''
        obs: List[float] = []
        obs.extend(
            [
                float(math.sin(self.lam_omega * self.t + self.lam_phase)),  # 标记系统负载的周期信号
                float(len(self.Qul)),
                float(len(self.Aul)),
                float(len(self.Qcl)),
            ]
        )

        # active uplink users sorted by arrival time
        active_sorted = [req for _, req in sorted(self.Aul.items(), key=lambda kv: (kv[1].t_arr, kv[0]))] # 对Aul集合排个序，让变量位置语义稳定
        # Aul 中请求的状态固定用K个槽，无请求的槽就padding
        for idx in range(self.K):
            if idx < len(active_sorted):
                req = active_sorted[idx]
                tau_ul_rem = 0.0
                if req.in_ul_flight and req.ul_end_t is not None:
                    tau_ul_rem = max(req.ul_end_t - self.t, 0.0)
                obs.extend(
                    [
                        1.0,  # 代表这个槽位有真实请求，不是padding
                        float(req.need_tx_start),  # 当前是否需要启动
                        float(np.log10(req.mu + 1e-30)),  # 
                        float(np.log10(req.sigma2 + 1e-30)),
                        float(req.m),
                        float(req.gamma_acc),
                        float(self.t - req.t_arr),
                        float(tau_ul_rem),
                    ]
                )
            else:
                # 
                obs.extend([0.0] * 8)

        # 只取 Qul 和 Qcl 的头部几个请求的等待时长。
        obs.extend(self._head_waits(self.Qul, self.q_ul_obs_cap))
        obs.extend(self._head_waits(self.Qcl, self.q_cl_obs_cap))

        for gpu in self.gpus:
            if gpu.current_batch_id is None:
                obs.extend([0.0] * 2)
            else:
                batch = self.batches[gpu.current_batch_id]  # 这才是gpu运行剩余时间的来源，gpu.rem_time似乎没用到
                obs.extend(
                    [
                        1.0,  # is_busy==True
                        float(max(batch.end_t - self.t, 0.0))
                    ]
                )

        return np.asarray(obs, dtype=np.float32)
    
    def _qcl_sort_key(self, req_id: int):
        req = self.requests[req_id]
        t_arr = req.t_arr if req.t_arr is not None else float("inf")
        t_cl = req.t_cl if req.t_cl is not None else float("inf")
        return (t_arr, t_cl, req_id)


    def _insert_into_qcl_sorted(self, req_id: int) -> None:
        """
        Keep Qcl sorted by:
        1) t_arr ascending
        2) t_cl ascending
        3) req_id ascending
        """
        new_key = self._qcl_sort_key(req_id)

        for idx, old_req_id in enumerate(self.Qcl):
            if new_key < self._qcl_sort_key(old_req_id):
                self.Qcl.insert(idx, req_id)
                return

        self.Qcl.append(req_id)