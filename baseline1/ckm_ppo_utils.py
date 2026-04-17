from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import sys

from ckm_batching_environment import CKM, CKMBatchingEnv, PrefillModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        y = (x - self.mean.astype(np.float32)) / std
        return np.clip(y, -clip, clip)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def make_ckm_env(
    gain_npz_path: str,
    env_kwargs: Dict[str, Any],
    ckm_kwargs: Optional[Dict[str, Any]] = None,
    prefill_kwargs: Optional[Dict[str, Any]] = None,
) -> CKMBatchingEnv:
    data = np.load(gain_npz_path)
    gain_map = data["arr_0"]

    ckm_default = {
        "xlim": (0.0, 400.0),
        "ylim": (0.0, 400.0),
        "building_db_threshold": -150.0,
        "additional_gain": 33.0,
        "sample_only_non_building": True,
        "window_size": 21,
        "cv2_min": 0.05,
        "cv2_max": 0.50,
        "mu_floor_linear": 1e-30,
    }
    if ckm_kwargs is not None:
        ckm_default.update(ckm_kwargs)
    ckm = CKM(gain_map=gain_map, **ckm_default)

    prefill_default = {
        "beta0": 8e-3,
        "beta1": 10e-6,
        "beta2": 9e-9,
        "delta_p": 140.0,
        "N_token": 256,
        "k1": 1.4,
        "k2": 3.5,
    }
    if prefill_kwargs is not None:
        prefill_default.update(prefill_kwargs)
    prefill_model = PrefillModel(**prefill_default)

    return CKMBatchingEnv(
        ckm=ckm,
        prefill_model=prefill_model,
        **env_kwargs,
    )


def collect_rollout(
    env: CKMBatchingEnv,
    agent: Any,
    obs_rms: RunningMeanStd,
    batch_size: int,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    obs_list = []
    raw_obs_list = []
    power_mask_list = []
    batch_mask_list = []
    power_list = []
    batch_list = []
    logp_list = []
    rew_list = []
    done_list = []
    val_list = []

    # env.lam_phase = np.random.uniform(0.0, 2.0 * np.pi)
    obs = env._obs()
    last_done = 1.0

    while len(obs_list) < batch_size:
        # 这里先不做obs_rms.update，收集各进程数据再update。
        raw_obs = np.asarray(obs, dtype=np.float32).copy()
        raw_obs_list.append(raw_obs)

        power_mask, batch_mask = env.extract_action_masks(raw_obs)
        obs_n = obs_rms.normalize(raw_obs)

        env_action, logp, value, aux = agent.act(
            obs_n,
            power_mask=power_mask,
            batch_mask=batch_mask,
            deterministic=deterministic,
        )
        next_obs, reward, done, _ = env.step(env_action)

        obs_list.append(obs_n)
        power_mask_list.append(aux["power_mask"])
        batch_mask_list.append(aux["batch_mask"])
        power_list.append(aux["power_action"])
        batch_list.append(aux["batch_action"])
        logp_list.append(logp)
        rew_list.append(reward)
        done_list.append(float(done))
        val_list.append(value)

        obs = next_obs
        last_done = float(done)
        if done:
            obs = env.reset()  # 这里只是要重置episode状态，但不重置随机流状态

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "raw_obs": np.asarray(raw_obs_list, dtype=np.float32),  # 保存未归一化的数据
        "power_mask": np.asarray(power_mask_list, dtype=np.float32),
        "batch_mask": np.asarray(batch_mask_list, dtype=np.float32),
        "power_action": np.asarray(power_list, dtype=np.float32),
        "batch_action": np.asarray(batch_list, dtype=np.int64),
        "logp": np.asarray(logp_list, dtype=np.float32),
        "rew": np.asarray(rew_list, dtype=np.float32),
        "done": np.asarray(done_list, dtype=np.float32),
        "val": np.asarray(val_list, dtype=np.float32),
        "last_obs": np.asarray(obs, dtype=np.float32),  # 这里是未归一化的
        "last_obs_n": obs_rms.normalize(obs).astype(np.float32),  # 这个是归一化的，且用的是也是旧obs_rms
        "last_done": last_done,
    }

def sync_obs_rms(obs_rms: RunningMeanStd, obs_batch: np.ndarray, accelerator) -> None:
    x = torch.as_tensor(obs_batch, dtype=torch.float32, device=accelerator.device)

    batch_count = torch.tensor([x.shape[0]], dtype=torch.float32, device=accelerator.device)
    batch_sum = x.sum(dim=0)
    batch_sumsq = (x * x).sum(dim=0)

    batch_count = accelerator.reduce(batch_count, reduction="sum")
    batch_sum = accelerator.reduce(batch_sum, reduction="sum")
    batch_sumsq = accelerator.reduce(batch_sumsq, reduction="sum")

    total_count = max(int(batch_count.item()), 1)
    mean = (batch_sum / batch_count.clamp_min(1.0)).cpu().numpy()
    var = (batch_sumsq / batch_count.clamp_min(1.0)).cpu().numpy() - mean ** 2
    var = np.maximum(var, 1e-8)

    obs_rms._update_from_moments(mean, var, total_count)
    
def reduce_mean_dict(stats: Dict[str, float], accelerator) -> Dict[str, float]:
    # 把字典型指标做跨进程平均
    out = {}
    for k, v in stats.items():
        t = torch.tensor(float(v), dtype=torch.float32, device=accelerator.device)  # 把值做reduce
        t = accelerator.reduce(t, reduction="sum")
        out[k] = float((t / accelerator.num_processes).item())
    return out

def run_episode(
    env: CKMBatchingEnv,
    agent: Any,
    obs_rms: RunningMeanStd,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs = env.reset(seed=seed)
    done = False

    trace: Dict[str, List[Any]] = {
        "t": [],
        "reward": [],
        "Qul": [],
        "Aul": [],
        "Qcl": [],
        "NSLA_t": [],
        "Eul_t": [],
        "Epf_t": [],
        "n_tx_launch": [],
        "n_batch_launch": [],
        "tx_power_mean": [],
        "batch_size_sum": [],
        "batch_size_mean": [],
        "gpu0_rem": [],
    }

    while not done:
        power_mask, batch_mask = env.extract_action_masks(obs)
        obs_n = obs_rms.normalize(obs)
        action, _, _, _ = agent.act(
            obs_n,
            power_mask=power_mask,
            batch_mask=batch_mask,
            deterministic=deterministic,
        )
        parsed = env.parse_obs(obs)
        next_obs, reward, done, info = env.step(action)

        trace["t"].append(info["t"])
        trace["reward"].append(reward)
        trace["Qul"].append(info["queue_lengths"]["Qul"])
        trace["Aul"].append(info["queue_lengths"]["Aul"])
        trace["Qcl"].append(info["queue_lengths"]["Qcl"])
        trace["NSLA_t"].append(info["reward_terms"]["NSLA_t"])
        trace["Eul_t"].append(info["reward_terms"]["Eul_t"])
        trace["Epf_t"].append(info["reward_terms"]["Epf_t"])
        trace["n_tx_launch"].append(len(info["launched_uplink"]))
        trace["n_batch_launch"].append(len(info["launched_batches"]))
        trace["gpu0_rem"].append(float(parsed["gpu"][0]["gpu_rem"]) if env.N > 0 else 0.0)

        if len(info["launched_uplink"]) > 0:
            trace["tx_power_mean"].append(float(np.mean([x["power"] for x in info["launched_uplink"]])))
        else:
            trace["tx_power_mean"].append(0.0)

        if len(info["launched_batches"]) > 0:
            batch_sizes = [x["batch_size"] for x in info["launched_batches"]]
            trace["batch_size_sum"].append(int(np.sum(batch_sizes)))
            trace["batch_size_mean"].append(float(np.mean(batch_sizes)))
        else:
            trace["batch_size_sum"].append(0)
            trace["batch_size_mean"].append(0.0)

        obs = next_obs

    metrics = env.get_metrics()
    trace_np = {k: np.asarray(v) for k, v in trace.items()}
    return trace_np, metrics

def tail_mean_by_ttft(
    ttfts: np.ndarray,
    values: np.ndarray,
    tail_ratio: float,
) -> float:
    ttfts = np.asarray(ttfts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    n = ttfts.size
    if n == 0:
        return 0.0

    k = max(1, int(np.ceil(tail_ratio * n)))
    idx = np.argpartition(ttfts, n - k)[-k:]
    return float(np.mean(values[idx]))

def evaluate_policy(
    env: CKMBatchingEnv,
    agent: Any,
    obs_rms: RunningMeanStd,
    episodes: int,
    deterministic: bool = True,
    seed_base: int = 10000,
) -> Dict[str, float]:
    returns = []
    throughputs = []
    effective_throughputs = []
    avg_ttfts = []
    p95_ttfts = []
    p99_ttfts = []
    all_ttfts = []
    all_tx_side_times = []
    all_compute_side_times = []
    avg_ul_energies = []
    avg_gpu_energies = []
    served = []
    arrivals = []
    batches = []

    for ep in range(episodes):
        obs = env.reset(seed=seed_base + ep)
        done = False
        ep_return = 0.0
        # debug_count = 0
        while not done:
            with torch.no_grad():
                power_mask, batch_mask = env.extract_action_masks(obs)
                obs_n = obs_rms.normalize(obs)
                action, _, _, _ = agent.act(
                    obs_n,
                    power_mask=power_mask,
                    batch_mask=batch_mask,
                    deterministic=deterministic,
                )
                obs, reward, done, info = env.step(action)
                ep_return += reward
            
            # if debug_count<10:
            #     debug_count = debug_count + 1
            #     print("power_mask =", power_mask)
            #     print("batch_mask =", batch_mask)
            #     print("launched_uplink =", len(info["launched_uplink"]))
            #     print("launched_batches =", len(info["launched_batches"]))
        metrics = info["episode_metrics"]
        for rid in env.finished_request_ids:
            req = env.requests[rid]
            all_ttfts.append(float(req.t_fin - req.t_arr))
            all_tx_side_times.append(float(req.t_cl - req.t_arr))
            all_compute_side_times.append(float(req.t_fin - req.t_cl))
        returns.append(ep_return)
        throughputs.append(metrics["throughput"])
        effective_throughputs.append(metrics["effective_throughput"])
        avg_ttfts.append(metrics["avg_ttft"])
        p95_ttfts.append(metrics["ttft_p95"])
        p99_ttfts.append(metrics["ttft_p99"])
        avg_ul_energies.append(metrics["avg_ul_energy"])
        avg_gpu_energies.append(metrics["avg_gpu_energy"])
        served.append(metrics["Nsrv"])
        arrivals.append(metrics["N_arrivals"])
        batches.append(metrics["total_batches"])

    all_ttfts = np.asarray(all_ttfts, dtype=np.float64)
    all_tx_side_times = np.asarray(all_tx_side_times, dtype=np.float64)
    all_compute_side_times = np.asarray(all_compute_side_times, dtype=np.float64)

    ttft_tail5_avg_tx_time = tail_mean_by_ttft(all_ttfts, all_tx_side_times, 0.05)
    ttft_tail5_avg_compute_time = tail_mean_by_ttft(all_ttfts, all_compute_side_times, 0.05)
    ttft_tail1_avg_tx_time = tail_mean_by_ttft(all_ttfts, all_tx_side_times, 0.01)
    ttft_tail1_avg_compute_time = tail_mean_by_ttft(all_ttfts, all_compute_side_times, 0.01)
    return {
        "avg_return": float(np.mean(returns)),
        "avg_throughput": float(np.mean(throughputs)),
        "avg_effective_throughput": float(np.mean(effective_throughputs)),
        "avg_ttft": float(np.mean(avg_ttfts)),
        "avg_ttft_p95": float(np.mean(p95_ttfts)),
        "avg_ttft_p99": float(np.mean(p99_ttfts)),
        "ttft_tail5_avg_tx_time": float(ttft_tail5_avg_tx_time),
        "ttft_tail5_avg_compute_time": float(ttft_tail5_avg_compute_time),
        "ttft_tail1_avg_tx_time": float(ttft_tail1_avg_tx_time),
        "ttft_tail1_avg_compute_time": float(ttft_tail1_avg_compute_time),
        "avg_ul_energy": float(np.mean(avg_ul_energies)),
        "avg_gpu_energy": float(np.mean(avg_gpu_energies)),
        "avg_served": float(np.mean(served)),
        "avg_arrivals": float(np.mean(arrivals)),
        "avg_batches": float(np.mean(batches)),
        "episodes": int(episodes),
    }


def plot_training_curves(history: Dict[str, List[Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    x = np.asarray(history["updates"], dtype=np.float32)

    def _plot_single_curve(
        y_key: str,
        title: str,
        ylabel: str,
        save_path: str,
        annotate_points: bool = False,
    ) -> None:
        if y_key not in history or len(history[y_key]) == 0:
            return

        y = np.asarray(history[y_key], dtype=np.float32)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel("update")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.4)

        if annotate_points:
            for xi, yi in zip(x, y):
                plt.annotate(
                    f"({int(xi)}, {yi:.4f})",
                    xy=(xi, yi),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    # 这些量：单独保存图片，并在每个点标注坐标
    metric_specs = [
        ("eval_return", "Eval return", "return", "eval_return.png", True),
        ("avg_effective_throughput", "Avg effective throughput", "throughput", "avg_effective_throughput.png", True),
        ("avg_ttft", "Avg TTFT", "TTFT", "avg_ttft.png", True),
        ("avg_ttft_p95", "Avg TTFT P95", "TTFT P95", "avg_ttft_p95.png", True),
        ("avg_ttft_p99", "Avg TTFT P99", "TTFT P99", "avg_ttft_p99.png", True),
        ("avg_ul_energy", "Avg UL energy", "UL energy", "avg_ul_energy.png", True),
        ("ttft_tail5_avg_tx_time", "Tail 5% Avg TX-side Time", "time", "ttft_tail5_avg_tx_time.png", True),
        ("ttft_tail5_avg_compute_time", "Tail 5% Avg Compute-side Time", "time", "ttft_tail5_avg_compute_time.png", True),
        ("ttft_tail1_avg_tx_time", "Tail 1% Avg TX-side Time", "time", "ttft_tail1_avg_tx_time.png", True),
        ("ttft_tail1_avg_compute_time", "Tail 1% Avg Compute-side Time", "time", "ttft_tail1_avg_compute_time.png", True),
    ]

    for y_key, title, ylabel, filename, annotate_points in metric_specs:
        _plot_single_curve(
            y_key=y_key,
            title=title,
            ylabel=ylabel,
            save_path=os.path.join(out_dir, filename),
            annotate_points=annotate_points,
        )

    # update_info 里的量：也单独保存图片，但不标注坐标
    update_metric_specs = [
        ("actor_loss", "Actor loss", "loss", "actor_loss.png"),
        ("critic_loss", "Critic loss", "loss", "critic_loss.png"),
        ("entropy", "Entropy", "entropy", "entropy.png"),
        ("kl", "KL divergence", "kl", "kl.png"),
    ]

    for y_key, title, ylabel, filename in update_metric_specs:
        _plot_single_curve(
            y_key=y_key,
            title=title,
            ylabel=ylabel,
            save_path=os.path.join(out_dir, filename),
            annotate_points=False,
        )


def plot_rollout(trace: Dict[str, np.ndarray], out_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    t = trace["t"]

    plt.figure()
    plt.plot(t, trace["Qul"], label="|Qul|")
    plt.plot(t, trace["Aul"], label="|Aul|")
    plt.plot(t, trace["Qcl"], label="|Qcl|")
    plt.xlabel("time (s)")
    plt.ylabel("queue length")
    plt.title("Queue evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_queues.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(t, trace["reward"])
    plt.xlabel("time (s)")
    plt.ylabel("reward")
    plt.title("Per-step reward")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_reward.png", dpi=160)
    plt.close()

    plt.figure()
    plt.step(t, trace["tx_power_mean"], where="post", label="mean launched uplink power")
    plt.step(t, trace["batch_size_sum"], where="post", label="sum launched batch sizes")
    plt.xlabel("time (s)")
    plt.ylabel("action magnitude")
    plt.title("Executed actions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_actions.png", dpi=160)
    plt.close()

    plt.figure()
    plt.step(t, trace["Eul_t"], where="post", label="Eul_t")
    plt.step(t, trace["Epf_t"], where="post", label="Epf_t")
    plt.step(t, trace["NSLA_t"], where="post", label="NSLA_t")
    plt.xlabel("time (s)")
    plt.ylabel("per-step term")
    plt.title("Reward components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_reward_terms.png", dpi=160)
    plt.close()


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

from typing import Any, Tuple

def load_checkpoint(path: str, agent, obs_rms, accelerator, device: str = "cpu") -> Tuple[int, int]:
    
    assert accelerator is not None, "accelerator must be passed in"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")

    agent.load_state_dict(ckpt["agent"], accelerator=accelerator)
    obs_rms.load_state_dict(ckpt["obs_rms"])

    update_idx = int(ckpt.get("update_idx", 0))
    total_steps = int(ckpt.get("total_steps", 0))

    logging.info(
        f"Resumed from {path} | "
        f"update_idx={update_idx}, total_steps={total_steps}"
    )
    return update_idx, total_steps

def split_episodes(total_episodes: int, rank: int, world_size: int) -> Tuple[int, int]:
    base = total_episodes // world_size
    rem = total_episodes % world_size
    local_episodes = base + (1 if rank < rem else 0)
    start_ep_idx = rank * base + min(rank, rem)
    return local_episodes, start_ep_idx


def reduce_sum_dict(stats: Dict[str, float], accelerator) -> Dict[str, float]:
    out = {}
    for k, v in stats.items():
        t = torch.tensor(float(v), dtype=torch.float32, device=accelerator.device)
        t = accelerator.reduce(t, reduction="sum")
        out[k] = float(t.item())
    return out


def gather_1d_float_tensor(x: torch.Tensor, accelerator) -> torch.Tensor:
    x = x.reshape(-1).to(device=accelerator.device, dtype=torch.float32)

    local_n = torch.tensor([x.numel()], dtype=torch.long, device=accelerator.device)
    all_n = accelerator.gather(local_n).view(-1)

    if all_n.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    max_n = int(all_n.max().item())
    if max_n == 0:
        return torch.empty(0, dtype=torch.float32)

    padded = torch.zeros(max_n, dtype=torch.float32, device=accelerator.device)
    if x.numel() > 0:
        padded[: x.numel()] = x

    gathered = accelerator.gather(padded)  # shape = [world_size * max_n]

    chunks = []
    world_size = accelerator.num_processes
    for r in range(world_size):
        n_r = int(all_n[r].item())
        start = r * max_n
        chunks.append(gathered[start : start + n_r])

    if len(chunks) == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def evaluate_policy_distributed(
    env: CKMBatchingEnv,
    agent: Any,
    obs_rms: RunningMeanStd,
    total_episodes: int,
    accelerator,
    deterministic: bool = True,
    seed_base: int = 10000,
) -> Dict[str, float]:
    local_episodes, start_ep_idx = split_episodes(
        total_episodes=total_episodes,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    # 这些量按“episode 求平均”
    local_sum_return = 0.0
    local_sum_throughput = 0.0
    local_sum_effective_throughput = 0.0
    local_sum_arrivals = 0.0
    local_sum_batches = 0.0

    # 这些量按“served request 求平均”，更合理
    local_sum_served = 0.0
    local_sum_ttft_weighted = 0.0
    local_sum_ul_energy_weighted = 0.0
    local_sum_gpu_energy_weighted = 0.0

    # 为了精确算全局 p95 / p99，要收集所有 served request 的 TTFT
    local_ttfts = []
    local_tx_side_times = []
    local_compute_side_times = []

    for i in range(local_episodes):
        ep_seed = seed_base + start_ep_idx + i
        obs = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                power_mask, batch_mask = env.extract_action_masks(obs)
                obs_n = obs_rms.normalize(obs)
                action, _, _, _ = agent.act(
                    obs_n,
                    power_mask=power_mask,
                    batch_mask=batch_mask,
                    deterministic=deterministic,
                )
                obs, reward, done, info = env.step(action)
                ep_return += reward

        metrics = info["episode_metrics"]

        local_sum_return += ep_return
        local_sum_throughput += metrics["throughput"]
        local_sum_effective_throughput += metrics["effective_throughput"]
        local_sum_arrivals += metrics["N_arrivals"]
        local_sum_batches += metrics["total_batches"]

        nsrv = float(metrics["Nsrv"])
        local_sum_served += nsrv
        local_sum_ttft_weighted += metrics["avg_ttft"] * nsrv
        local_sum_ul_energy_weighted += metrics["avg_ul_energy"] * nsrv
        local_sum_gpu_energy_weighted += metrics["avg_gpu_energy"] * nsrv

        # 从环境中精确提取本 episode 所有 served request 的 TTFT
        for rid in env.finished_request_ids:
            req = env.requests[rid]
            local_ttfts.append(float(req.t_fin - req.t_arr))
            local_tx_side_times.append(float(req.t_cl - req.t_arr))
            local_compute_side_times.append(float(req.t_fin - req.t_cl))
            
    local_stats = {
        "sum_return": local_sum_return,
        "sum_throughput": local_sum_throughput,
        "sum_effective_throughput": local_sum_effective_throughput,
        "sum_arrivals": local_sum_arrivals,
        "sum_batches": local_sum_batches,
        "sum_served": local_sum_served,
        "sum_ttft_weighted": local_sum_ttft_weighted,
        "sum_ul_energy_weighted": local_sum_ul_energy_weighted,
        "sum_gpu_energy_weighted": local_sum_gpu_energy_weighted,
        "episodes": float(local_episodes),
    }

    global_stats = reduce_sum_dict(local_stats, accelerator)

    global_ttfts = gather_1d_float_tensor(
        torch.tensor(local_ttfts, dtype=torch.float32, device=accelerator.device),
        accelerator,
    ).cpu().numpy()
    global_tx_side_times = gather_1d_float_tensor(
        torch.tensor(local_tx_side_times, dtype=torch.float32, device=accelerator.device),
        accelerator,
    ).cpu().numpy()

    global_compute_side_times = gather_1d_float_tensor(
        torch.tensor(local_compute_side_times, dtype=torch.float32, device=accelerator.device),
        accelerator,
    ).cpu().numpy()

    total_eps = max(int(global_stats["episodes"]), 1)
    total_served = max(float(global_stats["sum_served"]), 1.0)

    if global_ttfts.size > 0:
        ttft_p95 = float(np.quantile(global_ttfts, 0.95))
        ttft_p99 = float(np.quantile(global_ttfts, 0.99))
        ttft_tail5_avg_tx_time = tail_mean_by_ttft(global_ttfts, global_tx_side_times, 0.05)
        ttft_tail5_avg_compute_time = tail_mean_by_ttft(global_ttfts, global_compute_side_times, 0.05)
        ttft_tail1_avg_tx_time = tail_mean_by_ttft(global_ttfts, global_tx_side_times, 0.01)
        ttft_tail1_avg_compute_time = tail_mean_by_ttft(global_ttfts, global_compute_side_times, 0.01)
    else:
        ttft_p95 = 0.0
        ttft_p99 = 0.0
        ttft_tail5_avg_tx_time = 0.0
        ttft_tail5_avg_compute_time = 0.0
        ttft_tail1_avg_tx_time = 0.0
        ttft_tail1_avg_compute_time = 0.0
        
    return {
        "avg_return": float(global_stats["sum_return"] / total_eps),
        "avg_throughput": float(global_stats["sum_throughput"] / total_eps),
        "avg_effective_throughput": float(global_stats["sum_effective_throughput"] / total_eps),
        "avg_ttft": float(global_stats["sum_ttft_weighted"] / total_served),
        "ttft_p95": ttft_p95,
        "ttft_p99": ttft_p99,
        "ttft_tail5_avg_tx_time" : ttft_tail5_avg_tx_time,
        "ttft_tail5_avg_compute_time" : ttft_tail5_avg_compute_time,
        "ttft_tail1_avg_tx_time" : ttft_tail1_avg_tx_time,
        "ttft_tail1_avg_compute_time" : ttft_tail1_avg_compute_time,
        "avg_ul_energy": float(global_stats["sum_ul_energy_weighted"] / total_served),
        "avg_gpu_energy": float(global_stats["sum_gpu_energy_weighted"] / total_served),
        "avg_served": float(global_stats["sum_served"] / total_eps),
        "avg_arrivals": float(global_stats["sum_arrivals"] / total_eps),
        "avg_batches": float(global_stats["sum_batches"] / total_eps),
        "episodes": int(global_stats["episodes"]),
    }
    
def setup_logging(rank: int, log_path: str = "trainlog.log"):
    root = logging.getLogger()
    # 1) clear old handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 2) formatter with rank tag
    fmt = logging.Formatter(
        "%(asctime)s [Rank {:>2}] %(levelname)s: %(message)s".format(rank)
    )

    # 3) console handler (optional)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 4) single shared file handler
    file_h = logging.FileHandler(log_path, mode="a")
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    root.setLevel(logging.INFO)