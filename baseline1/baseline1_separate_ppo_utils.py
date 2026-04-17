from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline1_separate_ppo import BatchPPOAgent, PowerPPOAgent
from ckm_batching_environment import CKMBatchingEnv
from ckm_ppo_agent import PPOConfig
from ckm_ppo_utils import (
    RunningMeanStd,
    gather_1d_float_tensor,
    make_ckm_env,
    plot_rollout,
    reduce_mean_dict,
    reduce_sum_dict,
    save_json,
    set_seed,
    setup_logging,
    split_episodes,
    sync_obs_rms,
    tail_mean_by_ttft,
)


def get_split_obs_dims(env: CKMBatchingEnv) -> Tuple[int, int]:
    tx_obs_dim = 2 + 8 * env.K + env.q_ul_obs_cap
    compute_obs_dim = 1 + env.q_cl_obs_cap + 2 * env.N
    return tx_obs_dim, compute_obs_dim


def extract_tx_obs(env: CKMBatchingEnv, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    idx = 4
    active_ul = obs[idx : idx + 8 * env.K]
    idx += 8 * env.K
    q_ul_waits = obs[idx : idx + env.q_ul_obs_cap]
    tx_obs = np.concatenate(
        [
            obs[1:3],   # len_Qul, len_Aul
            active_ul,
            q_ul_waits,
        ],
        axis=0,
    )
    return tx_obs.astype(np.float32)


def extract_compute_obs(env: CKMBatchingEnv, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    idx = 4 + 8 * env.K + env.q_ul_obs_cap
    q_cl_waits = obs[idx : idx + env.q_cl_obs_cap]
    idx += env.q_cl_obs_cap
    gpu = obs[idx : idx + 2 * env.N]
    compute_obs = np.concatenate(
        [
            obs[3:4],   # len_Qcl
            q_cl_waits,
            gpu,
        ],
        axis=0,
    )
    return compute_obs.astype(np.float32)


def build_joint_action(power_action: np.ndarray, batch_action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(power_action, dtype=np.float32),
            np.asarray(batch_action, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


class Baseline1Policy:
    def __init__(
        self,
        env: CKMBatchingEnv,
        tx_agent: PowerPPOAgent,
        compute_agent: BatchPPOAgent,
        tx_obs_rms: RunningMeanStd,
        compute_obs_rms: RunningMeanStd,
    ) -> None:
        self.env = env
        self.tx_agent = tx_agent
        self.compute_agent = compute_agent
        self.tx_obs_rms = tx_obs_rms
        self.compute_obs_rms = compute_obs_rms

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        power_mask, batch_mask = self.env.extract_action_masks(obs)

        tx_obs = extract_tx_obs(self.env, obs)
        compute_obs = extract_compute_obs(self.env, obs)
        tx_obs_n = self.tx_obs_rms.normalize(tx_obs)
        compute_obs_n = self.compute_obs_rms.normalize(compute_obs)

        power_action, tx_logp, tx_value, tx_aux = self.tx_agent.act(
            tx_obs_n,
            power_mask=power_mask,
            deterministic=deterministic,
        )
        batch_action, batch_logp, batch_value, batch_aux = self.compute_agent.act(
            compute_obs_n,
            batch_mask=batch_mask,
            deterministic=deterministic,
        )

        action = build_joint_action(power_action, batch_action)
        aux = {
            "tx_logp": tx_logp,
            "batch_logp": batch_logp,
            "tx_value": tx_value,
            "batch_value": batch_value,
            "tx_aux": tx_aux,
            "batch_aux": batch_aux,
            "power_mask": power_mask,
            "batch_mask": batch_mask,
            "tx_obs": tx_obs,
            "compute_obs": compute_obs,
            "tx_obs_n": tx_obs_n,
            "compute_obs_n": compute_obs_n,
        }
        return action, aux


def collect_rollout_separate(
    env: CKMBatchingEnv,
    tx_agent: PowerPPOAgent,
    compute_agent: BatchPPOAgent,
    tx_obs_rms: RunningMeanStd,
    compute_obs_rms: RunningMeanStd,
    batch_size: int,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    tx_obs_list = []
    tx_raw_obs_list = []
    power_mask_list = []
    power_action_list = []
    tx_logp_list = []
    tx_val_list = []

    compute_obs_list = []
    compute_raw_obs_list = []
    batch_mask_list = []
    batch_action_list = []
    batch_logp_list = []
    batch_val_list = []

    rew_list = []
    done_list = []

    obs = env._obs()
    last_done = 1.0

    while len(tx_obs_list) < batch_size:
        raw_obs = np.asarray(obs, dtype=np.float32).copy()
        power_mask, batch_mask = env.extract_action_masks(raw_obs)

        tx_raw_obs = extract_tx_obs(env, raw_obs)
        compute_raw_obs = extract_compute_obs(env, raw_obs)
        tx_obs_n = tx_obs_rms.normalize(tx_raw_obs)
        compute_obs_n = compute_obs_rms.normalize(compute_raw_obs)

        power_action, tx_logp, tx_value, tx_aux = tx_agent.act(
            tx_obs_n,
            power_mask=power_mask,
            deterministic=deterministic,
        )
        batch_action, batch_logp, batch_value, batch_aux = compute_agent.act(
            compute_obs_n,
            batch_mask=batch_mask,
            deterministic=deterministic,
        )
        joint_action = build_joint_action(power_action, batch_action)

        next_obs, reward, done, _ = env.step(joint_action)

        tx_obs_list.append(tx_obs_n)
        tx_raw_obs_list.append(tx_raw_obs)
        power_mask_list.append(tx_aux["power_mask"])
        power_action_list.append(tx_aux["power_action"])
        tx_logp_list.append(tx_logp)
        tx_val_list.append(tx_value)

        compute_obs_list.append(compute_obs_n)
        compute_raw_obs_list.append(compute_raw_obs)
        batch_mask_list.append(batch_aux["batch_mask"])
        batch_action_list.append(batch_aux["batch_action"])
        batch_logp_list.append(batch_logp)
        batch_val_list.append(batch_value)

        rew_list.append(reward)
        done_list.append(float(done))

        obs = next_obs
        last_done = float(done)
        if done:
            obs = env.reset()

    tx_last_obs_raw = extract_tx_obs(env, obs)
    compute_last_obs_raw = extract_compute_obs(env, obs)

    return {
        "tx_obs": np.asarray(tx_obs_list, dtype=np.float32),
        "tx_raw_obs": np.asarray(tx_raw_obs_list, dtype=np.float32),
        "power_mask": np.asarray(power_mask_list, dtype=np.float32),
        "power_action": np.asarray(power_action_list, dtype=np.float32),
        "tx_logp": np.asarray(tx_logp_list, dtype=np.float32),
        "tx_val": np.asarray(tx_val_list, dtype=np.float32),
        "tx_last_obs": tx_last_obs_raw.astype(np.float32),
        "tx_last_obs_n": tx_obs_rms.normalize(tx_last_obs_raw).astype(np.float32),

        "compute_obs": np.asarray(compute_obs_list, dtype=np.float32),
        "compute_raw_obs": np.asarray(compute_raw_obs_list, dtype=np.float32),
        "batch_mask": np.asarray(batch_mask_list, dtype=np.float32),
        "batch_action": np.asarray(batch_action_list, dtype=np.int64),
        "batch_logp": np.asarray(batch_logp_list, dtype=np.float32),
        "batch_val": np.asarray(batch_val_list, dtype=np.float32),
        "compute_last_obs": compute_last_obs_raw.astype(np.float32),
        "compute_last_obs_n": compute_obs_rms.normalize(compute_last_obs_raw).astype(np.float32),

        "rew": np.asarray(rew_list, dtype=np.float32),
        "done": np.asarray(done_list, dtype=np.float32),
        "last_done": last_done,
    }


@torch.no_grad()
def run_episode_separate(
    env: CKMBatchingEnv,
    tx_agent: PowerPPOAgent,
    compute_agent: BatchPPOAgent,
    tx_obs_rms: RunningMeanStd,
    compute_obs_rms: RunningMeanStd,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    policy = Baseline1Policy(env, tx_agent, compute_agent, tx_obs_rms, compute_obs_rms)
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
        parsed = env.parse_obs(obs)
        action, _ = policy.act(obs, deterministic=deterministic)
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


@torch.no_grad()
def evaluate_policy_distributed_separate(
    env: CKMBatchingEnv,
    tx_agent: PowerPPOAgent,
    compute_agent: BatchPPOAgent,
    tx_obs_rms: RunningMeanStd,
    compute_obs_rms: RunningMeanStd,
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

    local_sum_return = 0.0
    local_sum_throughput = 0.0
    local_sum_effective_throughput = 0.0
    local_sum_arrivals = 0.0
    local_sum_batches = 0.0

    local_sum_served = 0.0
    local_sum_ttft_weighted = 0.0
    local_sum_ul_energy_weighted = 0.0
    local_sum_gpu_energy_weighted = 0.0

    local_ttfts = []
    local_tx_side_times = []
    local_compute_side_times = []

    policy = Baseline1Policy(env, tx_agent, compute_agent, tx_obs_rms, compute_obs_rms)

    for i in range(local_episodes):
        ep_seed = seed_base + start_ep_idx + i
        obs = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0

        while not done:
            action, _ = policy.act(obs, deterministic=deterministic)
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
        "ttft_tail5_avg_tx_time": ttft_tail5_avg_tx_time,
        "ttft_tail5_avg_compute_time": ttft_tail5_avg_compute_time,
        "ttft_tail1_avg_tx_time": ttft_tail1_avg_tx_time,
        "ttft_tail1_avg_compute_time": ttft_tail1_avg_compute_time,
        "avg_ul_energy": float(global_stats["sum_ul_energy_weighted"] / total_served),
        "avg_gpu_energy": float(global_stats["sum_gpu_energy_weighted"] / total_served),
        "avg_served": float(global_stats["sum_served"] / total_eps),
        "avg_arrivals": float(global_stats["sum_arrivals"] / total_eps),
        "avg_batches": float(global_stats["sum_batches"] / total_eps),
        "episodes": int(global_stats["episodes"]),
    }


def reduce_agent_stats_with_prefix(
    stats: Dict[str, float],
    prefix: str,
    accelerator,
) -> Dict[str, float]:
    reduced = reduce_mean_dict(stats, accelerator)
    return {f"{prefix}_{k}": v for k, v in reduced.items()}


def global_normalize_adv(adv: np.ndarray, accelerator, device: torch.device) -> torch.Tensor:
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    adv_sum = accelerator.reduce(adv_t.sum(), reduction="sum")
    adv_sumsq = accelerator.reduce((adv_t * adv_t).sum(), reduction="sum")
    adv_count = accelerator.reduce(
        torch.tensor(float(adv_t.numel()), dtype=torch.float32, device=device),
        reduction="sum",
    )

    adv_mean = adv_sum / adv_count
    adv_var = adv_sumsq / adv_count - adv_mean * adv_mean
    adv_std = torch.sqrt(torch.clamp(adv_var, min=1e-8))
    return (adv_t - adv_mean) / adv_std


def load_split_checkpoint(
    path: str,
    tx_agent: PowerPPOAgent,
    compute_agent: BatchPPOAgent,
    tx_obs_rms: RunningMeanStd,
    compute_obs_rms: RunningMeanStd,
    accelerator,
    device: str = "cpu",
) -> Tuple[int, int, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")

    tx_agent.load_state_dict(ckpt["tx_agent"], accelerator=accelerator)
    compute_agent.load_state_dict(ckpt["compute_agent"], accelerator=accelerator)
    tx_obs_rms.load_state_dict(ckpt["tx_obs_rms"])
    compute_obs_rms.load_state_dict(ckpt["compute_obs_rms"])

    update_idx = int(ckpt.get("update_idx", 0))
    total_steps = int(ckpt.get("total_steps", 0))

    logging.info(
        f"Resumed from {path} | update_idx={update_idx}, total_steps={total_steps}"
    )
    return update_idx, total_steps, ckpt


def build_tx_agent_from_state(state: Dict[str, Any], device: str) -> PowerPPOAgent:
    cfg = PPOConfig(**state["cfg"])
    cfg.device = device
    agent = PowerPPOAgent(
        obs_dim=state["obs_dim"],
        power_dim=state["power_dim"],
        power_low=state["power_low"],
        power_high=state["power_high"],
        cfg=cfg,
    )
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if "actor_opt" in state:
        agent.actor_opt.load_state_dict(state["actor_opt"])
    if "critic_opt" in state:
        agent.critic_opt.load_state_dict(state["critic_opt"])
    return agent


def build_compute_agent_from_state(state: Dict[str, Any], device: str) -> BatchPPOAgent:
    cfg = PPOConfig(**state["cfg"])
    cfg.device = device
    agent = BatchPPOAgent(
        obs_dim=state["obs_dim"],
        batch_dim=state["batch_dim"],
        batch_action_values=state["batch_action_values"],
        cfg=cfg,
    )
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if "actor_opt" in state:
        agent.actor_opt.load_state_dict(state["actor_opt"])
    if "critic_opt" in state:
        agent.critic_opt.load_state_dict(state["critic_opt"])
    return agent
