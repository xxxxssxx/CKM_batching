from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ckm_batching_environment import CKMBatchingEnv, PrefillModel, CKM


@dataclass
class TrivialPolicy:
    """
    A placeholder policy with the same act() signature as PPOAgent.

    It is deliberately simple:
      - use a fixed high uplink power for all launchable users,
      - when the cloud queue is non-empty, request a moderate batch size on idle GPUs,
      - otherwise request zero batch.

    This is not meant for training quality; it only verifies that the environment,
    rollout, and plotting pipelines are wired correctly.
    """
    env: CKMBatchingEnv
    power_ratio: float = 0.85
    default_batch: int = 32

    def act(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, float, float]:
        parsed = self.env.parse_obs(obs)
        action = np.zeros(self.env.act_dim, dtype=np.float32)

        # uplink powers
        action[: self.env.K] = np.float32(self.power_ratio * self.env.p_max)

        # batch sizes, one per GPU
        qcl_len = int(round(parsed["global"]["len_Qcl"]))
        if qcl_len > 0:
            target_b = max(1, min(self.default_batch, self.env.Bmax, qcl_len))
            action[self.env.K :] = np.float32(target_b)
        else:
            action[self.env.K :] = 0.0

        return action, 0.0, 0.0


def run_rollout(env: CKMBatchingEnv, policy: Any, seed: int = 0) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
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
        "tx_power_mean": [],  # 当前时刻的平均发射功率
        "batch_size_sum": [],
        "gpu0_rem": [],
    }

    while not done:
        # 这个rollout不用以序列的batchsize当停止条件，因为仿真时长是固定的。
        parsed = env.parse_obs(obs)
        action, _, _ = policy.act(obs, deterministic=True)
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

        if len(info["launched_uplink"]) > 0:
            trace["tx_power_mean"].append(float(np.mean([x["power"] for x in info["launched_uplink"]])))
        else:
            trace["tx_power_mean"].append(0.0)

        if len(info["launched_batches"]) > 0:
            trace["batch_size_sum"].append(int(np.sum([x["batch_size"] for x in info["launched_batches"]])))
        else:
            trace["batch_size_sum"].append(0)

        trace["gpu0_rem"].append(float(parsed["gpu"][0]["gpu_rem"]) if env.N > 0 else 0.0)
        obs = next_obs

    metrics = env.get_metrics()
    trace_np = {k: np.asarray(v) for k, v in trace.items()}
    return trace_np, metrics


def plot_rollout(trace: Dict[str, np.ndarray], out_prefix: str) -> None:
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


def main() -> None:
    data = np.load('./gain.npz')
    gain_map = data['arr_0']

    ckm = CKM(
        gain_map=gain_map,
        xlim=(0.0, 400.0),
        ylim=(0.0, 400.0),
        building_db_threshold=-150.0,
        additional_gain=33,
        sample_only_non_building=True,
        window_size=21,
        cv2_min=0.05,
        cv2_max=0.50,
        mu_floor_linear=1e-30,
)
    env = CKMBatchingEnv(
        T=6.0,
        delta_t=0.0001,
        lam=800.0,
        K=4,
        N=4,
        Bmax=64,
        L=8192.0,
        R=10*1e6,
        N0=3.98*1e-18,
        Bw=10*1e6,
        p_max=0.4,
        tau_SLA=0.15,
        alpha=2.0,
        beta=1.0,
        gamma=0.05,
        ckm=ckm,
        prefill_model=PrefillModel(
            beta0=8*1e-3,
            beta1=10*1e-6,
            beta2=9*1e-9,
            delta_p=140.0,
            N_token=256,
            k1=1.4,
            k2=3.5,
        ),
        q_ul_obs_cap=8,
        q_cl_obs_cap=8,
        seed_value=2026,
    )

    policy = TrivialPolicy(env=env, power_ratio=1.0, default_batch=64)
    trace, metrics = run_rollout(env, policy, seed=2026)

    out_prefix = "./rollout/"
    plot_rollout(trace, out_prefix)

    with open("./rollout/rollout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
