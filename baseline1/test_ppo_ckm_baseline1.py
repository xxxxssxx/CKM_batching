from __future__ import annotations

import argparse
import os

import torch

from baseline1_separate_ppo_utils import (
    build_compute_agent_from_state,
    build_tx_agent_from_state,
    evaluate_policy_distributed_separate,
    run_episode_separate,
)
from ckm_ppo_utils import RunningMeanStd, make_ckm_env, plot_rollout, save_json


def _default_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        print("using npu")
        return "npu:0"
    if torch.cuda.is_available():
        print("using gpu")
        return "cuda"
    print("using cpu")
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./ckm_ppo_baseline1_out/baseline1_ckm_ppo_0004.pt")
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="./ckm_ppo_baseline1_eval")
    parser.add_argument("--gain_npz", type=str, default="", help="optional override")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gain_npz = args.gain_npz if args.gain_npz else ckpt["gain_npz"]
    env = make_ckm_env(
        gain_npz_path=gain_npz,
        env_kwargs=ckpt["env_kwargs"],
        ckm_kwargs=ckpt["ckm_kwargs"],
        prefill_kwargs=ckpt["prefill_kwargs"],
    )

    tx_agent = build_tx_agent_from_state(ckpt["tx_agent"], device=args.device)
    compute_agent = build_compute_agent_from_state(ckpt["compute_agent"], device=args.device)

    tx_obs_rms = RunningMeanStd(shape=(ckpt["tx_agent"]["obs_dim"],))
    tx_obs_rms.load_state_dict(ckpt["tx_obs_rms"])
    compute_obs_rms = RunningMeanStd(shape=(ckpt["compute_agent"]["obs_dim"],))
    compute_obs_rms.load_state_dict(ckpt["compute_obs_rms"])

    class _SingleProcessAcceleratorShim:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)
            self.num_processes = 1
            self.process_index = 0

        def reduce(self, tensor, reduction="sum"):
            return tensor

        def gather(self, tensor):
            return tensor

    accelerator = _SingleProcessAcceleratorShim(args.device)

    tx_agent.actor.eval()
    tx_agent.critic.eval()
    compute_agent.actor.eval()
    compute_agent.critic.eval()

    eval_summary = evaluate_policy_distributed_separate(
        env=env,
        tx_agent=tx_agent,
        compute_agent=compute_agent,
        tx_obs_rms=tx_obs_rms,
        compute_obs_rms=compute_obs_rms,
        total_episodes=args.episodes,
        accelerator=accelerator,
        deterministic=True,
        seed_base=args.seed,
    )
    save_json(eval_summary, os.path.join(args.out_dir, "eval_summary.json"))

    trace, metrics = run_episode_separate(
        env=env,
        tx_agent=tx_agent,
        compute_agent=compute_agent,
        tx_obs_rms=tx_obs_rms,
        compute_obs_rms=compute_obs_rms,
        seed=args.seed,
        deterministic=True,
    )
    plot_rollout(trace, os.path.join(args.out_dir, "rollout"))
    save_json(metrics, os.path.join(args.out_dir, "rollout_metrics.json"))

    print("Evaluation summary:")
    for k, v in eval_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
