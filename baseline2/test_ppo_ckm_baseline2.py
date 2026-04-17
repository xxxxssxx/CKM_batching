from __future__ import annotations

import argparse
import os

import torch
from ckm_ppo_agent import CKMPPOAgent, PPOConfig
from ckm_ppo_utils import (
    RunningMeanStd,
    evaluate_policy,
    plot_rollout,
    run_episode,
    save_json,
)
from baseline2_environment import make_baseline2_env


def _default_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        print("using npu")
        return "npu:0"
    if torch.cuda.is_available():
        print("using gpu")
        return "cuda"
    print("using cpu")
    return "cpu"


def build_agent_from_checkpoint(ckpt: dict, device: str) -> CKMPPOAgent:
    agent_state = ckpt["agent"]
    cfg = PPOConfig(**agent_state["cfg"])
    cfg.device = device
    agent = CKMPPOAgent(
        obs_dim=agent_state["obs_dim"],
        power_dim=agent_state["power_dim"],
        batch_dim=agent_state["batch_dim"],
        power_low=agent_state["power_low"],
        power_high=agent_state["power_high"],
        batch_action_values=agent_state["batch_action_values"],
        cfg=cfg,
    )
    agent.actor.load_state_dict(agent_state["actor"])
    agent.critic.load_state_dict(agent_state["critic"])
    if "actor_opt" in agent_state:
        agent.actor_opt.load_state_dict(agent_state["actor_opt"])
    if "critic_opt" in agent_state:
        agent.critic_opt.load_state_dict(agent_state["critic_opt"])
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./baseline2_ppo_out/baseline2_ppo_0004.pt")
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="./baseline2_ppo_eval")
    parser.add_argument("--gain_npz", type=str, default="", help="optional override")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gain_npz = args.gain_npz if args.gain_npz else ckpt["gain_npz"]
    env = make_baseline2_env(
        gain_npz_path=gain_npz,
        env_kwargs=ckpt["env_kwargs"],
        ckm_kwargs=ckpt["ckm_kwargs"],
        prefill_kwargs=ckpt["prefill_kwargs"],
        region_avg_valid_only=ckpt.get("baseline2_region_avg_valid_only", True),
    )
    agent = build_agent_from_checkpoint(ckpt, device=args.device)
    obs_rms = RunningMeanStd(shape=(env.obs_dim,))
    obs_rms.load_state_dict(ckpt["obs_rms"])

    print("evaluation start")

    eval_summary = evaluate_policy(
        env=env,
        agent=agent,
        obs_rms=obs_rms,
        episodes=args.episodes,
        deterministic=True,
        seed_base=args.seed,
    )
    save_json(eval_summary, os.path.join(args.out_dir, "eval_summary.json"))

    print("evaluation finish, rollout start")

    trace, metrics = run_episode(
        env=env,
        agent=agent,
        obs_rms=obs_rms,
        seed=args.seed,
        deterministic=True,
    )
    plot_rollout(trace, os.path.join(args.out_dir, "rollout"))
    save_json(metrics, os.path.join(args.out_dir, "rollout_metrics.json"))

    print("rollout finish, evaluation summary:")
    for k, v in eval_summary.items():
        print(f"{k}: {v}")

    if "baseline2_common_mu" in ckpt and "baseline2_common_sigma2" in ckpt:
        print("baseline2_common_mu:", ckpt["baseline2_common_mu"])
        print("baseline2_common_sigma2:", ckpt["baseline2_common_sigma2"])


if __name__ == "__main__":
    main()
