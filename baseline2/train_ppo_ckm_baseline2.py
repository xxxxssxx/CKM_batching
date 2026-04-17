from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import math
from accelerate import Accelerator
import logging

from ckm_ppo_agent import CKMPPOAgent, PPOConfig, compute_gae
from ckm_ppo_utils import (
    RunningMeanStd,
    collect_rollout,
    evaluate_policy_distributed,
    plot_training_curves,
    save_json,
    set_seed,
    load_checkpoint,
    sync_obs_rms,
    reduce_mean_dict,
    setup_logging,
)
from baseline2_environment import make_baseline2_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gain_npz", type=str, default="./gain.npz")
    parser.add_argument("--out_dir", type=str, default="./baseline2_ppo_out")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--reset_history", action="store_true")

    # env
    parser.add_argument("--T", type=float, default=120)
    parser.add_argument("--delta_t", type=float, default=1 * 1e-4)
    parser.add_argument("--lam", type=float, default=550.0)
    parser.add_argument("--lam_amp", type=float, default=0.0)
    parser.add_argument("--lam_omega", type=float, default=2 * 3.1415926 * 5 / 120)
    parser.add_argument("--lam_phase", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--Bmax", type=int, default=32)
    parser.add_argument("--batch_step", type=int, default=2)
    parser.add_argument("--L", type=float, default=8192.0)
    parser.add_argument("--R", type=float, default=10e6)
    parser.add_argument("--N0", type=float, default=3.98e-18)
    parser.add_argument("--Bw", type=float, default=10e6)
    parser.add_argument("--p_max", type=float, default=0.4)
    parser.add_argument("--tau_SLA", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--gamma_reward", type=float, default=0.1)
    parser.add_argument("--q_ul_obs_cap", type=int, default=32)
    parser.add_argument("--q_cl_obs_cap", type=int, default=32)

    # CKM / prefill
    parser.add_argument("--additional_gain", type=float, default=33.0)
    parser.add_argument("--building_db_threshold", type=float, default=-150.0)
    parser.add_argument("--window_size", type=int, default=21)
    parser.add_argument("--cv2_min", type=float, default=0.05)
    parser.add_argument("--cv2_max", type=float, default=0.50)
    parser.add_argument("--beta0", type=float, default=8e-3)
    parser.add_argument("--beta1", type=float, default=10e-6)
    parser.add_argument("--beta2", type=float, default=9e-9)
    parser.add_argument("--N_token", type=float, default=256.0)
    parser.add_argument("--delta_p", type=float, default=140.0)
    parser.add_argument("--k1", type=float, default=1.6)
    parser.add_argument("--k2", type=float, default=5)
    parser.add_argument(
        "--region_avg_valid_only",
        type=int,
        default=1,
        help="1: average common mu/sigma2 over valid non-building pixels; 0: average over the full map.",
    )

    # PPO
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=8192 * 16 * 8)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # train
    parser.add_argument("--total_steps", type=int, default=30 * 1.2 * 1e6 * 8)
    parser.add_argument("--rollout_steps", type=int, default=1.2 * 1e6 * 8)

    # evaluate and save
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=1 * 8)
    parser.add_argument("--eval_T", type=int, default=30)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    setup_logging(rank, log_path="logtrain_baseline2.log")
    logging.info("Start Baseline 2 training")

    set_seed(args.seed + rank)

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    os.makedirs(args.out_dir, exist_ok=True)

    env_kwargs = {
        "T": args.T,
        "delta_t": args.delta_t,
        "lam": args.lam,
        "lam_amp": args.lam_amp,
        "lam_omega": args.lam_omega,
        "lam_phase": args.lam_phase,
        "K": args.K,
        "N": args.N,
        "Bmax": args.Bmax,
        "L": args.L,
        "R": args.R,
        "N0": args.N0,
        "Bw": args.Bw,
        "p_max": args.p_max,
        "tau_SLA": args.tau_SLA,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma_reward,
        "q_ul_obs_cap": args.q_ul_obs_cap,
        "q_cl_obs_cap": args.q_cl_obs_cap,
        "seed_value": args.seed,
    }
    train_env_kwargs = dict(env_kwargs)
    train_env_kwargs["seed_value"] = args.seed + rank

    ckm_kwargs = {
        "additional_gain": args.additional_gain,
        "building_db_threshold": args.building_db_threshold,
        "window_size": args.window_size,
        "cv2_min": args.cv2_min,
        "cv2_max": args.cv2_max,
    }
    prefill_kwargs = {
        "beta0": args.beta0,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "N_token": args.N_token,
        "delta_p": args.delta_p,
        "k1": args.k1,
        "k2": args.k2,
    }

    region_avg_valid_only = bool(args.region_avg_valid_only)

    env = make_baseline2_env(
        gain_npz_path=args.gain_npz,
        env_kwargs=train_env_kwargs,
        ckm_kwargs=ckm_kwargs,
        prefill_kwargs=prefill_kwargs,
        region_avg_valid_only=region_avg_valid_only,
    )

    obs_dim = env.obs_dim
    power_low = np.zeros(env.K, dtype=np.float32)
    power_high = np.full(env.K, env.p_max, dtype=np.float32)

    rollout_steps_per_rank = math.ceil(args.rollout_steps / accelerator.num_processes)
    minibatch_size_per_rank = max(1, math.ceil(args.minibatch_size / accelerator.num_processes))
    ppo_cfg = PPOConfig(
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        update_epochs=args.update_epochs,
        minibatch_size=minibatch_size_per_rank,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        device=str(device),
    )

    batch_action_values = np.arange(0, env.Bmax + 1, args.batch_step, dtype=np.int64)
    if batch_action_values[-1] != env.Bmax:
        batch_action_values = np.append(batch_action_values, env.Bmax)
    agent = CKMPPOAgent(
        obs_dim=obs_dim,
        power_dim=env.K,
        batch_dim=env.N,
        power_low=power_low,
        power_high=power_high,
        batch_action_values=batch_action_values,
        cfg=ppo_cfg,
    )
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    agent.prepare(accelerator)
    env.reset(seed=train_env_kwargs["seed_value"])

    history = {
        "updates": [],
        "eval_return": [],
        "avg_effective_throughput": [],
        "avg_ttft": [],
        "ttft_p95": [],
        "ttft_p99": [],
        "ttft_tail5_avg_tx_time": [],
        "ttft_tail5_avg_compute_time": [],
        "ttft_tail1_avg_tx_time": [],
        "ttft_tail1_avg_compute_time": [],
        "avg_ul_energy": [],
        "avg_gpu_energy": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "kl": [],
    }

    total_steps = 0
    update_idx = 0

    if args.resume_path is not None:
        update_idx, total_steps = load_checkpoint(
            args.resume_path, agent, obs_rms, accelerator, device
        )
        if accelerator.is_main_process and (not args.reset_history):
            ckpt = torch.load(args.resume_path, map_location="cpu", weights_only=False)
            history = ckpt.get("history", history)

    accelerator.wait_for_everyone()

    while total_steps < args.total_steps:
        logging.info(f"rollout number {update_idx}, start.")

        batch_np = collect_rollout(
            env=env,
            agent=agent,
            obs_rms=obs_rms,
            batch_size=rollout_steps_per_rank,
            deterministic=False,
        )

        logging.info(f"rollout number {update_idx}, finish.")

        with torch.no_grad():
            last_obs_n = torch.tensor(batch_np["last_obs_n"], dtype=torch.float32, device=agent.device).unsqueeze(0)
            last_value = agent.value(last_obs_n).item()

        if batch_np["last_done"] > 0.5:
            last_value = 0.0

        adv, ret = compute_gae(
            rewards=batch_np["rew"],
            dones=batch_np["done"],
            values=batch_np["val"],
            last_value=last_value,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )

        sync_obs_rms(obs_rms, batch_np["raw_obs"], accelerator)

        adv_t = torch.tensor(adv, dtype=torch.float32, device=agent.device)
        adv_sum = accelerator.reduce(adv_t.sum(), reduction="sum")
        adv_sumsq = accelerator.reduce((adv_t * adv_t).sum(), reduction="sum")
        adv_count = accelerator.reduce(
            torch.tensor(float(adv_t.numel()), device=agent.device), reduction="sum"
        )

        adv_mean = adv_sum / adv_count
        adv_var = adv_sumsq / adv_count - adv_mean * adv_mean
        adv_std = torch.sqrt(torch.clamp(adv_var, min=1e-8))
        adv_t = (adv_t - adv_mean) / adv_std

        torch_batch = {
            "obs": torch.tensor(batch_np["obs"], dtype=torch.float32, device=agent.device),
            "power_mask": torch.tensor(batch_np["power_mask"], dtype=torch.float32, device=agent.device),
            "batch_mask": torch.tensor(batch_np["batch_mask"], dtype=torch.float32, device=agent.device),
            "power_action": torch.tensor(batch_np["power_action"], dtype=torch.float32, device=agent.device),
            "batch_action": torch.tensor(batch_np["batch_action"], dtype=torch.long, device=agent.device),
            "logp": torch.tensor(batch_np["logp"], dtype=torch.float32, device=agent.device),
            "adv": adv_t,
            "ret": torch.tensor(ret, dtype=torch.float32, device=agent.device),
        }

        logging.info(f"update number {update_idx}, start.")
        update_info = agent.update(torch_batch, accelerator=accelerator)
        logging.info(f"update number {update_idx}, finish.")

        update_info = reduce_mean_dict(update_info, accelerator)

        global_rollout_steps = torch.tensor(batch_np["obs"].shape[0], dtype=torch.int, device=device)
        global_rollout_steps = int(accelerator.reduce(global_rollout_steps, reduction="sum").item())
        total_steps += global_rollout_steps


        if update_idx % args.eval_every == 0 or total_steps >= args.total_steps:
            accelerator.wait_for_everyone()

            eval_env_kwargs = dict(env_kwargs)
            eval_env_kwargs["T"] = args.eval_T
            eval_env_kwargs["seed_value"] = args.seed + update_idx * 2026 + rank * 100003

            eval_env = make_baseline2_env(
                gain_npz_path=args.gain_npz,
                env_kwargs=eval_env_kwargs,
                ckm_kwargs=ckm_kwargs,
                prefill_kwargs=prefill_kwargs,
                region_avg_valid_only=region_avg_valid_only,
            )

            agent.actor.eval()
            agent.critic.eval()

            logging.info(f"evaluation number {update_idx}, start.")
            eval_info = evaluate_policy_distributed(
                env=eval_env,
                agent=agent,
                obs_rms=obs_rms,
                total_episodes=args.eval_episodes,
                accelerator=accelerator,
                deterministic=True,
                seed_base=args.seed + update_idx * 100000,
            )
            logging.info(f"evaluation number {update_idx}, finish.")

            agent.actor.train()
            agent.critic.train()

            if accelerator.is_main_process:
                history["updates"].append(update_idx)
                history["eval_return"].append(eval_info["avg_return"])
                history["avg_effective_throughput"].append(eval_info["avg_effective_throughput"])
                history["avg_ttft"].append(eval_info["avg_ttft"])
                history["ttft_p95"].append(eval_info["ttft_p95"])
                history["ttft_p99"].append(eval_info["ttft_p99"])
                history["ttft_tail5_avg_tx_time"].append(eval_info["ttft_tail5_avg_tx_time"])
                history["ttft_tail5_avg_compute_time"].append(eval_info["ttft_tail5_avg_compute_time"])
                history["ttft_tail1_avg_tx_time"].append(eval_info["ttft_tail1_avg_tx_time"])
                history["ttft_tail1_avg_compute_time"].append(eval_info["ttft_tail1_avg_compute_time"])
                history["avg_ul_energy"].append(eval_info["avg_ul_energy"])
                history["avg_gpu_energy"].append(eval_info["avg_gpu_energy"])
                history["actor_loss"].append(update_info["actor_loss"])
                history["critic_loss"].append(update_info["critic_loss"])
                history["entropy"].append(update_info["entropy"])
                history["kl"].append(update_info["kl"])

                logging.info(
                    f"[update {update_idx:04d} | steps {total_steps:07d}] "
                    f"return={eval_info['avg_return']:.3f} "
                    f"eta_eff={eval_info['avg_effective_throughput']:.3f} "
                    f"ttft={eval_info['avg_ttft']:.4f} "
                    f"ttft_p95={eval_info['ttft_p95']:.4f} "
                    f"ttft_p99={eval_info['ttft_p99']:.4f} "
                    f"tail5_tx={eval_info['ttft_tail5_avg_tx_time']:.4f} "
                    f"tail5_cl={eval_info['ttft_tail5_avg_compute_time']:.4f} "
                    f"tail1_tx={eval_info['ttft_tail1_avg_tx_time']:.4f} "
                    f"tail1_cl={eval_info['ttft_tail1_avg_compute_time']:.4f} "
                    f"e_ul={eval_info['avg_ul_energy']:.6f} "
                    f"e_gpu={eval_info['avg_gpu_energy']:.4f} "
                    f"pi_loss={update_info['actor_loss']:.4f} "
                    f"v_loss={update_info['critic_loss']:.4f} "
                    f"ent={update_info['entropy']:.4f} "
                    f"kl={update_info['kl']:.6f}"
                )

                plot_training_curves(history, args.out_dir)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process and (update_idx % args.save_every == 0 or total_steps >= args.total_steps):
            ckpt = {
                "agent": agent.state_dict(accelerator=accelerator),
                "obs_rms": obs_rms.state_dict(),
                "env_kwargs": env_kwargs,
                "ckm_kwargs": ckm_kwargs,
                "prefill_kwargs": prefill_kwargs,
                "gain_npz": args.gain_npz,
                "seed": args.seed,
                "update_idx": update_idx,
                "total_steps": total_steps,
                "history": history,
                "baseline_name": "baseline2_joint_without_ckm",
                "baseline2_region_avg_valid_only": region_avg_valid_only,
                "baseline2_common_mu": env.common_mu,
                "baseline2_common_sigma2": env.common_sigma2,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"baseline2_ppo_{update_idx:04d}.pt"))
        
        update_idx += 1
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_ckpt = {
            "agent": agent.state_dict(accelerator=accelerator),
            "obs_rms": obs_rms.state_dict(),
            "env_kwargs": env_kwargs,
            "ckm_kwargs": ckm_kwargs,
            "prefill_kwargs": prefill_kwargs,
            "gain_npz": args.gain_npz,
            "seed": args.seed,
            "update_idx": update_idx,
            "total_steps": total_steps,
            "history": history,
            "baseline_name": "baseline2_joint_without_ckm",
            "baseline2_region_avg_valid_only": region_avg_valid_only,
            "baseline2_common_mu": env.common_mu,
            "baseline2_common_sigma2": env.common_sigma2,
        }
        torch.save(final_ckpt, os.path.join(args.out_dir, "baseline2_ppo_final.pt"))
        save_json(history, os.path.join(args.out_dir, "train_history.json"))


if __name__ == "__main__":
    main()
