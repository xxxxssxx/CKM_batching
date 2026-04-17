from __future__ import annotations

import argparse
import math
import os
import logging

import numpy as np
import torch
from accelerate import Accelerator

from baseline1_separate_ppo import BatchPPOAgent, PowerPPOAgent
from baseline1_separate_ppo_utils import (
    collect_rollout_separate,
    evaluate_policy_distributed_separate,
    get_split_obs_dims,
    global_normalize_adv,
    load_split_checkpoint,
    reduce_agent_stats_with_prefix,
)
from ckm_ppo_agent import PPOConfig, compute_gae
from ckm_ppo_utils import (
    RunningMeanStd,
    make_ckm_env,
    save_json,
    set_seed,
    setup_logging,
    sync_obs_rms,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gain_npz", type=str, default="./gain.npz")
    parser.add_argument("--out_dir", type=str, default="./ckm_ppo_baseline1_out")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--reset_history", action="store_true")

    parser.add_argument("--T", type=float, default=120)
    parser.add_argument("--delta_t", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=550.0)
    parser.add_argument("--lam_amp", type=float, default=0.0)
    parser.add_argument("--lam_omega", type=float, default=2 * np.pi * 5 / 120)
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
    parser.add_argument("--k2", type=float, default=5.0)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=8192*16*8)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--total_steps", type=int, default=30 * 1.2 * 1e6 * 8)
    parser.add_argument("--rollout_steps", type=int, default=1.2 * 1e6 * 8)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=1 * 8)
    parser.add_argument("--eval_T", type=float, default=30)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    setup_logging(rank, log_path="logtrain_baseline1.log")
    logging.info("Start baseline1 training.")

    set_seed(args.seed + rank)

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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

    env = make_ckm_env(
        gain_npz_path=args.gain_npz,
        env_kwargs=train_env_kwargs,
        ckm_kwargs=ckm_kwargs,
        prefill_kwargs=prefill_kwargs,
    )

    tx_obs_dim, compute_obs_dim = get_split_obs_dims(env)
    power_low = np.zeros(env.K, dtype=np.float32)
    power_high = np.full(env.K, env.p_max, dtype=np.float32)
    batch_action_values = np.arange(0, env.Bmax + 1, args.batch_step, dtype=np.int64)
    if batch_action_values[-1] != env.Bmax:
        batch_action_values = np.append(batch_action_values, env.Bmax)

    rollout_steps_per_rank = math.ceil(args.rollout_steps / accelerator.num_processes)
    minibatch_size_per_rank = max(1, math.ceil(args.minibatch_size / accelerator.num_processes))
    ppo_cfg = PPOConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
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

    tx_agent = PowerPPOAgent(
        obs_dim=tx_obs_dim,
        power_dim=env.K,
        power_low=power_low,
        power_high=power_high,
        cfg=ppo_cfg,
    )
    compute_agent = BatchPPOAgent(
        obs_dim=compute_obs_dim,
        batch_dim=env.N,
        batch_action_values=batch_action_values,
        cfg=ppo_cfg,
    )
    tx_obs_rms = RunningMeanStd(shape=(tx_obs_dim,))
    compute_obs_rms = RunningMeanStd(shape=(compute_obs_dim,))

    tx_agent.prepare(accelerator)
    compute_agent.prepare(accelerator)
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
        "tx_actor_loss": [],
        "tx_critic_loss": [],
        "tx_entropy": [],
        "tx_kl": [],
        "compute_actor_loss": [],
        "compute_critic_loss": [],
        "compute_entropy": [],
        "compute_kl": [],
    }

    total_steps = 0
    update_idx = 0

    if args.resume_path is not None:
        update_idx, total_steps, ckpt = load_split_checkpoint(
            args.resume_path,
            tx_agent,
            compute_agent,
            tx_obs_rms,
            compute_obs_rms,
            accelerator,
            device=str(device),
        )
        if accelerator.is_main_process and (not args.reset_history):
            history = ckpt.get("history", history)
    accelerator.wait_for_everyone()

    while total_steps < args.total_steps:
        logging.info(f"rollout number {update_idx}, start.")
        batch_np = collect_rollout_separate(
            env=env,
            tx_agent=tx_agent,
            compute_agent=compute_agent,
            tx_obs_rms=tx_obs_rms,
            compute_obs_rms=compute_obs_rms,
            batch_size=rollout_steps_per_rank,
            deterministic=False,
        )
        logging.info(f"rollout number {update_idx}, finish.")

        with torch.no_grad():
            tx_last_obs_n = torch.tensor(batch_np["tx_last_obs_n"], dtype=torch.float32, device=tx_agent.device).unsqueeze(0)
            compute_last_obs_n = torch.tensor(batch_np["compute_last_obs_n"], dtype=torch.float32, device=compute_agent.device).unsqueeze(0)
            tx_last_value = tx_agent.value(tx_last_obs_n).item()
            compute_last_value = compute_agent.value(compute_last_obs_n).item()

        if batch_np["last_done"] > 0.5:
            tx_last_value = 0.0
            compute_last_value = 0.0

        tx_adv, tx_ret = compute_gae(
            rewards=batch_np["rew"],
            dones=batch_np["done"],
            values=batch_np["tx_val"],
            last_value=tx_last_value,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )
        compute_adv, compute_ret = compute_gae(
            rewards=batch_np["rew"],
            dones=batch_np["done"],
            values=batch_np["batch_val"],
            last_value=compute_last_value,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )

        sync_obs_rms(tx_obs_rms, batch_np["tx_raw_obs"], accelerator)
        sync_obs_rms(compute_obs_rms, batch_np["compute_raw_obs"], accelerator)

        tx_adv_t = global_normalize_adv(tx_adv, accelerator, tx_agent.device)
        compute_adv_t = global_normalize_adv(compute_adv, accelerator, compute_agent.device)

        tx_batch = {
            "obs": torch.tensor(batch_np["tx_obs"], dtype=torch.float32, device=tx_agent.device),
            "power_mask": torch.tensor(batch_np["power_mask"], dtype=torch.float32, device=tx_agent.device),
            "power_action": torch.tensor(batch_np["power_action"], dtype=torch.float32, device=tx_agent.device),
            "logp": torch.tensor(batch_np["tx_logp"], dtype=torch.float32, device=tx_agent.device),
            "adv": tx_adv_t,
            "ret": torch.tensor(tx_ret, dtype=torch.float32, device=tx_agent.device),
        }
        compute_batch = {
            "obs": torch.tensor(batch_np["compute_obs"], dtype=torch.float32, device=compute_agent.device),
            "batch_mask": torch.tensor(batch_np["batch_mask"], dtype=torch.float32, device=compute_agent.device),
            "batch_action": torch.tensor(batch_np["batch_action"], dtype=torch.long, device=compute_agent.device),
            "logp": torch.tensor(batch_np["batch_logp"], dtype=torch.float32, device=compute_agent.device),
            "adv": compute_adv_t,
            "ret": torch.tensor(compute_ret, dtype=torch.float32, device=compute_agent.device),
        }

        logging.info(f"update number {update_idx}, tx start.")
        tx_update_info = tx_agent.update(tx_batch, accelerator=accelerator)
        logging.info(f"update number {update_idx}, tx finish.")

        logging.info(f"update number {update_idx}, compute start.")
        compute_update_info = compute_agent.update(compute_batch, accelerator=accelerator)
        logging.info(f"update number {update_idx}, compute finish.")

        tx_update_info = reduce_agent_stats_with_prefix(tx_update_info, prefix="tx", accelerator=accelerator)
        compute_update_info = reduce_agent_stats_with_prefix(compute_update_info, prefix="compute", accelerator=accelerator)

        global_rollout_steps = torch.tensor(batch_np["tx_obs"].shape[0], dtype=torch.int32, device=device)
        global_rollout_steps = int(accelerator.reduce(global_rollout_steps, reduction="sum").item())
        total_steps += global_rollout_steps

        if update_idx % args.eval_every == 0 or total_steps >= args.total_steps:
            accelerator.wait_for_everyone()
            eval_env_kwargs = dict(env_kwargs)
            eval_env_kwargs["T"] = args.eval_T
            eval_env_kwargs["seed_value"] = args.seed + update_idx * 2026 + rank * 100003

            eval_env = make_ckm_env(
                gain_npz_path=args.gain_npz,
                env_kwargs=eval_env_kwargs,
                ckm_kwargs=ckm_kwargs,
                prefill_kwargs=prefill_kwargs,
            )

            tx_agent.actor.eval()
            tx_agent.critic.eval()
            compute_agent.actor.eval()
            compute_agent.critic.eval()

            logging.info(f"evaluation number {update_idx}, start.")
            eval_info = evaluate_policy_distributed_separate(
                env=eval_env,
                tx_agent=tx_agent,
                compute_agent=compute_agent,
                tx_obs_rms=tx_obs_rms,
                compute_obs_rms=compute_obs_rms,
                total_episodes=args.eval_episodes,
                accelerator=accelerator,
                deterministic=True,
                seed_base=args.seed + update_idx * 100000,
            )
            logging.info(f"evaluation number {update_idx}, finish.")

            tx_agent.actor.train()
            tx_agent.critic.train()
            compute_agent.actor.train()
            compute_agent.critic.train()

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
                history["tx_actor_loss"].append(tx_update_info["tx_actor_loss"])
                history["tx_critic_loss"].append(tx_update_info["tx_critic_loss"])
                history["tx_entropy"].append(tx_update_info["tx_entropy"])
                history["tx_kl"].append(tx_update_info["tx_kl"])
                history["compute_actor_loss"].append(compute_update_info["compute_actor_loss"])
                history["compute_critic_loss"].append(compute_update_info["compute_critic_loss"])
                history["compute_entropy"].append(compute_update_info["compute_entropy"])
                history["compute_kl"].append(compute_update_info["compute_kl"])

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
                    f"tx_pi_loss={tx_update_info['tx_actor_loss']:.4f} "
                    f"tx_v_loss={tx_update_info['tx_critic_loss']:.4f} "
                    f"tx_ent={tx_update_info['tx_entropy']:.4f} "
                    f"tx_kl={tx_update_info['tx_kl']:.6f} "
                    f"cl_pi_loss={compute_update_info['compute_actor_loss']:.4f} "
                    f"cl_v_loss={compute_update_info['compute_critic_loss']:.4f} "
                    f"cl_ent={compute_update_info['compute_entropy']:.4f} "
                    f"cl_kl={compute_update_info['compute_kl']:.6f}"
                )

            accelerator.wait_for_everyone()

        if accelerator.is_main_process and (update_idx % args.save_every == 0 or total_steps >= args.total_steps):
            ckpt = {
                "tx_agent": tx_agent.state_dict(accelerator=accelerator),
                "compute_agent": compute_agent.state_dict(accelerator=accelerator),
                "tx_obs_rms": tx_obs_rms.state_dict(),
                "compute_obs_rms": compute_obs_rms.state_dict(),
                "env_kwargs": env_kwargs,
                "ckm_kwargs": ckm_kwargs,
                "prefill_kwargs": prefill_kwargs,
                "gain_npz": args.gain_npz,
                "seed": args.seed,
                "update_idx": update_idx,
                "total_steps": total_steps,
                "history": history,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"baseline1_ckm_ppo_{update_idx:04d}.pt"))
        
        update_idx += 1
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_ckpt = {
            "tx_agent": tx_agent.state_dict(accelerator=accelerator),
            "compute_agent": compute_agent.state_dict(accelerator=accelerator),
            "tx_obs_rms": tx_obs_rms.state_dict(),
            "compute_obs_rms": compute_obs_rms.state_dict(),
            "env_kwargs": env_kwargs,
            "ckm_kwargs": ckm_kwargs,
            "prefill_kwargs": prefill_kwargs,
            "gain_npz": args.gain_npz,
            "seed": args.seed,
            "update_idx": update_idx,
            "total_steps": total_steps,
            "history": history,
        }
        torch.save(final_ckpt, os.path.join(args.out_dir, "baseline1_ckm_ppo_final.pt"))
        save_json(history, os.path.join(args.out_dir, "train_history.json"))


if __name__ == "__main__":
    main()
