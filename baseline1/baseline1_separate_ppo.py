from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ckm_ppo_agent import PPOConfig, atanh, orthogonal_init, ResidualMLPBlock, compute_gae


class PowerActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        power_dim: int,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, expansion=2) for _ in range(depth)]
        )
        self.power_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, power_dim),
        )
        self.power_log_std = nn.Parameter(torch.zeros(power_dim))

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.power_head[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.input_proj(obs)
        feat = self.backbone(feat)
        power_mean = self.power_head(feat)
        power_log_std = torch.clamp(self.power_log_std, -5.0, 2.0)
        return power_mean, power_log_std


class BatchActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        batch_dim: int,
        batch_action_size: int,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.batch_dim = int(batch_dim)
        self.batch_action_size = int(batch_action_size)

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, expansion=2) for _ in range(depth)]
        )
        self.batch_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, batch_dim * batch_action_size),
        )

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.batch_head[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(obs)
        feat = self.backbone(feat)
        logits = self.batch_head(feat)
        return logits.view(-1, self.batch_dim, self.batch_action_size)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, depth: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, expansion=2) for _ in range(depth)]
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.value_head[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(obs)
        feat = self.backbone(feat)
        return self.value_head(feat).squeeze(-1)


class PowerPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        power_dim: int,
        power_low: np.ndarray,
        power_high: np.ndarray,
        cfg: PPOConfig,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.power_dim = int(power_dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.power_low = torch.tensor(power_low, dtype=torch.float32, device=self.device)
        self.power_high = torch.tensor(power_high, dtype=torch.float32, device=self.device)
        self.power_scale = (self.power_high - self.power_low) / 2.0
        self.power_bias = (self.power_high + self.power_low) / 2.0

        self.actor = PowerActor(
            obs_dim=obs_dim,
            hidden_dim=cfg.hidden_dim,
            power_dim=power_dim,
            depth=cfg.depth,
        ).to(self.device)
        self.critic = Critic(obs_dim=obs_dim, hidden_dim=cfg.hidden_dim, depth=cfg.depth).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def prepare(self, accelerator) -> None:
        self.actor, self.critic, self.actor_opt, self.critic_opt = accelerator.prepare(
            self.actor, self.critic, self.actor_opt, self.critic_opt
        )
        self.device = accelerator.device
        self.power_low = self.power_low.to(self.device)
        self.power_high = self.power_high.to(self.device)
        self.power_scale = (self.power_high - self.power_low) / 2.0
        self.power_bias = (self.power_high + self.power_low) / 2.0

    def _to_env_power(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.power_scale + self.power_bias

    def _from_env_power(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.power_bias) / self.power_scale

    def _power_log_prob_terms_from_u(self, dist: Normal, u: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(u)
        logp_u = dist.log_prob(u)
        log_det = torch.log(1.0 - z * z + 1e-6)
        log_scale = torch.log(self.power_scale).view(1, -1)
        return logp_u - log_det - log_scale

    def _power_log_prob_from_u(
        self,
        dist: Normal,
        u: torch.Tensor,
        power_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logp_terms = self._power_log_prob_terms_from_u(dist, u)
        if power_mask is not None:
            logp_terms = logp_terms * power_mask
        return logp_terms.sum(dim=-1)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        power_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, np.ndarray]]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        power_mask_t = torch.tensor(power_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        power_mean, power_log_std = self.actor(obs_t)
        power_dist = Normal(power_mean, torch.exp(power_log_std))
        value = self.critic(obs_t)

        if deterministic:
            u = power_dist.mean
        else:
            u = power_dist.rsample()

        power_env = self._to_env_power(torch.tanh(u))
        power_logp = self._power_log_prob_from_u(power_dist, u, power_mask=power_mask_t)

        aux = {
            "power_action": power_env.squeeze(0).cpu().numpy().astype(np.float32),
            "power_mask": np.asarray(power_mask, dtype=np.float32),
        }
        return (
            power_env.squeeze(0).cpu().numpy().astype(np.float32),
            float(power_logp.item()),
            float(value.item()),
            aux,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        power_action: torch.Tensor,
        power_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        power_mean, power_log_std = self.actor(obs)
        power_dist = Normal(power_mean, torch.exp(power_log_std))
        value = self.critic(obs)

        z = torch.clamp(self._from_env_power(power_action), -0.999999, 0.999999)
        u = atanh(z)
        logp = self._power_log_prob_from_u(power_dist, u, power_mask=power_mask)

        u_ent = power_dist.rsample()
        power_entropy_terms = -self._power_log_prob_terms_from_u(power_dist, u_ent)
        entropy = (power_entropy_terms * power_mask).sum(dim=-1)
        return logp, entropy, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def update(self, batch: Dict[str, torch.Tensor], accelerator) -> Dict[str, float]:
        obs = batch["obs"]
        power_mask = batch["power_mask"]
        power_action = batch["power_action"]
        old_logp = batch["logp"]
        adv = batch["adv"]
        ret = batch["ret"]

        n = obs.shape[0]
        indices = torch.arange(n, device=self.device)
        actor_losses = []
        critic_losses = []
        entropies = []
        kls = []

        for _ in range(self.cfg.update_epochs):
            perm = indices[torch.randperm(n, device=indices.device)]
            for start in range(0, n, self.cfg.minibatch_size):
                mb = perm[start:start + self.cfg.minibatch_size]

                new_logp, entropy, value = self.evaluate_actions(
                    obs[mb],
                    power_action[mb],
                    power_mask[mb],
                )
                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[mb]
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = torch.mean((value - ret[mb]) ** 2)

                self.actor_opt.zero_grad(set_to_none=True)
                actor_obj = actor_loss - self.cfg.ent_coef * torch.mean(entropy)
                accelerator.backward(actor_obj)
                accelerator.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad(set_to_none=True)
                critic_obj = self.cfg.vf_coef * critic_loss
                accelerator.backward(critic_obj)
                accelerator.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kl = torch.mean(old_logp[mb] - new_logp)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(torch.mean(entropy).item())
                kls.append(approx_kl.item())

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "kl": float(np.mean(kls)) if kls else 0.0,
        }

    def state_dict(self, accelerator) -> Dict[str, object]:
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "power_dim": self.power_dim,
            "power_low": self.power_low.detach().cpu().numpy(),
            "power_high": self.power_high.detach().cpu().numpy(),
        }

    def load_state_dict(self, state: Dict[str, object], accelerator) -> None:
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])


class BatchPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        batch_dim: int,
        batch_action_values: np.ndarray,
        cfg: PPOConfig,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.batch_dim = int(batch_dim)
        self.batch_action_values = np.asarray(batch_action_values, dtype=np.int64)
        self.batch_action_size = int(len(self.batch_action_values))
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = BatchActor(
            obs_dim=obs_dim,
            hidden_dim=cfg.hidden_dim,
            batch_dim=batch_dim,
            batch_action_size=self.batch_action_size,
            depth=cfg.depth,
        ).to(self.device)
        self.critic = Critic(obs_dim=obs_dim, hidden_dim=cfg.hidden_dim, depth=cfg.depth).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def prepare(self, accelerator) -> None:
        self.actor, self.critic, self.actor_opt, self.critic_opt = accelerator.prepare(
            self.actor, self.critic, self.actor_opt, self.critic_opt
        )
        self.device = accelerator.device

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        batch_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, np.ndarray]]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch_mask_t = torch.tensor(batch_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        value = self.critic(obs_t)

        if deterministic:
            batch_action = torch.argmax(dist.logits, dim=-1)
        else:
            batch_action = dist.sample()

        logp = (dist.log_prob(batch_action) * batch_mask_t).sum(dim=-1)
        batch_action_np = batch_action.squeeze(0).cpu().numpy().astype(np.int64)
        batch_action_env = self.batch_action_values[batch_action_np].astype(np.float32)

        aux = {
            "batch_action": batch_action_np,
            "batch_mask": np.asarray(batch_mask, dtype=np.float32),
        }
        return batch_action_env, float(logp.item()), float(value.item()), aux

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        batch_action: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        value = self.critic(obs)

        logp = (dist.log_prob(batch_action) * batch_mask).sum(dim=-1)
        entropy = (dist.entropy() * batch_mask).sum(dim=-1)
        return logp, entropy, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def update(self, batch: Dict[str, torch.Tensor], accelerator) -> Dict[str, float]:
        obs = batch["obs"]
        batch_mask = batch["batch_mask"]
        batch_action = batch["batch_action"]
        old_logp = batch["logp"]
        adv = batch["adv"]
        ret = batch["ret"]

        n = obs.shape[0]
        indices = torch.arange(n, device=self.device)
        actor_losses = []
        critic_losses = []
        entropies = []
        kls = []

        for _ in range(self.cfg.update_epochs):
            perm = indices[torch.randperm(n, device=indices.device)]
            for start in range(0, n, self.cfg.minibatch_size):
                mb = perm[start:start + self.cfg.minibatch_size]

                new_logp, entropy, value = self.evaluate_actions(
                    obs[mb],
                    batch_action[mb],
                    batch_mask[mb],
                )
                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[mb]
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = torch.mean((value - ret[mb]) ** 2)

                self.actor_opt.zero_grad(set_to_none=True)
                actor_obj = actor_loss - self.cfg.ent_coef * torch.mean(entropy)
                accelerator.backward(actor_obj)
                accelerator.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad(set_to_none=True)
                critic_obj = self.cfg.vf_coef * critic_loss
                accelerator.backward(critic_obj)
                accelerator.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kl = torch.mean(old_logp[mb] - new_logp)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(torch.mean(entropy).item())
                kls.append(approx_kl.item())

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "kl": float(np.mean(kls)) if kls else 0.0,
        }

    def state_dict(self, accelerator) -> Dict[str, object]:
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "batch_dim": self.batch_dim,
            "batch_action_values": self.batch_action_values,
            "batch_action_size": self.batch_action_size,
        }

    def load_state_dict(self, state: Dict[str, object], accelerator) -> None:
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])
