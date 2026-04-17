from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import math


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HybridActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_power_actions: int,
        num_batch_actions: int,
        batch_action_size: int,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_batch_actions = num_batch_actions
        self.batch_action_size = batch_action_size

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
            nn.Linear(hidden_dim, num_power_actions),
        )

        # 全局 log_std，训练更稳
        self.power_log_std = nn.Parameter(torch.zeros(num_power_actions))

        self.batch_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_batch_actions * batch_action_size),
        )

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.power_head[-1], gain=0.01)
        orthogonal_init(self.batch_head[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.input_proj(obs)
        feat = self.backbone(feat)

        power_mean = self.power_head(feat)
        power_log_std = torch.clamp(self.power_log_std, -5.0, 2.0)
        batch_logits = self.batch_head(feat).view(-1, self.num_batch_actions, self.batch_action_size)
        return power_mean, power_log_std, batch_logits


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


@dataclass
class PPOConfig:
    hidden_dim: int = 256
    depth: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"


class CKMPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        power_dim: int,
        batch_dim: int,
        power_low: np.ndarray,
        power_high: np.ndarray,
        batch_action_values: np.ndarray,
        cfg: PPOConfig,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.power_dim = int(power_dim)
        self.batch_dim = int(batch_dim)
        self.batch_action_values = np.asarray(batch_action_values, dtype=np.int64)
        self.batch_action_size = int(len(self.batch_action_values))
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.power_low = torch.tensor(power_low, dtype=torch.float32, device=self.device)
        self.power_high = torch.tensor(power_high, dtype=torch.float32, device=self.device)
        self.power_scale = (self.power_high - self.power_low) / 2.0
        self.power_bias = (self.power_high + self.power_low) / 2.0

        self.actor = HybridActor(
            obs_dim=obs_dim,
            hidden_dim=cfg.hidden_dim,
            num_power_actions=power_dim,
            num_batch_actions=batch_dim,
            batch_action_size=self.batch_action_size,
            depth=cfg.depth
        ).to(self.device)
        self.critic = Critic(obs_dim=obs_dim, hidden_dim=cfg.hidden_dim, depth=cfg.depth).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        
    def prepare(self, accelerator) -> None:
        # 多卡准备
        self.actor, self.critic, self.actor_opt, self.critic_opt = accelerator.prepare(
            self.actor, self.critic, self.actor_opt, self.critic_opt
        )
        self.device = accelerator.device
        self.power_low = self.power_low.to(self.device)
        self.power_high = self.power_high.to(self.device)
        self.power_scale = (self.power_high - self.power_low) / 2.0
        self.power_bias = (self.power_high + self.power_low) / 2.0

    def _squash_power(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _to_env_power(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.power_scale + self.power_bias

    def _from_env_power(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.power_bias) / self.power_scale

    def _power_log_prob_terms_from_u(self, dist: Normal, u: torch.Tensor) -> torch.Tensor:
        """
        返回每个 power 维度各自的 log-prob contribution，shape = [B, K]
        """
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
    def act(self, obs: np.ndarray, power_mask: np.ndarray, batch_mask: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, Dict[str, np.ndarray]]:
        
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        power_mask_t = torch.tensor(power_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch_mask_t = torch.tensor(batch_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        power_mean, power_log_std, batch_logits = self.actor(obs_t)
        power_dist = Normal(power_mean, torch.exp(power_log_std))
        batch_dist = Categorical(logits=batch_logits)
        value = self.critic(obs_t)

        if deterministic:
            u = power_dist.mean
            batch_action = torch.argmax(batch_dist.logits, dim=-1)
        else:
            u = power_dist.rsample()  # 1. 先在无界空间中采样
            batch_action = batch_dist.sample()

        power_env = self._to_env_power(self._squash_power(u))  # 2. 用tanh压到-1~1。  3. 从-1~1映射到环境动作区间

        power_logp = self._power_log_prob_from_u(power_dist, u, power_mask=power_mask_t)
        batch_logp = (batch_dist.log_prob(batch_action) * batch_mask_t).sum(dim=-1)
        logp = power_logp + batch_logp
        
        batch_action_np = batch_action.squeeze(0).cpu().numpy().astype(np.int64)  # 动作索引。维度是batch_action_size
        batch_action_env = self.batch_action_values[batch_action_np]  # 作用于环境的真实batch_size

        env_action = np.concatenate(
            [
                power_env.squeeze(0).cpu().numpy(),
                batch_action_env.astype(np.float32),
            ]
        ).astype(np.float32)
        
        aux = {
            "power_action": power_env.squeeze(0).cpu().numpy().astype(np.float32),
            "batch_action": batch_action_np,
            "power_mask": np.asarray(power_mask, dtype=np.float32),
            "batch_mask": np.asarray(batch_mask, dtype=np.float32),
        }
        return env_action, float(logp.item()), float(value.item()), aux

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        power_action: torch.Tensor,
        batch_action: torch.Tensor,
        power_mask: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        power_mean, power_log_std, batch_logits = self.actor(obs)
        power_dist = Normal(power_mean, torch.exp(power_log_std))
        batch_dist = Categorical(logits=batch_logits)
        value = self.critic(obs)

        z = torch.clamp(self._from_env_power(power_action), -0.999999, 0.999999)
        u = atanh(z)
        
        power_logp = self._power_log_prob_from_u(power_dist, u, power_mask=power_mask)
        batch_logp = (batch_dist.log_prob(batch_action) * batch_mask).sum(dim=-1)
        logp = power_logp + batch_logp
        
        u_ent = power_dist.rsample()  # entropy bonus的估计：从当前策略重新采样用一个样本（因为minibatch里的样本是旧策略rollout出来的），用于熵的蒙特卡罗估计
        power_entropy_terms = -self._power_log_prob_terms_from_u(power_dist, u_ent)
        power_entropy = (power_entropy_terms * power_mask).sum(dim=-1)

        batch_entropy = (batch_dist.entropy() * batch_mask).sum(dim=-1)
        entropy = power_entropy + batch_entropy
        
        return logp, entropy, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def update(self, batch: Dict[str, torch.Tensor], accelerator) -> Dict[str, float]:
        
        assert accelerator is not None, "accelerator must be passed in"
        
        power_mask = batch["power_mask"]
        batch_mask = batch["batch_mask"]
        obs = batch["obs"]
        power_action = batch["power_action"]
        batch_action = batch["batch_action"]
        old_logp = batch["logp"]
        adv = batch["adv"]
        ret = batch["ret"]
        
        # adv不在这里标准化

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
                mb_power_mask = power_mask[mb]
                mb_batch_mask = batch_mask[mb]
                mb_obs = obs[mb]
                mb_power = power_action[mb]
                mb_batch = batch_action[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_ret = ret[mb]

                new_logp, entropy, value = self.evaluate_actions(mb_obs, mb_power, mb_batch, mb_power_mask, mb_batch_mask)
                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = torch.mean((value - mb_ret) ** 2)

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
                    approx_kl = torch.mean(mb_old_logp - new_logp)

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
        
        assert accelerator is not None, "accelerator must be passed in"
        
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)
        
        return {
            "actor": actor.state_dict(),  # 返回unwrap的模型
            "critic": critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "power_dim": self.power_dim,
            "batch_dim": self.batch_dim,
            "batch_action_size": self.batch_action_size,
            "batch_action_values": self.batch_action_values,
            "obs_dim": self.obs_dim,
            "power_low": self.power_low.detach().cpu().numpy(),
            "power_high": self.power_high.detach().cpu().numpy(),
        }

    def load_state_dict(self, state: Dict[str, object], accelerator) -> None:
        
        assert accelerator is not None, "accelerator must be passed in"
        
        # checkpoint里保存的已经是unwarp的模型。这里的unwrap是把要被加载进去参数的模型unwrap，使其能load_state_dict
        actor = accelerator.unwrap_model(self.actor)
        critic = accelerator.unwrap_model(self.critic)

        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]  # 当前步的优势
        last_gae = delta + gamma * lam * next_non_terminal * last_gae  # GAE advantage的bootstrap
        adv[t] = last_gae
    ret = adv + values  # (T) 这个量是给critic / value loss用的
    return adv, ret
