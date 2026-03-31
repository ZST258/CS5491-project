from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .env import DynamicNavigationEnv
from .models.common import BasePolicy


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000
    rollout_steps: int = 512
    update_epochs: int = 2
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    log_interval: int = 1


@dataclass
class RolloutBatch:
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)


class PPOTrainer:
    def __init__(self, env: DynamicNavigationEnv, model: BasePolicy, config: PPOConfig, seed: int = 0):
        self.env = env
        self.model = model
        self.config = config
        self.device = model.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.seed = seed
        self.global_step = 0
        self.training_history: list[dict[str, float]] = []

        # AMP：只要 device 是 cuda 就自动开启（train.py 无需改）
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self) -> dict[str, float]:
        observation, _ = self.env.reset(seed=self.seed)
        episode_return = 0.0
        last_logs: dict[str, float] = {}
        rollout_index = 0

        while self.global_step < self.config.total_timesteps:
            batch = RolloutBatch()

            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self.model.act(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                batch.observations.append(observation)
                batch.actions.append(action)
                batch.log_probs.append(log_prob)
                batch.rewards.append(reward)
                batch.dones.append(done)
                batch.values.append(value)

                episode_return += reward
                self.global_step += 1
                observation = next_observation

                if done:
                    batch.episode_returns.append(episode_return)
                    observation, _ = self.env.reset(seed=self.seed + self.global_step)
                    episode_return = 0.0

                if self.global_step >= self.config.total_timesteps:
                    break

            with torch.no_grad():
                _, _, next_value = self.model.act(observation)

            returns, advantages = self._compute_returns_and_advantages(batch, next_value)
            last_logs = self._update(batch, returns, advantages)

            if batch.episode_returns:
                last_logs["episode_return"] = float(np.mean(batch.episode_returns))

            if rollout_index % self.config.log_interval == 0:
                history_row = {"global_step": float(self.global_step), **last_logs}
                self.training_history.append(history_row)

            rollout_index += 1

        return last_logs

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "metadata": metadata or {}, "global_step": self.global_step},
            output_path,
        )
        return output_path

    def _compute_returns_and_advantages(self, batch: RolloutBatch, next_value: float):
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        values = np.asarray(batch.values + [next_value], dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * values[step + 1] * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + values[:-1]
        return returns, advantages

    def _update(self, batch: RolloutBatch, returns: np.ndarray, advantages: np.ndarray) -> dict[str, float]:
        num_steps = len(batch.actions)
        batch_indices = np.arange(num_steps, dtype=np.int64)

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        actions_np = np.asarray(batch.actions, dtype=np.int64)
        actions_tensor = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)

        old_log_probs = torch.as_tensor(np.asarray(batch.log_probs, dtype=np.float32), dtype=torch.float32, device=self.device)
        dones_np = np.asarray(batch.dones, dtype=np.bool_)

        all_action_masks = torch.stack(
            [torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device) for obs in batch.observations],
            dim=0,
        )

        logs: dict[str, float] = {}
        aux_loss_executed = False

        can_use_latent_path = hasattr(self.model, "encode") and hasattr(self.model, "evaluate_actions_from_latent")
        all_latents_detached = None

        # 缓存无梯度 latent，避免重复编码 + 避免二次 backward 图复用错误
        if can_use_latent_path:
            with torch.no_grad():
                all_latents_detached = self.model.encode(batch.observations).detach()

        for epoch in range(self.config.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, num_steps, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_idx = batch_indices[start:end]

                # 前向（AMP autocast）
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    if all_latents_detached is not None:
                        log_probs, entropy, values = self.model.evaluate_actions_from_latent(
                            latents=all_latents_detached[mb_idx],
                            action_masks=all_action_masks[mb_idx],
                            actions=actions_tensor[mb_idx],
                        )
                    else:
                        observations_mb = [batch.observations[i] for i in mb_idx]
                        log_probs, entropy, values = self.model.evaluate_actions(observations_mb, actions_tensor[mb_idx])

                    ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                    unclipped = ratio * advantages_tensor[mb_idx]
                    clipped = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * advantages_tensor[mb_idx]

                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * ((values - returns_tensor[mb_idx]) ** 2).mean()
                    entropy_loss = entropy.mean()

                    if not aux_loss_executed:
                        # aux 单独重新带梯度算一次，避免图复用问题
                        aux_loss = self.model.auxiliary_loss(batch.observations, actions_np, dones_np)
                        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss + aux_loss
                        logs["aux_loss"] = float(aux_loss.item())
                        aux_loss_executed = True
                    else:
                        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                if epoch == self.config.update_epochs - 1:
                    logs.update(
                        {
                            "loss": float(loss.item()),
                            "policy_loss": float(policy_loss.item()),
                            "value_loss": float(value_loss.item()),
                            "entropy": float(entropy_loss.item()),
                        }
                    )

        return logs