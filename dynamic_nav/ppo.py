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
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    log_interval: int = 1
    target_kl: float | None = None
    policy_lr: float | None = None
    value_lr: float | None = None
    lr_schedule: bool = False
    # Note: returns normalization option removed. The critic should learn
    # values on the original reward scale so advantages remain calibrated.
    value_weight_decay: float = 0.0
    # PredictivePolicy 专用：前N步不训练预测器，等encoder收敛
    aux_warmup_percent: float = 0.15
    # PopArt (always enabled)
    popart_beta: float = 0.975
    popart_eps: float = 1e-5
    popart_min_sigma: float = 1e-3


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
        self.seed = seed
        self.global_step = 0
        self.training_history: list[dict[str, float]] = []
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # 独立 scaler 供 aux optimizer 使用
        self.aux_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.aux_param_names = ("obstacle_gru", "step_embed", "_logit_gamma")
        # ── 主 optimizer ─────────────────────────────────────────
        if config.policy_lr is not None or config.value_lr is not None:
            policy_lr = config.policy_lr or config.learning_rate
            value_lr = config.value_lr or config.learning_rate
            policy_params, value_params = [], []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "value" in name or "critic" in name:
                    value_params.append(p)
                else:
                    policy_params.append(p)
            param_groups = []
            if policy_params:
                param_groups.append({"params": policy_params, "lr": policy_lr})
            if value_params:
                param_groups.append({"params": value_params, "lr": value_lr})
            self.optimizer = torch.optim.Adam(param_groups)
        else:
            main_params = [
                p for name, p in self.model.named_parameters()
                if p.requires_grad and not name.startswith(self.aux_param_names)
            ]
            self.optimizer = torch.optim.Adam(main_params, lr=config.learning_rate)

        # ── aux optimizer：只更新预测器参数，与主optimizer完全独立 ──
        self.aux_optimizer = None
        if getattr(self.model, "model_name", None) == "predictive":
            # predictor-specific parameters (obstacle_gru is the predictor module)
            aux_params = [
                p for name, p in self.model.named_parameters()
                if any(name.startswith(n) for n in self.aux_param_names)
                and p.requires_grad
            ]
            if aux_params:
                self.aux_optimizer = torch.optim.Adam(
                    aux_params, lr=config.policy_lr or config.learning_rate
                )

        if self.config.lr_schedule:
            # Compute total optimizer steps expected across training:
            # total_opt_steps = total_timesteps * update_epochs / minibatch_size
            total_opt_steps = max(
                1,
                int((self.config.total_timesteps * max(1, self.config.update_epochs)) // max(1, self.config.minibatch_size)),
            )
            # Lambda receives optimizer step index (0-based). Use that to schedule linearly.
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(1.0 - float(step) / float(total_opt_steps), 0.0),
            )
        else:
            self.lr_scheduler = None

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
                output = self.model.forward_batch([observation])
                next_value = float(
                    self.model.popart_denormalize(output.value.squeeze(0)).item()
                )

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
            {
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
                "global_step": self.global_step,
            },
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

        # --- PopArt: update running stats using RAW returns (once per rollout update) ---
        self.model.popart_update(
            targets_raw=returns_tensor,
            beta=float(self.config.popart_beta),
            eps=float(self.config.popart_eps),
            min_sigma=float(self.config.popart_min_sigma),
        )
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        # normalize advantages for stable learning but keep returns on the
        # original reward scale so the critic learns true value magnitudes.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        actions_np = np.asarray(batch.actions, dtype=np.int64)
        actions_tensor = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            np.asarray(batch.log_probs, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        dones_np = np.asarray(batch.dones, dtype=np.bool_)
        all_action_masks = torch.stack(
            [torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device)
             for obs in batch.observations],
            dim=0,
        )

        logs: dict[str, float] = {}
        is_predictive = getattr(self.model, "model_name", None) == "predictive"
        aux_warmup = getattr(self.config, "aux_warmup_percent", 0)

        # --- Diagnostics: normalized returns statistics (for debugging) ---
        eps = float(self.config.popart_eps)
        min_sigma = float(self.config.popart_min_sigma)
        returns_norm_dbg = self.model.popart_normalize(returns_tensor, eps=eps, min_sigma=min_sigma)
        logs["returns_norm_mean"] = float(returns_norm_dbg.mean().cpu().numpy())
        logs["returns_norm_std"] = float(returns_norm_dbg.std().cpu().numpy())

        # ── Aux loss：原始rollout顺序，独立optimizer，只更新预测器参数 ──
        if is_predictive and self.aux_optimizer is not None:
            if self.global_step >= aux_warmup * self.config.total_timesteps:
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    aux_loss = self.model.auxiliary_loss(batch.observations, actions_np, dones_np)

                self.aux_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.aux_scaler.scale(aux_loss).backward()
                    self.aux_scaler.unscale_(self.aux_optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for g in self.aux_optimizer.param_groups for p in g["params"]],
                        self.config.max_grad_norm,
                    )
                    self.aux_scaler.step(self.aux_optimizer)
                    self.aux_scaler.update()
                else:
                    aux_loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for g in self.aux_optimizer.param_groups for p in g["params"]],
                        self.config.max_grad_norm,
                    )
                    self.aux_optimizer.step()

                # 清零encoder等参数上的残留梯度，防止污染后续PPO更新
                self.optimizer.zero_grad(set_to_none=True)
                logs["aux_loss"] = float(aux_loss.item())

                # If the predictive model recorded diagnostics during aux loss
                # calculation, merge them into the logs so they appear in training history.
                if hasattr(self.model, "_last_aux_diag") and isinstance(self.model._last_aux_diag, dict):
                    # only copy known scalar keys to avoid polluting logs with tensors
                    for k, v in self.model._last_aux_diag.items():
                        # ensure JSON serializable floats/ints
                        try:
                            logs[str(k)] = float(v)
                        except Exception:
                            try:
                                logs[str(k)] = int(v)
                            except Exception:
                                logs[str(k)] = v
            else:
                logs["aux_loss"] = 0.0

        # ── PPO minibatch 更新 ────────────────────────────────────
        has_encode = hasattr(self.model, "encode")
        fwd_from_latent_impl = (
                getattr(self.model.__class__, "forward_from_latent", None)
                is not BasePolicy.forward_from_latent
        )

        # Allow using BasePolicy.evaluate_actions_from_latent (it is valid as long as forward_from_latent exists)
        can_use_latent_path = has_encode and fwd_from_latent_impl

        # auxiliary_loss_from_latent is a no-op in BasePolicy. Detect whether
        # the concrete model class overrides it so we avoid calling the
        # BasePolicy placeholder.
        aux_from_latent_impl = (
            getattr(self.model.__class__, "auxiliary_loss_from_latent", None)
            is not BasePolicy.auxiliary_loss_from_latent
        )

        continue_training = True

        for epoch in range(self.config.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, num_steps, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_idx = batch_indices[start:end]

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    observations_mb = [batch.observations[i] for i in mb_idx]

                    if can_use_latent_path:
                        # encode once and reuse latents for both evaluation and
                        # computing the critic's normalized output to avoid
                        # duplicated encoder work.
                        latents_mb = self.model.encode(observations_mb)
                        forward_out = self.model.forward_from_latent(latents_mb)
                    else:
                        forward_out = self.model.forward_batch(observations_mb)

                    dist = torch.distributions.Categorical(
                        logits=forward_out.logits.masked_fill(
                            all_action_masks[mb_idx] <= 0,
                            torch.finfo(forward_out.logits.dtype).min
                            if forward_out.logits.dtype in (torch.float16, torch.bfloat16)
                            else -1e9,
                        )
                    )
                    log_probs = dist.log_prob(actions_tensor[mb_idx])
                    entropy = dist.entropy()
                    values = self.model.popart_denormalize(forward_out.value).squeeze(-1)

                    log_ratio = log_probs - old_log_probs[mb_idx]
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * advantages_tensor[mb_idx]
                    clipped = (
                        torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                        * advantages_tensor[mb_idx]
                    )
                    policy_loss = -torch.min(unclipped, clipped).mean()

                    # --- PopArt value loss (normalized space) ---
                    # values / old_values_tensor / returns_tensor are RAW values
                    eps = float(self.config.popart_eps)
                    min_sigma = float(self.config.popart_min_sigma)

                    # Use the model's normalized-value output (v_norm) as the
                    # prediction for the normalized-space value loss. Forward
                    # through the policy network (latent or non-latent path)
                    # using the minibatch observations to get the normalized
                    # values directly from the critic head.
                    # Note: values (from evaluate_actions) are raw (denormalized)
                    # values returned for logging; we compute the loss in the
                    # normalized space via forward_batch.
                    values_pred_norm = forward_out.value.squeeze(-1)

                    returns_norm_mb = self.model.popart_normalize(
                        returns_tensor[mb_idx],
                        eps=eps,
                        min_sigma=min_sigma,
                    )

                    value_loss = 0.5 * ((values_pred_norm - returns_norm_mb) ** 2).mean()
                    entropy_loss = entropy.mean()

                    # predictive的aux loss已由独立optimizer处理，minibatch内置0
                    if is_predictive:
                        aux_loss_mb = torch.zeros((), device=self.device)
                    elif can_use_latent_path and aux_from_latent_impl:
                        aux_loss_mb = self.model.auxiliary_loss_from_latent(
                            latents_mb, actions_np[mb_idx], dones_np[mb_idx]
                        )
                    else:
                        aux_loss_mb = self.model.auxiliary_loss(
                            observations_mb, actions_np[mb_idx], dones_np[mb_idx]
                        )

                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy_loss
                        + aux_loss_mb
                    )

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    prev_scale = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    # Apply value weight decay when using AMP unscaled gradients
                    if self.config.value_weight_decay and self.config.value_weight_decay > 0.0:
                        for name, p in self.model.named_parameters():
                            if p.grad is None:
                                continue
                            if "value" in name or "critic" in name:
                                p.grad.add_(self.config.value_weight_decay * p.detach())
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.lr_scheduler is not None and self.scaler.get_scale() >= prev_scale:
                        try:
                            self.lr_scheduler.step()
                        except Exception:
                            pass
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    if self.config.value_weight_decay and self.config.value_weight_decay > 0.0:
                        for name, p in self.model.named_parameters():
                            if p.grad is None:
                                continue
                            if "value" in name or "critic" in name:
                                p.grad.add_(self.config.value_weight_decay * p.detach())
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        try:
                            self.lr_scheduler.step()
                        except Exception:
                            pass

                with torch.no_grad():
                    approx_kl = float(torch.mean((ratio - 1) - log_ratio).cpu().numpy())
                    clip_fraction = float(
                        torch.mean((torch.abs(ratio - 1) > self.config.clip_coef).float()).cpu().numpy()
                    )

                    # raw value stats (true reward scale)
                    value_pred_raw_mean = float(values.mean().cpu().numpy())
                    value_pred_raw_std = float(values.std().cpu().numpy())

                    # normalized value stats (should be O(1))
                    eps = float(self.config.popart_eps)
                    min_sigma = float(self.config.popart_min_sigma)
                    # diagnostics: use the network's normalized outputs on this
                    # minibatch for mean/std reporting
                    values_pred_norm_dbg = forward_out.value.squeeze(-1)
                    value_pred_mean = float(values_pred_norm_dbg.mean().cpu().numpy())
                    value_pred_std = float(values_pred_norm_dbg.std().cpu().numpy())

                if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
                    continue_training = False
                    logs.update({
                        "approx_kl": approx_kl,
                        "clip_fraction": clip_fraction,
                        "value_pred_mean": value_pred_mean,
                        "value_pred_std": value_pred_std,
                        "value_pred_raw_mean": value_pred_raw_mean,
                        "value_pred_raw_std": value_pred_raw_std,
                        "popart_mu": float(self.model.popart_mu.item()),
                        "popart_sigma": float(self.model.popart_sigma.item()),
                        "returns_mean": float(returns_tensor.mean().cpu().numpy()),
                        "returns_std": float(returns_tensor.std().cpu().numpy()),
                    })
                    break

                if epoch == self.config.update_epochs - 1:
                    logs.update({
                        "loss": float(loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                        "value_pred_mean": value_pred_mean,
                        "value_pred_std": value_pred_std,
                        "value_pred_raw_mean": value_pred_raw_mean,
                        "value_pred_raw_std": value_pred_raw_std,
                        "popart_mu": float(self.model.popart_mu.item()),
                        "popart_sigma": float(self.model.popart_sigma.item()),
                        "returns_mean": float(returns_tensor.mean().cpu().numpy()),
                        "returns_std": float(returns_tensor.std().cpu().numpy()),
                    })

            if not continue_training:
                break

        return logs
