from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from dynamic_nav.observation import flatten_observation


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


def observation_to_tensors(observation: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.as_tensor(observation["node_features"], dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(observation["edge_index"], dtype=torch.long, device=device),
        "global_features": torch.as_tensor(observation["global_features"], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=device),
        "node_count": torch.as_tensor(
            observation.get("node_count", [observation["node_features"].shape[0]]),
            dtype=torch.long,
            device=device,
        ),
    }


def batch_to_tensors(observations: list[dict[str, Any]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [observation_to_tensors(observation, device=device) for observation in observations]


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
    # AMP(fp16) 安全：避免 -1e9 在 half 下溢出
    if logits.dtype in (torch.float16, torch.bfloat16):
        mask_value = torch.finfo(logits.dtype).min  # fp16约 -65504
    else:
        mask_value = -1e9
    masked_logits = logits.masked_fill(action_mask <= 0, mask_value)
    return torch.distributions.Categorical(logits=masked_logits)


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Expanded MLP block
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class BasePolicy(nn.Module):
    model_name = "base"

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # --- PopArt buffers ---
        # Internal network value output is treated as normalized value v_norm.
        # The PPO loop receives denormalized raw value: v_raw = sigma * v_norm + mu.
        self.register_buffer("popart_mu", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("popart_sigma", torch.ones(1, dtype=torch.float32))

        self.to(self.device)

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        raise NotImplementedError

    def forward_from_latent(self, latents: torch.Tensor) -> PolicyOutput:
        raise NotImplementedError("forward_from_latent is not implemented for this policy.")

    # ---------------- PopArt API ----------------
    def get_value_last_layer(self) -> nn.Linear | None:
        """
        Return the LAST nn.Linear that outputs the scalar value (normalized space).
        Subclasses should override. If None, PopArt becomes a no-op.
        """
        return None

    @torch.no_grad()
    def popart_denormalize(self, value_norm: torch.Tensor) -> torch.Tensor:
        return value_norm * self.popart_sigma + self.popart_mu

    @torch.no_grad()
    def popart_normalize(self, value_raw: torch.Tensor, eps: float = 1e-5, min_sigma: float = 1e-3) -> torch.Tensor:
        # Do not add eps to the denominator here. The epsilon is used when
        # computing batch statistics but shouldn't bias the normalized value.
        sigma = torch.clamp(self.popart_sigma, min=min_sigma)
        return (value_raw - self.popart_mu) / sigma

    @torch.no_grad()
    def popart_update(self, targets_raw: torch.Tensor, beta: float, eps: float, min_sigma: float) -> None:
        """
        EMA update running (mu, sigma) using batch raw targets (returns),
        and compensate the value head last layer so raw predictions stay invariant.

        Compensation:
          W' = (old_sigma / new_sigma) * W
          b' = (old_sigma * b + old_mu - new_mu) / new_sigma
        """
        layer = self.get_value_last_layer()
        if layer is None:
            return

        t = targets_raw.detach().float().view(-1)
        if t.numel() == 0:
            return

        old_mu = self.popart_mu.clone()
        old_sigma = torch.clamp(self.popart_sigma.clone(), min=min_sigma)

        batch_mu = t.mean()
        batch_var = torch.mean((t - batch_mu) ** 2)
        batch_sigma = torch.sqrt(batch_var + eps)

        new_mu = beta * old_mu + (1.0 - beta) * batch_mu
        new_sigma = beta * old_sigma + (1.0 - beta) * batch_sigma
        new_sigma = torch.clamp(new_sigma, min=min_sigma)

        # compensate last linear layer
        w = layer.weight.data
        scale = (old_sigma / new_sigma).to(dtype=w.dtype)  # shape [1]
        w.mul_(scale)

        if layer.bias is not None:
            b = layer.bias.data
            b.copy_(((old_sigma.to(b.dtype) * b) + old_mu.to(b.dtype) - new_mu.to(b.dtype)) / new_sigma.to(b.dtype))

        self.popart_mu.copy_(new_mu)
        self.popart_sigma.copy_(new_sigma)

    # -----------------------------------------------------------

    def auxiliary_loss(self, observations: list[dict[str, Any]], actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        return torch.zeros((), device=self.device)

    def auxiliary_loss_from_latent(self, latents: torch.Tensor, actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        return torch.zeros((), device=self.device)

    def act(self, observation: dict[str, Any]) -> tuple[int, float, float]:
        output = self.forward_batch([observation])
        action_mask = torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = masked_categorical(output.logits, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return RAW value to the PPO loop
        value_raw = self.popart_denormalize(output.value.squeeze(0))
        return int(action.item()), float(log_prob.item()), float(value_raw.item())

    def evaluate_actions(
        self,
        observations: list[dict[str, Any]],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_batch(observations)
        action_masks = torch.stack(
            [torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device) for obs in observations], dim=0
        )
        dist = masked_categorical(output.logits, action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Return RAW values
        values_raw = self.popart_denormalize(output.value).squeeze(-1)
        return log_probs, entropy, values_raw

    def evaluate_actions_from_latent(
        self,
        latents: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_from_latent(latents)
        dist = masked_categorical(output.logits, action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Return RAW values
        values_raw = self.popart_denormalize(output.value).squeeze(-1)
        return log_probs, entropy, values_raw


def flatten_batch(observations: list[dict[str, Any]], max_nodes: int, device: torch.device) -> torch.Tensor:
    flat = [flatten_observation(observation, max_nodes=max_nodes) for observation in observations]
    return torch.as_tensor(np.asarray(flat), dtype=torch.float32, device=device)


# RoleAwareConvEncoder has been removed. If you need an explicit role-aware
# convolutional encoder, reintroduce it in a dedicated module. The previous
# implementation lived here and was intentionally deleted per the user's
# request to remove legacy role-aware code.
