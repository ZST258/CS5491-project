from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from dynamic_nav.config import NODE_FEATURE_DIM, NUM_ACTIONS
from .common import BasePolicy, GraphAttentionEncoder, PolicyOutput


class PredictivePolicy(BasePolicy):
    model_name = "predictive"

    def __init__(self, latent_dim: int = 96, horizon: int = 3, aux_coef: float = 0.2, device: str = "cpu"):
        super().__init__(device=device)
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.aux_coef = aux_coef

        self.encoder = GraphAttentionEncoder(NODE_FEATURE_DIM, hidden_dim=latent_dim, output_dim=latent_dim)
        self.action_embedding = nn.Embedding(NUM_ACTIONS, latent_dim)
        self.rollout_cell = nn.GRUCell(latent_dim * 2, latent_dim)

        feature_dim = latent_dim * (horizon + 1)
        self.actor = nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, NUM_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, 1))
        self.context_head = nn.Linear(latent_dim, NUM_ACTIONS)
        self.to(self.device)

    def encode(self, observations: list[dict[str, Any]]) -> torch.Tensor:
        return self.encoder.forward_batch(observations, device=self.device)

    def rollout_predictions(self, latent: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        bsz = latent.shape[0]
        if self.horizon <= 0:
            return torch.empty(bsz, 0, self.latent_dim, device=self.device, dtype=latent.dtype)

        predictions = torch.empty(bsz, self.horizon, self.latent_dim, device=self.device, dtype=latent.dtype)
        hidden = latent
        for step in range(self.horizon):
            hidden = self.rollout_cell(torch.cat([hidden, action_embeddings[:, step]], dim=-1), hidden)
            predictions[:, step] = hidden
        return predictions

    def forward_from_latent(self, latent: torch.Tensor) -> PolicyOutput:
        action_prior = torch.softmax(self.context_head(latent), dim=-1)  # [B, A]
        expected_action_emb = action_prior @ self.action_embedding.weight  # [B, D]
        action_embeddings = expected_action_emb.unsqueeze(1).expand(-1, self.horizon, -1)  # [B, H, D]
        future_latents = self.rollout_predictions(latent, action_embeddings)  # [B, H, D]
        features = torch.cat([latent.unsqueeze(1), future_latents], dim=1).reshape(latent.shape[0], -1)
        return PolicyOutput(logits=self.actor(features), value=self.critic(features))

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        latent = self.encode(observations)
        return self.forward_from_latent(latent)

    def auxiliary_loss(self, observations: list[dict[str, Any]], actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        latents = self.encode(observations)
        return self.auxiliary_loss_from_latent(latents, actions, dones)

    def auxiliary_loss_from_latent(self, latents: torch.Tensor, actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        batch_size = latents.shape[0]
        if self.horizon <= 0 or batch_size <= self.horizon:
            return torch.zeros((), device=self.device, requires_grad=True)

        dones_t = torch.as_tensor(dones.astype(np.bool_), device=self.device)
        num_candidates = batch_size - self.horizon
        valid_mask = torch.ones(num_candidates, dtype=torch.bool, device=self.device)

        for h in range(self.horizon):
            valid_mask &= ~dones_t[h: h + num_candidates]

        v_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if v_idx.numel() == 0:
            return torch.zeros((), device=self.device, requires_grad=True)

        latent = latents[v_idx]  # [V, D]

        offsets_tgt = torch.arange(1, self.horizon + 1, device=self.device).unsqueeze(0)  # [1, H]
        target_indices = v_idx.unsqueeze(1) + offsets_tgt  # [V, H]
        target_tensor = latents[target_indices].detach()  # [V, H, D]

        actions_t = torch.as_tensor(actions.astype(np.int64), dtype=torch.long, device=self.device)
        offsets_act = torch.arange(0, self.horizon, device=self.device).unsqueeze(0)  # [1, H]
        action_indices = v_idx.unsqueeze(1) + offsets_act
        action_tensor = actions_t[action_indices]  # [V, H]
        action_embeddings = self.action_embedding(action_tensor)  # [V, H, D]

        predictions = self.rollout_predictions(latent, action_embeddings)  # [V, H, D]
        mse_loss = torch.mean((predictions - target_tensor) ** 2)
        return self.aux_coef * mse_loss