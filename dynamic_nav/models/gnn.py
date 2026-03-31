from __future__ import annotations

from torch import nn

from dynamic_nav.config import NODE_FEATURE_DIM, NUM_ACTIONS

from .common import BasePolicy, GraphAttentionEncoder, PolicyOutput


class GNNPolicy(BasePolicy):
    model_name = "gnn"

    def __init__(self, latent_dim: int = 128, device: str = "cpu"):
        super().__init__(device=device)
        self.encoder = GraphAttentionEncoder(NODE_FEATURE_DIM, hidden_dim=latent_dim, output_dim=latent_dim)
        self.policy_head = nn.Sequential(nn.Tanh(), nn.Linear(latent_dim, NUM_ACTIONS))
        self.value_head = nn.Sequential(nn.Tanh(), nn.Linear(latent_dim, 1))
        self.to(self.device)

    def encode(self, observations: list[dict]):
        return self.encoder.forward_batch(observations, device=self.device)

    def forward_batch(self, observations: list[dict]) -> PolicyOutput:
        latent = self.encode(observations)
        return PolicyOutput(logits=self.policy_head(latent), value=self.value_head(latent))
