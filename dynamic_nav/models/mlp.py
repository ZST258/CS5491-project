from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynamic_nav.config import NUM_ACTIONS

from .common import BasePolicy, MLPBlock, PolicyOutput, flatten_batch


class MLPPolicy(BasePolicy):
    """Flattened MLP baseline with larger MLPBlock capacity.

    This restores the original flattened observation baseline (no per-node
    MLP/pooling) and increases the encoder hidden dimension to 1024 while
    keeping the improved MLPBlock (depth, LayerNorm, Dropout, residuals).
    """

    model_name = "mlp"

    def __init__(self, max_nodes: int, hidden_dim: int = 128, device: str = "cpu"):
        self.max_nodes = max_nodes
        # input_dim = flattened nodes + global features + action mask
        # NODE_FEATURE_DIM and GLOBAL_FEATURE_DIM are defined in config; import
        from dynamic_nav.config import NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM

        input_dim = max_nodes * NODE_FEATURE_DIM + GLOBAL_FEATURE_DIM + NUM_ACTIONS
        super().__init__(device=device)
        # encoder: larger hidden dimension
        self.encoder = MLPBlock(input_dim, hidden_dim)
        # extra intermediate layer before heads for capacity
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, NUM_ACTIONS))
        # Increase capacity of the value head: add an extra hidden layer so the
        # value estimator has more modeling power without changing the shared
        # encoder. Add LayerNorm to stabilize training and reduce covariate shift
        # for the value predictions.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        # flatten_batch will use dynamic_nav.observation.flatten_observation
        encoded = self.encoder(flatten_batch(observations, max_nodes=self.max_nodes, device=self.device))
        return PolicyOutput(logits=self.policy_head(encoded), value=self.value_head(encoded))

    def get_value_last_layer(self) -> nn.Linear | None:
        return self.value_head[-1]
