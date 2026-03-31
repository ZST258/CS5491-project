from __future__ import annotations

from typing import Any

from torch import nn

from dynamic_nav.config import NUM_ACTIONS

from .common import BasePolicy, MLPBlock, PolicyOutput, flatten_batch


class MLPPolicy(BasePolicy):
    model_name = "mlp"

    def __init__(self, max_nodes: int, hidden_dim: int = 128, device: str = "cpu"):
        self.max_nodes = max_nodes
        input_dim = max_nodes * 5 + 2 + NUM_ACTIONS
        super().__init__(device=device)
        self.encoder = MLPBlock(input_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.to(self.device)

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        encoded = self.encoder(flatten_batch(observations, max_nodes=self.max_nodes, device=self.device))
        return PolicyOutput(logits=self.policy_head(encoded), value=self.value_head(encoded))
