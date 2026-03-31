from __future__ import annotations

from dynamic_nav.config import DIFFICULTY_CONFIGS

from .gnn import GNNPolicy
from .mlp import MLPPolicy
from .predictive import PredictivePolicy


def build_model(model_name: str, difficulty: str, device: str = "cpu", **model_kwargs):
    config = DIFFICULTY_CONFIGS[difficulty]
    if model_name == "mlp":
        return MLPPolicy(max_nodes=config.max_nodes, device=device)
    if model_name == "gnn":
        return GNNPolicy(device=device)
    if model_name == "predictive":
        return PredictivePolicy(device=device, **model_kwargs)
    raise ValueError(f"Unsupported model: {model_name}")
