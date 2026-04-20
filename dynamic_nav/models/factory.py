from __future__ import annotations

from dynamic_nav.config import DIFFICULTY_CONFIGS

from .gnn import GNNPolicy
from .mlp import MLPPolicy
from .predictive import PredictivePolicy


def build_model(model_name: str, difficulty: str, device: str = "cpu", **model_kwargs):
    config = DIFFICULTY_CONFIGS[difficulty]
    if model_name == "mlp":
        # Pass through model-specific kwargs for consistency with other builders
        return MLPPolicy(max_nodes=config.max_nodes, device=device, **model_kwargs)
    if model_name == "gnn":
        return GNNPolicy(device=device, **model_kwargs)
    if model_name == "predictive":
        # Pass through encoder-related kwargs (latent_dim, num_layers, dropout)
        return PredictivePolicy(device=device, **model_kwargs)
    raise ValueError(f"Unsupported model: {model_name}")
