from __future__ import annotations

import torch

from dynamic_nav.env import DynamicNavigationEnv
from dynamic_nav.models import build_model


def test_all_models_accept_shared_observation_schema():
    env = DynamicNavigationEnv("easy")
    observation, _ = env.reset(seed=0)
    for model_name in ["mlp", "gnn", "predictive"]:
        model = build_model(model_name, "easy")
        output = model.forward_batch([observation])
        assert output.logits.shape == (1, 5)
        assert output.value.shape == (1, 1)


def test_predictive_auxiliary_loss_is_finite():
    env = DynamicNavigationEnv("easy")
    model = build_model("predictive", "easy")
    observations = []
    actions = []
    dones = []
    observation, _ = env.reset(seed=5)
    for _ in range(6):
        observations.append(observation)
        action = 0
        observation, _, terminated, truncated, _ = env.step(action)
        actions.append(action)
        dones.append(terminated or truncated)
    loss = model.auxiliary_loss(observations, actions=torch.tensor(actions).numpy(), dones=torch.tensor(dones).numpy())
    assert torch.isfinite(loss)
