from __future__ import annotations

from pathlib import Path

from dynamic_nav.env import DynamicNavigationEnv
from dynamic_nav.models import build_model
from dynamic_nav.ppo import PPOConfig, PPOTrainer


def test_ppo_trainer_saves_checkpoint(tmp_path: Path):
    env = DynamicNavigationEnv("easy")
    model = build_model("mlp", "easy")
    trainer = PPOTrainer(env=env, model=model, config=PPOConfig(total_timesteps=64, rollout_steps=32, minibatch_size=16))
    logs = trainer.train()
    checkpoint_path = trainer.save_checkpoint(tmp_path / "mlp_easy_seed0.pt", metadata={"model": "mlp"})
    assert checkpoint_path.exists()
    assert "loss" in logs
