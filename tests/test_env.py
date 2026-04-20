from __future__ import annotations

from dynamic_nav.env import DynamicNavigationEnv
from dynamic_nav.config import NODE_FEATURE_DIM


def test_reset_produces_legal_positions():
    env = DynamicNavigationEnv("easy")
    observation, info = env.reset(seed=7)
    # Use the environment's configured max_nodes and the global NODE_FEATURE_DIM
    # rather than hard-coding (12, 5) so the test remains robust to feature
    # dimension/extensions.
    assert observation["node_features"].shape == (env.config.max_nodes, NODE_FEATURE_DIM)
    assert observation["edge_index"].shape[0] == 2
    assert info["status"] == "running"
    positions = env.agent_pos, env.goal_pos, *(obstacle.position for obstacle in env.obstacles)
    assert all(0 <= x < env.current_spec.grid_size and 0 <= y < env.current_spec.grid_size for x, y in positions)


def test_seeded_reset_is_reproducible():
    env_a = DynamicNavigationEnv("medium")
    env_b = DynamicNavigationEnv("medium")
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert env_a.agent_pos == env_b.agent_pos
    assert env_a.goal_pos == env_b.goal_pos
    assert [ob.position for ob in env_a.obstacles] == [ob.position for ob in env_b.obstacles]
    assert (obs_a["node_features"] == obs_b["node_features"]).all()


def test_step_updates_status_consistently():
    env = DynamicNavigationEnv("easy")
    env.reset(seed=42)
    _, _, terminated, truncated, info = env.step(0)
    assert info["status"] in {"running", "success", "collision", "timeout"}
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
