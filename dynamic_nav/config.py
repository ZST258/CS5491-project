from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    grid_size: int
    num_obstacles: int
    max_steps: int
    knn_k: int
    max_obstacles: int

    @property
    def max_nodes(self) -> int:
        return self.max_obstacles + 2


DIFFICULTY_CONFIGS = {
    "easy": DifficultyConfig(
        name="easy",
        grid_size=8,
        num_obstacles=3,
        max_steps=30,
        knn_k=3,
        max_obstacles=10,
    ),
    "medium": DifficultyConfig(
        name="medium",
        grid_size=10,
        num_obstacles=6,
        max_steps=45,
        knn_k=4,
        max_obstacles=10,
    ),
    "hard": DifficultyConfig(
        name="hard",
        grid_size=12,
        num_obstacles=10,
        max_steps=60,
        knn_k=6,
        max_obstacles=10,
    ),
}

ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

TYPE_AGENT = 0.0
TYPE_GOAL = 1.0
TYPE_OBSTACLE = 2.0

NUM_ACTIONS = len(ACTION_DELTAS)
NODE_FEATURE_DIM = 7   # 新增: is_cv(idx5), is_rw(idx6)
GLOBAL_FEATURE_DIM = 10  # norm_step, norm_node_count, norm_min_obs_dist, norm_goal_dist, prev_dx, prev_dy, goal_dx, goal_dy, nearest_obs_dx, nearest_obs_dy
