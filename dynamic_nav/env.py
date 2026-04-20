from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .config import (
    ACTION_DELTAS,
    DIFFICULTY_CONFIGS,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_ACTIONS,
    TYPE_AGENT,
    TYPE_GOAL,
    TYPE_OBSTACLE,
)
from .datasets import generate_episode_spec
from .graph_utils import build_knn_edge_index
from .gym_compat import gym, spaces
from .types import EpisodeSpec, ObstacleSpec


class DynamicNavigationEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, difficulty: str = "easy", episode_spec: EpisodeSpec | None = None):
        if difficulty not in DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.config = DIFFICULTY_CONFIGS[difficulty]
        self.difficulty = difficulty
        self.episode_spec = deepcopy(episode_spec)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.max_nodes, NODE_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(low=-1, high=self.config.max_nodes, shape=(2, self.config.max_nodes * self.config.knn_k), dtype=np.int64),
                "global_features": spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_FEATURE_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(NUM_ACTIONS,), dtype=np.float32),
            }
        )
        self.rng = np.random.default_rng(0)
        self.current_spec: EpisodeSpec | None = None
        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.obstacles: list[ObstacleSpec] = []
        self.step_count = 0
        self.last_path_length = 0
        self._last_goal_dist: int = 0
        self._prev_agent_pos: tuple[int, int] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "episode_spec" in options:
            self.current_spec = deepcopy(options["episode_spec"])
        elif self.episode_spec is not None:
            self.current_spec = deepcopy(self.episode_spec)
        else:
            next_seed = int(seed if seed is not None else self.rng.integers(0, 1_000_000))
            self.current_spec = generate_episode_spec(self.difficulty, next_seed)
        self.agent_pos = self.current_spec.agent_start
        self.goal_pos = self.current_spec.goal
        self.obstacles = deepcopy(self.current_spec.obstacles)
        self.step_count = 0
        self.last_path_length = 0
        self._prev_agent_pos = self.agent_pos
        self._last_goal_dist = (
                abs(self.agent_pos[0] - self.goal_pos[0]) +
                abs(self.agent_pos[1] - self.goal_pos[1])
        )
        return self._build_observation(), self._build_info(status="running")

    def step(self, action: int):
        if self.current_spec is None:
            raise RuntimeError("reset() must be called before step().")
        action = int(action)
        self._prev_agent_pos = self.agent_pos
        proposed = self._move_within_bounds(self.agent_pos, ACTION_DELTAS[action])
        if proposed != self.agent_pos:
            self.last_path_length += 1
        self.agent_pos = proposed
        obstacle_collision = self._advance_obstacles()
        self.step_count += 1
        success = self.agent_pos == self.goal_pos
        collision = obstacle_collision or any(obstacle.position == self.agent_pos for obstacle in self.obstacles)
        timeout = self.step_count >= self.current_spec.max_steps and not success and not collision
        terminated = success or collision
        truncated = timeout
        reward = self._reward(success=success, collision=collision)
        status = "success" if success else "collision" if collision else "timeout" if timeout else "running"
        return self._build_observation(), reward, terminated, truncated, self._build_info(status=status)

    def render(self):
        if self.current_spec is None:
            return "<uninitialized env>"
        grid = [["." for _ in range(self.current_spec.grid_size)] for _ in range(self.current_spec.grid_size)]
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ax][ay] = "A"
        grid[gx][gy] = "G"
        for obstacle in self.obstacles:
            x, y = obstacle.position
            grid[x][y] = "X"
        return "\n".join(" ".join(row) for row in grid)

    def _reward(self, success: bool, collision: bool) -> float:
        goal_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        base = -0.1 - 0.01 * goal_distance
        if success:
            return base + 10.0
        if collision:
            return base - 10.0
        return base

    def _advance_obstacles(self) -> bool:
        collision = False
        updated: list[ObstacleSpec] = []
        for obstacle in self.obstacles:
            updated_obstacle = deepcopy(obstacle)
            if updated_obstacle.pattern == "constant_velocity":
                next_position, next_velocity = self._advance_constant_velocity(updated_obstacle.position, updated_obstacle.velocity)
                updated_obstacle.position = next_position
                updated_obstacle.velocity = next_velocity
            else:
                walk_delta = updated_obstacle.walk_deltas[min(self.step_count, len(updated_obstacle.walk_deltas) - 1)]
                updated_obstacle.position = self._move_within_bounds(updated_obstacle.position, walk_delta)
            if updated_obstacle.position == self.agent_pos:
                collision = True
            updated.append(updated_obstacle)
        self.obstacles = updated
        return collision

    def _advance_constant_velocity(
        self, position: tuple[int, int], velocity: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        x, y = position
        vx, vy = velocity
        next_x, next_y = x + vx, y + vy
        if not (0 <= next_x < self.current_spec.grid_size):
            vx *= -1
            next_x = x + vx
        if not (0 <= next_y < self.current_spec.grid_size):
            vy *= -1
            next_y = y + vy
        next_x = min(max(next_x, 0), self.current_spec.grid_size - 1)
        next_y = min(max(next_y, 0), self.current_spec.grid_size - 1)
        return (next_x, next_y), (vx, vy)

    def _move_within_bounds(self, position: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        x = min(max(position[0] + delta[0], 0), self.current_spec.grid_size - 1)
        y = min(max(position[1] + delta[1], 0), self.current_spec.grid_size - 1)
        return x, y

    def _build_observation(self) -> dict[str, np.ndarray]:
        grid_size = float(self.current_spec.grid_size)
        node_features = [
            [self.agent_pos[0] / grid_size, self.agent_pos[1] / grid_size,
             0.0, 0.0, TYPE_AGENT, 0.0, 0.0],
            [self.goal_pos[0] / grid_size, self.goal_pos[1] / grid_size,
             0.0, 0.0, TYPE_GOAL, 0.0, 0.0],
        ]
        positions = [self.agent_pos, self.goal_pos]
        for obstacle in self.obstacles:
            vx, vy = obstacle.velocity
            is_cv = 1.0 if obstacle.pattern == "constant_velocity" else 0.0
            is_rw = 0.0 if obstacle.pattern == "constant_velocity" else 1.0
            node_features.append([
                obstacle.position[0] / grid_size,
                obstacle.position[1] / grid_size,
                float(vx) / self.current_spec.grid_size,  # 归一化
                float(vy) / self.current_spec.grid_size,
                TYPE_OBSTACLE,
                is_cv,
                is_rw,
            ])
            positions.append(obstacle.position)
        node_array = np.asarray(node_features, dtype=np.float32)
        edge_index = build_knn_edge_index(np.asarray(positions, dtype=np.float32), self.config.knn_k)
        max_edges = self.config.max_nodes * self.config.knn_k
        padded_edges = np.full((2, max_edges), -1, dtype=np.int64)
        length = min(edge_index.shape[1], max_edges)
        if length:
            padded_edges[:, :length] = edge_index[:, :length]
        action_mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        for action, delta in ACTION_DELTAS.items():
            if self._move_within_bounds(self.agent_pos, delta) == self.agent_pos and delta != (0, 0):
                action_mask[action] = 0.0
        node_count_feature = float(node_array.shape[0]) / float(self.config.max_nodes)
        if self.obstacles:
            min_obs_dist = min(
                abs(self.agent_pos[0] - obs.position[0]) +
                abs(self.agent_pos[1] - obs.position[1])
                for obs in self.obstacles
            )
            norm_min_dist = min_obs_dist / (self.current_spec.grid_size * 2.0)
        else:
            norm_min_dist = 1.0

        goal_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        norm_goal_dist = goal_dist / (self.current_spec.grid_size * 2.0)
        if self.goal_pos[0] == self.agent_pos[0]:
            goal_dx = 0.0
        else:
            goal_dx = (self.goal_pos[0] - self.agent_pos[0]) / grid_size
        if self.goal_pos[1] == self.agent_pos[1]:
            goal_dy = 0.0
        else:
            goal_dy = (self.goal_pos[1] - self.agent_pos[1]) / grid_size

        nearest_obs_dx = 0.0
        nearest_obs_dy = 0.0
        if self._prev_agent_pos is None:
            prev_dx = 0.0
            prev_dy = 0.0
        else:
            prev_dx = (self.agent_pos[0] - self._prev_agent_pos[0]) / grid_size
            prev_dy = (self.agent_pos[1] - self._prev_agent_pos[1]) / grid_size

        if self.obstacles:
            nearest_obs = min(
                self.obstacles,
                key=lambda obs: abs(self.agent_pos[0] - obs.position[0]) + abs(self.agent_pos[1] - obs.position[1]),
            )
            nearest_obs_dx = (nearest_obs.position[0] - self.agent_pos[0]) / grid_size
            nearest_obs_dy = (nearest_obs.position[1] - self.agent_pos[1]) / grid_size

        global_features = np.asarray(
            [
                self.step_count / self.current_spec.max_steps,
                node_count_feature,
                norm_min_dist,  # 新增第4维
                norm_goal_dist,
                prev_dx,
                prev_dy,
                goal_dx,
                goal_dy,
                nearest_obs_dx,
                nearest_obs_dy,
            ],
            dtype=np.float32,
        )
        padded_nodes = np.zeros((self.config.max_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        padded_nodes[: node_array.shape[0]] = node_array
        return {
            "node_features": padded_nodes,
            "edge_index": padded_edges,
            "global_features": global_features,
            "action_mask": action_mask,
            "node_count": np.asarray([node_array.shape[0]], dtype=np.int64),
            "grid_size": np.asarray(self.current_spec.grid_size, dtype=np.int64),  # 新增
            "difficulty": self.difficulty,  # 新增
        }

    def _build_info(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "path_length": self.last_path_length,
            "difficulty": self.difficulty,
            "seed": self.current_spec.seed if self.current_spec else None,
        }


class MultiDifficultyEnv:
    def __init__(self, difficulties: list[str] | None = None):
        self.difficulties = difficulties or ["easy", "medium", "hard"]
        self.envs = [DynamicNavigationEnv(difficulty=difficulty) for difficulty in self.difficulties]
        self.current_index = -1
        self.active_env = self.envs[0]
        self.action_space = self.active_env.action_space
        self.observation_space = self.active_env.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.current_index = (self.current_index + 1) % len(self.envs)
        self.active_env = self.envs[self.current_index]
        env_seed = None if seed is None else seed + self.current_index
        return self.active_env.reset(seed=env_seed, options=options)

    def step(self, action: int):
        return self.active_env.step(action)
