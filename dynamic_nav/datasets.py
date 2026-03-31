from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import DIFFICULTY_CONFIGS
from .types import EpisodeSpec, ObstacleSpec


def _sample_free_position(rng: np.random.Generator, grid_size: int, occupied: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        candidate = (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate


def generate_episode_spec(difficulty: str, seed: int) -> EpisodeSpec:
    config = DIFFICULTY_CONFIGS[difficulty]
    rng = np.random.default_rng(seed)
    occupied: set[tuple[int, int]] = set()
    agent_start = _sample_free_position(rng, config.grid_size, occupied)
    goal = _sample_free_position(rng, config.grid_size, occupied)
    obstacles = []
    for obstacle_index in range(config.num_obstacles):
        position = _sample_free_position(rng, config.grid_size, occupied)
        pattern = "constant_velocity" if obstacle_index % 2 == 0 else "random_walk"
        if pattern == "constant_velocity":
            velocity_options = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            velocity = velocity_options[int(rng.integers(0, len(velocity_options)))]
            walk_deltas = []
        else:
            velocity = (0, 0)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            walk_deltas = [moves[int(rng.integers(0, len(moves)))] for _ in range(config.max_steps)]
        obstacles.append(
            ObstacleSpec(position=position, velocity=velocity, pattern=pattern, walk_deltas=walk_deltas)
        )
    return EpisodeSpec(
        difficulty=difficulty,
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        agent_start=agent_start,
        goal=goal,
        obstacles=obstacles,
        seed=seed,
    )


def generate_eval_suite(episodes_per_tier: int = 100, base_seed: int = 2026) -> dict[str, list[dict]]:
    suite: dict[str, list[dict]] = {}
    for tier_index, difficulty in enumerate(("easy", "medium", "hard")):
        suite[difficulty] = [
            generate_episode_spec(difficulty, base_seed + tier_index * 10_000 + episode).to_dict()
            for episode in range(episodes_per_tier)
        ]
    return suite


def save_eval_suite(path: str | Path, episodes_per_tier: int = 100, base_seed: int = 2026) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suite = generate_eval_suite(episodes_per_tier=episodes_per_tier, base_seed=base_seed)
    output_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return output_path


def load_eval_suite(path: str | Path) -> dict[str, list[EpisodeSpec]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        difficulty: [EpisodeSpec.from_dict(item) for item in items]
        for difficulty, items in raw.items()
    }
