from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ObstacleSpec:
    position: tuple[int, int]
    velocity: tuple[int, int]
    pattern: str
    walk_deltas: list[tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObstacleSpec":
        return cls(
            position=tuple(data["position"]),
            velocity=tuple(data["velocity"]),
            pattern=data["pattern"],
            walk_deltas=[tuple(item) for item in data.get("walk_deltas", [])],
        )


@dataclass
class EpisodeSpec:
    difficulty: str
    grid_size: int
    max_steps: int
    agent_start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: list[ObstacleSpec]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "grid_size": self.grid_size,
            "max_steps": self.max_steps,
            "agent_start": self.agent_start,
            "goal": self.goal,
            "seed": self.seed,
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeSpec":
        return cls(
            difficulty=data["difficulty"],
            grid_size=data["grid_size"],
            max_steps=data["max_steps"],
            agent_start=tuple(data["agent_start"]),
            goal=tuple(data["goal"]),
            seed=data["seed"],
            obstacles=[ObstacleSpec.from_dict(item) for item in data["obstacles"]],
        )
