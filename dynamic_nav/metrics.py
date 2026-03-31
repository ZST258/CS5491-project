from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    success: bool
    collision: bool
    timeout: bool
    path_length: int
    oracle_length: int | None
    episode_length: int

    @property
    def path_efficiency(self) -> float | None:
        if not self.success or self.oracle_length is None or self.path_length <= 0:
            return None
        return self.oracle_length / float(self.path_length)


def summarize_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float | int | None]:
    total = len(episodes)
    success_rate = sum(item.success for item in episodes) / total if total else 0.0
    collision_rate = sum(item.collision for item in episodes) / total if total else 0.0
    avg_episode_length = sum(item.episode_length for item in episodes) / total if total else 0.0
    valid_efficiencies = [item.path_efficiency for item in episodes if item.path_efficiency is not None]
    avg_efficiency = sum(valid_efficiencies) / len(valid_efficiencies) if valid_efficiencies else None
    return {
        "episodes": total,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_episode_length": avg_episode_length,
        "path_efficiency": avg_efficiency,
    }
