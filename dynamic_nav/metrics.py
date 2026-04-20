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
    oracle_move_count: int | None = None

    # path_efficiency (legacy) removed per request. Use time_efficiency or move_efficiency.
    @property
    def time_efficiency(self) -> float | None:
        """
        Time-based efficiency: oracle shortest time (oracle_length) divided by
        actual episode time (episode_length). Both are in time steps so the
        ratio should be <= 1 for successful episodes when oracle is optimal.
        """
        if not self.success or self.oracle_length is None or self.episode_length <= 0:
            return None
        return self.oracle_length / float(self.episode_length)

    @property
    def move_efficiency(self) -> float | None:
        """
        Move-based efficiency: oracle_move_count divided by agent's move count
        (path_length). Both are counts of position changes. Returns None if
        move counts are not available or zero.
        """
        if (
            not self.success
            or self.oracle_move_count is None
            or self.path_length <= 0
        ):
            return None
        return self.oracle_move_count / float(self.path_length)


def summarize_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float | int | None]:
    total = len(episodes)
    success_rate = sum(item.success for item in episodes) / total if total else 0.0
    collision_rate = sum(item.collision for item in episodes) / total if total else 0.0
    avg_episode_length = sum(item.episode_length for item in episodes) / total if total else 0.0
    valid_time_eff = [item.time_efficiency for item in episodes if item.time_efficiency is not None]
    avg_time_eff = sum(valid_time_eff) / len(valid_time_eff) if valid_time_eff else None
    valid_move_eff = [item.move_efficiency for item in episodes if item.move_efficiency is not None]
    avg_move_eff = sum(valid_move_eff) / len(valid_move_eff) if valid_move_eff else None
    return {
        "episodes": total,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_episode_length": avg_episode_length,
        "time_efficiency": avg_time_eff,
        "move_efficiency": avg_move_eff,
    }
