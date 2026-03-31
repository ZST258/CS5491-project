from __future__ import annotations

from dynamic_nav.oracle import oracle_shortest_path_length, time_expanded_a_star
from dynamic_nav.types import EpisodeSpec, ObstacleSpec


def test_oracle_returns_manhattan_distance_without_obstacles():
    spec = EpisodeSpec(
        difficulty="easy",
        grid_size=5,
        max_steps=20,
        agent_start=(0, 0),
        goal=(2, 2),
        obstacles=[],
        seed=0,
    )
    assert oracle_shortest_path_length(spec, dynamic=False) == 4
    assert len(time_expanded_a_star(spec, dynamic=False)) == 5


def test_oracle_detects_unreachable_goal_when_goal_is_always_blocked():
    spec = EpisodeSpec(
        difficulty="easy",
        grid_size=4,
        max_steps=6,
        agent_start=(0, 0),
        goal=(3, 3),
        obstacles=[
            ObstacleSpec(position=(3, 3), velocity=(0, 0), pattern="random_walk", walk_deltas=[(0, 0)] * 6),
        ],
        seed=1,
    )
    assert oracle_shortest_path_length(spec, dynamic=True) is None
