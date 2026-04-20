from __future__ import annotations

from dynamic_nav.oracle import (
    time_expanded_a_star,
    oracle_shortest_path_length,
    oracle_move_count_from_path,
)
from dynamic_nav.types import EpisodeSpec
from dynamic_nav.metrics import EpisodeMetrics


def test_efficiencies_no_obstacles():
    spec = EpisodeSpec(
        difficulty="easy",
        grid_size=5,
        max_steps=20,
        agent_start=(0, 0),
        goal=(2, 2),
        obstacles=[],
        seed=0,
    )
    path = time_expanded_a_star(spec, dynamic=False)
    assert path is not None
    oracle_len = oracle_shortest_path_length(spec, dynamic=False)
    assert oracle_len == 4
    move_count = oracle_move_count_from_path(path)
    # without stays the move count equals oracle_len
    assert move_count == oracle_len

    metrics = EpisodeMetrics(
        success=True,
        collision=False,
        timeout=False,
        path_length=oracle_len,
        oracle_length=oracle_len,
        episode_length=oracle_len,
        oracle_move_count=move_count,
    )
    assert metrics.time_efficiency == 1.0
    assert metrics.move_efficiency == 1.0


def test_oracle_move_count_with_stays():
    # create a path with explicit stays to test move counting
    path = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0)]
    move_count = oracle_move_count_from_path(path)
    # moves occur when position changes: (0,0)->(1,0) and (1,0)->(2,0)
    assert move_count == 2
