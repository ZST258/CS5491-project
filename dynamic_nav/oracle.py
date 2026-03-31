from __future__ import annotations

import heapq
from copy import deepcopy

from .config import ACTION_DELTAS, NUM_ACTIONS
from .types import EpisodeSpec, ObstacleSpec


def _advance_constant_velocity(position: tuple[int, int], velocity: tuple[int, int], grid_size: int):
    x, y = position
    vx, vy = velocity
    next_x, next_y = x + vx, y + vy
    if not (0 <= next_x < grid_size):
        vx *= -1
        next_x = x + vx
    if not (0 <= next_y < grid_size):
        vy *= -1
        next_y = y + vy
    next_x = min(max(next_x, 0), grid_size - 1)
    next_y = min(max(next_y, 0), grid_size - 1)
    return (next_x, next_y), (vx, vy)


def rollout_obstacles(spec: EpisodeSpec, steps: int | None = None) -> list[set[tuple[int, int]]]:
    steps = steps or spec.max_steps
    obstacles = deepcopy(spec.obstacles)
    occupancy = [set(obstacle.position for obstacle in obstacles)]
    for step in range(steps):
        updated: list[ObstacleSpec] = []
        for obstacle in obstacles:
            next_obstacle = deepcopy(obstacle)
            if next_obstacle.pattern == "constant_velocity":
                next_position, next_velocity = _advance_constant_velocity(
                    next_obstacle.position, next_obstacle.velocity, spec.grid_size
                )
                next_obstacle.position = next_position
                next_obstacle.velocity = next_velocity
            else:
                delta = next_obstacle.walk_deltas[min(step, len(next_obstacle.walk_deltas) - 1)]
                next_obstacle.position = (
                    min(max(next_obstacle.position[0] + delta[0], 0), spec.grid_size - 1),
                    min(max(next_obstacle.position[1] + delta[1], 0), spec.grid_size - 1),
                )
            updated.append(next_obstacle)
        obstacles = updated
        occupancy.append(set(obstacle.position for obstacle in obstacles))
    return occupancy


def time_expanded_a_star(spec: EpisodeSpec, dynamic: bool = True) -> list[tuple[int, int]] | None:
    obstacle_occupancy = rollout_obstacles(spec, steps=spec.max_steps) if dynamic else [set()] * (spec.max_steps + 1)
    start_state = (0, spec.agent_start)
    frontier: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(frontier, (_heuristic(spec.agent_start, spec.goal), 0, spec.agent_start))
    parent: dict[tuple[int, tuple[int, int]], tuple[int, tuple[int, int]] | None] = {start_state: None}
    best_cost = {start_state: 0}
    while frontier:
        _, time_step, position = heapq.heappop(frontier)
        state = (time_step, position)
        if position == spec.goal:
            return _reconstruct_path(parent, state)
        if time_step >= spec.max_steps:
            continue
        for action in range(NUM_ACTIONS):
            delta = ACTION_DELTAS[action]
            next_position = (
                min(max(position[0] + delta[0], 0), spec.grid_size - 1),
                min(max(position[1] + delta[1], 0), spec.grid_size - 1),
            )
            next_time = time_step + 1
            if next_position in obstacle_occupancy[next_time]:
                continue
            next_state = (next_time, next_position)
            new_cost = time_step + 1
            if new_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = new_cost
                parent[next_state] = state
                priority = new_cost + _heuristic(next_position, spec.goal)
                heapq.heappush(frontier, (priority, next_time, next_position))
    return None


def oracle_shortest_path_length(spec: EpisodeSpec, dynamic: bool = True) -> int | None:
    path = time_expanded_a_star(spec, dynamic=dynamic)
    if path is None:
        return None
    return max(len(path) - 1, 0)


def _heuristic(start: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def _reconstruct_path(parent: dict, state: tuple[int, tuple[int, int]]) -> list[tuple[int, int]]:
    path = []
    while state is not None:
        _, position = state
        path.append(position)
        state = parent[state]
    return list(reversed(path))
