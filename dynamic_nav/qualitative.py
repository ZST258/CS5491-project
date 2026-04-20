from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ACTION_DELTAS, ACTION_NAMES
from .env import DynamicNavigationEnv
from .oracle import oracle_shortest_path_length, time_expanded_a_star
from .reporting import write_json


DELTA_TO_ACTION = {delta: action for action, delta in ACTION_DELTAS.items()}


def greedy_action(model, observation: dict[str, Any]) -> int:
    import torch

    with torch.no_grad():
        output = model.forward_batch([observation])
        logits = output.logits.squeeze(0).detach().cpu().numpy()
        masked_logits = np.where(np.asarray(observation["action_mask"]) > 0, logits, -1e9)
        return int(np.argmax(masked_logits))


def obstacle_positions(env: DynamicNavigationEnv) -> list[tuple[int, int]]:
    return [obstacle.position for obstacle in env.obstacles]


def nearest_obstacle_distance(agent_pos: tuple[int, int], obstacles: list[tuple[int, int]]) -> float:
    if not obstacles:
        return float("inf")
    return float(min(abs(agent_pos[0] - x) + abs(agent_pos[1] - y) for x, y in obstacles))


def goal_distance(agent_pos: tuple[int, int], goal_pos: tuple[int, int]) -> int:
    return abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])


def select_case_indices(rows: list[dict[str, Any]], case_types: list[str], limit: int) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    used_episode_indices: set[int] = set()
    for case_type in case_types:
        candidates = list(rows)
        if case_type == "success":
            candidates = [row for row in candidates if int(row["success"]) == 1]
            # prefer cases with higher time efficiency and shorter episode length
            candidates.sort(key=lambda row: (-float_or_default(row.get("time_efficiency", "")), int(row["episode_length"])))
        elif case_type == "failure":
            candidates = [row for row in candidates if int(row["success"]) == 0]
            candidates.sort(key=lambda row: (int(row["timeout"]) - int(row["collision"]), -int(row["episode_length"])))
        elif case_type == "detour":
            candidates = [
                row
                for row in candidates
                if int(row["success"]) == 1 and row["oracle_length"] not in {"", None} and int(row["path_length"]) > int(row["oracle_length"])
            ]
            candidates.sort(key=lambda row: (int(row["path_length"]) - int(row["oracle_length"])), reverse=True)
        elif case_type == "closest_collision":
            candidates = [row for row in candidates if int(row["collision"]) == 1]
            candidates.sort(key=lambda row: int(row["episode_length"]), reverse=True)
        else:
            continue
        added = 0
        for row in candidates:
            episode_index = int(row["episode_index"])
            if episode_index in used_episode_indices:
                continue
            selections.append({"case_type": case_type, "episode_index": episode_index})
            used_episode_indices.add(episode_index)
            added += 1
            if added >= limit:
                break
    return selections


def float_or_default(value: Any, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    return float(value)


def action_from_positions(current: tuple[int, int], next_pos: tuple[int, int]) -> int:
    delta = (next_pos[0] - current[0], next_pos[1] - current[1])
    return DELTA_TO_ACTION.get(delta, 0)


def build_turning_points(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turning_points = []
    previous_action = None
    previous_goal_distance = None
    for step in steps:
        reasons = []
        if previous_action is not None and step["action_name"] != previous_action:
            reasons.append("action_change")
        if step["min_obstacle_distance"] <= 1.0:
            reasons.append("near_obstacle")
        if previous_goal_distance is not None and step["goal_distance"] > previous_goal_distance:
            reasons.append("detour")
        if reasons:
            turning_points.append({"step": step["step"], "reasons": reasons, "status": step["status"]})
        previous_action = step["action_name"]
        previous_goal_distance = step["goal_distance"]
    return turning_points[:6]


def classify_failure(steps: list[dict[str, Any]], final_status: str) -> str:
    if final_status == "success":
        return "success"
    if final_status == "timeout":
        return "timeout_due_to_detour"
    recent = steps[-3:] if len(steps) >= 3 else steps
    if any(step["min_obstacle_distance"] <= 1.0 for step in recent):
        return "late_avoidance"
    return "collision_under_crossing"


def trajectory_svg(
    path: Path,
    grid_size: int,
    agent_positions: list[tuple[int, int]],
    goal_pos: tuple[int, int],
    obstacle_starts: list[tuple[int, int]],
    title: str,
):
    cell = 36
    margin = 40
    width = margin * 2 + grid_size * cell
    height = margin * 2 + grid_size * cell + 30
    points = []
    for x, y in agent_positions:
        px = margin + y * cell + cell / 2
        py = margin + x * cell + cell / 2
        points.append(f"{px},{py}")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="Helvetica">{title}</text>',
    ]
    for row in range(grid_size + 1):
        y = margin + row * cell
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{margin + grid_size * cell}" y2="{y}" stroke="#d9d9d9"/>')
    for col in range(grid_size + 1):
        x = margin + col * cell
        svg.append(f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + grid_size * cell}" stroke="#d9d9d9"/>')
    for x, y in obstacle_starts:
        px = margin + y * cell + 6
        py = margin + x * cell + 6
        svg.append(f'<rect x="{px}" y="{py}" width="{cell - 12}" height="{cell - 12}" fill="#bbbbbb" />')
    goal_x = margin + goal_pos[1] * cell + 6
    goal_y = margin + goal_pos[0] * cell + 6
    svg.append(f'<rect x="{goal_x}" y="{goal_y}" width="{cell - 12}" height="{cell - 12}" fill="#2ca02c" />')
    if points:
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#d62728" stroke-width="3"/>')
        start_x, start_y = agent_positions[0]
        sx = margin + start_y * cell + cell / 2
        sy = margin + start_x * cell + cell / 2
        end_x, end_y = agent_positions[-1]
        ex = margin + end_y * cell + cell / 2
        ey = margin + end_x * cell + cell / 2
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="6" fill="#1f77b4" />')
        svg.append(f'<circle cx="{ex}" cy="{ey}" r="6" fill="#d62728" />')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def replay_episode(spec, model=None, model_name: str = "policy") -> dict[str, Any]:
    env = DynamicNavigationEnv(difficulty=spec.difficulty, episode_spec=spec)
    observation, _ = env.reset()
    initial_frame = env.render()
    frames = [f"Step 0\n{initial_frame}"]
    steps = []
    oracle_path = time_expanded_a_star(spec, dynamic=True)
    oracle_length = oracle_shortest_path_length(spec, dynamic=True)
    path_cursor = 1
    done = False
    step_index = 0
    previous_positions = [env.agent_pos]
    while not done:
        before = env.agent_pos
        if model_name == "oracle":
            if oracle_path:
                target = oracle_path[path_cursor] if path_cursor < len(oracle_path) else oracle_path[-1]
                action = action_from_positions(before, target)
                path_cursor += 1
            else:
                action = 0
        else:
            action = greedy_action(model, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        after = env.agent_pos
        obstacles = obstacle_positions(env)
        step_record = {
            "step": step_index + 1,
            "from_pos": list(before),
            "to_pos": list(after),
            "action": int(action),
            "action_name": ACTION_NAMES[int(action)],
            "reward": float(reward),
            "status": info["status"],
            "path_length": int(info["path_length"]),
            "min_obstacle_distance": nearest_obstacle_distance(after, obstacles),
            "goal_distance": goal_distance(after, env.goal_pos),
            "obstacles": [list(item) for item in obstacles],
        }
        steps.append(step_record)
        previous_positions.append(after)
        frames.append(f"Step {step_index + 1} [{ACTION_NAMES[int(action)]}]\n{env.render()}")
        step_index += 1
        done = terminated or truncated
    turning_points = build_turning_points(steps)
    failure_type = classify_failure(steps, steps[-1]["status"] if steps else "timeout")
    return {
        "difficulty": spec.difficulty,
        "seed": spec.seed,
        "grid_size": spec.grid_size,
        "goal": list(spec.goal),
        "agent_start": list(spec.agent_start),
        "obstacle_starts": [list(obstacle.position) for obstacle in spec.obstacles],
        "oracle_length": oracle_length,
        "status": steps[-1]["status"] if steps else "timeout",
        "failure_type": failure_type,
        "path_length": int(steps[-1]["path_length"]) if steps else 0,
        "episode_length": len(steps),
        "turning_points": turning_points,
        "steps": steps,
        "frames": frames,
        "agent_positions": [list(item) for item in previous_positions],
    }


def export_rollout_assets(rollout: dict[str, Any], output_dir: str | Path, prefix: str) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{prefix}_rollout.json"
    steps_path = output_path / f"{prefix}_steps.csv"
    frames_path = output_path / f"{prefix}_frames.txt"
    summary_path = output_path / f"{prefix}_summary.md"
    svg_path = output_path / f"{prefix}_trajectory.svg"
    write_json(json_path, rollout)
    with steps_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in rollout["steps"][0].keys() if key != "obstacles"] if rollout["steps"] else ["step"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in rollout["steps"]:
            writer.writerow({key: value for key, value in step.items() if key in fieldnames})
    frames_path.write_text("\n\n".join(rollout["frames"]), encoding="utf-8")
    trajectory_svg(
        svg_path,
        grid_size=int(rollout["grid_size"]),
        agent_positions=[tuple(item) for item in rollout["agent_positions"]],
        goal_pos=tuple(rollout["goal"]),
        obstacle_starts=[tuple(item) for item in rollout["obstacle_starts"]],
        title=f"{prefix}: {rollout['status']}",
    )
    summary_lines = [
        f"# Case {prefix}",
        "",
        f"- Outcome: `{rollout['status']}`",
        f"- Failure type: `{rollout['failure_type']}`",
        f"- Path length: `{rollout['path_length']}`",
        f"- Oracle length: `{rollout['oracle_length']}`",
        f"- Episode length: `{rollout['episode_length']}`",
        "",
        "## Turning Points",
    ]
    if rollout["turning_points"]:
        for item in rollout["turning_points"]:
            summary_lines.append(f"- Step {item['step']}: {', '.join(item['reasons'])} ({item['status']})")
    else:
        summary_lines.append("- No major turning points detected.")
    summary_lines.extend(
        [
            "",
            "## Assets",
            "",
            f"- JSON: `{json_path}`",
            f"- Steps CSV: `{steps_path}`",
            f"- Frames TXT: `{frames_path}`",
            f"- Trajectory SVG: `{svg_path}`",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return {
        "json": str(json_path),
        "steps_csv": str(steps_path),
        "frames_txt": str(frames_path),
        "trajectory_svg": str(svg_path),
        "summary_md": str(summary_path),
    }


def export_case_manifest(
    run_name: str,
    cases: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{run_name}_cases.json"
    md_path = output_path / f"{run_name}_cases.md"
    write_json(json_path, {"run_name": run_name, "cases": cases})
    lines = [f"# Failure And Qualitative Cases For {run_name}", ""]
    if not cases:
        lines.append("- No cases exported yet.")
    for case in cases:
        lines.extend(
            [
                f"## {case['case_name']}",
                "",
                f"- Difficulty: `{case['difficulty']}`",
                f"- Episode index: `{case['episode_index']}`",
                f"- Case type: `{case['case_type']}`",
                f"- Outcome: `{case['status']}`",
                f"- Failure type: `{case['failure_type']}`",
                f"- Summary: `{case['assets']['summary_md']}`",
                f"- Trajectory: `{case['assets']['trajectory_svg']}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
