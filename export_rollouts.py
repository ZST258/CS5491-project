from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch

from dynamic_nav.checkpoints import load_model_checkpoint
from dynamic_nav.datasets import load_eval_suite
from dynamic_nav.qualitative import export_case_manifest, export_rollout_assets, replay_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Export qualitative rollout assets for a single episode.")
    parser.add_argument("--model", choices=["mlp", "gnn", "predictive", "oracle"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--eval-suite", type=str, default="configs/eval_suite.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/qualitative")
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument(
        "--case-types",
        nargs="+",
        default=["success", "failure", "closest_collision"],
        choices=["success", "failure", "detour", "closest_collision"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def float_or_default(value, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    return float(value)


def parse_frame(frame_text: str) -> dict[str, object]:
    lines = frame_text.splitlines()
    if not lines:
        return {"step": 0, "action": None, "grid": [], "agent": None, "goal": None, "obstacles": []}
    header = lines[0].strip()
    step_match = re.search(r"Step\s+(\d+)", header)
    step = int(step_match.group(1)) if step_match else 0
    action_match = re.search(r"\[(.*?)\]", header)
    action = action_match.group(1) if action_match else None
    grid_lines = [line.strip() for line in lines[1:] if line.strip()]
    grid = [row.split() for row in grid_lines]
    agent = None
    goal = None
    obstacles: list[tuple[int, int]] = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "A":
                agent = (r, c)
            elif cell == "G":
                goal = (r, c)
            elif cell == "X":
                obstacles.append((r, c))
    return {"step": step, "action": action, "grid": grid, "agent": agent, "goal": goal, "obstacles": obstacles}


def write_gif(rollout: dict[str, object], output_path: Path) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("GIF export requires Pillow (PIL) to be installed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        title_font = font
    frames = [str(frame) for frame in rollout.get("frames", [])]
    agent_positions = [tuple(pos) for pos in rollout.get("agent_positions", [])]
    final_status = str(rollout.get("status", ""))
    parsed_frames = [parse_frame(frame) for frame in frames]
    cell = 72
    margin = 24
    title_h = 60

    def draw_star(draw, center, radius, fill, outline=None):
        cx, cy = center
        points = []
        for i in range(10):
            angle = -90 + i * 36
            rad = radius if i % 2 == 0 else radius * 0.42
            x = cx + rad * math.cos(math.radians(angle))
            y = cy + rad * math.sin(math.radians(angle))
            points.append((x, y))
        draw.polygon(points, fill=fill, outline=outline)

    def draw_cross(draw, center, size, color, width=6):
        cx, cy = center
        draw.line((cx - size, cy - size, cx + size, cy + size), fill=color, width=width)
        draw.line((cx - size, cy + size, cx + size, cy - size), fill=color, width=width)

    images = []
    trail: list[tuple[int, int]] = []
    def render_agent(draw, center, mode: str = "normal"):
        cx, cy = center
        if mode == "success_pulse":
            draw.ellipse((cx - 22, cy - 18, cx + 22, cy + 18), fill=(196, 237, 202), outline=(93, 166, 105), width=2)
            draw.ellipse((cx - 16, cy - 24, cx + 16, cy + 24), fill=(24, 98, 52), outline=(8, 62, 30), width=3)
        elif mode == "collision":
            draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(120, 20, 20), outline=(70, 10, 10), width=3)
            draw_cross(draw, (cx, cy), 18, color=(255, 235, 235), width=5)
        else:
            draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(24, 98, 52), outline=(8, 62, 30), width=3)

    def render_frame(frame: dict[str, object], step_idx: int, agent_pos: tuple[int, int] | None, pulse_mode: str | None = None) -> Image.Image:
        grid = frame["grid"]
        grid_h = len(grid)
        grid_w = len(grid[0])
        width = margin * 2 + grid_w * cell
        height = margin * 2 + title_h + grid_h * cell
        image = Image.new("RGB", (width, height), (247, 249, 252))
        draw = ImageDraw.Draw(image)
        title = f"Step {frame['step']}" + (f" [{frame['action']}]" if frame.get("action") else "")
        draw.text((margin, 10), title, fill=(40, 40, 40), font=title_font)

        grid_top = margin + title_h
        for r in range(grid_h):
            for c in range(grid_w):
                x0 = margin + c * cell
                y0 = grid_top + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(220, 224, 230))

        if agent_pos is not None:
            trail.append(agent_pos)
        for pos in trail[:-1]:
            r, c = pos
            x0 = margin + c * cell + 8
            y0 = grid_top + r * cell + 8
            x1 = x0 + cell - 16
            y1 = y0 + cell - 16
            draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=(188, 234, 193), outline=None)

        for r, c in frame["obstacles"]:
            x0 = margin + c * cell + 6
            y0 = grid_top + r * cell + 6
            x1 = x0 + cell - 12
            y1 = y0 + cell - 12
            draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(116, 124, 135), outline=(90, 96, 106), width=2)

        goal = frame["goal"]
        if goal is not None:
            r, c = goal
            cx = margin + c * cell + cell // 2
            cy = grid_top + r * cell + cell // 2
            draw_star(draw, (cx, cy), cell * 0.31, fill=(245, 194, 66), outline=(176, 123, 12))

        if agent_pos is not None:
            r, c = agent_pos
            cx = margin + c * cell + cell // 2
            cy = grid_top + r * cell + cell // 2
            if pulse_mode == "success_pulse":
                render_agent(draw, (cx, cy), mode="success_pulse")
            elif pulse_mode == "collision":
                render_agent(draw, (cx, cy), mode="collision")
            else:
                render_agent(draw, (cx, cy), mode="normal")
        return image

    for idx, frame in enumerate(parsed_frames):
        grid = frame["grid"]
        if not grid:
            continue
        agent_pos = agent_positions[idx] if idx < len(agent_positions) else frame["agent"]
        images.append(render_frame(frame, idx, agent_pos, pulse_mode="collision" if idx == len(parsed_frames) - 1 and final_status == "collision" else None))

    if final_status == "success" and parsed_frames:
        final_frame = parsed_frames[-1]
        final_agent_pos = agent_positions[-1] if agent_positions else final_frame["agent"]
        # Add a few celebration frames so success reads clearly in the GIF.
        images.extend(
            [
                render_frame(final_frame, len(parsed_frames) - 1, final_agent_pos, pulse_mode="success_pulse"),
                render_frame(final_frame, len(parsed_frames) - 1, final_agent_pos, pulse_mode="success_pulse"),
                render_frame(final_frame, len(parsed_frames) - 1, final_agent_pos, pulse_mode="normal"),
            ]
        )

    if not images:
        images = [Image.new("RGB", (240, 120), "white")]
    if len(images) == 1:
        images = images * 2
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=520,
        loop=0,
        optimize=False,
    )
    return str(output_path)


def load_episode_rows(run_name: str, difficulty: str) -> list[dict[str, str]]:
    csv_candidates = [
        Path("outputs") / "eval" / f"{run_name}_{difficulty}_episodes.csv",
    ]
    archive_root = Path("outputs") / "archive" / "eval"
    csv_candidates.extend(sorted(archive_root.glob(f"{run_name}_{difficulty}_episodes_*.csv"), reverse=True))
    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    raise FileNotFoundError(
        f"Could not find episodes CSV for {run_name} {difficulty}. Expected outputs/eval/{run_name}_{difficulty}_episodes.csv or an archived copy."
    )


def manhattan_distance(spec) -> int:
    agent = spec.agent_start
    goal = spec.goal
    ax, ay = int(agent[0]), int(agent[1])
    gx, gy = int(goal[0]), int(goal[1])
    return abs(ax - gx) + abs(ay - gy)


def select_auto_cases(
    rows: list[dict[str, str]],
    suite: dict[str, list[dict[str, object]]],
    case_types: list[str],
    limit: int,
) -> list[dict[str, int | str]]:
    selections: list[dict[str, int | str]] = []
    used_episode_indices: set[int] = set()

    def add_case(case_type: str, episode_index: int):
        if episode_index in used_episode_indices:
            return
        selections.append({"case_type": case_type, "episode_index": episode_index})
        used_episode_indices.add(episode_index)

    for case_type in case_types:
        if case_type == "success":
            candidates = [row for row in rows if int(row["success"]) == 1]
            if not candidates:
                continue
            # Choose the success case whose initial agent-goal Manhattan
            # distance is largest; use path length/time efficiency as tiebreaks.
            candidates.sort(
                key=lambda row: (
                    -manhattan_distance(suite[row.get("difficulty", "hard")][int(row["episode_index"])]),
                    -int(row["path_length"]),
                    -float_or_default(row.get("time_efficiency")),
                    int(row["episode_index"]),
                )
            )
            add_case("success", int(candidates[0]["episode_index"]))
        elif case_type == "failure":
            candidates = [row for row in rows if int(row["success"]) == 0]
            if not candidates:
                continue
            candidates.sort(key=lambda row: (int(row["timeout"]) - int(row["collision"]), -int(row["episode_length"]), int(row["episode_index"])))
            add_case("failure", int(candidates[0]["episode_index"]))
        elif case_type == "detour":
            candidates = [row for row in rows if int(row["success"]) == 1 and row["oracle_length"] not in {"", None} and int(row["path_length"]) > int(row["oracle_length"])]
            if not candidates:
                continue
            candidates.sort(key=lambda row: (int(row["path_length"]) - int(row["oracle_length"])), reverse=True)
            add_case("detour", int(candidates[0]["episode_index"]))
        elif case_type == "closest_collision":
            candidates = [row for row in rows if int(row["collision"]) == 1]
            if not candidates:
                continue
            candidates.sort(key=lambda row: int(row["episode_length"]))
            add_case("closest_collision", int(candidates[0]["episode_index"]))
            if len(candidates) > 1:
                add_case("closest_collision", int(candidates[-1]["episode_index"]))

    return selections


def main():
    args = parse_args()
    suite = load_eval_suite(args.eval_suite)
    if args.model == "oracle":
        model = None
    else:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.checkpoint_dir) / f"{args.run_name}.pt"
        model, _, _ = load_model_checkpoint(
            checkpoint_path, model_name=args.model, difficulty=args.difficulty, device=args.device
        )

    if args.episode_index is not None:
        spec = suite[args.difficulty][args.episode_index]
        rollout = replay_episode(spec, model=model, model_name=args.model)
        case_name = f"{args.run_name}_{args.difficulty}_episode{args.episode_index}"
        assets = export_rollout_assets(rollout, Path(args.output_dir) / args.run_name / args.difficulty, case_name)
        assets["animation_gif"] = write_gif(rollout, Path(assets["json"]).with_name(f"{case_name}_animation.gif"))
        print(json.dumps({"case_name": case_name, "assets": assets}, indent=2))
        return

    rows = load_episode_rows(args.run_name, args.difficulty)
    selected_cases = select_auto_cases(rows, suite, args.case_types, args.case_limit)
    exported_cases: list[dict] = []
    difficulty_dir = Path(args.output_dir) / args.run_name / args.difficulty
    for case in selected_cases:
        episode_index = int(case["episode_index"])
        spec = suite[args.difficulty][episode_index]
        rollout = replay_episode(spec, model=model, model_name=args.model)
        case_name = f"{args.run_name}_{args.difficulty}_{case['case_type']}_{episode_index}"
        assets = export_rollout_assets(rollout, difficulty_dir, case_name)
        assets["animation_gif"] = write_gif(rollout, Path(assets["json"]).with_name(f"{case_name}_animation.gif"))
        exported_cases.append(
            {
                "case_name": case_name,
                "run_name": args.run_name,
                "difficulty": args.difficulty,
                "episode_index": episode_index,
                "case_type": case["case_type"],
                "status": rollout["status"],
                "failure_type": rollout["failure_type"],
                "assets": assets,
            }
        )
    export_case_manifest(args.run_name, exported_cases, Path(args.output_dir) / args.run_name)
    print(json.dumps({"run_name": args.run_name, "cases": exported_cases}, indent=2))


if __name__ == "__main__":
    main()
