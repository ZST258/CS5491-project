from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import shutil

from dynamic_nav.checkpoints import load_model_checkpoint
from dynamic_nav.datasets import load_eval_suite, save_eval_suite
from dynamic_nav.env import DynamicNavigationEnv
from dynamic_nav.metrics import EpisodeMetrics, summarize_metrics
from dynamic_nav.oracle import oracle_shortest_path_length, time_expanded_a_star, oracle_move_count
from dynamic_nav.qualitative import export_case_manifest, export_rollout_assets, replay_episode, select_case_indices
from dynamic_nav.reporting import utc_now_iso, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agents on a frozen test suite.")
    parser.add_argument("--model", choices=["mlp", "gnn", "predictive", "oracle"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--eval-suite", type=str, default="configs/eval_suite.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-group", type=str, default="adhoc")
    parser.add_argument("--report-section", type=str, default="adhoc")
    parser.add_argument("--export-cases", action="store_true")
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument(
        "--case-types",
        nargs="+",
        default=["success", "failure", "detour", "closest_collision"],
        choices=["success", "failure", "detour", "closest_collision"],
    )
    parser.add_argument("--qualitative-dir", type=str, default="outputs/qualitative")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def greedy_action(model, observation: dict) -> int:
    with torch.no_grad():
        output = model.forward_batch([observation])
        logits = output.logits.squeeze(0).detach().cpu().numpy()
        masked_logits = np.where(np.asarray(observation["action_mask"]) > 0, logits, -1e9)
        return int(np.argmax(masked_logits))


def evaluate_oracle(spec):
    oracle_path = time_expanded_a_star(spec, dynamic=True)
    if oracle_path is None:
        return EpisodeMetrics(
            success=False,
            collision=False,
            timeout=True,
            path_length=0,
            oracle_length=None,
            episode_length=spec.max_steps,
            oracle_move_count=None,
        )
    path_length = max(len(oracle_path) - 1, 0)
    move_count = oracle_move_count_from_path = oracle_move_count(spec, dynamic=True)
    return EpisodeMetrics(
        success=True,
        collision=False,
        timeout=False,
        path_length=path_length,
        oracle_length=path_length,
        episode_length=path_length,
        oracle_move_count=move_count,
    )


def evaluate_policy(model, spec):
    env = DynamicNavigationEnv(difficulty=spec.difficulty, episode_spec=spec)
    observation, _ = env.reset()
    done = False
    info = {"status": "running", "path_length": 0}
    steps = 0
    while not done:
        action = greedy_action(model, observation)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    oracle_length = oracle_shortest_path_length(spec, dynamic=True)
    move_count = oracle_move_count(spec, dynamic=True)
    return EpisodeMetrics(
        success=info["status"] == "success",
        collision=info["status"] == "collision",
        timeout=info["status"] == "timeout",
        path_length=info["path_length"],
        oracle_length=oracle_length,
        episode_length=steps,
        oracle_move_count=move_count,
    )


def main():
    args = parse_args()
    suite_path = Path(args.eval_suite)
    if not suite_path.exists():
        save_eval_suite(suite_path)
    suite = load_eval_suite(suite_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.model}_{args.difficulty}_seed{args.seed}"
    if args.model != "oracle":
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.checkpoint_dir) / f"{run_name}.pt"
        model_difficulty = args.difficulty if args.difficulty != "all" else "hard"
        model, checkpoint_metadata, _ = load_model_checkpoint(
            checkpoint_path, model_name=args.model, difficulty=model_difficulty, device=args.device
        )
    else:
        model = None
        checkpoint_path = None
        checkpoint_metadata = {}

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    summary: dict[str, dict] = {}
    exported_cases: list[dict] = []
    for difficulty in difficulties:
        episodes = []
        rows = []
        for episode_index, spec in enumerate(suite[difficulty]):
            metrics = evaluate_oracle(spec) if args.model == "oracle" else evaluate_policy(model, spec)
            episodes.append(metrics)
            rows.append(
                {
                    "episode_index": episode_index,
                    "success": int(metrics.success),
                    "collision": int(metrics.collision),
                    "timeout": int(metrics.timeout),
                    "path_length": metrics.path_length,
                    "oracle_length": metrics.oracle_length if metrics.oracle_length is not None else "",
                    "episode_length": metrics.episode_length,
                    # clearer metrics
                    "time_efficiency": metrics.time_efficiency if metrics.time_efficiency is not None else "",
                    "move_efficiency": metrics.move_efficiency if metrics.move_efficiency is not None else "",
                }
            )
        summary[difficulty] = summarize_metrics(episodes)
        csv_path = output_dir / f"{run_name}_{difficulty}_episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if args.export_cases:
            difficulty_dir = Path(args.qualitative_dir) / run_name / difficulty
            selected_cases = select_case_indices(rows, args.case_types, args.case_limit)
            for case in selected_cases:
                episode_index = case["episode_index"]
                spec = suite[difficulty][episode_index]
                rollout = replay_episode(spec, model=model, model_name=args.model)
                case_name = f"{run_name}_{difficulty}_{case['case_type']}_{episode_index}"
                assets = export_rollout_assets(rollout, difficulty_dir, case_name)
                exported_cases.append(
                    {
                        "case_name": case_name,
                        "run_name": run_name,
                        "difficulty": difficulty,
                        "episode_index": episode_index,
                        "case_type": case["case_type"],
                        "status": rollout["status"],
                        "failure_type": rollout["failure_type"],
                        "assets": assets,
                    }
                )
    summary_path = output_dir / f"{run_name}_summary.json"
    cases_manifest = (
        export_case_manifest(run_name, exported_cases, Path(args.qualitative_dir) / run_name) if args.export_cases else None
    )
    # Archive existing summary and episode CSVs if present
    try:
        archive_root = Path("outputs") / "archive"
        archive_eval = archive_root / "eval"
        archive_eval.mkdir(parents=True, exist_ok=True)
        ts = utc_now_iso().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        if summary_path.exists():
            dest = archive_eval / f"{run_name}_summary_{ts}.json"
            shutil.copy2(str(summary_path), str(dest))
        # archive per-difficulty CSVs
        for difficulty in ( [args.difficulty] if args.difficulty != "all" else ["easy","medium","hard"] ):
            csv_path = output_dir / f"{run_name}_{difficulty}_episodes.csv"
            if csv_path.exists():
                dest = archive_eval / f"{run_name}_{difficulty}_episodes_{ts}.csv"
                shutil.copy2(str(csv_path), str(dest))
    except Exception:
        print("Warning: failed to archive existing eval outputs")

    payload = {
        "run_name": run_name,
        "summary_type": "evaluation",
        "created_at_utc": utc_now_iso(),
        "metadata": {
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "run_group": args.run_group,
            "report_section": args.report_section,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_metadata": checkpoint_metadata,
            "eval_suite_path": str(suite_path),
            "output_dir": str(output_dir),
            "qualitative_dir": str(args.qualitative_dir),
            "exported_cases": len(exported_cases),
        },
        "summary": summary,
    }
    if cases_manifest is not None:
        payload["cases_manifest"] = cases_manifest
    write_json(summary_path, payload)
    print(json.dumps({"run_name": run_name, "summary": summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
