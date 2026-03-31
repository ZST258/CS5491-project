from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dynamic_nav.env import DynamicNavigationEnv, MultiDifficultyEnv
from dynamic_nav.models import build_model
from dynamic_nav.ppo import PPOConfig, PPOTrainer
from dynamic_nav.reporting import utc_now_iso, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agents for dynamic navigation.")
    parser.add_argument("--model", choices=["mlp", "gnn", "predictive"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="outputs/train_logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-group", type=str, default="adhoc")
    parser.add_argument("--report-section", type=str, default="adhoc")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--aux-coef", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    env = DynamicNavigationEnv(difficulty=args.difficulty) if args.difficulty != "all" else MultiDifficultyEnv()
    model_difficulty = args.difficulty if args.difficulty != "all" else "hard"
    model_kwargs = {}
    if args.model == "predictive":
        model_kwargs = {"horizon": args.horizon, "aux_coef": args.aux_coef}
    model = build_model(model_name=args.model, difficulty=model_difficulty, device=args.device, **model_kwargs)
    trainer = PPOTrainer(
        env=env,
        model=model,
        config=PPOConfig(total_timesteps=args.total_timesteps),
        seed=args.seed,
    )
    logs = trainer.train()
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"{args.model}_{args.difficulty}_seed{args.seed}"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    trainer.save_checkpoint(
        checkpoint_path,
        metadata={
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "run_name": run_name,
            "model_kwargs": model_kwargs,
            "run_group": args.run_group,
            "report_section": args.report_section,
            "created_at_utc": utc_now_iso(),
        },
    )
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / f"{run_name}_train_summary.json"
    history_path = log_dir / f"{run_name}_train_history.csv"
    summary_payload = {
        "run_name": run_name,
        "summary_type": "training",
        "created_at_utc": utc_now_iso(),
        "checkpoint": str(checkpoint_path),
        "history_path": str(history_path),
        "logs": logs,
        "metadata": {
            "run_group": args.run_group,
            "report_section": args.report_section,
            "device": args.device,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
        },
        "config": {
            "model": args.model,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "model_kwargs": model_kwargs,
        },
    }
    write_json(summary_path, summary_payload)
    if trainer.training_history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trainer.training_history[0].keys()))
            writer.writeheader()
            writer.writerows(trainer.training_history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "summary_path": str(summary_path), "logs": logs}, indent=2))


if __name__ == "__main__":
    main()
