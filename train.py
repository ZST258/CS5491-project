from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import shutil

from dynamic_nav.env import DynamicNavigationEnv, MultiDifficultyEnv
from dynamic_nav.models import build_model
from dynamic_nav.ppo import PPOConfig, PPOTrainer
from dynamic_nav.reporting import utc_now_iso, write_json

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agents for dynamic navigation.")
    parser.add_argument("--model", choices=["mlp", "gnn", "predictive"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="outputs/train_logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-group", type=str, default="adhoc")
    parser.add_argument("--report-section", type=str, default="adhoc")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--aux-coef", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    # PPO / training related CLI options (kept minimal and synced with PPOConfig)
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--policy-lr", type=float, default=None)
    parser.add_argument("--value-lr", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    # --clip-range-vf removed: value-function clipping is incompatible with
    # PopArt normalization and was causing the critic to be locked to a
    # moving baseline. Do not allow this option via CLI.
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--value-weight-decay", type=float, default=None)
    parser.add_argument("--popart-beta", type=float, default=None)
    parser.add_argument("--popart-eps", type=float, default=None)
    parser.add_argument("--popart-min-sigma", type=float, default=None)
    parser.add_argument("--aux-warmup-percent", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    # normalize_returns option removed (returns are kept on original scale)
    parser.add_argument("--lr-schedule", action="store_true")
    # note: keep boolean normalize_returns tri-state (None = default)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    env = DynamicNavigationEnv(difficulty=args.difficulty) if args.difficulty != "all" else MultiDifficultyEnv()
    model_difficulty = args.difficulty if args.difficulty != "all" else "hard"
    model_kwargs = {}
    if args.model == "predictive":
        # allow predictive to receive encoder-related kwargs as well
        model_kwargs = {"horizon": args.horizon, "aux_coef": args.aux_coef}
        if args.latent_dim is not None:
            model_kwargs.update({"latent_dim": args.latent_dim})
        if args.num_layers is not None:
            model_kwargs.update({"num_layers": args.num_layers})
        if args.dropout is not None:
            model_kwargs.update({"dropout": args.dropout})
    if args.model == "gnn" and args.latent_dim is not None:
        model_kwargs.update({"latent_dim": args.latent_dim})
    if args.model == "gnn" and args.num_layers is not None:
        model_kwargs.update({"num_layers": args.num_layers})
    if args.model == "gnn" and args.dropout is not None:
        model_kwargs.update({"dropout": args.dropout})
    model = build_model(model_name=args.model, difficulty=model_difficulty, device=args.device, **model_kwargs)
    # Build PPOConfig from defaults, overriding with provided CLI args where set
    cfg_kwargs = {}
    # map CLI names to PPOConfig fields when provided
    if args.total_timesteps is not None:
        cfg_kwargs["total_timesteps"] = args.total_timesteps
    if args.rollout_steps is not None:
        cfg_kwargs["rollout_steps"] = args.rollout_steps
    if args.update_epochs is not None:
        cfg_kwargs["update_epochs"] = args.update_epochs
    if args.minibatch_size is not None:
        cfg_kwargs["minibatch_size"] = args.minibatch_size
    if args.learning_rate is not None:
        cfg_kwargs["learning_rate"] = args.learning_rate
    if args.policy_lr is not None:
        cfg_kwargs["policy_lr"] = args.policy_lr
    if args.value_lr is not None:
        cfg_kwargs["value_lr"] = args.value_lr
    if args.clip_coef is not None:
        cfg_kwargs["clip_coef"] = args.clip_coef
    if args.target_kl is not None:
        cfg_kwargs["target_kl"] = args.target_kl
    if args.value_coef is not None:
        cfg_kwargs["value_coef"] = args.value_coef
    if args.entropy_coef is not None:
        cfg_kwargs["entropy_coef"] = args.entropy_coef
    if args.lr_schedule:
        cfg_kwargs["lr_schedule"] = True

    # additional PPOConfig fields
    if args.gamma is not None:
        cfg_kwargs["gamma"] = args.gamma
    if args.gae_lambda is not None:
        cfg_kwargs["gae_lambda"] = args.gae_lambda
    if args.max_grad_norm is not None:
        cfg_kwargs["max_grad_norm"] = args.max_grad_norm
    if args.log_interval is not None:
        cfg_kwargs["log_interval"] = args.log_interval
    if args.value_weight_decay is not None:
        cfg_kwargs["value_weight_decay"] = args.value_weight_decay
    if args.popart_beta is not None:
        cfg_kwargs["popart_beta"] = args.popart_beta
    if args.popart_eps is not None:
        cfg_kwargs["popart_eps"] = args.popart_eps
    if args.popart_min_sigma is not None:
        cfg_kwargs["popart_min_sigma"] = args.popart_min_sigma
    if args.aux_warmup_percent is not None:
        cfg_kwargs["aux_warmup_percent"] = args.aux_warmup_percent

    trainer = PPOTrainer(env=env, model=model, config=PPOConfig(**cfg_kwargs), seed=args.seed)
    logs = trainer.train()
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"{args.model}_{args.difficulty}_seed{args.seed}"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"

    # Backup existing checkpoint if present (copy to checkpoints/backup)
    try:
        if checkpoint_path.exists():
            backup_dir = checkpoint_dir / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = utc_now_iso().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
            backup_path = backup_dir / f"{run_name}_{ts}.pt"
            shutil.copy2(str(checkpoint_path), str(backup_path))
    except Exception:
        # Do not fail training on backup errors; warn and continue
        print(f"Warning: failed to backup existing checkpoint {checkpoint_path}")

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
    # Backup existing summary/history files if present (move to outputs/archive/train_logs)
    try:
        archive_root = Path("outputs") / "archive"
        archive_train_logs = archive_root / "train_logs"
        archive_train_logs.mkdir(parents=True, exist_ok=True)
        ts = utc_now_iso().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        if summary_path.exists():
            dest = archive_train_logs / f"{run_name}_train_summary_{ts}.json"
            shutil.move(str(summary_path), str(dest))
        if history_path.exists():
            dest = archive_train_logs / f"{run_name}_train_history_{ts}.csv"
            shutil.move(str(history_path), str(dest))
    except Exception:
        print("Warning: failed to archive existing train logs")
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
        # Write training history as CSV using the union of all keys across
        # rows to ensure new diagnostic fields added during training (e.g.
        # approx_kl, clip_fraction) are included and missing values are
        # represented as empty strings. This avoids ValueError when rows
        # contain differing keys.
        # Collect union of keys
        all_keys = set()
        for row in trainer.training_history:
            all_keys.update(row.keys())
        # Keep a stable order: global_step first if present, then sorted keys
        fieldnames = []
        if "global_step" in all_keys:
            fieldnames.append("global_step")
        for k in sorted(all_keys):
            if k != "global_step":
                fieldnames.append(k)

        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in trainer.training_history:
                # fill missing keys with empty string
                out_row = {k: (row.get(k, "") if row.get(k, None) is not None else "") for k in fieldnames}
                writer.writerow(out_row)
    print(json.dumps({"checkpoint": str(checkpoint_path), "summary_path": str(summary_path), "logs": logs}, indent=2))


if __name__ == "__main__":
    main()
