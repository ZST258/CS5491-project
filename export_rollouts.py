from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dynamic_nav.checkpoints import load_model_checkpoint
from dynamic_nav.datasets import load_eval_suite
from dynamic_nav.qualitative import export_rollout_assets, replay_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Export qualitative rollout assets for a single episode.")
    parser.add_argument("--model", choices=["mlp", "gnn", "predictive", "oracle"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--eval-suite", type=str, default="configs/eval_suite.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/qualitative")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    suite = load_eval_suite(args.eval_suite)
    spec = suite[args.difficulty][args.episode_index]
    if args.model == "oracle":
        model = None
    else:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.checkpoint_dir) / f"{args.run_name}.pt"
        model, _, _ = load_model_checkpoint(
            checkpoint_path, model_name=args.model, difficulty=args.difficulty, device=args.device
        )
    rollout = replay_episode(spec, model=model, model_name=args.model)
    case_name = f"{args.run_name}_{args.difficulty}_episode{args.episode_index}"
    assets = export_rollout_assets(rollout, Path(args.output_dir) / args.run_name / args.difficulty, case_name)
    print(json.dumps({"case_name": case_name, "assets": assets}, indent=2))


if __name__ == "__main__":
    main()
