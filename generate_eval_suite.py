from __future__ import annotations

import argparse

from dynamic_nav.datasets import save_eval_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a frozen evaluation suite for the dynamic navigation project.")
    parser.add_argument("--output", type=str, default="configs/eval_suite.json")
    parser.add_argument("--episodes-per-tier", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = save_eval_suite(args.output, episodes_per_tier=args.episodes_per_tier, base_seed=args.base_seed)
    print(output_path)


if __name__ == "__main__":
    main()
