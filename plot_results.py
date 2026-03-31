from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a markdown summary table from evaluation CSV files.")
    parser.add_argument("--input-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob("*_episodes.csv"))
    if not csv_files:
        raise SystemExit("No evaluation CSV files found.")
    print("| File | Episodes | Success | Collision | Avg Path Efficiency |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        episodes = len(rows)
        success = sum(int(row["success"]) for row in rows) / episodes
        collision = sum(int(row["collision"]) for row in rows) / episodes
        efficiencies = [float(row["path_efficiency"]) for row in rows if row["path_efficiency"]]
        avg_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        print(f"| {csv_file.name} | {episodes} | {success:.3f} | {collision:.3f} | {avg_eff:.3f} |")


if __name__ == "__main__":
    main()
