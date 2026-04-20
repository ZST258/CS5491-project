from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dynamic_nav.reporting import load_summary_payload, summary_metrics, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment summaries into report-ready tables.")
    parser.add_argument("--config", type=str, default="configs/experiments.json")
    parser.add_argument("--eval-dir", type=str, default="outputs/eval")
    parser.add_argument("--train-log-dir", type=str, default="outputs/train_logs")
    parser.add_argument("--qualitative-dir", type=str, default="outputs/qualitative")
    parser.add_argument("--output-dir", type=str, default="outputs/report_assets")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def difficulty_order(value: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2}.get(value, 99)


def main_label(row: dict[str, Any]) -> str:
    if row["model"] == "predictive":
        horizon = row.get("horizon", 3)
        return f"predictive_h{horizon}"
    return row["model"]


def ablation_label(row: dict[str, Any]) -> str:
    if row["model"] == "gnn":
        return "gnn"
    return f"predictive_h{row.get('horizon', 3)}"


def pivot_rows(rows: list[dict[str, Any]], label_fn, difficulties: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["difficulty"]][label_fn(row)] = row
    labels = sorted({label_fn(row) for row in rows})
    metrics = ["success_rate", "collision_rate", "time_efficiency", "move_efficiency", "avg_episode_length"]
    pivot = []
    for difficulty in difficulties:
        result = {"difficulty": difficulty}
        for label in labels:
            source = grouped.get(difficulty, {}).get(label, {})
            for metric in metrics:
                value = source.get(metric, "")
                result[f"{label}_{metric}"] = value
        pivot.append(result)
    return pivot


def summarize_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["difficulty"], row["label"])].append(row)
    summary_rows = []
    for (difficulty, label), items in sorted(grouped.items(), key=lambda item: (difficulty_order(item[0][0]), item[0][1])):
        summary_row = {"difficulty": difficulty, "label": label, "count": len(items)}
        for metric in ["success_rate", "collision_rate", "time_efficiency", "move_efficiency", "avg_episode_length"]:
            values = [float(item[metric]) for item in items if item[metric] not in {"", None}]
            if values:
                summary_row[f"{metric}_mean"] = mean(values)
                summary_row[f"{metric}_min"] = min(values)
                summary_row[f"{metric}_max"] = max(values)
            else:
                summary_row[f"{metric}_mean"] = ""
                summary_row[f"{metric}_min"] = ""
                summary_row[f"{metric}_max"] = ""
        summary_rows.append(summary_row)
    return summary_rows


def gather_cases(qualitative_dir: Path) -> dict[str, Any]:
    case_files = sorted(qualitative_dir.rglob("*_cases.json")) if qualitative_dir.exists() else []
    cases = []
    for path in case_files:
        payload = load_json(path)
        cases.extend(payload.get("cases", []))
    return {"count": len(cases), "cases": cases}


def build_results_narrative(report_numbers: dict[str, Any]) -> str:
    lines = [
        "# Results Narrative Draft",
        "",
        "This section is generated from `report_numbers.json` and is safe to revise once final long runs finish.",
        "",
    ]
    main_numbers = report_numbers.get("main", {})
    if not main_numbers:
        lines.append("Main comparison results are not available yet. To be updated after final runs.")
    else:
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_block = main_numbers.get(difficulty, {})
            if not difficulty_block:
                continue
            available_models = ", ".join(sorted(difficulty_block.keys()))
            lines.append(
                f"For the **{difficulty}** setting, the available models are {available_models}. Replace this sentence with the strongest supported takeaway after final runs."
            )
    lines.extend(
        [
            "",
            "## Ablation Draft",
            "",
            "Use the medium and hard comparisons among `gnn`, `predictive_h1`, and `predictive_h3` to explain whether forecasting horizon helps beyond relational encoding alone.",
            "",
            "## Stability Draft",
            "",
            "Use the predictive-hard multi-seed summary to describe variance conservatively. If gains are small, focus on consistency and qualitative behavior instead of claiming broad superiority.",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_json(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_dir)
    train_log_dir = Path(args.train_log_dir)
    qualitative_dir = Path(args.qualitative_dir)
    main_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for run in config["runs"]:
        run_name = run["run_name"]
        summary_path = eval_dir / f"{run_name}_summary.json"
        if not summary_path.exists():
            continue
        summary_payload = load_summary_payload(summary_path)
        summary = summary_metrics(summary_payload)
        train_summary_path = train_log_dir / f"{run_name}_train_summary.json"
        train_summary = load_json(train_summary_path) if train_summary_path.exists() else None
        if train_summary is not None:
            train_rows.append(
                {
                    "run_name": run_name,
                    "group": run["group"],
                    "report_section": run["report_section"],
                    "model": run["model"],
                    "difficulty": run["difficulty"],
                    "seed": run["seed"],
                    "total_timesteps": run.get("total_timesteps", ""),
                    "horizon": run.get("model_kwargs", {}).get("horizon", ""),
                    "final_loss": train_summary["logs"].get("loss", ""),
                    "episode_return": train_summary["logs"].get("episode_return", ""),
                    "aux_loss": train_summary["logs"].get("aux_loss", ""),
                    "history_path": train_summary.get("history_path", ""),
                }
            )
        for difficulty, metrics in summary.items():
            row = {
                "run_name": run_name,
                "group": run["group"],
                "report_section": run["report_section"],
                "model": run["model"],
                "difficulty": difficulty,
                "seed": run["seed"],
                "horizon": run.get("model_kwargs", {}).get("horizon", ""),
                "label": main_label({"model": run["model"], "horizon": run.get("model_kwargs", {}).get("horizon", 3)}),
                "success_rate": metrics["success_rate"],
                "collision_rate": metrics["collision_rate"],
            "time_efficiency": metrics["time_efficiency"] if metrics["time_efficiency"] is not None else "",
                "move_efficiency": metrics.get("move_efficiency", "") if metrics.get("move_efficiency") is not None else "",
                "avg_episode_length": metrics["avg_episode_length"],
            }
            if run["group"] == "main":
                main_rows.append(row)
            if (run["model"] == "gnn" and difficulty in {"medium", "hard"}) or run["group"] == "ablation":
                ablation_rows.append({**row, "label": ablation_label(row)})
            if run["group"] == "stability" or run_name == "predictive_hard_seed0_h3_main":
                stability_rows.append({**row, "label": "predictive_h3"})

    main_rows.sort(key=lambda row: (difficulty_order(row["difficulty"]), row["model"], row["run_name"]))
    ablation_rows.sort(key=lambda row: (difficulty_order(row["difficulty"]), row["label"], row["run_name"]))
    stability_rows.sort(key=lambda row: (difficulty_order(row["difficulty"]), row["run_name"]))

    main_pivot = pivot_rows(main_rows, lambda row: row["model"], ["easy", "medium", "hard"])
    ablation_pivot = pivot_rows(ablation_rows, lambda row: row["label"], ["medium", "hard"])
    stability_summary = summarize_stability(stability_rows)
    cases_payload = gather_cases(qualitative_dir)

    write_csv(output_dir / "main_results.csv", main_rows)
    write_csv(output_dir / "main_results_pivot.csv", main_pivot)
    write_csv(output_dir / "ablation_results.csv", ablation_rows)
    write_csv(output_dir / "ablation_pivot.csv", ablation_pivot)
    write_csv(output_dir / "training_summary.csv", train_rows)
    write_csv(output_dir / "stability_results.csv", stability_rows)
    write_csv(output_dir / "stability_summary.csv", stability_summary)

    report_numbers = {
        "main": {
            difficulty: {row["model"]: {metric: row[metric] for metric in ["success_rate", "collision_rate", "time_efficiency", "move_efficiency", "avg_episode_length"]} for row in main_rows if row["difficulty"] == difficulty}
            for difficulty in ["easy", "medium", "hard"]
        },
        "ablation": {
            difficulty: {row["label"]: {metric: row[metric] for metric in ["success_rate", "collision_rate", "time_efficiency", "move_efficiency", "avg_episode_length"]} for row in ablation_rows if row["difficulty"] == difficulty}
            for difficulty in ["medium", "hard"]
        },
        "stability": {
            row["difficulty"]: {key: row[key] for key in row if key != "difficulty"} for row in stability_summary
        },
        "paths": {
            "main_results_csv": str(output_dir / "main_results.csv"),
            "main_results_pivot_csv": str(output_dir / "main_results_pivot.csv"),
            "ablation_pivot_csv": str(output_dir / "ablation_pivot.csv"),
            "stability_summary_csv": str(output_dir / "stability_summary.csv"),
            "failure_cases_md": str(output_dir / "failure_cases.md"),
        },
    }
    write_json(output_dir / "report_numbers.json", report_numbers)

    table_lines = [
        "# Final Report Assets",
        "",
        "## Main Results",
        "",
        "| Run | Difficulty | Success | Collision | Time Efficiency | Move Efficiency | Avg Episode Length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        time_eff = "NA" if row["time_efficiency"] == "" else f"{float(row['time_efficiency']):.3f}"
        move_eff = "NA" if row.get("move_efficiency", "") == "" else f"{float(row['move_efficiency']):.3f}"
        table_lines.append(
            f"| {row['run_name']} | {row['difficulty']} | {float(row['success_rate']):.3f} | {float(row['collision_rate']):.3f} | {time_eff} | {move_eff} | {float(row['avg_episode_length']):.3f} |"
        )
    table_lines.extend(
        [
            "",
            "## Ablation Focus",
            "",
            "- Compare `gnn` against `predictive_h1` and `predictive_h3` for medium and hard.",
            "- Use `stability_summary.csv` to summarize predictive-hard seed variance.",
        ]
    )
    (output_dir / "report_tables.md").write_text("\n".join(table_lines), encoding="utf-8")
    (output_dir / "results_narrative.md").write_text(build_results_narrative(report_numbers), encoding="utf-8")

    failure_md_lines = ["# Failure Cases", ""]
    if cases_payload["cases"]:
        for case in cases_payload["cases"]:
            failure_md_lines.extend(
                [
                    f"## {case['case_name']}",
                    "",
                    f"- Difficulty: `{case['difficulty']}`",
                    f"- Episode index: `{case['episode_index']}`",
                    f"- Case type: `{case['case_type']}`",
                    f"- Outcome: `{case['status']}`",
                    f"- Failure type: `{case['failure_type']}`",
                    f"- Trajectory: `{case['assets']['trajectory_svg']}`",
                    "",
                ]
            )
    else:
        failure_md_lines.append("No qualitative cases exported yet. Run `eval.py --export-cases` or `export_rollouts.py` after producing evaluation outputs.")
    write_json(output_dir / "failure_cases.json", cases_payload)
    (output_dir / "failure_cases.md").write_text("\n".join(failure_md_lines), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
