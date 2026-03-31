from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dynamic_nav.reporting import utc_now_iso, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run the formal CS5491 experiment matrix.")
    parser.add_argument("--config", type=str, default="configs/experiments.json")
    parser.add_argument("--mode", choices=["dry-run", "run"], default="dry-run")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected run_name values.")
    parser.add_argument("--only-tag", choices=["main", "ablation", "stability"], default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--export-cases", action="store_true")
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument(
        "--case-types",
        nargs="+",
        default=["success", "failure", "detour", "closest_collision"],
        choices=["success", "failure", "detour", "closest_collision"],
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_train_command(run: dict, python_exe: str, layout: dict, device: str | None) -> list[str]:
    command = [
        python_exe,
        "train.py",
        "--model",
        run["model"],
        "--difficulty",
        run["difficulty"],
        "--seed",
        str(run["seed"]),
        "--total-timesteps",
        str(run["total_timesteps"]),
        "--checkpoint-dir",
        layout["checkpoints"],
        "--log-dir",
        layout["training_logs"],
        "--run-name",
        run["run_name"],
        "--run-group",
        run["group"],
        "--report-section",
        run["report_section"],
    ]
    model_kwargs = run.get("model_kwargs", {})
    if "horizon" in model_kwargs:
        command.extend(["--horizon", str(model_kwargs["horizon"])])
    if "aux_coef" in model_kwargs:
        command.extend(["--aux-coef", str(model_kwargs["aux_coef"])])
    if device:
        command.extend(["--device", device])
    return command


def build_eval_command(run: dict, python_exe: str, layout: dict, device: str | None, args) -> list[str]:
    command = [
        python_exe,
        "eval.py",
        "--model",
        run["model"],
        "--difficulty",
        run["difficulty"],
        "--seed",
        str(run["seed"]),
        "--eval-suite",
        layout.get("eval_suite", "configs/eval_suite.json"),
        "--output-dir",
        layout["evaluation_outputs"],
        "--run-name",
        run["run_name"],
        "--run-group",
        run["group"],
        "--report-section",
        run["report_section"],
        "--qualitative-dir",
        layout["qualitative_outputs"],
    ]
    if args.export_cases:
        command.append("--export-cases")
        command.extend(["--case-limit", str(args.case_limit)])
        command.extend(["--case-types", *args.case_types])
    if run["model"] != "oracle":
        command.extend(["--checkpoint-dir", layout["checkpoints"]])
    if device:
        command.extend(["--device", device])
    return command


def expected_outputs_exist(run: dict) -> bool:
    expected_outputs = run.get("expected_outputs", [])
    return bool(expected_outputs) and all(Path(path).exists() for path in expected_outputs)


def manifest_entry(run: dict, commands: list[list[str]]) -> dict:
    model_kwargs = run.get("model_kwargs", {})
    return {
        "run_name": run["run_name"],
        "kind": run["kind"],
        "group": run["group"],
        "report_section": run["report_section"],
        "model": run["model"],
        "difficulty": run["difficulty"],
        "seed": run["seed"],
        "horizon": model_kwargs.get("horizon"),
        "aux_coef": model_kwargs.get("aux_coef"),
        "total_timesteps": run.get("total_timesteps"),
        "expected_outputs": run.get("expected_outputs", []),
        "existing_outputs": [path for path in run.get("expected_outputs", []) if Path(path).exists()],
        "commands": commands,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    layout = config["output_layout"]
    layout["eval_suite"] = config.get("eval_suite", "configs/eval_suite.json")
    for value in layout.values():
        if value.endswith(".json"):
            continue
        Path(value).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(layout["evaluation_outputs"]) / "experiment_manifest.json"
    previous_manifest = []
    if manifest_path.exists():
        previous_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_manifest = previous_payload.get("runs", previous_payload if isinstance(previous_payload, list) else [])
    previous_names = {entry["run_name"] for entry in previous_manifest}
    selected = []
    for run in config["runs"]:
        if args.only and run["run_name"] not in args.only:
            continue
        if args.only_tag and run.get("group") != args.only_tag:
            continue
        if args.resume_failed and previous_manifest and run["run_name"] not in previous_names:
            continue
        if args.resume_failed and expected_outputs_exist(run):
            continue
        selected.append(run)
    manifest = []
    for run in selected:
        commands = []
        if run["kind"] == "train_eval":
            commands.append(build_train_command(run, args.python, layout, args.device))
        commands.append(build_eval_command(run, args.python, layout, args.device, args))
        entry = manifest_entry(run, commands)
        entry["status"] = "skipped_existing" if args.skip_existing and expected_outputs_exist(run) else "pending"
        manifest.append(entry)
        if args.skip_existing and expected_outputs_exist(run):
            print(f"# skipping {run['run_name']} because expected outputs already exist")
            continue
        for command in commands:
            print("$", " ".join(command))
            if args.mode == "run":
                subprocess.run(command, check=True)
        if args.mode == "run":
            entry["status"] = "completed" if expected_outputs_exist(run) else "incomplete"
    write_json(
        manifest_path,
        {
            "created_at_utc": utc_now_iso(),
            "config_path": str(args.config),
            "mode": args.mode,
            "selected_count": len(selected),
            "export_cases": args.export_cases,
            "runs": manifest,
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
