from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dynamic_nav.datasets import generate_episode_spec


def write_eval_suite(path: Path):
    suite = {
        "easy": [generate_episode_spec("easy", 123).to_dict()],
        "medium": [generate_episode_spec("medium", 456).to_dict()],
        "hard": [generate_episode_spec("hard", 789).to_dict()],
    }
    path.write_text(json.dumps(suite), encoding="utf-8")


def test_experiment_config_contains_required_metadata_fields():
    config = json.loads(Path("configs/experiments.json").read_text(encoding="utf-8"))
    run_names = {run["run_name"] for run in config["runs"]}
    assert "oracle_all_seed0" in run_names
    assert "predictive_hard_seed0_h3_main" in run_names
    for run in config["runs"]:
        assert "group" in run
        assert "report_section" in run
        assert "expected_outputs" in run


def test_run_experiments_dry_run_supports_tag_filter_and_skip_existing(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    train_dir = tmp_path / "train_logs"
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    existing_checkpoint = checkpoints / "predictive_hard_seed0_h3_main.pt"
    existing_checkpoint.write_text("placeholder", encoding="utf-8")
    config = {
        "eval_suite": "configs/eval_suite.json",
        "output_layout": {
            "checkpoints": str(checkpoints),
            "training_logs": str(train_dir),
            "evaluation_outputs": str(eval_dir),
            "qualitative_outputs": str(tmp_path / "qualitative"),
            "report_assets": str(tmp_path / "report_assets"),
            "figures": str(tmp_path / "figures"),
        },
        "runs": [
            {
                "run_name": "predictive_hard_seed0_h3_main",
                "kind": "train_eval",
                "group": "main",
                "report_section": "main_results",
                "model": "predictive",
                "difficulty": "hard",
                "seed": 0,
                "total_timesteps": 128,
                "model_kwargs": {"horizon": 3, "aux_coef": 0.2},
                "expected_outputs": [str(existing_checkpoint)],
            },
            {
                "run_name": "predictive_hard_seed1_h3_stability",
                "kind": "train_eval",
                "group": "stability",
                "report_section": "stability",
                "model": "predictive",
                "difficulty": "hard",
                "seed": 1,
                "total_timesteps": 128,
                "model_kwargs": {"horizon": 3, "aux_coef": 0.2},
                "expected_outputs": [str(checkpoints / "predictive_hard_seed1_h3_stability.pt")],
            },
        ],
    }
    config_path = tmp_path / "experiments.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "run_experiments.py",
            "--config",
            str(config_path),
            "--mode",
            "dry-run",
            "--only-tag",
            "main",
            "--skip-existing",
            "--export-cases",
            "--case-limit",
            "1",
        ],
        cwd=Path.cwd(),
        check=True,
    )
    manifest_payload = json.loads((eval_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["selected_count"] == 1
    assert manifest_payload["export_cases"] is True
    assert manifest_payload["runs"][0]["group"] == "main"
    assert manifest_payload["runs"][0]["status"] == "skipped_existing"


def test_train_summary_and_eval_summary_include_metadata(tmp_path: Path):
    suite_path = tmp_path / "eval_suite.json"
    write_eval_suite(suite_path)
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "train_logs"
    eval_dir = tmp_path / "eval"
    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--model",
            "mlp",
            "--difficulty",
            "easy",
            "--seed",
            "0",
            "--total-timesteps",
            "32",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--log-dir",
            str(log_dir),
            "--run-name",
            "mlp_easy_seed0_meta",
            "--run-group",
            "main",
            "--report-section",
            "main_results",
        ],
        cwd=Path.cwd(),
        check=True,
    )
    train_summary = json.loads((log_dir / "mlp_easy_seed0_meta_train_summary.json").read_text(encoding="utf-8"))
    assert train_summary["summary_type"] == "training"
    assert train_summary["metadata"]["run_group"] == "main"
    assert Path(train_summary["checkpoint"]).exists()
    subprocess.run(
        [
            sys.executable,
            "eval.py",
            "--model",
            "mlp",
            "--difficulty",
            "easy",
            "--seed",
            "0",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--eval-suite",
            str(suite_path),
            "--output-dir",
            str(eval_dir),
            "--run-name",
            "mlp_easy_seed0_meta",
            "--run-group",
            "main",
            "--report-section",
            "main_results",
        ],
        cwd=Path.cwd(),
        check=True,
    )
    eval_summary = json.loads((eval_dir / "mlp_easy_seed0_meta_summary.json").read_text(encoding="utf-8"))
    assert eval_summary["summary_type"] == "evaluation"
    assert eval_summary["metadata"]["run_group"] == "main"
    assert eval_summary["metadata"]["eval_suite_path"] == str(suite_path)


def test_eval_export_cases_and_rollout_export_generate_assets(tmp_path: Path):
    suite_path = tmp_path / "eval_suite.json"
    write_eval_suite(suite_path)
    eval_dir = tmp_path / "eval"
    qualitative_dir = tmp_path / "qualitative"
    subprocess.run(
        [
            sys.executable,
            "eval.py",
            "--model",
            "oracle",
            "--difficulty",
            "easy",
            "--run-name",
            "oracle_easy_cases",
            "--eval-suite",
            str(suite_path),
            "--output-dir",
            str(eval_dir),
            "--export-cases",
            "--case-limit",
            "1",
            "--qualitative-dir",
            str(qualitative_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    summary_payload = json.loads((eval_dir / "oracle_easy_cases_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["metadata"]["exported_cases"] >= 1
    assert (qualitative_dir / "oracle_easy_cases" / "oracle_easy_cases_cases.md").exists()
    subprocess.run(
        [
            sys.executable,
            "export_rollouts.py",
            "--model",
            "oracle",
            "--difficulty",
            "easy",
            "--episode-index",
            "0",
            "--run-name",
            "oracle_easy_cases",
            "--eval-suite",
            str(suite_path),
            "--output-dir",
            str(qualitative_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    exported = list((qualitative_dir / "oracle_easy_cases" / "easy").glob("*_trajectory.svg"))
    assert exported


def test_aggregate_results_and_generate_report_assets(tmp_path: Path):
    config = {
        "runs": [
            {
                "run_name": "mlp_easy_seed0_main",
                "group": "main",
                "report_section": "main_results",
                "model": "mlp",
                "difficulty": "easy",
                "seed": 0,
            },
            {
                "run_name": "predictive_hard_seed0_h3_main",
                "group": "main",
                "report_section": "main_results",
                "model": "predictive",
                "difficulty": "hard",
                "seed": 0,
                "model_kwargs": {"horizon": 3},
            },
            {
                "run_name": "predictive_hard_seed1_h3_stability",
                "group": "stability",
                "report_section": "stability",
                "model": "predictive",
                "difficulty": "hard",
                "seed": 1,
                "model_kwargs": {"horizon": 3},
            },
        ]
    }
    config_path = tmp_path / "experiments.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    eval_dir = tmp_path / "eval"
    train_dir = tmp_path / "train_logs"
    qualitative_dir = tmp_path / "qualitative"
    output_dir = tmp_path / "report_assets"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    eval_dir.mkdir()
    train_dir.mkdir()
    qualitative_dir.mkdir()
    reports_dir.mkdir()
    (eval_dir / "mlp_easy_seed0_main_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "easy": {
                        "success_rate": 0.5,
                        "collision_rate": 0.2,
                        "time_efficiency": 0.9,
                        "avg_episode_length": 10.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "predictive_hard_seed0_h3_main_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "hard": {
                        "success_rate": 0.6,
                        "collision_rate": 0.1,
                        "time_efficiency": 0.8,
                        "avg_episode_length": 12.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "predictive_hard_seed1_h3_stability_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "hard": {
                        "success_rate": 0.4,
                        "collision_rate": 0.2,
                        "time_efficiency": 0.7,
                        "avg_episode_length": 14.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "predictive_hard_seed0_h3_main_train_summary.json").write_text(
        json.dumps(
            {
                "logs": {"loss": 1.0, "episode_return": 2.0, "aux_loss": 0.1},
                "history_path": str(train_dir / "predictive_hard_seed0_h3_main_train_history.csv"),
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "predictive_hard_seed0_h3_main_train_history.csv").write_text(
        "global_step,episode_return\n32,1.0\n64,2.0\n",
        encoding="utf-8",
    )
    (qualitative_dir / "sample_cases.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "case_1",
                        "difficulty": "hard",
                        "episode_index": 0,
                        "case_type": "failure",
                        "status": "collision",
                        "failure_type": "late_avoidance",
                        "assets": {"trajectory_svg": "outputs/qualitative/case_1.svg"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "aggregate_results.py",
            "--config",
            str(config_path),
            "--eval-dir",
            str(eval_dir),
            "--train-log-dir",
            str(train_dir),
            "--qualitative-dir",
            str(qualitative_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    assert (output_dir / "main_results_pivot.csv").exists()
    assert (output_dir / "ablation_pivot.csv").exists()
    assert (output_dir / "stability_summary.csv").exists()
    assert (output_dir / "report_numbers.json").exists()
    subprocess.run(
        [
            sys.executable,
            "make_figures.py",
            "--report-dir",
            str(output_dir),
            "--train-log-dir",
            str(train_dir),
            "--qualitative-dir",
            str(qualitative_dir),
            "--output-dir",
            str(figures_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    assert (figures_dir / "success_rate.svg").exists() or (figures_dir / "success_rate.png").exists()
    subprocess.run(
        [
            sys.executable,
            "generate_report_assets.py",
            "--report-dir",
            str(output_dir),
            "--figures-dir",
            str(figures_dir),
            "--reports-dir",
            str(reports_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    assert (reports_dir / "final_report_draft.md").exists()
    assert (reports_dir / "presentation_draft.md").exists()
