from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate draft report and presentation assets from report outputs.")
    parser.add_argument("--report-dir", type=str, default="outputs/report_assets")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures")
    parser.add_argument("--reports-dir", type=str, default="reports")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def maybe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def metric_text(block: dict, model: str, metric: str, default: str = "TBD") -> str:
    if not block or model not in block:
        return default
    value = block[model].get(metric)
    if value in {"", None}:
        return default
    return f"{float(value):.3f}"


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_numbers = load_json(report_dir / "report_numbers.json")
    failure_cases_md = maybe_read(report_dir / "failure_cases.md")
    results_narrative = maybe_read(report_dir / "results_narrative.md")
    main = report_numbers.get("main", {})
    easy = main.get("easy", {})
    medium = main.get("medium", {})
    hard = main.get("hard", {})

    report_lines = [
        "# CS5491 Final Report Draft",
        "",
        "## Title",
        "Graph-Based Predictive Agent for Decision Making in Dynamic Environments",
        "",
        "## 1. Introduction / Problem",
        "Dynamic obstacle navigation is challenging because the agent must react to moving hazards while still making progress toward a goal. In flat observation settings, standard reinforcement learning policies often become reactive rather than proactive. This project studies whether structured relational encoding and short-horizon latent prediction can improve decision making in dynamic environments.",
        "",
        "Our working hypothesis is that graph-based state representations should outperform a plain MLP baseline, and that a lightweight predictive module may offer additional gains in harder settings. This draft is designed so the final numbers can be updated directly from the generated report assets.",
        "",
        "## 2. Method",
        "We model each environment state as a graph over the agent, goal, and moving obstacles. The `MLP-PPO` baseline consumes a flattened observation, `GNN-PPO` applies graph attention to obtain a relational latent state, and `Predictive(GAT+GRU+PPO)` rolls the latent forward for a short forecasting horizon before choosing an action.",
        "",
        "The predictive branch is intentionally lightweight: it uses a GRU in latent space rather than a heavy world model. This keeps the implementation compatible with the current course-project codebase while still testing the main idea from the proposal.",
        "",
        "## 3. Environment And Metrics",
        "The environment is a dynamic grid world with constant-velocity and random-walk obstacles. We evaluate on fixed easy, medium, and hard tiers using a frozen evaluation suite so that all methods face the same test episodes.",
        "",
    "We report success rate, collision rate, timeout/move failure breakdown, average episode length, and path efficiency relative to a time-expanded A* oracle. Path efficiency is computed only on successful episodes, which makes it a path-quality metric rather than a failure-penalized aggregate.",
        "",
        "## 4. Experimental Setup",
        "The formal experiment matrix is defined in `configs/experiments.json`. The main comparison includes `oracle`, `mlp`, `gnn`, and `predictive`, followed by two ablations: `gnn` versus `predictive`, and `predictive(H=1)` versus `predictive(H=3)`.",
        "",
        "To be updated after final runs: insert actual hardware information, runtime budget, and the exact command sequence used to execute the locked experiment plan.",
        "",
        "## 5. Main Results",
    f"On the easy split, the current placeholders are: MLP success `{metric_text(easy, 'mlp', 'success_rate')}`, GNN success `{metric_text(easy, 'gnn', 'success_rate')}`, Predictive success `{metric_text(easy, 'predictive', 'success_rate')}`, and Predictive move efficiency `{metric_text(easy, 'predictive', 'move_efficiency')}`.",
    f"On the medium split, the current placeholders are: MLP success `{metric_text(medium, 'mlp', 'success_rate')}`, GNN success `{metric_text(medium, 'gnn', 'success_rate')}`, Predictive success `{metric_text(medium, 'predictive', 'success_rate')}`, and Predictive move efficiency `{metric_text(medium, 'predictive', 'move_efficiency')}`.",
    f"On the hard split, the current placeholders are: MLP success `{metric_text(hard, 'mlp', 'success_rate')}`, GNN success `{metric_text(hard, 'gnn', 'success_rate')}`, Predictive success `{metric_text(hard, 'predictive', 'success_rate')}`, and Predictive move efficiency `{metric_text(hard, 'predictive', 'move_efficiency')}`.",
        "",
        f"Insert the pivot table from `{report_dir / 'main_results_pivot.csv'}` and the figures from `{figures_dir / 'group_main_metrics.png'}` (includes success, failure breakdown, time & move efficiency). Replace these placeholder sentences with the strongest supported claim after final runs.",
        "",
        "## 6. Ablation And Failure Analysis",
        "The ablation section should compare whether prediction adds value beyond relational encoding alone, especially on medium and hard difficulties. Use `ablation_pivot.csv` and `stability_summary.csv` to discuss whether longer-horizon forecasting is useful and whether the predictive model is stable across seeds.",
        "",
        "Use the qualitative exports to discuss representative failure cases. If predictive gains are small, frame the analysis around what structured reasoning helps with, where forecasting still breaks down, and why the hardest collisions remain difficult.",
        "",
        "### Failure Case Notes",
        failure_cases_md if failure_cases_md else "To be updated after qualitative case export.",
        "",
        "## 7. Conclusion",
        "This project provides a reproducible comparison among flat, relational, and predictive policies in a dynamic navigation setting. The final conclusion should restate only the strongest result actually supported by the main table and qualitative analysis.",
        "",
        "If predictive gains remain limited, the contribution should be framed as a careful study of structured policy design and short-horizon forecasting in a course-scale dynamic environment, together with a reproducible experimental pipeline.",
        "",
        "## Appendix",
        f"- Main table asset: `{report_dir / 'report_tables.md'}`",
        f"- Narrative notes: `{report_dir / 'results_narrative.md'}`",
        f"- Figure manifest: `{figures_dir / 'figure_manifest.json'}`",
        f"- Experiment matrix: `configs/experiments.json`",
    ]

    slide_lines = [
        "# CS5491 Presentation Draft",
        "",
        "## Slide 1. Title",
        "- Graph-Based Predictive Agent for Decision Making in Dynamic Environments",
        "- Team members",
        "- Goal: improve proactive navigation in dynamic obstacle fields",
        "",
        "## Slide 2. Motivation",
        "- Dynamic environments require both relational reasoning and short-horizon anticipation.",
        "- Flat vector policies are often reactive and brittle under dense motion.",
        "- Our project asks whether graph structure plus latent prediction helps.",
        "",
        "## Slide 3. Task Setup",
        "- Grid world with moving obstacles, fixed goal, and three difficulty tiers.",
        "- Frozen evaluation suite ensures fair comparison across all methods.",
        "- Metrics: success, collision, path efficiency, average episode length.",
        "",
        "## Slide 4. Methods",
        "- `MLP-PPO`: flat baseline.",
        "- `GNN-PPO`: graph attention for relational encoding.",
        "- `Predictive(GAT+GRU+PPO)`: latent rollout before action selection.",
        "",
        "## Slide 5. Main Results",
        f"- Use `{figures_dir / 'group_main_metrics.png'}` and `{report_dir / 'main_results_pivot.csv'}`.",
        f"- Easy placeholder: predictive success `{metric_text(easy, 'predictive', 'success_rate')}`, predictive move efficiency `{metric_text(easy, 'predictive', 'move_efficiency')}`.",
        f"- Hard placeholder: predictive success `{metric_text(hard, 'predictive', 'success_rate')}`, predictive move efficiency `{metric_text(hard, 'predictive', 'move_efficiency')}`.",
        "- Replace with final claim after full runs.",
        "",
        "## Slide 6. Ablation",
        f"- Use `{report_dir / 'ablation_pivot.csv'}`.",
        "- Compare `gnn`, `predictive_h1`, and `predictive_h3` on medium and hard.",
        "- State whether forecasting horizon helps beyond relational encoding.",
        "",
        "## Slide 7. Failure Cases",
        f"- Use `{report_dir / 'failure_cases.md'}` and copied qualitative SVGs from `{figures_dir}`.",
        "- Show 2-3 representative collisions, detours, or timeout cases.",
        "- Explain which failure modes remain unresolved.",
        "",
        "## Slide 8. Reproducibility",
        "- Locked experiment matrix in `configs/experiments.json`.",
        "- Unified train/eval entrypoints and frozen evaluation suite.",
        "- Aggregated report numbers and figures are generated from outputs, not hand-copied.",
        "",
        "## Slide 9. Conclusion",
        "- Summarize the strongest supported result.",
        "- If gains are mixed, emphasize what the graph structure helped and what prediction did not solve.",
        "- Mention future improvements: stronger world models or richer dynamic scenarios.",
        "",
        "## Slide 10. Backup",
        results_narrative if results_narrative else "- Add extra talking points here after final runs.",
    ]

    (reports_dir / "final_report_draft.md").write_text("\n".join(report_lines), encoding="utf-8")
    (reports_dir / "presentation_draft.md").write_text("\n".join(slide_lines), encoding="utf-8")
    print(json.dumps({"report": str(reports_dir / 'final_report_draft.md'), "slides": str(reports_dir / 'presentation_draft.md')}, indent=2))


if __name__ == "__main__":
    main()
