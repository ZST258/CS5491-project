from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report figures from aggregated experiment results.")
    parser.add_argument("--report-dir", type=str, default="outputs/report_assets")
    parser.add_argument("--train-log-dir", type=str, default="outputs/train_logs")
    parser.add_argument("--qualitative-dir", type=str, default="outputs/qualitative")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_svg_bar_chart(path: Path, title: str, categories: list[str], series: list[tuple[str, list[float]]], y_label: str):
    width = 920
    height = 460
    margin_left = 80
    margin_right = 30
    margin_top = 55
    plot_height = height - margin_top - 90
    plot_width = width - margin_left - margin_right
    max_candidates = [max(values) for _, values in series if values]
    max_value = max(max_candidates + [1.0])
    band_width = plot_width / max(len(categories), 1)
    bar_width = band_width / max(len(series) + 1, 2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Helvetica">{title}</text>',
        f'<text x="22" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="14" font-family="Helvetica" transform="rotate(-90 22 {margin_top + plot_height / 2})">{y_label}</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_height - (value / max_value) * plot_height
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="Helvetica">{value:.2f}</text>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333"/>')
    for category_index, category in enumerate(categories):
        base_x = margin_left + category_index * band_width + band_width * 0.15
        for series_index, (_, values) in enumerate(series):
            value = values[category_index]
            bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
            x = base_x + series_index * bar_width
            y = margin_top + plot_height - bar_height
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{bar_height}" fill="{colors[series_index % len(colors)]}"/>')
        label_x = margin_left + category_index * band_width + band_width / 2
        svg.append(f'<text x="{label_x}" y="{height - 45}" text-anchor="middle" font-size="12" font-family="Helvetica">{category}</text>')
    legend_x = width - margin_right - 150
    legend_y = margin_top + 10
    for index, (label, _) in enumerate(series):
        y = legend_y + index * 20
        svg.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{colors[index % len(colors)]}"/>')
        svg.append(f'<text x="{legend_x + 20}" y="{y}" font-size="12" font-family="Helvetica">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def save_svg_line_chart(path: Path, title: str, series: list[tuple[str, list[tuple[float, float]]]], y_label: str):
    width = 980
    height = 460
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_x = [x for _, points in series for x, _ in points]
    all_y = [y for _, points in series for _, y in points]
    max_x = max(all_x) if all_x else 1.0
    min_y = min(all_y) if all_y else 0.0
    max_y = max(all_y) if all_y else 1.0
    span_y = max(max_y - min_y, 1.0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Helvetica">{title}</text>',
        f'<text x="22" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="14" font-family="Helvetica" transform="rotate(-90 22 {margin_top + plot_height / 2})">{y_label}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = min_y + span_y * tick / 5
        y = margin_top + plot_height - ((value - min_y) / span_y) * plot_height
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="Helvetica">{value:.2f}</text>')
    for index, (label, points) in enumerate(series):
        if not points:
            continue
        coords = []
        for x_value, y_value in points:
            x = margin_left + (x_value / max_x) * plot_width
            y = margin_top + plot_height - ((y_value - min_y) / span_y) * plot_height
            coords.append(f"{x},{y}")
        svg.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        legend_y = margin_top + 10 + index * 20
        legend_x = width - margin_right - 200
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{colors[index % len(colors)]}"/>')
        svg.append(f'<text x="{legend_x + 18}" y="{legend_y}" font-size="12" font-family="Helvetica">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def build_grouped_rows(main_rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped = {}
    for row in main_rows:
        grouped.setdefault(row["difficulty"], {})[row["model"]] = row
    return grouped


def load_train_history_points(train_log_dir: Path) -> list[tuple[str, list[tuple[float, float]]]]:
    training_summary_rows = load_rows(train_log_dir.parent / "report_assets" / "training_summary.csv")
    series = []
    for row in training_summary_rows:
        history_path = Path(row["history_path"]) if row.get("history_path") else None
        if not history_path or not history_path.exists():
            continue
        history_rows = load_rows(history_path)
        points = []
        for history_row in history_rows:
            if history_row.get("episode_return", "") == "":
                continue
            points.append((float(history_row["global_step"]), float(history_row["episode_return"])))
        if points:
            series.append((row["run_name"], points))
    return series


def copy_qualitative_assets(qualitative_dir: Path, output_dir: Path):
    assets = sorted(qualitative_dir.rglob("*_trajectory.svg")) if qualitative_dir.exists() else []
    gallery_lines = ["# Qualitative Gallery", ""]
    if not assets:
        gallery_lines.append("No qualitative trajectory assets exported yet.")
    for index, asset in enumerate(assets[:3], start=1):
        target = output_dir / f"qualitative_{index}.svg"
        shutil.copyfile(asset, target)
        gallery_lines.append(f"- Case {index}: `{target}`")
    (output_dir / "qualitative_gallery.md").write_text("\n".join(gallery_lines), encoding="utf-8")


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    train_log_dir = Path(args.train_log_dir)
    qualitative_dir = Path(args.qualitative_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = load_rows(report_dir / "main_results.csv")
    grouped = build_grouped_rows(main_rows)
    difficulties = ["easy", "medium", "hard"]
    models = ["oracle", "mlp", "gnn", "predictive"]
    metrics = [
        ("success_rate", "Success Rate by Difficulty", "success_rate"),
        ("collision_rate", "Collision Rate by Difficulty", "collision_rate"),
        ("path_efficiency", "Path Efficiency by Difficulty", "path_efficiency"),
    ]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    for metric_key, title, stem in metrics:
        series = []
        for model in models:
            values = []
            for difficulty in difficulties:
                row = grouped.get(difficulty, {}).get(model)
                cell = row.get(metric_key, "") if row else ""
                values.append(float(cell) if cell not in {"", "NA"} else 0.0)
            series.append((model, values))
        if plt is None:
            save_svg_bar_chart(output_dir / f"{stem}.svg", title, difficulties, series, y_label=title.split(" by ")[0])
        else:
            x = list(range(len(difficulties)))
            width = 0.2
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for model_index, (model, values) in enumerate(series):
                offsets = [item + (model_index - 1.5) * width for item in x]
                ax.bar(offsets, values, width=width, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels(difficulties)
            ax.set_ylabel(title.split(" by ")[0])
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{stem}.png", dpi=200)
            plt.close(fig)

    history_series = load_train_history_points(train_log_dir)
    if history_series:
        if plt is None:
            save_svg_line_chart(output_dir / "training_curve.svg", "Training Return Curve", history_series, y_label="Episode Return")
        else:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            for label, points in history_series:
                ax.plot([point[0] for point in points], [point[1] for point in points], label=label)
            ax.set_title("Training Return Curve")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Episode Return")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "training_curve.png", dpi=200)
            plt.close(fig)
    else:
        snapshot_rows = load_rows(report_dir / "training_summary.csv")
        labels = [row["run_name"] for row in snapshot_rows]
        values = [float(row["episode_return"]) if row["episode_return"] != "" else 0.0 for row in snapshot_rows]
        if not labels:
            labels = ["pending"]
            values = [0.0]
        if plt is None:
            save_svg_bar_chart(output_dir / "training_curve_snapshot.svg", "Training Return Snapshot", labels, [("episode_return", values)], "Episode Return")
        else:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(labels, values)
            ax.set_ylabel("Episode Return")
            ax.set_title("Training Return Snapshot")
            ax.tick_params(axis="x", rotation=75)
            fig.tight_layout()
            fig.savefig(output_dir / "training_curve_snapshot.png", dpi=200)
            plt.close(fig)

    copy_qualitative_assets(qualitative_dir, output_dir)
    manifest_path = output_dir / "figure_manifest.json"
    figures = [str(path) for path in sorted(output_dir.glob("*")) if path.is_file()]
    manifest_path.write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
