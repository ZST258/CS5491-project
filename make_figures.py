from __future__ import annotations

import argparse
import csv
import json
import shutil
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    USE_MPL = True
except Exception:
    plt = None
    USE_MPL = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence
    USE_PIL = True
except Exception:
    Image = ImageDraw = ImageFont = None
    USE_PIL = False


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


def _ensure_sorted(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return sorted(points, key=lambda p: p[0])


def save_png_bar_chart_pil(path: Path, title: str, categories: list[str], series: list[tuple[str, list[float]]], y_label: str, width: int = 1200, height: int = 640):
    # simple bar chart renderer using Pillow
    if not USE_PIL:
        raise RuntimeError("Pillow is required for PNG fallback drawing but is not available")
    margin_left = 100
    margin_right = 40
    margin_top = 60
    margin_bottom = 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    # compute max value
    max_candidates = [max(values) for _, values in series if values]
    max_value = max(max_candidates + [1.0])
    band_width = plot_width / max(len(categories), 1)
    bar_width = band_width / max(len(series) + 1, 2)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
    # title
    draw.text((width // 2 - 200, 12), title, fill=(20, 20, 20), font=title_font)
    # y ticks
    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_height - (value / max_value) * plot_height
        draw.line((margin_left, y, width - margin_right, y), fill=(230, 230, 230))
        draw.text((margin_left - 10, y - 8), f"{value:.2f}", fill=(0, 0, 0), font=title_font, anchor="rm")
    colors = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(140,86,75)]
    for category_index, category in enumerate(categories):
        base_x = margin_left + category_index * band_width + band_width * 0.1
        for series_index, (_, values) in enumerate(series):
            value = values[category_index]
            bar_h = 0 if max_value == 0 else (value / max_value) * plot_height
            x = base_x + series_index * bar_width
            y = margin_top + plot_height - bar_h
            draw.rectangle((x, y, x + bar_width * 0.8, margin_top + plot_height), fill=colors[series_index % len(colors)])
        label_x = margin_left + category_index * band_width + band_width / 2
        draw.text((label_x, height - margin_bottom + 10), category, fill=(0, 0, 0), font=title_font, anchor="mm")
    # legend
    legend_x = width - margin_right - 200
    legend_y = margin_top + 10
    for i, (label, _) in enumerate(series):
        y = legend_y + i * 22
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=colors[i % len(colors)])
        draw.text((legend_x + 20, y), label, fill=(0, 0, 0), font=title_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(200, 200))


def save_png_line_chart_pil(path: Path, title: str, series: List[Tuple[str, List[Tuple[float, float]]]], y_label: str, width: int = 1200, height: int = 640):
    if not USE_PIL:
        raise RuntimeError("Pillow is required for PNG fallback drawing but is not available")
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_x = [x for _, points in series for x, _ in points]
    all_y = [y for _, points in series for _, y in points]
    max_x = max(all_x) if all_x else 100.0
    min_y = min(all_y) if all_y else 0.0
    max_y = max(all_y) if all_y else 1.0
    span_y = max(max_y - min_y, 1e-6)
    colors = [(31,119,180),(255,127,14),(44,160,44),(214,39,40)]

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
    draw.text((width // 2 - 200, 12), title, fill=(20, 20, 20), font=title_font)
    # grid and ticks
    for tick in range(6):
        value = min_y + span_y * tick / 5
        y = margin_top + plot_height - ((value - min_y) / span_y) * plot_height
        draw.line((margin_left, y, width - margin_right, y), fill=(230, 230, 230))
        draw.text((margin_left - 10, y - 8), f"{value:.2f}", fill=(0, 0, 0), font=title_font, anchor="rm")
    # draw lines
    for idx, (label, points) in enumerate(series):
        pts = _ensure_sorted(points)
        coords = []
        for x_val, y_val in pts:
            x = margin_left + (x_val / max_x) * plot_width
            y = margin_top + plot_height - ((y_val - min_y) / span_y) * plot_height
            coords.append((x, y))
        if coords:
            draw.line(coords, fill=colors[idx % len(colors)], width=3)
    # legend (simple)
    legend_x = width - margin_right - 220
    legend_y = margin_top + 10
    for idx, (label, _) in enumerate(series):
        y = legend_y + idx * 20
        draw.rectangle((legend_x, y, legend_x + 12, y + 12), fill=colors[idx % len(colors)])
        draw.text((legend_x + 18, y), label, fill=(0,0,0), font=title_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(200,200))


def build_grouped_rows(main_rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped = {}
    for row in main_rows:
        grouped.setdefault(row["difficulty"], {})[row["model"]] = row
    return grouped


def build_eval_index(eval_dir: Path) -> dict:
    """
    Build a simple index of evaluation summary JSONs for fallback lookups.
    Keys are (label, difficulty, seed_str) where label is e.g. 'predictive_h3' or 'mlp'.
    Also populate a (label, difficulty, None) entry for a seed-agnostic fallback.
    """
    index: dict[tuple[str, str, str | None], dict] = {}
    if not eval_dir.exists():
        return index
    for path in eval_dir.glob("*_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        md = payload.get("metadata", {})
        model = md.get("model")
        # prefer checkpoint metadata horizon when available
        horizon = None
        if isinstance(md.get("checkpoint_metadata"), dict):
            horizon = md.get("checkpoint_metadata", {}).get("model_kwargs", {}).get("horizon")
        if model == "predictive" and horizon is not None:
            label = f"predictive_h{horizon}"
        else:
            label = model
        seed = md.get("seed")
        seed_str = f"seed{seed}" if seed is not None else None
        for difficulty, metrics in payload.get("summary", {}).items():
            key = (label, difficulty, seed_str)
            index[key] = metrics
            # also insert seed-agnostic fallback if not present
            agg_key = (label, difficulty, None)
            if agg_key not in index:
                index[agg_key] = metrics
    return index


def _hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{int(max(0,min(255,c))):02x}" for c in rgb)


def _lighten_hex(hexstr: str, amount: float = 0.5) -> str:
    """Lighten a hex color by blending it with white by given amount (0..1)."""
    r, g, b = _hex_to_rgb(hexstr)
    r2 = int(r + (255 - r) * amount)
    g2 = int(g + (255 - g) * amount)
    b2 = int(b + (255 - b) * amount)
    return _rgb_to_hex((r2, g2, b2))


def _model_color_pairs(labels: list[str]) -> list[tuple[str, str]]:
    """Return a list of (collision_color, timeout_color) for each label.
    Uses a base palette and creates a lighter variant for timeout.
    """
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    pairs = []
    for i, _ in enumerate(labels):
        base = base_colors[i % len(base_colors)]
        light = _lighten_hex(base, amount=0.5)
        pairs.append((base, light))
    return pairs


def _get_metric_from_sources(label: str, difficulty: str, metric_key: str, *,
                             grouped: dict, report_numbers: dict | None, eval_index: dict) -> float | None:
    """
    Try to obtain metric value for (label, difficulty, metric_key) from multiple sources.
    Order: grouped CSV rows (main_results), report_numbers.json, eval_index (per-run summary JSONs).
    Returns float or None if not found.
    """
    # 1) grouped rows (main_results.csv) - grouped uses model keys like 'predictive'
    # allow requesting 'predictive_h3' -> grouped key 'predictive'
    candidate_row = None
    # direct model key
    if grouped.get(difficulty, {}).get(label):
        candidate_row = grouped[difficulty][label]
    # special-case predictive_hX -> grouped key 'predictive'
    if candidate_row is None and label and label.startswith("predictive_h"):
        cand = grouped.get(difficulty, {}).get("predictive")
        if cand:
            candidate_row = cand
    if candidate_row is not None:
        val = candidate_row.get(metric_key, "")
        if val not in {"", None, "NA"}:
            try:
                return float(val)
            except Exception:
                pass
        # if csv row has run_name we can fallback to eval JSON for this specific run
        run_name = candidate_row.get("run_name")
        if run_name:
            eval_path = Path("outputs") / "eval" / f"{run_name}_summary.json"
            if eval_path.exists():
                try:
                    payload = json.loads(eval_path.read_text(encoding="utf-8"))
                    metrics = payload.get("summary", {}).get(difficulty, {})
                    if metric_key in metrics:
                        return float(metrics[metric_key])
                except Exception:
                    pass

    # 2) report_numbers.json (aggregated blocks)
    if report_numbers:
        # main block
        main_block = report_numbers.get("main", {}).get(difficulty, {})
        if main_block:
            # map predictive_h3 -> predictive in main block
            if label == "predictive_h3":
                src = main_block.get("predictive", {})
            else:
                src = main_block.get(label)
            if isinstance(src, dict) and metric_key in src:
                try:
                    return float(src[metric_key])
                except Exception:
                    pass
        # ablation block
        ablation_block = report_numbers.get("ablation", {}).get(difficulty, {})
        if ablation_block and label in ablation_block:
            try:
                return float(ablation_block[label][metric_key])
            except Exception:
                pass

    # 3) eval_index: try exact seed-agnostic then seed-specific if label contains seed
    # try seed-agnostic
    if (label, difficulty, None) in eval_index:
        metrics = eval_index[(label, difficulty, None)]
        if metric_key in metrics:
            try:
                return float(metrics[metric_key])
            except Exception:
                pass
    # try seed-specific keys (seed0..seed9)
    for s in ("seed0", "seed1", "seed2", "seed3", None):
        key = (label, difficulty, s)
        if key in eval_index:
            metrics = eval_index[key]
            if metric_key in metrics:
                try:
                    return float(metrics[metric_key])
                except Exception:
                    pass

    return None


def load_train_histories(train_log_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    # returns mapping run_name -> {metric: [(percent, value), ...], ...}
    training_summary_rows = load_rows(train_log_dir.parent / "report_assets" / "training_summary.csv")
    histories: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    metrics_to_collect = [
        "episode_return",
        "loss",
        "value_loss",
        "policy_loss",
        "entropy",
        "value_pred_mean",
        "returns_mean",
    ]
    for row in training_summary_rows:
        history_path = Path(row["history_path"]) if row.get("history_path") else None
        if not history_path or not history_path.exists():
            continue
        history_rows = load_rows(history_path)
        if not history_rows:
            continue
        # determine normalization denominator: prefer total_timesteps if present
        try:
            total_timesteps = float(row.get("total_timesteps")) if row.get("total_timesteps") else None
        except Exception:
            total_timesteps = None
        # gather global steps
        steps = [float(r["global_step"]) for r in history_rows if r.get("global_step", "") != ""]
        if not steps:
            continue
        max_step = max(steps)
        denom = total_timesteps if total_timesteps and total_timesteps > 0 else max_step
        if denom <= 0:
            denom = max_step
        hist_dict: Dict[str, List[Tuple[float, float]]] = {m: [] for m in metrics_to_collect}
        # compute percent per row and collect metrics
        for hr in history_rows:
            if hr.get("global_step", "") == "":
                continue
            step = float(hr["global_step"])
            percent = 100.0 * (step / denom)
            if percent > 100.0:
                percent = 100.0
            for m in metrics_to_collect:
                v = hr.get(m, "")
                if v in {"", None}:
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                hist_dict[m].append((percent, val))
            # also build abs diff if both present
            v1 = hr.get("value_pred_mean", "")
            v2 = hr.get("returns_mean", "")
            if v1 not in {"", None} and v2 not in {"", None}:
                try:
                    val1 = float(v1)
                    val2 = float(v2)
                    hist_dict.setdefault("abs_value_return_diff", []).append((percent, abs(val1 - val2)))
                except Exception:
                    pass
        histories[row["run_name"]] = hist_dict
    return histories


def process_disagreements(disagreement_root: Path, output_dir: Path):
    """
    Read outputs/disagreement/{a,b,c} and for each pattern create a 2x2 PNG
    that contains the final frame (as PNG) of the gifs found for each model.
    Expected structure (flexible): outputs/disagreement/<pattern>/<run_name>/.../*.gif
    """
    if not disagreement_root.exists():
        print(f"No disagreement directory found at {disagreement_root}; skipping qualitative figures")
        return
    patterns = [p.name for p in disagreement_root.iterdir() if p.is_dir()]
    target_models = ["mlp", "gnn", "predictive_h3", "predictive_h1"]

    for pattern in sorted(patterns):
        pattern_dir = disagreement_root / pattern
        gif_files = list(pattern_dir.rglob("*.gif"))
        if not gif_files:
            print(f"No GIFs found under {pattern_dir}; skipping")
            continue
        # find gif per target model
        found: Dict[str, Path] = {}
        for gf in gif_files:
            name = gf.name.lower()
            # try to map to predictive_h3/h1 first
            if "predictive" in name and ("_h3" in name or "h3" in name):
                found.setdefault("predictive_h3", gf)
                continue
            if "predictive" in name and ("_h1" in name or "h1" in name):
                found.setdefault("predictive_h1", gf)
                continue
            if name.startswith("mlp") or "mlp_" in name or "_mlp" in name:
                found.setdefault("mlp", gf)
                continue
            if name.startswith("gnn") or "gnn_" in name or "_gnn" in name:
                found.setdefault("gnn", gf)
                continue
            # fallback: if filename contains 'predictive' but no horizon label
            if "predictive" in name and "predictive_h3" not in found and "predictive_h1" not in found:
                # prefer h3 if ambiguity
                found.setdefault("predictive_h3", gf)

        # build 2x2 grid; load last frames
        panel_w = 560
        panel_h = 420
        grid_w = panel_w * 2
        grid_h = panel_h * 2
        canvas = Image.new("RGB", (grid_w, grid_h), "white") if USE_PIL else None
        draw = ImageDraw.Draw(canvas) if USE_PIL else None

        positions = [(0, 0), (panel_w, 0), (0, panel_h), (panel_w, panel_h)]
        labels = ["MLP", "GNN", "PRED(H=3)", "PRED(H=1)"]

        for idx, model_key in enumerate(target_models):
            pos = positions[idx]
            label = labels[idx]
            gif_path = found.get(model_key)
            # layout parameters: small top padding, label area below image; bottom padding larger for bottom row
            top_padding = 8
            label_gap = 8
            label_height = 22
            bottom_padding = 12 if idx < 2 else 32
            max_image_h = panel_h - top_padding - label_gap - label_height - bottom_padding
            if max_image_h < 16:
                max_image_h = max(16, panel_h - 60)

            if not USE_PIL:
                print(f"Pillow is required to extract frames from GIFs for disagreement figures; skipping {pattern}")
                break

            if not gif_path:
                # draw placeholder in image area and label below
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except Exception:
                    font = ImageFont.load_default()
                # placeholder center within allotted image area
                img_center_x = pos[0] + panel_w // 2
                img_center_y = pos[1] + top_padding + max_image_h // 2
                draw.text((img_center_x, img_center_y), "No GIF", fill=(0, 0, 0), font=font, anchor="mm")
                # label below
                label_x = pos[0] + panel_w // 2
                label_y = pos[1] + top_padding + max_image_h + label_gap
                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font, anchor="mm")
                continue

            # extract composed last frame from gif
            try:
                im = Image.open(gif_path)
                last = None
                # compose frames to handle partial-frame GIFs
                for frame in ImageSequence.Iterator(im):
                    frm = frame.convert("RGBA")
                    if last is None:
                        last = Image.new("RGBA", frm.size)
                    last = Image.alpha_composite(last, frm)
                if last is None:
                    im.seek(max(0, im.n_frames - 1))
                    last = im.convert("RGBA")
                last_rgb = last.convert("RGB")
            except Exception as exc:
                print(f"Failed to extract last frame from {gif_path}: {exc}")
                continue

            # resize to fit within panel width and max_image_h while preserving aspect
            w, h = last_rgb.size
            scale = min(panel_w / w, max_image_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = last_rgb.resize((new_w, new_h), Image.LANCZOS)

            # compute paste position: horizontally centered, vertically within top area
            paste_x = pos[0] + (panel_w - new_w) // 2
            paste_y = pos[1] + top_padding + (max_image_h - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))

            # draw label centered below the image
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            label_x = pos[0] + panel_w // 2
            label_y = paste_y + new_h + label_gap
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font, anchor="mm")

        # save composite
        out_path = output_dir / f"disagreement_{pattern}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if canvas:
            canvas.save(out_path, dpi=(200, 200))
        print(f"Wrote disagreement figure: {out_path}")


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
        ("time_efficiency", "Time Efficiency by Difficulty", "time_efficiency"),
        ("move_efficiency", "Move Efficiency by Difficulty", "move_efficiency"),
    ]

    # build eval index for fallback metric lookup
    eval_index = build_eval_index(Path("outputs") / "eval")
    # load report_numbers.json if present for aggregated lookups
    report_numbers = None
    try:
        report_numbers = json.loads((report_dir / "report_numbers.json").read_text(encoding="utf-8"))
    except Exception:
        report_numbers = None

    # We no longer generate single-metric figures (e.g., success_rate.png) because
    # group composite figures (group_main_metrics, group_ablation_metrics, group_stability_metrics)
    # contain the needed comparisons. Remove any existing single-metric files in the output dir.
    for fname in ["success_rate.png", "collision_rate.png", "time_efficiency.png", "move_efficiency.png"]:
        p = output_dir / fname
        if p.exists():
            try:
                p.unlink()
                print(f"Removed single-metric figure: {p}")
            except Exception as exc:
                print(f"Failed to remove {p}: {exc}")

    histories = load_train_histories(train_log_dir)
    if histories:
        # training return curve
        series = []
        for run_name, hist in histories.items():
            pts = hist.get("episode_return", [])
            if pts:
                series.append((run_name, _ensure_sorted(pts)))
        out_path = output_dir / "training_curve.png"
        if USE_MPL:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            for label, points in series:
                xs = [p for p, _ in points]
                ys = [v for _, v in points]
                ax.plot(xs, ys, label=label)
            ax.set_title("Training Return Curve")
            ax.set_xlabel("Training Progress (%)")
            ax.set_ylabel("Episode Return")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
        else:
            save_png_line_chart_pil(out_path, "Training Return Curve", series, y_label="Episode Return")

        # Loss figure: 1x3 for loss, value_loss, policy_loss
        panels = ["loss", "value_loss", "policy_loss"]
        if USE_MPL:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            for ax_idx, metric in enumerate(panels):
                ax = axes[ax_idx]
                for run_name, hist in histories.items():
                    pts = hist.get(metric, [])
                    if not pts:
                        continue
                    xs = [p for p, _ in pts]
                    ys = [v for _, v in pts]
                    ax.plot(xs, ys, label=run_name)
                ax.set_title(metric)
                ax.set_xlabel("Training Progress (%)")
            axes[0].set_ylabel("Loss")
            axes[0].legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(output_dir / "loss_panels.png", dpi=200)
            plt.close(fig)
        else:
            # build PIL multi-panel image
            width, height = 1800, 480
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            try:
                title_font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                title_font = ImageFont.load_default()
            panel_w = width // 3
            for i, metric in enumerate(panels):
                # collect series for this metric
                s = []
                for run_name, hist in histories.items():
                    pts = hist.get(metric, [])
                    if pts:
                        s.append((run_name, _ensure_sorted(pts)))
                # draw small panel
                if not s:
                    continue
                # draw axes
                margin = 50
                left = i * panel_w + margin
                top = 40
                pw = panel_w - margin * 2
                ph = height - 100
                # find y range
                all_vals = [v for _, pts in s for _, v in pts]
                min_y = min(all_vals)
                max_y = max(all_vals)
                span = max(max_y - min_y, 1e-6)
                for run_idx, (label, pts) in enumerate(s):
                    coords = []
                    for p, v in pts:
                        x = left + (p / 100.0) * pw
                        y = top + ph - ((v - min_y) / span) * ph
                        coords.append((x, y))
                    draw.line(coords, fill=(31,119,180) if run_idx==0 else (255,127,14), width=2)
                draw.text((i * panel_w + 10, 10), metric, fill=(0,0,0), font=title_font)
            outp = output_dir / "loss_panels.png"
            img.save(outp, dpi=(200,200))

        # PopArt quality: value_pred_mean, returns_mean, abs diff
        panels = ["value_pred_mean", "returns_mean", "abs_value_return_diff"]
        if USE_MPL:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            for ax_idx, metric in enumerate(panels):
                ax = axes[ax_idx]
                for run_name, hist in histories.items():
                    pts = hist.get(metric, [])
                    if not pts:
                        continue
                    xs = [p for p, _ in pts]
                    ys = [v for _, v in pts]
                    ax.plot(xs, ys, label=run_name)
                ax.set_title(metric)
                ax.set_xlabel("Training Progress (%)")
            axes[0].set_ylabel("Value / Returns")
            axes[0].legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(output_dir / "popart_quality.png", dpi=200)
            plt.close(fig)
        else:
            # simple PIL rendering
            width, height = 1800, 480
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            try:
                title_font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                title_font = ImageFont.load_default()
            panel_w = width // 3
            for i, metric in enumerate(panels):
                s = []
                for run_name, hist in histories.items():
                    pts = hist.get(metric, [])
                    if pts:
                        s.append((run_name, _ensure_sorted(pts)))
                if not s:
                    continue
                left = i * panel_w + 50
                top = 40
                pw = panel_w - 100
                ph = height - 100
                all_vals = [v for _, pts in s for _, v in pts]
                min_y = min(all_vals)
                max_y = max(all_vals)
                span = max(max_y - min_y, 1e-6)
                for run_idx, (label, pts) in enumerate(s):
                    coords = []
                    for p, v in pts:
                        x = left + (p / 100.0) * pw
                        y = top + ph - ((v - min_y) / span) * ph
                        coords.append((x, y))
                    draw.line(coords, fill=(31,119,180) if run_idx==0 else (255,127,14), width=2)
                draw.text((i * panel_w + 10, 10), metric, fill=(0,0,0), font=title_font)
            outp = output_dir / "popart_quality.png"
            img.save(outp, dpi=(200,200))

        # Entropy plot
        series = []
        for run_name, hist in histories.items():
            pts = hist.get("entropy", [])
            if pts:
                series.append((run_name, _ensure_sorted(pts)))
        out_path = output_dir / "entropy.png"
        if USE_MPL:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            for label, points in series:
                xs = [p for p, _ in points]
                ys = [v for _, v in points]
                ax.plot(xs, ys, label=label)
            ax.set_title("Policy Entropy")
            ax.set_xlabel("Training Progress (%)")
            ax.set_ylabel("Entropy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
        else:
            save_png_line_chart_pil(out_path, "Policy Entropy", series, y_label="Entropy")

    else:
        # snapshot fallback (when no detailed histories exist)
        snapshot_rows = load_rows(report_dir / "training_summary.csv")
        labels = [row["run_name"] for row in snapshot_rows]
        values = [float(row["episode_return"]) if row["episode_return"] != "" else 0.0 for row in snapshot_rows]
        if not labels:
            labels = ["pending"]
            values = [0.0]
        out_path = output_dir / "training_curve_snapshot.png"
        if USE_MPL:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(labels, values)
            ax.set_ylabel("Episode Return")
            ax.set_title("Training Return Snapshot")
            ax.tick_params(axis="x", rotation=75)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
        else:
            # use simple PIL bar chart
            save_png_bar_chart_pil(out_path, "Training Return Snapshot", labels, [("episode_return", values)], "Episode Return")

    # Instead of copying qualitative SVGs, build disagreement composites from
    # outputs/disagreement/{a,b,c} which should contain GIFs for each model.
    process_disagreements(Path("outputs") / "disagreement", output_dir)
    # Build group composite figures (2x2: success_rate, collision_rate, time_efficiency, move_efficiency)
    def build_group_composite(group_name: str, categories: list[str], labels: list[str], out_name: str):
        # categories: x-axis categories (difficulties/seeds)
        # labels: models or variant labels to include as series
        panels = [
            ("success_rate", "Success Rate"),
            ("collision_rate", "Collision Rate"),
            ("time_efficiency", "Time Efficiency"),
            ("move_efficiency", "Move Efficiency"),
        ]
        # collect series per panel. For collision_rate we build stacked components:
        # collision and timeout (= 1 - success_rate - collision_rate).
        panel_series = []
        for metric_key, metric_title in panels:
            if metric_key != "collision_rate":
                series = []
                for label in labels:
                    values = []
                    for cat in categories:
                        if group_name == "stability":
                            difficulty = "hard"
                            seed = cat
                            val = _get_metric_from_sources(label, difficulty, metric_key, grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                            if val is None and (label, difficulty, seed) in eval_index:
                                try:
                                    val = float(eval_index[(label, difficulty, seed)].get(metric_key))
                                except Exception:
                                    val = None
                        else:
                            difficulty = cat
                            val = _get_metric_from_sources(label, difficulty, metric_key, grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                        if val is None:
                            print(f"Warning: missing {metric_key} for label={label} category={cat} in group={group_name}; using 0.0")
                            val = 0.0
                        values.append(val)
                    series.append((label, values))
                panel_series.append({"title": metric_title, "kind": "grouped", "series": series})
            else:
                # build stacked components for failure breakdown
                collision_by_label = []
                timeout_by_label = []
                for label in labels:
                    coll_vals = []
                    to_vals = []
                    for cat in categories:
                        if group_name == "stability":
                            difficulty = "hard"
                            seed = cat
                            coll = _get_metric_from_sources(label, difficulty, "collision_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                            succ = _get_metric_from_sources(label, difficulty, "success_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                            # try seed-specific eval entries if needed
                            if coll is None and (label, difficulty, seed) in eval_index:
                                try:
                                    coll = float(eval_index[(label, difficulty, seed)].get("collision_rate", 0.0))
                                except Exception:
                                    coll = None
                            if succ is None and (label, difficulty, seed) in eval_index:
                                try:
                                    succ = float(eval_index[(label, difficulty, seed)].get("success_rate", 0.0))
                                except Exception:
                                    succ = None
                        else:
                            difficulty = cat
                            coll = _get_metric_from_sources(label, difficulty, "collision_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                            succ = _get_metric_from_sources(label, difficulty, "success_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                        if coll is None:
                            coll = 0.0
                        if succ is None:
                            succ = 0.0
                        timeout = 1.0 - succ - coll
                        if timeout < 0:
                            timeout = 0.0
                        coll_vals.append(coll)
                        to_vals.append(timeout)
                    collision_by_label.append(coll_vals)
                    timeout_by_label.append(to_vals)
                panel_series.append({
                    "title": "Failure Breakdown (Collision vs Timeout)",
                    "kind": "stacked",
                    "labels": labels,
                    "collision_by_label": collision_by_label,
                    "timeout_by_label": timeout_by_label,
                })

        # Now render 2x2 composite using matplotlib if available, else PIL
        out_path = output_dir / out_name
        # Special-case: stability group only needs one combined plot (all metrics across seeds)
        if group_name == "stability":
            # build a single panel where x-axis is seeds and series are the four metrics
            metrics = ["success_rate", "collision_rate", "time_efficiency", "move_efficiency"]
            metric_titles = ["Success Rate", "Failure Breakdown", "Time Efficiency", "Move Efficiency"]
            # compute values per metric per seed; collision is handled as stacked (collision + timeout)
            non_collision_metrics = ["success_rate", "time_efficiency", "move_efficiency"]
            non_coll_vals = {m: [] for m in non_collision_metrics}
            coll_vals = []
            timeout_vals = []
            model_label = labels[0] if labels else "model"
            # color pairs for the model(s)
            model_color_pairs = _model_color_pairs(labels)
            for cat in categories:
                seed = cat
                # for non-collision metrics
                for m in non_collision_metrics:
                    v = None
                    if (model_label, "hard", seed) in eval_index:
                        try:
                            v = float(eval_index[(model_label, "hard", seed)].get(m))
                        except Exception:
                            v = None
                    if v is None:
                        v = _get_metric_from_sources(model_label, "hard", m, grouped=grouped, report_numbers=report_numbers, eval_index=eval_index)
                    if v is None:
                        print(f"Warning: missing {m} for stability seed={seed}; using 0.0")
                        v = 0.0
                    non_coll_vals[m].append(v)
                # collision + timeout
                coll = None
                succ = None
                if (model_label, "hard", seed) in eval_index:
                    try:
                        coll = float(eval_index[(model_label, "hard", seed)].get("collision_rate", 0.0))
                    except Exception:
                        coll = None
                    try:
                        succ = float(eval_index[(model_label, "hard", seed)].get("success_rate", 0.0))
                    except Exception:
                        succ = None
                if coll is None:
                    coll = _get_metric_from_sources(model_label, "hard", "collision_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index) or 0.0
                if succ is None:
                    succ = _get_metric_from_sources(model_label, "hard", "success_rate", grouped=grouped, report_numbers=report_numbers, eval_index=eval_index) or 0.0
                to = 1.0 - succ - coll
                if to < 0:
                    to = 0.0
                coll_vals.append(coll)
                timeout_vals.append(to)

            # prepare plotting
            if USE_MPL:
                fig, ax = plt.subplots(figsize=(10, 5))
                x = list(range(len(categories)))
                num_metrics = len(metrics)
                width = 0.18
                # choose metric colors for non-collision metrics
                # choose metric colors; use teal for success_rate to avoid clashing with model collision color
                metric_palette = ["#17becf", "#2ca02c", "#9467bd"]
                metric_idx = 0
                for i, metric_key in enumerate(metrics):
                    offsets = [xi + (i - (num_metrics-1)/2) * width for xi in x]
                    if metric_key == "collision_rate":
                        # stacked bar using model color pair (only one model_label expected here)
                        coll_col, to_col = model_color_pairs[0]
                        ax.bar(offsets, coll_vals, width=width, color=coll_col)
                        ax.bar(offsets, timeout_vals, width=width, bottom=coll_vals, color=to_col)
                    else:
                        vals = non_coll_vals[metric_key]
                        col = metric_palette[metric_idx % len(metric_palette)]
                        ax.bar(offsets, vals, width=width, label=metric_titles[i] if metric_key != "collision_rate" else "Failure Breakdown", color=col)
                        metric_idx += 1
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.set_title(f"Stability Metrics ({model_label})")
                ax.set_ylabel("Value")
                # build legend: non-collision metrics + collision components
                from matplotlib.patches import Patch
                metric_handles = [Patch(facecolor=metric_palette[i], label=non_collision_metrics[i].replace("_", " ").title()) for i in range(len(non_collision_metrics))]
                comp_handles = [Patch(facecolor=model_color_pairs[0][0], label=f"{model_label} Collision"), Patch(facecolor=model_color_pairs[0][1], label=f"{model_label} Timeout")]
                handles = metric_handles + comp_handles
                ax.legend(handles=handles, fontsize="small")
                fig.tight_layout()
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
            else:
                # PIL fallback: draw grouped bars where collision is stacked
                panel_w = 1000
                panel_h = 480
                img = Image.new("RGB", (panel_w, panel_h), "white")
                draw = ImageDraw.Draw(img)
                try:
                    title_font = ImageFont.truetype("arial.ttf", 16)
                except Exception:
                    title_font = ImageFont.load_default()
                band_w = (panel_w - 160) / max(len(categories), 1)
                metric_palette = ["#17becf", "#2ca02c", "#9467bd"]
                for ci, cat in enumerate(categories):
                    base_x = 80 + ci * band_w
                    for i, metric_key in enumerate(metrics):
                        bar_w = band_w / (len(metrics) + 1)
                        x = base_x + i * bar_w
                        if metric_key == "collision_rate":
                            coll = coll_vals[ci]
                            to = timeout_vals[ci]
                            # scale to panel height
                            max_v = max((v for sub in non_coll_vals.values() for v in sub) + coll_vals + timeout_vals) or 1.0
                            coll_h = 0 if max_v == 0 else (coll / max_v) * (panel_h - 160)
                            to_h = 0 if max_v == 0 else (to / max_v) * (panel_h - 160)
                            y_coll = panel_h - 80 - coll_h
                            y_to = y_coll - to_h
                            coll_col, to_col = model_color_pairs[0]
                            draw.rectangle((x, y_coll, x + bar_w * 0.8, panel_h - 80), fill=coll_col)
                            draw.rectangle((x, y_to, x + bar_w * 0.8, y_coll), fill=to_col)
                        else:
                            vals = non_coll_vals[metric_key]
                            val = vals[ci]
                            max_v = max((v for sub in non_coll_vals.values() for v in sub) + coll_vals + timeout_vals) or 1.0
                            bar_h = 0 if max_v == 0 else (val / max_v) * (panel_h - 160)
                            y = panel_h - 80 - bar_h
                            col = metric_palette.pop(0) if metric_palette else "#777777"
                            draw.rectangle((x, y, x + bar_w * 0.8, panel_h - 80), fill=col)
                    draw.text((80 + ci * band_w + band_w/2, panel_h - 60), cat, fill=(0,0,0), font=title_font, anchor="mm")
                draw.text((10, 10), f"Stability Metrics ({model_label})", fill=(0,0,0), font=title_font)
                img.save(out_path, dpi=(200,200))
            print(f"Wrote group composite figure: {out_path}")
            return

        if USE_MPL:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
            x = list(range(len(categories)))
            for ax_idx, panel in enumerate(panel_series):
                ax = axes_flat[ax_idx]
                if panel["kind"] == "grouped":
                    series = panel["series"]
                    width = 0.15
                    for i, (label, values) in enumerate(series):
                        offsets = [item + (i - (len(series)-1)/2) * width for item in x]
                        ax.bar(offsets, values, width=width, label=label)
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.set_title(panel["title"])
                    if ax_idx == 0 or ax_idx == 2:
                        ax.set_ylabel(panel["title"])
                    ax.legend(fontsize="small")
                elif panel["kind"] == "stacked":
                    # stacked per-model bars where each bar is collision (bottom) + timeout (top)
                    labels_local = panel["labels"]
                    coll_by_label = panel["collision_by_label"]
                    to_by_label = panel["timeout_by_label"]
                    width = 0.15
                    # get color pairs per model: (collision_color, timeout_color)
                    color_pairs = _model_color_pairs(labels_local)
                    # plot stacked bars for each model grouping using model-specific colors
                    for i, lab in enumerate(labels_local):
                        coll_vals = coll_by_label[i]
                        to_vals = to_by_label[i]
                        offsets = [item + (i - (len(labels_local)-1)/2) * width for item in x]
                        coll_col, to_col = color_pairs[i]
                        ax.bar(offsets, coll_vals, width=width, color=coll_col)
                        ax.bar(offsets, to_vals, width=width, bottom=coll_vals, color=to_col)
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.set_title(panel["title"])
                    ax.set_ylabel("Rate")
                    # add legend: components and model proxies
                    from matplotlib.patches import Patch
                    comp_handles = [Patch(facecolor="#555555", label="Collision (model-colored bottom)"), Patch(facecolor="#bbbbbb", label="Timeout (model-colored top)")]
                    comp_legend = ax.legend(handles=comp_handles, fontsize="small", title="Failure components", loc="upper right")
                    # model proxies for legend (show labels with their colors)
                    model_handles = [Patch(facecolor=color_pairs[i][0], label=lab) for i, lab in enumerate(labels_local)]
                    model_legend = ax.legend(handles=model_handles, fontsize="small", loc="upper left")
                    ax.add_artist(comp_legend)
            fig.suptitle(f"{group_name.capitalize()} Metrics")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
        else:
            # Simple PIL layout: place 2x2 panels vertically
            panel_w = 640
            panel_h = 480
            canvas_w = panel_w * 2
            canvas_h = panel_h * 2
            img = Image.new("RGB", (canvas_w, canvas_h), "white")
            draw = ImageDraw.Draw(img)
            try:
                title_font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                title_font = ImageFont.load_default()
            for idx, panel in enumerate(panel_series):
                col = idx % 2
                row = idx // 2
                left = col * panel_w
                top = row * panel_h
                cats = categories
                if panel["kind"] == "grouped":
                    svals = panel["series"]
                    # compute max
                    all_vals = [v for _, vals in svals for v in vals]
                    max_v = max(all_vals) if all_vals else 1.0
                    band_w = (panel_w - 160) / max(len(cats), 1)
                    for ci, cat in enumerate(cats):
                        base_x = left + 80 + ci * band_w
                        for si, (_, vals) in enumerate(svals):
                            val = vals[ci]
                            bar_w = band_w / max(len(svals)+1, 2)
                            x = base_x + si * bar_w
                            bar_h = 0 if max_v == 0 else (val / max_v) * (panel_h - 140)
                            y = top + (panel_h - 100) - bar_h
                            draw.rectangle((x, y, x + bar_w*0.8, top + (panel_h - 100)), fill=(31,119,180))
                        draw.text((left + 80 + ci * band_w + band_w/2, top + panel_h - 80), cat, fill=(0,0,0), font=title_font, anchor="mm")
                    draw.text((left + 10, top + 10), panel["title"], fill=(0,0,0), font=title_font)
                else:
                    # stacked failure breakdown
                    labels_local = panel["labels"]
                    coll_by_label = panel["collision_by_label"]
                    to_by_label = panel["timeout_by_label"]
                    # compute max for stacking scale (use coll+to)
                    all_vals = [v for lab_vals in coll_by_label for v in lab_vals] + [v for lab_vals in to_by_label for v in lab_vals]
                    max_v = max(all_vals) if all_vals else 1.0
                    band_w = (panel_w - 160) / max(len(cats), 1)
                    color_pairs = _model_color_pairs(labels_local)
                    for ci, cat in enumerate(cats):
                        base_x = left + 80 + ci * band_w
                        for si, lab in enumerate(labels_local):
                            coll = coll_by_label[si][ci]
                            to = to_by_label[si][ci]
                            bar_w = band_w / max(len(labels_local)+1, 2)
                            x = base_x + si * bar_w
                            coll_h = 0 if max_v == 0 else (coll / max_v) * (panel_h - 140)
                            to_h = 0 if max_v == 0 else (to / max_v) * (panel_h - 140)
                            y_coll = top + (panel_h - 100) - coll_h
                            y_to = y_coll - to_h
                            coll_col, to_col = color_pairs[si]
                            draw.rectangle((x, y_coll, x + bar_w*0.8, top + (panel_h - 100)), fill=coll_col)
                            draw.rectangle((x, y_to, x + bar_w*0.8, y_coll), fill=to_col)
                        draw.text((left + 80 + ci * band_w + band_w/2, top + panel_h - 80), cat, fill=(0,0,0), font=title_font, anchor="mm")
                    draw.text((left + 10, top + 10), panel["title"], fill=(0,0,0), font=title_font)
            img.save(out_path, dpi=(200,200))
        print(f"Wrote group composite figure: {out_path}")

    # Main group: compare models (oracle, mlp, gnn, predictive_h3) across difficulties
    build_group_composite("main", ["easy", "medium", "hard"], ["oracle", "mlp", "gnn", "predictive_h3"], "group_main_metrics.png")
    # Ablation: compare predictive_h3 vs predictive_h1 (and gnn if present) across medium/hard
    build_group_composite("ablation", ["medium", "hard"], ["gnn", "predictive_h1", "predictive_h3"], "group_ablation_metrics.png")
    # Stability: compare predictive_h3 across seeds seed0/seed1/seed2
    build_group_composite("stability", ["seed0", "seed1", "seed2"], ["predictive_h3"], "group_stability_metrics.png")
    manifest_path = output_dir / "figure_manifest.json"
    figures = [str(path) for path in sorted(output_dir.glob("*")) if path.is_file()]
    manifest_path.write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
