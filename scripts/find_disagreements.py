from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Tuple


def find_episode_files(eval_dir: Path) -> List[Path]:
    # match files ending with _hard_episodes.csv but exclude those starting with oracle
    files = []
    for path in sorted(eval_dir.glob("*_hard_episodes.csv")):
        if path.name.startswith("oracle"):
            continue
        files.append(path)
    return files


def read_episode_csv(path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                idx = int(r.get("episode_index", r.get("index", "0")))
            except Exception:
                continue
            rows[idx] = r
    return rows


def main():
    eval_dir = Path("outputs/eval")
    files = find_episode_files(eval_dir)
    if not files:
        print("No matching _hard_episodes.csv files found in outputs/eval")
        return

    # map: model_run_name -> {episode_index: row}
    data: Dict[str, Dict[int, Dict[str, str]]] = {}
    for path in files:
        name = path.stem  # e.g. mlp_hard_seed0_main_hard_episodes
        # normalize to model name (assume naming <model>_...)
        model = name.split("_")[0]
        data[name] = read_episode_csv(path)

    # collect episode indices present across runs (intersection would be stricter,
    # but we'll consider union and check existence per model when evaluating conditions)
    all_indices = set()
    for rows in data.values():
        all_indices.update(rows.keys())

    # Helper to get boolean success for a given run name and episode index
    def success(run_name: str, idx: int) -> bool:
        rows = data.get(run_name, {})
        row = rows.get(idx)
        if not row:
            return False
        val = row.get("success", "0")
        try:
            return int(val) != 0
        except Exception:
            return False

    # We need per-model run names for mlp, gnn, predictive variants (h3 and h1)
    # Find candidate run name keys by matching prefix
    def find_run(prefix: str) -> List[str]:
        return [name for name in data.keys() if name.startswith(prefix)]

    mlp_runs = find_run("mlp")
    gnn_runs = find_run("gnn")
    predictive_runs = [name for name in data.keys() if name.startswith("predictive")]

    # identify predictive h3 and h1 runs
    pred_h3 = [n for n in predictive_runs if "_h3_" in n or n.endswith("_h3")]
    pred_h1 = [n for n in predictive_runs if "_h1_" in n or n.endswith("_h1")]

    # For simplicity, if multiple runs per model exist (e.g., different seeds), we will
    # consider any run that meets the success/failure pattern. We'll keep track of which
    # run matched for reporting.

    results = {"a": [], "b": [], "c": []}  # list of tuples (episode_index, distance, details)

    for idx in sorted(all_indices):
        # Evaluate candidate truth values across available runs
        # We'll treat success if ANY mlp run succeeded (for condition intent we need mlp fail ->
        # all mlp runs fail? The user's phrasing suggests 'mlp model failed' meaning the mlp run
        # corresponding to the main run failed. Here we will interpret by checking that there exists
        # at least one mlp run and we use the first one. To be safer we check ANY run per model and
        # pick the most common meaning: prefer run names that include "_seed0_" or "_main" if present.

        def pick_run(runs: List[str]) -> str | None:
            if not runs:
                return None
            for candidate in runs:
                if "_seed0_" in candidate or candidate.endswith("_main"):
                    return candidate
            return runs[0]

        mlp = pick_run(mlp_runs)
        gnn = pick_run(gnn_runs)
        ph3 = pick_run(pred_h3)
        ph1 = pick_run(pred_h1)

        # If any required run is missing, skip index for patterns that need it
        # Pattern a: mlp fail, gnn success, pred_h3 success, pred_h1 success
        if mlp and gnn and ph3 and ph1:
            mlp_suc = success(mlp, idx)
            gnn_suc = success(gnn, idx)
            ph3_suc = success(ph3, idx)
            ph1_suc = success(ph1, idx)
            if (not mlp_suc) and gnn_suc and ph3_suc and ph1_suc:
                # compute distance proxy: use oracle_length if present, else path_length
                # attempt to find oracle run's row for this idx
                oracle_row = None
                for key in data.keys():
                    if key.startswith("oracle"):
                        oracle_row = data[key].get(idx)
                        if oracle_row:
                            break
                distance = None
                if oracle_row and oracle_row.get("oracle_length"):
                    try:
                        distance = float(oracle_row["oracle_length"])
                    except Exception:
                        distance = None
                # fallback to predictive h3's path_length if available, otherwise mlp
                if distance is None:
                    for k in (ph3, ph1, gnn, mlp):
                        row = data.get(k, {}).get(idx)
                        if row and row.get("path_length"):
                            try:
                                distance = float(row["path_length"])
                                break
                            except Exception:
                                continue
                if distance is None:
                    distance = 0.0
                details = {"mlp": mlp, "gnn": gnn, "predictive_h3": ph3, "predictive_h1": ph1}
                results["a"].append((idx, distance, details))

        # Pattern b: mlp fail, gnn fail, pred_h3 success, pred_h1 success
        if mlp and gnn and ph3 and ph1:
            mlp_suc = success(mlp, idx)
            gnn_suc = success(gnn, idx)
            ph3_suc = success(ph3, idx)
            ph1_suc = success(ph1, idx)
            if (not mlp_suc) and (not gnn_suc) and ph3_suc and ph1_suc:
                # distance logic same as above
                oracle_row = None
                for key in data.keys():
                    if key.startswith("oracle"):
                        oracle_row = data[key].get(idx)
                        if oracle_row:
                            break
                distance = None
                if oracle_row and oracle_row.get("oracle_length"):
                    try:
                        distance = float(oracle_row["oracle_length"])
                    except Exception:
                        distance = None
                if distance is None:
                    for k in (ph3, ph1, gnn, mlp):
                        row = data.get(k, {}).get(idx)
                        if row and row.get("path_length"):
                            try:
                                distance = float(row["path_length"])
                                break
                            except Exception:
                                continue
                if distance is None:
                    distance = 0.0
                details = {"mlp": mlp, "gnn": gnn, "predictive_h3": ph3, "predictive_h1": ph1}
                results["b"].append((idx, distance, details))

        # Pattern c: mlp fail, gnn fail, pred_h3 success, pred_h1 fail
        if mlp and gnn and ph3 and ph1:
            mlp_suc = success(mlp, idx)
            gnn_suc = success(gnn, idx)
            ph3_suc = success(ph3, idx)
            ph1_suc = success(ph1, idx)
            if (not mlp_suc) and (not gnn_suc) and ph3_suc and (not ph1_suc):
                oracle_row = None
                for key in data.keys():
                    if key.startswith("oracle"):
                        oracle_row = data[key].get(idx)
                        if oracle_row:
                            break
                distance = None
                if oracle_row and oracle_row.get("oracle_length"):
                    try:
                        distance = float(oracle_row["oracle_length"])
                    except Exception:
                        distance = None
                if distance is None:
                    for k in (ph3, ph1, gnn, mlp):
                        row = data.get(k, {}).get(idx)
                        if row and row.get("path_length"):
                            try:
                                distance = float(row["path_length"])
                                break
                            except Exception:
                                continue
                if distance is None:
                    distance = 0.0
                details = {"mlp": mlp, "gnn": gnn, "predictive_h3": ph3, "predictive_h1": ph1}
                results["c"].append((idx, distance, details))

    # prepare output base
    out_base = Path("outputs") / "disagreement"
    out_base.mkdir(parents=True, exist_ok=True)

    def stem_to_run_name(stem: str) -> str:
        # remove trailing _<difficulty>_episodes if present
        return re.sub(r"_(?:easy|medium|hard)_episodes$", "", stem)

    # For each pattern, choose the episode with the maximum distance and export assets
    for key in ("a", "b", "c"):
        entries = results[key]
        if not entries:
            print(f"Pattern {key}: no matching episodes found")
            continue
        entries.sort(key=lambda t: t[1], reverse=True)
        idx, distance, details = entries[0]
        print(f"Pattern {key} - selected episode {idx} (distance={distance})")
        print("  runs:")
        # create pattern directory
        pattern_out = out_base / key
        pattern_out.mkdir(parents=True, exist_ok=True)
        for role, stem in details.items():
            run_name = stem_to_run_name(stem) if stem else None
            print(f"    {role}: {run_name}")
            if not run_name:
                print(f"      Skipping {role}: no run_name available")
                continue
            model = run_name.split("_")[0]
            # we don't handle oracle exports here per user instruction
            if model == "oracle":
                print(f"      Skipping export for oracle model {run_name}")
                continue
            # check checkpoint presence; if missing, warn and skip
            checkpoint_path = Path("checkpoints") / f"{run_name}.pt"
            if not checkpoint_path.exists():
                print(f"      Warning: checkpoint {checkpoint_path} not found for {run_name}; skipping export")
                continue
            # build command to call export_rollouts.py for this specific episode
            cmd = [
                sys.executable,
                "export_rollouts.py",
                "--model",
                model,
                "--difficulty",
                "hard",
                "--episode-index",
                str(idx),
                "--run-name",
                run_name,
                "--output-dir",
                str(pattern_out),
                "--checkpoint-dir",
                "checkpoints",
            ]
            print(f"      Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"      export_rollouts.py failed for {run_name}: {exc}")


if __name__ == "__main__":
    main()
