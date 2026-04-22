# CS5491 Dynamic Navigation Project

Course project scaffold for a graph-based predictive agent in dynamic environments.  
This repository includes:

- A reproducible grid-world environment with moving obstacles and three fixed difficulty tiers
  - **Easy**: 8×8 grid, 3 obstacles, 30 max steps, knn_k=3
  - **Medium**: 10×10 grid, 6 obstacles, 45 max steps, knn_k=4
  - **Hard**: 12×12 grid, 10 obstacles, 60 max steps, knn_k=6
  - All tiers use max_obstacles=10 for consistent padding
- A time-expanded A* oracle for path-efficiency evaluation
- Three PPO-compatible agents: `mlp`, `gnn`, and `predictive`
- Frozen evaluation-suite generation, evaluation export, and smoke tests
- A locked experiment matrix, result aggregation utilities, and report-asset generation

---

## Project Structure

- `dynamic_nav/`: environment, oracle, models, PPO trainer, and utilities
- `train.py`: unified training entrypoint
- `eval.py`: unified evaluation entrypoint
- `generate_eval_suite.py`: creates the frozen benchmark suite
- `run_experiments.py`: executes experiment matrix entries from config
- `aggregate_results.py`: consolidates summaries into report-ready tables
- `make_figures.py`: generates final comparison figures
- `export_rollouts.py`: exports qualitative episode assets
- `generate_report_assets.py`: builds report/presentation draft assets
- `tests/`: environment, oracle, model, and training smoke tests
- `configs/eval_suite.json`: shared frozen evaluation episodes
- `configs/experiments.json`: locked main runs, ablations, and stability runs
- `reports/`: report and presentation outlines/drafts
- `outputs/`: generated logs, summaries, figures, and qualitative artifacts

---

## Setup

Recommended environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code has local fallbacks when `gymnasium` or `torch-geometric` are unavailable, but full project experiments should use the recommended dependencies.

> **Note:** To facilitate reproducibility and ease of experimentation, the entire project is hosted on [**Google Colab**](https://colab.research.google.com/drive/12k9WIKx4LiWfpN34oWvNnonfYmlPtsUK?authuser=1#scrollTo=73964dea). All components of the pipeline—including data processing, model training, evaluation, and visualization—can be executed end-to-end in a single notebook environment without additional configuration.
---

## Two-Phase Workflow

### Phase 1: Pipeline Completion
Complete and verify scripts, report assets, figure generation, and qualitative export logic without running all long experiments.

### Phase 2: Final Experiment Push
Execute the locked matrix, aggregate outputs, regenerate figures, and update report text with final numbers.

Recommended output layout:

- `outputs/eval`: per-run evaluation summaries and per-episode CSVs
- `outputs/train_logs`: training summaries and training-history CSVs
- `outputs/report_assets`: pivot tables, report numbers, narratives, and failure-case summaries
- `outputs/figures`: final bar charts, training curves, and copied qualitative SVGs
- `outputs/qualitative`: rollout JSON, step CSVs, ASCII frames, and trajectory SVGs

---

## Pipeline Example (Recommended End-to-End)

This is the standard workflow for running the project pipeline.

### 1) Generate the frozen evaluation dataset

```bash
python generate_eval_suite.py
```

(Equivalent explicit form:)

```bash
python generate_eval_suite.py --output configs/eval_suite.json --episodes-per-tier 100 --base-seed 2026
```

### 2) Run the `main` experiment group

```bash
python run_experiments.py --mode run --only-tag main
```

### 3) Resume long runs with checkpoint-friendly skipping

Because experiments can take a long time, you can safely resume and skip completed outputs:

```bash
python run_experiments.py --mode run --only-tag main --skip-existing
```

### 4) Preview commands without executing (dry run)

```bash
python run_experiments.py --mode dry-run
```

You can also inspect only one tag in dry-run mode:

```bash
python run_experiments.py --mode dry-run --only-tag main
```

### 5) Aggregate and generate final assets

After runs finish:

```bash
python aggregate_results.py
python make_figures.py
python generate_report_assets.py
```

These commands automatically produce consolidated metrics, figures, and report-ready assets.

---

## Common Commands

### Generate frozen evaluation suite

```bash
python3 generate_eval_suite.py --output configs/eval_suite.json --episodes-per-tier 100 --base-seed 2026
```

### Train one model on one tier

```bash
python3 train.py --model mlp --difficulty easy --seed 0 --total-timesteps 30000 --checkpoint-dir checkpoints
python3 train.py --model gnn --difficulty medium --seed 0 --total-timesteps 60000 --checkpoint-dir checkpoints
python3 train.py --model predictive --difficulty hard --seed 0 --total-timesteps 163840 --checkpoint-dir checkpoints
```

Note: Training timesteps are scaled based on model complexity and difficulty. See `configs/experiments.json` for the full hyperparameter configurations used in the main experiments.

### Train predictive ablation (`H=1`)

```bash
python3 train.py --model predictive --difficulty hard --seed 0 --total-timesteps 163840 --horizon 1 --run-name predictive_hard_seed0_h1_ablation
```

Note: The predictive model uses horizon=3 by default. The ablation study tests horizon=1 to measure the impact of multi-step prediction.

### Train across all tiers (round-robin curriculum)

```bash
python3 train.py --model predictive --difficulty all --seed 0 --total-timesteps 200000 --checkpoint-dir checkpoints
```

Note: Multi-difficulty training uses a round-robin curriculum that cycles through easy/medium/hard episodes.

### Evaluate on frozen suite

```bash
python3 eval.py --model mlp --difficulty easy --seed 0 --checkpoint-dir checkpoints --eval-suite configs/eval_suite.json --output-dir outputs
python3 eval.py --model oracle --difficulty all --eval-suite configs/eval_suite.json --output-dir outputs
```

### Preview and run locked experiment matrix

```bash
python3 run_experiments.py --mode dry-run
python3 run_experiments.py --mode run --device cuda
python3 run_experiments.py --mode run --device cuda --only-tag main --skip-existing --export-cases --case-limit 2
```

### Aggregate tables and generate figures

```bash
python3 aggregate_results.py --eval-dir outputs/eval --train-log-dir outputs/train_logs --qualitative-dir outputs/qualitative --output-dir outputs/report_assets
python3 make_figures.py --report-dir outputs/report_assets --train-log-dir outputs/train_logs --qualitative-dir outputs/qualitative --output-dir outputs/figures
python3 generate_report_assets.py --report-dir outputs/report_assets --figures-dir outputs/figures --reports-dir reports
```

### Export qualitative cases

```bash
python3 eval.py --model predictive --difficulty hard --seed 0 --run-name predictive_hard_seed0_h3_main --checkpoint-dir checkpoints --eval-suite configs/eval_suite.json --output-dir outputs/eval --export-cases --qualitative-dir outputs/qualitative
python3 export_rollouts.py --model predictive --difficulty hard --episode-index 3 --run-name predictive_hard_seed0_h3_main --checkpoint-dir checkpoints --output-dir outputs/qualitative
```

### Run tests

```bash
pytest
```

---

## Observation Schema

Each environment step returns a dictionary with:

- `node_features`: padded array of shape `(max_nodes, 7)` containing `x, y, vx, vy, type_id, is_cv, is_rw`
  - `x, y`: normalized position coordinates (0-1)
  - `vx, vy`: normalized velocity components
  - `type_id`: 0.0 (agent), 1.0 (goal), 2.0 (obstacle)
  - `is_cv`: 1.0 if constant_velocity obstacle, 0.0 otherwise
  - `is_rw`: 1.0 if random_walk obstacle, 0.0 otherwise
- `edge_index`: padded directed kNN graph edges of shape `(2, max_nodes * knn_k)`
- `global_features`: 10-dimensional vector containing:
  1. `norm_step`: normalized timestep progress (step / max_steps)
  2. `norm_node_count`: normalized node count (node_count / max_nodes)
  3. `norm_min_obs_dist`: normalized distance to nearest obstacle
  4. `norm_goal_dist`: normalized Manhattan distance to goal
  5. `prev_dx`: x-direction displacement from previous step
  6. `prev_dy`: y-direction displacement from previous step
  7. `goal_dx`: normalized x-direction to goal
  8. `goal_dy`: normalized y-direction to goal
  9. `nearest_obs_dx`: normalized x-direction to nearest obstacle
  10. `nearest_obs_dy`: normalized y-direction to nearest obstacle
- `action_mask`: legal action indicator for `stay/up/down/left/right` (shape: `(5,)`)
- `node_count`: actual number of graph nodes before padding (scalar)
- `grid_size`: size of the grid for the current episode (scalar)
- `difficulty`: difficulty level string ("easy", "medium", or "hard")

---

## Notes for the Team

- `MLP-PPO` uses flattened padded observations as the non-structural baseline.
- `GNN-PPO` uses a lightweight pure-PyTorch graph attention encoder so the repo works even without PyG.
- `Predictive` adds a GRU world model with the following design:
  - **Discrete move prediction**: Predicts 5-class moves (stay/up/down/left/right) instead of continuous coordinates
  - **Recursive single-step prediction**: GRU predicts one step at a time, feeding predicted positions back for the next step
  - **Selective prediction**: Only predicts constant_velocity obstacles; random_walk obstacles keep their current position
  - **Soft rollout**: Uses softmax × delta_table to maintain differentiability for gradient flow
  - **Auxiliary loss**: Cross-entropy loss over cv obstacles only, with distance-based weighting
  - **Boundary awareness**: Detects and predicts boundary rebounds at each recursive step
  - Optimizes `PPO loss + auxiliary prediction loss` with configurable `aux_coef` (default 0.2)
- The evaluation pipeline uses the same frozen episodes for every method, which is the key fairness guarantee.
- Training timesteps vary by model complexity:
  - MLP: 30k-60k steps (scales with difficulty)
  - GNN: 40k-80k steps (scales with difficulty)
  - Predictive: 81k-163k steps (uses larger rollout_steps=2048, includes aux warmup)
- Final write-up scaffolds are under:
  - `reports/final_report_outline.md`
  - `reports/presentation_outline.md`

---

## Final Delivery Order

1. Generate or verify `configs/eval_suite.json`.
2. Run `python3 run_experiments.py --mode dry-run` and verify the locked matrix.
3. Execute selected runs after freezing the codebase.
4. Aggregate summaries into report assets.
5. Generate figures and qualitative trajectory exports.
6. Generate draft report/presentation files and revise English text with final numbers.