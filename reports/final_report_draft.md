# CS5491 Final Report Draft

## Title
Graph-Based Predictive Agent for Decision Making in Dynamic Environments

## 1. Introduction / Problem
Dynamic obstacle navigation is challenging because the agent must react to moving hazards while still making progress toward a goal. In flat observation settings, standard reinforcement learning policies often become reactive rather than proactive. This project studies whether structured relational encoding and short-horizon latent prediction can improve decision making in dynamic environments.

Our working hypothesis is that graph-based state representations should outperform a plain MLP baseline, and that a lightweight predictive module may offer additional gains in harder settings. This draft is designed so the final numbers can be updated directly from the generated report assets.

## 2. Method
We model each environment state as a graph over the agent, goal, and moving obstacles. The `MLP-PPO` baseline consumes a flattened observation, `GNN-PPO` applies graph attention to obtain a relational latent state, and `Predictive(GAT+GRU+PPO)` rolls the latent forward for a short forecasting horizon before choosing an action.

The predictive branch is intentionally lightweight: it uses a GRU in latent space rather than a heavy world model. This keeps the implementation compatible with the current course-project codebase while still testing the main idea from the proposal.

## 3. Environment And Metrics
The environment is a dynamic grid world with constant-velocity and random-walk obstacles. We evaluate on fixed easy, medium, and hard tiers using a frozen evaluation suite so that all methods face the same test episodes.

We report success rate, collision rate, average episode length, and path efficiency relative to a time-expanded A* oracle. Path efficiency is computed only on successful episodes, which makes it a path-quality metric rather than a failure-penalized aggregate.

## 4. Experimental Setup
The formal experiment matrix is defined in `configs/experiments.json`. The main comparison includes `oracle`, `mlp`, `gnn`, and `predictive`, followed by two ablations: `gnn` versus `predictive`, and `predictive(H=1)` versus `predictive(H=3)`.

To be updated after final runs: insert actual hardware information, runtime budget, and the exact command sequence used to execute the locked experiment plan.

## 5. Main Results
On the easy split, the current placeholders are: MLP success `0.080`, GNN success `0.140`, and Predictive success `0.140`.
On the medium split, the current placeholders are: MLP success `0.120`, GNN success `0.090`, and Predictive success `0.030`.
On the hard split, the current placeholders are: MLP success `0.020`, GNN success `0.020`, and Predictive success `0.070`.

Insert the pivot table from `outputs\report_assets\main_results_pivot.csv` and the figures from `outputs\figures\success_rate.svg`, `outputs\figures\collision_rate.svg`, and `outputs\figures\path_efficiency.svg`. Replace these placeholder sentences with the strongest supported claim after final runs.

## 6. Ablation And Failure Analysis
The ablation section should compare whether prediction adds value beyond relational encoding alone, especially on medium and hard difficulties. Use `ablation_pivot.csv` and `stability_summary.csv` to discuss whether longer-horizon forecasting is useful and whether the predictive model is stable across seeds.

Use the qualitative exports to discuss representative failure cases. If predictive gains are small, frame the analysis around what structured reasoning helps with, where forecasting still breaks down, and why the hardest collisions remain difficult.

### Failure Case Notes
# Failure Cases

No qualitative cases exported yet. Run `eval.py --export-cases` or `export_rollouts.py` after producing evaluation outputs.

## 7. Conclusion
This project provides a reproducible comparison among flat, relational, and predictive policies in a dynamic navigation setting. The final conclusion should restate only the strongest result actually supported by the main table and qualitative analysis.

If predictive gains remain limited, the contribution should be framed as a careful study of structured policy design and short-horizon forecasting in a course-scale dynamic environment, together with a reproducible experimental pipeline.

## Appendix
- Main table asset: `outputs\report_assets\report_tables.md`
- Narrative notes: `outputs\report_assets\results_narrative.md`
- Figure manifest: `outputs\figures\figure_manifest.json`
- Experiment matrix: `configs/experiments.json`