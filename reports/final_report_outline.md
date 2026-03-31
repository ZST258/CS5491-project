# CS5491 Final Report Template

## Title
Graph-Based Predictive Agent for Decision Making in Dynamic Environments

## Abstract
This project studies whether structured relational reasoning and short-horizon latent prediction can improve navigation in dynamic environments. We compare a flat `MLP-PPO` baseline, a graph-based `GNN-PPO` policy, and a predictive `GAT+GRU+PPO` agent on a frozen dynamic grid-world benchmark. Replace the metric placeholders with values from `outputs/report_assets/report_numbers.json` after final runs.

## 1. Introduction
Dynamic navigation is difficult because the agent must avoid moving obstacles while still planning toward a distant goal. In standard flat-state reinforcement learning, this often leads to reactive rather than proactive behavior. Our project investigates whether graph structure and latent-space prediction help the agent reason about both spatial relations and short-horizon future dynamics.

The central claim of this report should stay conservative: graph-based encoding is expected to outperform the flat baseline, while the predictive module is evaluated as an additional but lightweight improvement rather than a guaranteed win. If the final numbers are mixed, keep the framing focused on structured modeling and reproducible analysis.

## 2. Method
We represent each state as a graph over the agent, the goal, and moving obstacle nodes. The `MLP-PPO` baseline consumes the padded observation as a flat vector. `GNN-PPO` uses graph attention to produce a relational latent representation, and `Predictive(GAT+GRU+PPO)` rolls the latent state forward before choosing an action.

Include one system diagram here. The diagram should show graph encoding, latent rollout for horizon `H`, and the policy/value heads. If the final writeup stays concise, keep this section method-focused and avoid over-claiming novelty.

## 3. Environment And Metrics
The benchmark is a dynamic grid world with constant-velocity and random-walk obstacles, evaluated across easy, medium, and hard tiers. We use a frozen evaluation suite so that all policies see exactly the same test episodes.

We report success rate, collision rate, average episode length, and path efficiency relative to a time-expanded A* oracle. Path efficiency is defined only for successful episodes and therefore measures path quality rather than aggregate failure severity.

## 4. Experimental Setup
The locked experiment matrix is defined in `configs/experiments.json`. Main runs cover `oracle`, `mlp`, `gnn`, and `predictive`, while ablations compare `gnn` against predictive variants and compare `predictive(H=1)` against `predictive(H=3)`.

To be updated after final runs:
- hardware and runtime budget
- final dependency versions
- command sequence used to reproduce the main table

## 5. Main Results
Insert the main table from `outputs/report_assets/main_results_pivot.csv` and figures from `outputs/figures/`.

Suggested draft sentence:
"On easy, medium, and hard splits respectively, the current predictive success-rate placeholders are `{{main.easy.predictive.success_rate}}`, `{{main.medium.predictive.success_rate}}`, and `{{main.hard.predictive.success_rate}}`. Replace this sentence after final runs with the strongest supported comparison against `mlp` and `gnn`."

Suggested interpretation paragraph:
"If the final results show that `gnn` consistently improves over `mlp`, then the main conclusion should emphasize the value of structured relational encoding. If `predictive` improves only on harder splits, frame the forecasting module as a targeted gain under higher dynamic complexity rather than a universal improvement."

## 6. Ablation And Failure Analysis
Use `outputs/report_assets/ablation_pivot.csv` and `outputs/report_assets/stability_summary.csv` for this section. The main questions are whether forecasting helps beyond graph encoding alone and whether longer horizon prediction is useful in harder settings.

For failure analysis, reference `outputs/report_assets/failure_cases.md` and the qualitative trajectory assets under `outputs/qualitative/` or `outputs/figures/qualitative_*.svg`. Discuss failures using the fixed labels:
- `late_avoidance`
- `collision_under_crossing`
- `timeout_due_to_detour`

Suggested draft sentence:
"Even when predictive control does not dominate every baseline, the qualitative cases help show whether the model is failing because of delayed reaction, crossing conflicts, or overly conservative detours."

## 7. Conclusion
Restate only the strongest claim directly supported by the final metrics. If predictive gains are weak, frame the project as a careful empirical study of flat, relational, and predictive policies in a reproducible dynamic navigation benchmark. If predictive gains are stronger on hard instances, emphasize that short-horizon latent forecasting is most useful when obstacle dynamics are dense and uncertain.

## Appendix
- Main asset manifest: `outputs/report_assets/report_numbers.json`
- Tables: `outputs/report_assets/report_tables.md`
- Narrative notes: `outputs/report_assets/results_narrative.md`
- Figure manifest: `outputs/figures/figure_manifest.json`
- Experiment matrix: `configs/experiments.json`
