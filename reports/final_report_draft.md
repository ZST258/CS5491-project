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
On the easy split, the current placeholders are: MLP success `0.370`, GNN success `0.630`, and Predictive success `0.810`.
On the medium split, the current placeholders are: MLP success `0.190`, GNN success `0.630`, and Predictive success `0.830`.
On the hard split, the current placeholders are: MLP success `0.050`, GNN success `0.530`, and Predictive success `0.710`.

Insert the pivot table from `outputs\report_assets\main_results_pivot.csv` and the figures from `outputs\figures\success_rate.svg`, `outputs\figures\collision_rate.svg`, and `outputs\figures\time_efficiency.svg`. Replace these placeholder sentences with the strongest supported claim after final runs.

## 6. Ablation And Failure Analysis
The ablation section should compare whether prediction adds value beyond relational encoding alone, especially on medium and hard difficulties. Use `ablation_pivot.csv` and `stability_summary.csv` to discuss whether longer-horizon forecasting is useful and whether the predictive model is stable across seeds.

Use the qualitative exports to discuss representative failure cases. If predictive gains are small, frame the analysis around what structured reasoning helps with, where forecasting still breaks down, and why the hardest collisions remain difficult.

### Failure Case Notes
# Failure Cases

## gnn_easy_seed0_main_easy_success_90

- Difficulty: `easy`
- Episode index: `90`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_easy_seed0_main\easy\gnn_easy_seed0_main_easy_success_90_trajectory.svg`

## gnn_easy_seed0_main_easy_failure_62

- Difficulty: `easy`
- Episode index: `62`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_easy_seed0_main\easy\gnn_easy_seed0_main_easy_failure_62_trajectory.svg`

## gnn_easy_seed0_main_easy_detour_29

- Difficulty: `easy`
- Episode index: `29`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_easy_seed0_main\easy\gnn_easy_seed0_main_easy_detour_29_trajectory.svg`

## gnn_easy_seed0_main_easy_closest_collision_8

- Difficulty: `easy`
- Episode index: `8`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_easy_seed0_main\easy\gnn_easy_seed0_main_easy_closest_collision_8_trajectory.svg`

## gnn_hard_seed0_main_hard_success_90

- Difficulty: `hard`
- Episode index: `90`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_hard_seed0_main\hard\gnn_hard_seed0_main_hard_success_90_trajectory.svg`

## gnn_hard_seed0_main_hard_failure_34

- Difficulty: `hard`
- Episode index: `34`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_hard_seed0_main\hard\gnn_hard_seed0_main_hard_failure_34_trajectory.svg`

## gnn_hard_seed0_main_hard_detour_73

- Difficulty: `hard`
- Episode index: `73`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_hard_seed0_main\hard\gnn_hard_seed0_main_hard_detour_73_trajectory.svg`

## gnn_hard_seed0_main_hard_closest_collision_4

- Difficulty: `hard`
- Episode index: `4`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_hard_seed0_main\hard\gnn_hard_seed0_main_hard_closest_collision_4_trajectory.svg`

## gnn_medium_seed0_main_medium_success_50

- Difficulty: `medium`
- Episode index: `50`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_medium_seed0_main\medium\gnn_medium_seed0_main_medium_success_50_trajectory.svg`

## gnn_medium_seed0_main_medium_failure_39

- Difficulty: `medium`
- Episode index: `39`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_medium_seed0_main\medium\gnn_medium_seed0_main_medium_failure_39_trajectory.svg`

## gnn_medium_seed0_main_medium_detour_41

- Difficulty: `medium`
- Episode index: `41`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\gnn_medium_seed0_main\medium\gnn_medium_seed0_main_medium_detour_41_trajectory.svg`

## gnn_medium_seed0_main_medium_closest_collision_34

- Difficulty: `medium`
- Episode index: `34`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\gnn_medium_seed0_main\medium\gnn_medium_seed0_main_medium_closest_collision_34_trajectory.svg`

## predictive_easy_seed0_h3_main_easy_success_90

- Difficulty: `easy`
- Episode index: `90`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_easy_seed0_h3_main\easy\predictive_easy_seed0_h3_main_easy_success_90_trajectory.svg`

## predictive_easy_seed0_h3_main_easy_failure_48

- Difficulty: `easy`
- Episode index: `48`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_easy_seed0_h3_main\easy\predictive_easy_seed0_h3_main_easy_failure_48_trajectory.svg`

## predictive_easy_seed0_h3_main_easy_detour_86

- Difficulty: `easy`
- Episode index: `86`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_easy_seed0_h3_main\easy\predictive_easy_seed0_h3_main_easy_detour_86_trajectory.svg`

## predictive_easy_seed0_h3_main_easy_closest_collision_8

- Difficulty: `easy`
- Episode index: `8`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_easy_seed0_h3_main\easy\predictive_easy_seed0_h3_main_easy_closest_collision_8_trajectory.svg`

## predictive_easy_seed0_h3_main_easy_closest_collision_54

- Difficulty: `easy`
- Episode index: `54`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_easy_seed0_h3_main\easy\predictive_easy_seed0_h3_main_easy_closest_collision_54_trajectory.svg`

## predictive_hard_seed0_h1_ablation_hard_success_83

- Difficulty: `hard`
- Episode index: `83`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h1_ablation\hard\predictive_hard_seed0_h1_ablation_hard_success_83_trajectory.svg`

## predictive_hard_seed0_h1_ablation_hard_failure_90

- Difficulty: `hard`
- Episode index: `90`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h1_ablation\hard\predictive_hard_seed0_h1_ablation_hard_failure_90_trajectory.svg`

## predictive_hard_seed0_h1_ablation_hard_closest_collision_4

- Difficulty: `hard`
- Episode index: `4`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h1_ablation\hard\predictive_hard_seed0_h1_ablation_hard_closest_collision_4_trajectory.svg`

## predictive_hard_seed0_h3_main_hard_success_25

- Difficulty: `hard`
- Episode index: `25`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h3_main\hard\predictive_hard_seed0_h3_main_hard_success_25_trajectory.svg`

## predictive_hard_seed0_h3_main_hard_failure_81

- Difficulty: `hard`
- Episode index: `81`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h3_main\hard\predictive_hard_seed0_h3_main_hard_failure_81_trajectory.svg`

## predictive_hard_seed0_h3_main_hard_detour_57

- Difficulty: `hard`
- Episode index: `57`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h3_main\hard\predictive_hard_seed0_h3_main_hard_detour_57_trajectory.svg`

## predictive_hard_seed0_h3_main_hard_closest_collision_4

- Difficulty: `hard`
- Episode index: `4`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed0_h3_main\hard\predictive_hard_seed0_h3_main_hard_closest_collision_4_trajectory.svg`

## predictive_hard_seed1_h3_stability_hard_success_83

- Difficulty: `hard`
- Episode index: `83`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed1_h3_stability\hard\predictive_hard_seed1_h3_stability_hard_success_83_trajectory.svg`

## predictive_hard_seed1_h3_stability_hard_failure_81

- Difficulty: `hard`
- Episode index: `81`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed1_h3_stability\hard\predictive_hard_seed1_h3_stability_hard_failure_81_trajectory.svg`

## predictive_hard_seed1_h3_stability_hard_closest_collision_4

- Difficulty: `hard`
- Episode index: `4`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed1_h3_stability\hard\predictive_hard_seed1_h3_stability_hard_closest_collision_4_trajectory.svg`

## predictive_hard_seed2_h3_stability_hard_success_25

- Difficulty: `hard`
- Episode index: `25`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed2_h3_stability\hard\predictive_hard_seed2_h3_stability_hard_success_25_trajectory.svg`

## predictive_hard_seed2_h3_stability_hard_failure_81

- Difficulty: `hard`
- Episode index: `81`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed2_h3_stability\hard\predictive_hard_seed2_h3_stability_hard_failure_81_trajectory.svg`

## predictive_hard_seed2_h3_stability_hard_detour_16

- Difficulty: `hard`
- Episode index: `16`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_hard_seed2_h3_stability\hard\predictive_hard_seed2_h3_stability_hard_detour_16_trajectory.svg`

## predictive_hard_seed2_h3_stability_hard_closest_collision_4

- Difficulty: `hard`
- Episode index: `4`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_hard_seed2_h3_stability\hard\predictive_hard_seed2_h3_stability_hard_closest_collision_4_trajectory.svg`

## predictive_medium_seed0_h1_ablation_medium_success_81

- Difficulty: `medium`
- Episode index: `81`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h1_ablation\medium\predictive_medium_seed0_h1_ablation_medium_success_81_trajectory.svg`

## predictive_medium_seed0_h1_ablation_medium_failure_63

- Difficulty: `medium`
- Episode index: `63`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h1_ablation\medium\predictive_medium_seed0_h1_ablation_medium_failure_63_trajectory.svg`

## predictive_medium_seed0_h1_ablation_medium_detour_84

- Difficulty: `medium`
- Episode index: `84`
- Case type: `detour`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h1_ablation\medium\predictive_medium_seed0_h1_ablation_medium_detour_84_trajectory.svg`

## predictive_medium_seed0_h1_ablation_medium_closest_collision_34

- Difficulty: `medium`
- Episode index: `34`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h1_ablation\medium\predictive_medium_seed0_h1_ablation_medium_closest_collision_34_trajectory.svg`

## predictive_medium_seed0_h3_main_medium_success_81

- Difficulty: `medium`
- Episode index: `81`
- Case type: `success`
- Outcome: `success`
- Failure type: `success`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h3_main\medium\predictive_medium_seed0_h3_main_medium_success_81_trajectory.svg`

## predictive_medium_seed0_h3_main_medium_failure_62

- Difficulty: `medium`
- Episode index: `62`
- Case type: `failure`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h3_main\medium\predictive_medium_seed0_h3_main_medium_failure_62_trajectory.svg`

## predictive_medium_seed0_h3_main_medium_closest_collision_8

- Difficulty: `medium`
- Episode index: `8`
- Case type: `closest_collision`
- Outcome: `collision`
- Failure type: `late_avoidance`
- Trajectory: `outputs\qualitative\predictive_medium_seed0_h3_main\medium\predictive_medium_seed0_h3_main_medium_closest_collision_8_trajectory.svg`


## 7. Conclusion
This project provides a reproducible comparison among flat, relational, and predictive policies in a dynamic navigation setting. The final conclusion should restate only the strongest result actually supported by the main table and qualitative analysis.

If predictive gains remain limited, the contribution should be framed as a careful study of structured policy design and short-horizon forecasting in a course-scale dynamic environment, together with a reproducible experimental pipeline.

## Appendix
- Main table asset: `outputs\report_assets\report_tables.md`
- Narrative notes: `outputs\report_assets\results_narrative.md`
- Figure manifest: `outputs\figures\figure_manifest.json`
- Experiment matrix: `configs/experiments.json`