# CS5491 Presentation Template

## Slide 1. Title
- Graph-Based Predictive Agent for Decision Making in Dynamic Environments
- Team members
- One-sentence claim placeholder: structured and predictive policies for dynamic navigation

## Slide 2. Motivation
- Dynamic obstacles make short-sighted RL policies brittle.
- Flat vector observations hide relations among agent, goal, and obstacles.
- Our question: can graph structure plus short-horizon prediction improve behavior?

## Slide 3. Task Setup
- Grid world with moving obstacles
- Easy / medium / hard difficulty tiers
- Frozen evaluation suite for fair comparison
- Oracle A* used only for reference metrics

## Slide 4. Methods
- `MLP-PPO`: flat baseline
- `GNN-PPO`: graph attention over relational state
- `Predictive(GAT+GRU+PPO)`: latent rollout before policy choice
- Insert architecture figure here

## Slide 5. Main Results
- Use `outputs/figures/success_rate.*`
- Use `outputs/report_assets/main_results_pivot.csv`
- One-sentence takeaway placeholder:
  "Graph structure improves over the flat baseline, while prediction provides [targeted / limited / clear] gains on harder cases."

## Slide 6. Ablation
- Use `outputs/report_assets/ablation_pivot.csv`
- Compare `gnn`, `predictive_h1`, `predictive_h3`
- State whether horizon helps and where it helps

## Slide 7. Failure Cases
- Use `outputs/report_assets/failure_cases.md`
- Show 2-3 qualitative trajectories from `outputs/figures/qualitative_*.svg`
- Explain failure types:
  - late avoidance
  - crossing collision
  - detour timeout

## Slide 8. Reproducibility
- Locked experiment matrix
- Unified train/eval/export scripts
- Generated report numbers and figure manifest
- Same frozen evaluation suite for all methods

## Slide 9. Conclusion
- What clearly worked
- What remained hard
- What the final contribution is, even if predictive gains are mixed

## Slide 10. Backup
- Stability summary from `outputs/report_assets/stability_summary.csv`
- Extra qualitative examples
- Additional implementation details if asked
