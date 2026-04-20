# CS5491 Presentation Draft

## Slide 1. Title
- Graph-Based Predictive Agent for Decision Making in Dynamic Environments
- Team members
- Goal: improve proactive navigation in dynamic obstacle fields

## Slide 2. Motivation
- Dynamic environments require both relational reasoning and short-horizon anticipation.
- Flat vector policies are often reactive and brittle under dense motion.
- Our project asks whether graph structure plus latent prediction helps.

## Slide 3. Task Setup
- Grid world with moving obstacles, fixed goal, and three difficulty tiers.
- Frozen evaluation suite ensures fair comparison across all methods.
- Metrics: success, collision, path efficiency, average episode length.

## Slide 4. Methods
- `MLP-PPO`: flat baseline.
- `GNN-PPO`: graph attention for relational encoding.
- `Predictive(GAT+GRU+PPO)`: latent rollout before action selection.

## Slide 5. Main Results
- Use `outputs\figures\success_rate.svg` and `outputs\report_assets\main_results_pivot.csv`.
- Easy placeholder: predictive success `0.810`.
- Hard placeholder: predictive success `0.710`.
- Replace with final claim after full runs.

## Slide 6. Ablation
- Use `outputs\report_assets\ablation_pivot.csv`.
- Compare `gnn`, `predictive_h1`, and `predictive_h3` on medium and hard.
- State whether forecasting horizon helps beyond relational encoding.

## Slide 7. Failure Cases
- Use `outputs\report_assets\failure_cases.md` and copied qualitative SVGs from `outputs\figures`.
- Show 2-3 representative collisions, detours, or timeout cases.
- Explain which failure modes remain unresolved.

## Slide 8. Reproducibility
- Locked experiment matrix in `configs/experiments.json`.
- Unified train/eval entrypoints and frozen evaluation suite.
- Aggregated report numbers and figures are generated from outputs, not hand-copied.

## Slide 9. Conclusion
- Summarize the strongest supported result.
- If gains are mixed, emphasize what the graph structure helped and what prediction did not solve.
- Mention future improvements: stronger world models or richer dynamic scenarios.

## Slide 10. Backup
# Results Narrative Draft

This section is generated from `report_numbers.json` and is safe to revise once final long runs finish.

For the **easy** setting, the available models are gnn, mlp, oracle, predictive. Replace this sentence with the strongest supported takeaway after final runs.
For the **medium** setting, the available models are gnn, mlp, oracle, predictive. Replace this sentence with the strongest supported takeaway after final runs.
For the **hard** setting, the available models are gnn, mlp, oracle, predictive. Replace this sentence with the strongest supported takeaway after final runs.

## Ablation Draft

Use the medium and hard comparisons among `gnn`, `predictive_h1`, and `predictive_h3` to explain whether forecasting horizon helps beyond relational encoding alone.

## Stability Draft

Use the predictive-hard multi-seed summary to describe variance conservatively. If gains are small, focus on consistency and qualitative behavior instead of claiming broad superiority.