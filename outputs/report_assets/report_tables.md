# Final Report Assets

## Main Results

| Run | Difficulty | Success | Collision | Path Efficiency | Avg Episode Length |
| --- | --- | ---: | ---: | ---: | ---: |
| gnn_easy_seed0_main | easy | 0.140 | 0.350 | 1.000 | 18.850 |
| mlp_easy_seed0_main | easy | 0.080 | 0.390 | 1.000 | 20.050 |
| oracle_all_seed0 | easy | 1.000 | 0.000 | 1.000 | 5.580 |
| predictive_easy_seed0_h3_main | easy | 0.140 | 0.480 | 0.883 | 18.510 |
| gnn_medium_seed0_main | medium | 0.090 | 0.580 | 1.000 | 22.580 |
| mlp_medium_seed0_main | medium | 0.120 | 0.530 | 1.000 | 23.230 |
| oracle_all_seed0 | medium | 1.000 | 0.000 | 1.000 | 6.460 |
| predictive_medium_seed0_h3_main | medium | 0.030 | 0.520 | 1.000 | 26.160 |
| gnn_hard_seed0_main | hard | 0.020 | 0.730 | 1.000 | 25.540 |
| mlp_hard_seed0_main | hard | 0.020 | 0.730 | 1.000 | 25.540 |
| oracle_all_seed0 | hard | 1.000 | 0.000 | 1.000 | 7.860 |
| predictive_hard_seed0_h3_main | hard | 0.070 | 0.620 | 1.013 | 27.240 |

## Ablation Focus

- Compare `gnn` against `predictive_h1` and `predictive_h3` for medium and hard.
- Use `stability_summary.csv` to summarize predictive-hard seed variance.