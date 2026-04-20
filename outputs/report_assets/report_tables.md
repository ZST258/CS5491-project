# Final Report Assets

## Main Results

| Run | Difficulty | Success | Collision | Time Efficiency | Move Efficiency | Avg Episode Length |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gnn_easy_seed0_main | easy | 0.630 | 0.290 | 0.985 | 0.979 | 7.510 |
| mlp_easy_seed0_main | easy | 0.470 | 0.240 | 0.865 | 0.861 | 14.120 |
| oracle_all_seed0 | easy | 1.000 | 0.000 | 1.000 | 0.977 | 5.580 |
| predictive_easy_seed0_h3_main | easy | 0.810 | 0.190 | 0.971 | 0.964 | 5.070 |
| gnn_medium_seed0_main | medium | 0.630 | 0.230 | 0.962 | 0.961 | 13.330 |
| mlp_medium_seed0_main | medium | 0.240 | 0.470 | 0.881 | 0.885 | 20.820 |
| oracle_all_seed0 | medium | 1.000 | 0.000 | 1.000 | 0.992 | 6.460 |
| predictive_medium_seed0_h3_main | medium | 0.830 | 0.170 | 1.008 | 1.005 | 5.630 |
| gnn_hard_seed0_main | hard | 0.530 | 0.470 | 0.925 | 0.921 | 8.900 |
| mlp_hard_seed0_main | hard | 0.200 | 0.670 | 0.899 | 0.899 | 22.570 |
| oracle_all_seed0 | hard | 1.000 | 0.000 | 1.000 | 0.987 | 7.860 |
| predictive_hard_seed0_h3_main | hard | 0.710 | 0.290 | 0.996 | 0.992 | 6.270 |

## Ablation Focus

- Compare `gnn` against `predictive_h1` and `predictive_h3` for medium and hard.
- Use `stability_summary.csv` to summarize predictive-hard seed variance.