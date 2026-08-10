# Robustness experiments

The submitted course report identified parameter dependence and the original random train/test split as the two most important analytical limitations. The portfolio refresh turns both into reproducible experiments.

## Run

From the repository root:

```bash
python src/robustness_analysis.py
```

The canonical five-fold, seven-configuration run completed in about 318 seconds on the verification machine.

## Price-model validation

The same baseline and community-enhanced ridge models are evaluated on identical folds under three schemes:

1. **Random five-fold:** shuffled listing-level folds.
2. **Host-grouped five-fold:** all listings owned by a host remain in one fold. Generated artifacts record and validate zero train/test host overlap.
3. **Spatial-block five-fold:** projected Toronto coordinates are clustered into five compact geographic regions; each region is held out once.

The community-enhanced model produces a tiny positive mean raw R² change under all three schemes, but mean adjusted R² declines in all three. Raw R² improves in all five random folds, all five host-grouped folds, and three of five spatial folds. The evidence supports a small association, not a practically important predictive improvement.

Community labels are estimated once from the complete Graph C network before cross-validation. Price is not used to construct that graph. The evaluation is therefore transductive: it asks whether a price-free network representation remains associated with price across held-out rows, hosts, and areas. It does not test how a production system would assign a completely new listing without rebuilding or extending the graph.

Generated artifacts:

- `results/tables/price_model_cv_results.csv`: every fold and model
- `results/tables/price_model_cv_summary.csv`: mean and standard deviation by scheme/model
- `results/tables/price_model_cv_deltas.csv`: paired expanded-minus-baseline changes
- `results/figures/price_model_cv_comparison.png`: score and delta comparison
- `results/figures/price_model_cv_spatial_blocks.png`: geographic holdout design

## Graph C parameter sensitivity

The baseline uses a 500 m radius, five attribute neighbours, and spatial/host/attribute weights of 0.60/0.25/0.15. Six alternatives change one design dimension around that baseline:

- Radius: 300 m and 700 m
- Attribute neighbours: 3 and 10
- Weight profile: spatial-heavy 0.75/0.15/0.10 and attribute-heavy 0.45/0.20/0.35

Each configuration rebuilds Graph C and reruns seeded Louvain. The experiment records graph structure, community count and size, modularity, alignment with official neighbourhoods, and NMI/VI against the baseline partition.

All settings retain a small number of high-modularity full-graph communities, supporting the broad conclusion that the combined network reveals citywide segments rather than only local neighbourhood clusters. However, the alternatives produce 14–21 communities and baseline-partition NMI of 0.69–0.86. Exact labels and boundaries should therefore not be presented as immutable natural categories.

Generated artifacts:

- `results/tables/parameter_sensitivity_results.csv`
- `results/figures/parameter_sensitivity.png`
