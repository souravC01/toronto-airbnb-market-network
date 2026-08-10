# Script overview

## Canonical workflow

| Script | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Runs the complete experiment, regenerates portfolio figures, and validates the repository. |
| `scripts/validate_repository.py` | Checks result schemas, Louvain/Leiden coverage, required figures, and portfolio assets. |

## Core analysis

| Script | Purpose |
|---|---|
| `src/clean_dataset.py` | Recreates the cleaned Toronto dataset from a raw Inside Airbnb snapshot. |
| `src/airbnb_final_experiments.py` | Builds Graph A/B/C, runs Louvain and optional Leiden, evaluates alignment, creates community summaries, and compares price models. |
| `src/robustness_analysis.py` | Runs seven Graph C parameter settings plus paired random, host-grouped, and spatial-block five-fold price validation. |

## Visualizations

| Script | Purpose |
|---|---|
| `src/make_midterm_eda_plots.py` | Creates exploratory figures used by the research workflow. |
| `src/make_alignment_plots.py` | Creates NMI and VI comparison plots from the canonical alignment table. |
| `src/make_community_clarity_figures.py` | Creates interpretable Graph C community profile, price, and concentration figures. |
| `src/make_labelled_community_map.py` | Builds the categorical, labelled Graph C Louvain map from generated assignments. |
| `src/generate_graph_c_leiden_figures.py` | Optionally rebuilds only Graph C Leiden artifacts. |

## Interactive portfolio

The `portfolio/` application presents the project as a responsive case study. It includes interactive graph-layer, community-algorithm, validation-scheme, and sensitivity controls and uses selected canonical figures and result tables. Its local development and production commands are documented in `portfolio/README.md`.
