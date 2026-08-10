#!/usr/bin/env python3
"""Robustness experiments for the Toronto Airbnb network analysis.

This module adds two checks that were identified as future work in the
submitted course report:

1. Five-fold price-model validation under random, host-grouped, and spatial
   block splits.
2. One-at-a-time sensitivity tests around the baseline Graph C parameters.

Community membership is estimated once from the full price-free graph before
cross-validation. The price experiment is therefore transductive: it tests
whether the learned representation remains useful across held-out rows,
hosts, and areas, but it is not a fully inductive deployment simulation.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import airbnb_final_experiments as core


BASELINE_MODEL = "Baseline: listing + official neighbourhood"
EXPANDED_MODEL = "Expanded: baseline + network community"
COMMUNITY_COLUMN = "graph_c_louvain_community"

PRICE_NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "calculated_host_listings_count",
]

PRICE_CATEGORICAL_FEATURES = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "instant_bookable",
    "host_is_superhost",
]


@dataclass(frozen=True)
class SensitivityConfig:
    """One Graph C configuration in the one-at-a-time sensitivity design."""

    name: str
    varied_parameter: str
    radius_meters: int = 500
    attribute_k: int = 5
    alpha_spatial: float = 0.60
    alpha_host: float = 0.25
    alpha_attribute: float = 0.15


SENSITIVITY_CONFIGS = [
    SensitivityConfig("Baseline", "baseline"),
    SensitivityConfig("Radius 300 m", "spatial_radius", radius_meters=300),
    SensitivityConfig("Radius 700 m", "spatial_radius", radius_meters=700),
    SensitivityConfig("Attribute k = 3", "attribute_neighbours", attribute_k=3),
    SensitivityConfig("Attribute k = 10", "attribute_neighbours", attribute_k=10),
    SensitivityConfig(
        "Spatial-heavy weights",
        "edge_weight_profile",
        alpha_spatial=0.75,
        alpha_host=0.15,
        alpha_attribute=0.10,
    ),
    SensitivityConfig(
        "Attribute-heavy weights",
        "edge_weight_profile",
        alpha_spatial=0.45,
        alpha_host=0.20,
        alpha_attribute=0.35,
    ),
]


def project_coordinates_meters(frame: pd.DataFrame) -> np.ndarray:
    """Approximate Toronto latitude/longitude as local planar coordinates."""
    latitude = frame["latitude"].to_numpy(dtype=float)
    longitude = frame["longitude"].to_numpy(dtype=float)
    mean_latitude_radians = np.radians(latitude.mean())
    meters_per_degree = 111_320.0
    x = (longitude - longitude.mean()) * meters_per_degree * np.cos(mean_latitude_radians)
    y = (latitude - latitude.mean()) * meters_per_degree
    return np.column_stack([x, y])


def make_spatial_blocks(frame: pd.DataFrame, n_blocks: int = 5) -> np.ndarray:
    """Create deterministic, geographically compact validation blocks."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    if len(frame) < n_blocks:
        raise ValueError("n_blocks cannot exceed the number of rows")

    coordinates = project_coordinates_meters(frame)
    return KMeans(
        n_clusters=n_blocks,
        random_state=core.RANDOM_STATE,
        n_init=20,
    ).fit_predict(coordinates)


def make_validation_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Return comparable random, host-grouped, and spatial-block splits."""
    if frame["host_id"].nunique() < n_splits:
        raise ValueError("Not enough unique hosts for grouped cross-validation")

    random_cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=core.RANDOM_STATE,
    )
    random_splits = list(random_cv.split(frame))

    host_cv = GroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=core.RANDOM_STATE,
    )
    host_splits = list(host_cv.split(frame, groups=frame["host_id"]))

    spatial_blocks = make_spatial_blocks(frame, n_blocks=n_splits)
    spatial_splits = []
    all_indices = np.arange(len(frame))
    for block in sorted(np.unique(spatial_blocks)):
        test_indices = all_indices[spatial_blocks == block]
        train_indices = all_indices[spatial_blocks != block]
        spatial_splits.append((train_indices, test_indices))

    return {
        "random_5fold": random_splits,
        "host_grouped_5fold": host_splits,
        "spatial_block_5fold": spatial_splits,
    }, spatial_blocks


def prepare_price_frame(
    df_with_community: pd.DataFrame,
    community_col: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Prepare the modelling frame while retaining split identifiers."""
    numeric_features = [
        column for column in PRICE_NUMERIC_FEATURES if column in df_with_community.columns
    ]
    categorical_features = [
        column
        for column in PRICE_CATEGORICAL_FEATURES
        if column in df_with_community.columns
    ]
    required = [
        "log_price_w",
        "host_id",
        "latitude",
        "longitude",
        community_col,
    ]
    columns = required + numeric_features + categorical_features
    work = df_with_community[columns].copy()
    work = work.dropna(subset=["log_price_w", "host_id", community_col]).reset_index(drop=True)

    for column in numeric_features:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work[column] = work[column].fillna(work[column].median())

    for column in categorical_features + [community_col]:
        work[column] = work[column].fillna("Unknown").astype(str)

    return work, numeric_features, categorical_features


def make_price_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """Construct the same ridge-regression specification used in the report."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", core.make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def evaluate_price_cross_validation(
    df_with_community: pd.DataFrame,
    community_col: str,
    out_dir: Path,
    fig_dir: Path,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paired model comparisons under three validation schemes."""
    work, numeric_features, categorical_features = prepare_price_frame(
        df_with_community,
        community_col,
    )
    schemes, spatial_blocks = make_validation_splits(work, n_splits=n_splits)
    y = work["log_price_w"]

    model_specs = [
        (BASELINE_MODEL, categorical_features),
        (EXPANDED_MODEL, categorical_features + [community_col]),
    ]

    rows = []
    for scheme, splits in schemes.items():
        print(f"\nPrice validation: {scheme}")
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            train_hosts = set(work.iloc[train_indices]["host_id"])
            test_hosts = set(work.iloc[test_indices]["host_id"])
            host_overlap_count = len(train_hosts & test_hosts)

            for model_name, model_categorical_features in model_specs:
                feature_columns = numeric_features + model_categorical_features
                X = work[feature_columns]
                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]

                model = make_price_pipeline(numeric_features, model_categorical_features)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                r2 = r2_score(y_test, prediction)
                mae_log = mean_absolute_error(y_test, prediction)
                rmse_log = float(np.sqrt(mean_squared_error(y_test, prediction)))
                y_test_dollars = np.exp(y_test)
                prediction_dollars = np.exp(prediction)
                mae_dollars = mean_absolute_error(y_test_dollars, prediction_dollars)
                rmse_dollars = float(
                    np.sqrt(mean_squared_error(y_test_dollars, prediction_dollars))
                )
                transformed_train = model.named_steps["preprocessor"].transform(X_train)
                encoded_features = transformed_train.shape[1]

                rows.append(
                    {
                        "validation_scheme": scheme,
                        "fold": fold,
                        "model": model_name,
                        "test_r2": r2,
                        "test_adjusted_r2_approx": core.adjusted_r2_score(
                            r2,
                            len(y_test),
                            encoded_features,
                        ),
                        "mae_log_price": mae_log,
                        "rmse_log_price": rmse_log,
                        "mae_dollars_approx": mae_dollars,
                        "rmse_dollars_approx": rmse_dollars,
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        "train_hosts": len(train_hosts),
                        "test_hosts": len(test_hosts),
                        "host_overlap_count": host_overlap_count,
                        "encoded_features": encoded_features,
                        "test_spatial_block": (
                            int(np.unique(spatial_blocks[test_indices])[0])
                            if scheme == "spatial_block_5fold"
                            else np.nan
                        ),
                    }
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["validation_scheme", "model"], sort=False)
        .agg(
            folds=("fold", "nunique"),
            test_rows_mean=("test_rows", "mean"),
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", "std"),
            adjusted_r2_mean=("test_adjusted_r2_approx", "mean"),
            adjusted_r2_std=("test_adjusted_r2_approx", "std"),
            mae_log_mean=("mae_log_price", "mean"),
            mae_log_std=("mae_log_price", "std"),
            rmse_log_mean=("rmse_log_price", "mean"),
            rmse_log_std=("rmse_log_price", "std"),
            mae_dollars_mean=("mae_dollars_approx", "mean"),
            mae_dollars_std=("mae_dollars_approx", "std"),
            rmse_dollars_mean=("rmse_dollars_approx", "mean"),
            rmse_dollars_std=("rmse_dollars_approx", "std"),
            max_host_overlap=("host_overlap_count", "max"),
        )
        .reset_index()
    )

    delta_rows = []
    metrics = {
        "test_r2": "r2_delta",
        "test_adjusted_r2_approx": "adjusted_r2_delta",
        "mae_log_price": "mae_log_delta",
        "rmse_log_price": "rmse_log_delta",
        "mae_dollars_approx": "mae_dollars_delta",
        "rmse_dollars_approx": "rmse_dollars_delta",
    }
    for scheme, group in results.groupby("validation_scheme", sort=False):
        row = {"validation_scheme": scheme, "folds": int(group["fold"].nunique())}
        for metric, prefix in metrics.items():
            wide = group.pivot(index="fold", columns="model", values=metric)
            differences = wide[EXPANDED_MODEL] - wide[BASELINE_MODEL]
            row[f"{prefix}_mean"] = differences.mean()
            row[f"{prefix}_std"] = differences.std()
        r2_wide = group.pivot(index="fold", columns="model", values="test_r2")
        row["expanded_r2_wins"] = int(
            (r2_wide[EXPANDED_MODEL] > r2_wide[BASELINE_MODEL]).sum()
        )
        row["expanded_adjusted_r2_wins"] = int(
            (
                group.pivot(
                    index="fold",
                    columns="model",
                    values="test_adjusted_r2_approx",
                )[EXPANDED_MODEL]
                > group.pivot(
                    index="fold",
                    columns="model",
                    values="test_adjusted_r2_approx",
                )[BASELINE_MODEL]
            ).sum()
        )
        delta_rows.append(row)

    deltas = pd.DataFrame(delta_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "price_model_cv_results.csv", index=False)
    summary.to_csv(out_dir / "price_model_cv_summary.csv", index=False)
    deltas.to_csv(out_dir / "price_model_cv_deltas.csv", index=False)
    plot_price_cv(summary, deltas, fig_dir / "price_model_cv_comparison.png")
    plot_spatial_blocks(
        work,
        spatial_blocks,
        fig_dir / "price_model_cv_spatial_blocks.png",
    )
    return results, summary, deltas


def plot_price_cv(summary: pd.DataFrame, deltas: pd.DataFrame, out_path: Path) -> None:
    """Plot mean model scores and paired fold deltas."""
    schemes = ["random_5fold", "host_grouped_5fold", "spatial_block_5fold"]
    labels = ["Random", "Host-grouped", "Spatial blocks"]
    x = np.arange(len(schemes))
    width = 0.34

    baseline = summary.set_index(["validation_scheme", "model"]).loc[
        [(scheme, BASELINE_MODEL) for scheme in schemes]
    ]
    expanded = summary.set_index(["validation_scheme", "model"]).loc[
        [(scheme, EXPANDED_MODEL) for scheme in schemes]
    ]
    delta_frame = deltas.set_index("validation_scheme").loc[schemes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(
        x - width / 2,
        baseline["r2_mean"],
        width,
        yerr=baseline["r2_std"],
        label="Baseline",
        capsize=3,
    )
    axes[0, 0].bar(
        x + width / 2,
        expanded["r2_mean"],
        width,
        yerr=expanded["r2_std"],
        label="Baseline + community",
        capsize=3,
    )
    axes[0, 0].set_title("Mean test $R^2$ across five folds")
    axes[0, 0].set_ylabel("$R^2$ (higher is better)")
    axes[0, 0].legend()

    axes[0, 1].bar(
        x - width / 2,
        baseline["mae_dollars_mean"],
        width,
        yerr=baseline["mae_dollars_std"],
        label="Baseline",
        capsize=3,
    )
    axes[0, 1].bar(
        x + width / 2,
        expanded["mae_dollars_mean"],
        width,
        yerr=expanded["mae_dollars_std"],
        label="Baseline + community",
        capsize=3,
    )
    axes[0, 1].set_title("Approximate dollar MAE")
    axes[0, 1].set_ylabel("MAE ($; lower is better)")

    axes[1, 0].bar(x, delta_frame["r2_delta_mean"], color="#377eb8")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Paired $R^2$ change from community feature")
    axes[1, 0].set_ylabel("Expanded minus baseline")

    axes[1, 1].bar(x, delta_frame["mae_dollars_delta_mean"], color="#e41a1c")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Paired dollar-MAE change")
    axes[1, 1].set_ylabel("Expanded minus baseline ($)")

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Price-model robustness: three validation schemes", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spatial_blocks(
    frame: pd.DataFrame,
    spatial_blocks: np.ndarray,
    out_path: Path,
) -> None:
    """Show the five geographic holdout regions used in spatial validation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        frame["longitude"],
        frame["latitude"],
        c=spatial_blocks,
        s=4,
        alpha=0.65,
        cmap="tab10",
    )
    ax.set_title("Five geographic blocks used for spatial cross-validation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    colorbar = fig.colorbar(scatter, ax=ax, ticks=sorted(np.unique(spatial_blocks)))
    colorbar.set_label("Held-out block ID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def sensitivity_labels(
    df: pd.DataFrame,
    node_to_community: Dict[object, int],
) -> np.ndarray:
    """Return community labels in stable dataset order."""
    return np.asarray([node_to_community[node] for node in df["id"]], dtype=int)


def run_parameter_sensitivity(
    df: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    configs: Iterable[SensitivityConfig] = SENSITIVITY_CONFIGS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the one-at-a-time Graph C sensitivity experiment."""
    graph_config = core.GraphConfig(
        "Graph C: Spatial + shared host + attribute similarity",
        True,
        True,
        True,
    )
    baseline_labels = None
    baseline_assignment = None
    rows = []

    for config in configs:
        print("\n" + "=" * 70)
        print(f"Sensitivity configuration: {config.name}")
        start = time.time()
        graph = core.build_graph_variant(
            df=df,
            config=graph_config,
            radius_meters=config.radius_meters,
            attribute_k=config.attribute_k,
            alpha_spatial=config.alpha_spatial,
            alpha_host=config.alpha_host,
            alpha_attribute=config.alpha_attribute,
        )
        build_seconds = time.time() - start

        start = time.time()
        communities, assignment, modularity = core.run_louvain(graph)
        louvain_seconds = time.time() - start
        labels = sensitivity_labels(df, assignment)

        if baseline_labels is None:
            baseline_labels = labels.copy()
            baseline_assignment = core.make_assignment_dataframe(
                df,
                assignment,
                COMMUNITY_COLUMN,
            )

        official_labels = df["neighbourhood_cleansed"].astype(str).to_numpy()
        components = list(nx.connected_components(graph))
        largest_component = max((len(component) for component in components), default=0)
        community_sizes = np.asarray([len(community) for community in communities])

        rows.append(
            {
                "configuration": config.name,
                "varied_parameter": config.varied_parameter,
                "radius_meters": config.radius_meters,
                "attribute_k": config.attribute_k,
                "alpha_spatial": config.alpha_spatial,
                "alpha_host": config.alpha_host,
                "alpha_attribute": config.alpha_attribute,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "connected_components": len(components),
                "largest_component_fraction": largest_component
                / max(graph.number_of_nodes(), 1),
                "communities": len(communities),
                "largest_community": int(community_sizes.max()),
                "median_community_size": float(np.median(community_sizes)),
                "modularity": modularity,
                "nmi_vs_neighbourhood": normalized_mutual_info_score(
                    official_labels,
                    labels,
                ),
                "vi_vs_neighbourhood": core.variation_of_information(
                    official_labels,
                    labels,
                ),
                "nmi_vs_baseline": normalized_mutual_info_score(
                    baseline_labels,
                    labels,
                ),
                "vi_vs_baseline": core.variation_of_information(
                    baseline_labels,
                    labels,
                ),
                "build_seconds": build_seconds,
                "louvain_seconds": louvain_seconds,
                "total_seconds": build_seconds + louvain_seconds,
            }
        )
        del graph, communities, assignment
        gc.collect()

    if baseline_assignment is None:
        raise RuntimeError("The sensitivity configuration list did not produce a baseline graph")

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "parameter_sensitivity_results.csv", index=False)
    plot_parameter_sensitivity(
        results,
        fig_dir / "parameter_sensitivity.png",
    )
    return results, baseline_assignment


def plot_parameter_sensitivity(results: pd.DataFrame, out_path: Path) -> None:
    """Plot structural stability around the baseline Graph C design."""
    x = np.arange(len(results))
    labels = results["configuration"].str.replace(" weights", "", regex=False)
    baseline = results.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(x, results["communities"], color="#4c78a8")
    axes[0, 0].axhline(
        baseline["communities"], color="black", linestyle="--", linewidth=1
    )
    axes[0, 0].set_title("Detected communities")
    axes[0, 0].set_ylabel("Louvain communities")

    axes[0, 1].bar(x, results["modularity"], color="#59a14f")
    axes[0, 1].axhline(
        baseline["modularity"], color="black", linestyle="--", linewidth=1
    )
    axes[0, 1].set_title("Weighted modularity")
    axes[0, 1].set_ylabel("Modularity")

    axes[1, 0].bar(x, results["nmi_vs_baseline"], color="#f28e2b")
    axes[1, 0].set_title("Partition agreement with baseline")
    axes[1, 0].set_ylabel("NMI (higher is more stable)")
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].bar(x, results["nmi_vs_neighbourhood"], color="#b07aa1")
    axes[1, 1].axhline(
        baseline["nmi_vs_neighbourhood"], color="black", linestyle="--", linewidth=1
    )
    axes[1, 1].set_title("Agreement with official neighbourhoods")
    axes[1, 1].set_ylabel("NMI")

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Graph C one-at-a-time parameter sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Graph C sensitivity and grouped/spatial price validation."
    )
    parser.add_argument("--csv", default="data/toronto_listings_clean.csv")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--fig-dir", default="results/figures")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if args.folds != 5:
        parser.error("The canonical experiment uses exactly five folds")

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    print(f"Loading cleaned dataset: {csv_path}")
    df = core.load_clean_data(csv_path)
    print(f"Rows loaded: {len(df):,}")

    sensitivity, baseline_assignment = run_parameter_sensitivity(
        df,
        out_dir,
        fig_dir,
    )
    cv_results, cv_summary, cv_deltas = evaluate_price_cross_validation(
        baseline_assignment,
        COMMUNITY_COLUMN,
        out_dir,
        fig_dir,
        n_splits=args.folds,
    )

    print("\nRobustness experiments complete.")
    print(sensitivity[["configuration", "communities", "modularity", "nmi_vs_baseline"]])
    print(cv_summary[["validation_scheme", "model", "r2_mean", "mae_dollars_mean"]])
    print(cv_deltas[["validation_scheme", "r2_delta_mean", "mae_dollars_delta_mean"]])
    print(f"Detailed fold results: {out_dir / 'price_model_cv_results.csv'}")


if __name__ == "__main__":
    main()
