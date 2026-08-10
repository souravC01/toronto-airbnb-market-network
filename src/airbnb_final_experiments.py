#!/usr/bin/env python3
"""
EECS 4414 Final Project Experiments
Airbnb Listing Networks: Community Detection and Price Influence Across Toronto Neighbourhoods

This script produces the final-report experiment outputs.

Outputs:
  results/tables/
    graph_stats.csv
    community_results.csv
    alignment_results.csv
    price_model_results.csv
    run_summary.txt

    graph_c_louvain_assignments.csv
    graph_c_louvain_community_summary.csv

    If --run-leiden is used:
      graph_c_leiden_assignments.csv
      graph_c_leiden_community_summary.csv

  results/figures/
    graph_c_louvain_community_map.png
    graph_c_louvain_community_sizes.png
    graph_c_louvain_median_price_by_community.png

    If --run-leiden is used:
      graph_c_leiden_community_map.png
      graph_c_leiden_community_sizes.png
      graph_c_leiden_median_price_by_community.png

How to run from the repository root:
  python -m pip install -r requirements.txt
  python src/airbnb_final_experiments.py --run-leiden

Optional Leiden support:
  pip install igraph leidenalg
  python src/airbnb_final_experiments.py --run-leiden
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EARTH_RADIUS_METERS = 6_371_000
RANDOM_STATE = 42


@dataclass
class GraphConfig:
    name: str
    use_spatial: bool
    use_host: bool
    use_attribute: bool


GRAPH_CONFIGS = [
    GraphConfig("Graph A: Spatial only", True, False, False),
    GraphConfig("Graph B: Spatial + shared host", True, True, False),
    GraphConfig("Graph C: Spatial + shared host + attribute similarity", True, True, True),
]


def make_one_hot_encoder():
    """Handle both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned CSV: {missing}")


def load_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned Airbnb CSV and apply safety cleaning."""
    df = pd.read_csv(csv_path)

    required = [
        "id",
        "host_id",
        "latitude",
        "longitude",
        "neighbourhood_cleansed",
        "room_type",
        "property_type",
        "price_w",
        "log_price_w",
    ]
    ensure_columns(df, required)

    df = df.dropna(
        subset=["id", "host_id", "latitude", "longitude", "price_w", "log_price_w"]
    ).copy()
    df = df.drop_duplicates(subset=["id"]).copy()

    numeric_fill_cols = [
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

    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_fill_cols = [
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "instant_bookable",
        "host_is_superhost",
        "license",
    ]

    for col in categorical_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df.reset_index(drop=True)


def add_or_update_edge(G: nx.Graph, u, v, weight_add: float, edge_type: str) -> bool:
    """Add or update an edge and report whether a new edge was created."""
    if u == v or weight_add <= 0:
        return False

    if G.has_edge(u, v):
        G[u][v]["weight"] += float(weight_add)
        G[u][v]["types"].add(edge_type)
        return False
    else:
        G.add_edge(u, v, weight=float(weight_add), types={edge_type})
        return True


def init_graph_nodes(df: pd.DataFrame) -> nx.Graph:
    """Initialize graph nodes, one node per Airbnb listing."""
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(
            row["id"],
            host_id=row["host_id"],
            neighbourhood=row["neighbourhood_cleansed"],
            room_type=row["room_type"],
            property_type=row["property_type"],
            price_w=row.get("price_w", np.nan),
            log_price_w=row.get("log_price_w", np.nan),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )

    return G


def add_spatial_edges(
    df: pd.DataFrame,
    G: nx.Graph,
    radius_meters: int,
    alpha_spatial: float,
) -> None:
    """Add spatial-proximity edges using BallTree haversine radius search."""
    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = radius_meters / EARTH_RADIUS_METERS

    indices, distances = tree.query_radius(
        coords_rad,
        r=radius_rad,
        return_distance=True,
        sort_results=False,
    )

    listing_ids = df["id"].to_numpy()

    for i, (idxs, dists_rad) in enumerate(zip(indices, distances)):
        u = listing_ids[i]

        for j, dist_rad in zip(idxs, dists_rad):
            if j <= i:
                continue

            v = listing_ids[j]
            dist_m = float(dist_rad) * EARTH_RADIUS_METERS
            spatial_score = max(0.0, 1.0 - (dist_m / radius_meters))

            add_or_update_edge(
                G,
                u,
                v,
                alpha_spatial * spatial_score,
                "spatial",
            )


def add_shared_host_edges(
    df: pd.DataFrame,
    G: nx.Graph,
    alpha_host: float,
    same_host_k: int = 5,
) -> None:
    """
    Add sparse shared-host edges.

    Instead of connecting every pair of listings owned by the same host,
    each listing is connected to up to same_host_k nearest listings from
    the same host. This prevents very large host cliques from dominating
    the network and keeps runtime manageable.
    """
    processed_hosts = 0
    total_host_edges_added = 0

    for _, group in df.groupby("host_id"):
        ids = group["id"].to_numpy()
        n = len(ids)

        if n < 2:
            continue

        if n == 2:
            total_host_edges_added += int(
                add_or_update_edge(G, ids[0], ids[1], alpha_host, "shared_host")
            )
            continue

        coords_rad = np.radians(group[["latitude", "longitude"]].to_numpy())

        k = min(same_host_k + 1, n)
        tree = BallTree(coords_rad, metric="haversine")
        nearest = tree.query(coords_rad, k=k, return_distance=False)

        for i in range(n):
            u = ids[i]

            for j in nearest[i][1:]:
                v = ids[j]
                total_host_edges_added += int(
                    add_or_update_edge(G, u, v, alpha_host, "shared_host")
                )

        processed_hosts += 1

        if processed_hosts % 500 == 0:
            print(f"    Processed {processed_hosts} multi-listing hosts...")

    print(f"    Added approximately {total_host_edges_added:,} shared-host edges.")


def add_attribute_similarity_edges(
    df: pd.DataFrame,
    G: nx.Graph,
    k: int,
    alpha_attribute: float,
) -> None:
    """Add attribute-similarity edges using k-nearest neighbours."""
    numeric_features = [
        "accommodates",
        "bedrooms",
        "beds",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "reviews_per_month",
    ]

    categorical_features = [
        "room_type",
        "property_type",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    if not numeric_features and not categorical_features:
        warnings.warn("No usable attribute-similarity features found; skipping attribute edges.")
        return

    work = df[["id"] + numeric_features + categorical_features].copy()

    for col in numeric_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median())

    for col in categorical_features:
        work[col] = work[col].fillna("Unknown").astype(str)

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
    )

    X = transformer.fit_transform(work)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(work)), metric="cosine")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    listing_ids = work["id"].to_numpy()

    for i in range(len(work)):
        u = listing_ids[i]

        for dist, j in zip(distances[i][1:], indices[i][1:]):
            similarity = 1.0 - float(dist)

            if similarity <= 0:
                continue

            v = listing_ids[j]
            add_or_update_edge(
                G,
                u,
                v,
                alpha_attribute * similarity,
                "attribute",
            )


def build_graph_variant(
    df: pd.DataFrame,
    config: GraphConfig,
    radius_meters: int,
    attribute_k: int,
    alpha_spatial: float,
    alpha_host: float,
    alpha_attribute: float,
) -> nx.Graph:
    """Build one of the three graph variants used in the published analysis."""
    G = init_graph_nodes(df)

    if config.use_spatial:
        print(f"  Adding spatial edges: radius={radius_meters}m")
        add_spatial_edges(df, G, radius_meters, alpha_spatial)

    if config.use_host:
        print("  Adding shared-host edges")
        add_shared_host_edges(df, G, alpha_host)

    if config.use_attribute:
        print(f"  Adding attribute-similarity edges: k={attribute_k}")
        add_attribute_similarity_edges(df, G, attribute_k, alpha_attribute)

    return G


def graph_stats(
    G: nx.Graph,
    graph_name: str,
    build_seconds: float,
    clustering_sample_size: int,
) -> Dict[str, float]:
    """Compute graph statistics, estimating weighted clustering on a fixed sample."""
    components = list(nx.connected_components(G))
    largest_cc = max((len(c) for c in components), default=0)
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    weighted_degrees = np.array([deg for _, deg in G.degree(weight="weight")], dtype=float)

    nodes = list(G.nodes())
    sample_size = min(clustering_sample_size, len(nodes))
    if G.number_of_edges() and sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_indices = rng.choice(len(nodes), size=sample_size, replace=False)
        sample_nodes = [nodes[index] for index in sample_indices]
        clustering = nx.average_clustering(G, nodes=sample_nodes, weight="weight")
    else:
        clustering = 0.0

    return {
        "graph": graph_name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_degree": degrees.mean() if len(degrees) else 0,
        "median_degree": float(np.median(degrees)) if len(degrees) else 0,
        "average_weighted_degree": weighted_degrees.mean() if len(weighted_degrees) else 0,
        "connected_components": len(components),
        "largest_component_size": largest_cc,
        "largest_component_fraction": largest_cc / max(G.number_of_nodes(), 1),
        "average_clustering_weighted_approx": clustering,
        "clustering_sample_size": sample_size,
        "build_seconds": build_seconds,
    }


def run_louvain(G: nx.Graph) -> Tuple[List[set], Dict[object, int], float]:
    """Run Louvain community detection using NetworkX."""
    communities = nx.community.louvain_communities(
        G,
        weight="weight",
        seed=RANDOM_STATE,
    )

    node_to_comm = {}

    for cid, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = cid

    modularity = nx.community.modularity(G, communities, weight="weight")
    return [set(c) for c in communities], node_to_comm, modularity


def run_leiden_if_available(G: nx.Graph) -> Optional[Tuple[List[set], Dict[object, int], float]]:
    """
    Run Leiden community detection if optional packages are installed.

    Returns None if igraph/leidenalg is unavailable.
    """
    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:
        print(f"  Leiden skipped: {exc}")
        return None

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=weights,
        seed=RANDOM_STATE,
    )

    communities = []
    node_to_comm = {}

    for cid, members in enumerate(partition):
        comm_nodes = {nodes[i] for i in members}
        communities.append(comm_nodes)

        for node in comm_nodes:
            node_to_comm[node] = cid

    modularity = nx.community.modularity(G, communities, weight="weight")
    return communities, node_to_comm, modularity


def community_result_row(
    graph_name: str,
    algorithm: str,
    communities: List[set],
    modularity: float,
    run_seconds: float,
) -> Dict[str, float]:
    """Create one row for the community detection results table."""
    sizes = np.array([len(c) for c in communities], dtype=float)

    return {
        "graph": graph_name,
        "algorithm": algorithm,
        "communities": len(communities),
        "modularity": modularity,
        "largest_community": int(sizes.max()) if len(sizes) else 0,
        "median_community_size": float(np.median(sizes)) if len(sizes) else 0,
        "mean_community_size": float(sizes.mean()) if len(sizes) else 0,
        "singletons": int((sizes == 1).sum()) if len(sizes) else 0,
        "community_detection_seconds": run_seconds,
    }


def labels_from_assignment(
    df: pd.DataFrame,
    node_to_comm: Dict[object, int],
) -> Tuple[List[str], List[int]]:
    """Return neighbourhood labels and community labels in dataset order."""
    y_true = df["neighbourhood_cleansed"].astype(str).tolist()
    y_pred = [int(node_to_comm.get(node, -1)) for node in df["id"]]
    return y_true, y_pred


def entropy_from_counts(counts: np.ndarray) -> float:
    """Entropy helper for Variation of Information."""
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def variation_of_information(labels_a: Iterable, labels_b: Iterable) -> float:
    """Compute Variation of Information between two partitions."""
    a = pd.Series(list(labels_a), dtype="category")
    b = pd.Series(list(labels_b), dtype="category")

    contingency = pd.crosstab(a, b).to_numpy()
    total = contingency.sum()

    if total == 0:
        return 0.0

    row_counts = contingency.sum(axis=1)
    col_counts = contingency.sum(axis=0)

    h_a = entropy_from_counts(row_counts)
    h_b = entropy_from_counts(col_counts)

    nz = contingency > 0
    p_ij = contingency[nz] / total

    row_probs = row_counts / total
    col_probs = col_counts / total

    row_idx, col_idx = np.where(nz)
    mi = float((p_ij * np.log2(p_ij / (row_probs[row_idx] * col_probs[col_idx]))).sum())

    return h_a + h_b - 2.0 * mi


def alignment_row(
    graph_name: str,
    algorithm: str,
    df: pd.DataFrame,
    node_to_comm: Dict[object, int],
) -> Dict[str, float]:
    """Compute NMI and VI between detected communities and neighbourhood labels."""
    y_neigh, y_comm = labels_from_assignment(df, node_to_comm)

    return {
        "graph": graph_name,
        "algorithm": algorithm,
        "nmi_vs_neighbourhood": normalized_mutual_info_score(y_neigh, y_comm),
        "vi_vs_neighbourhood": variation_of_information(y_neigh, y_comm),
    }


def make_assignment_dataframe(
    df: pd.DataFrame,
    node_to_comm: Dict[object, int],
    column_name: str,
) -> pd.DataFrame:
    """Merge community assignment back into the listing dataframe."""
    assign = pd.DataFrame(
        {
            "id": list(node_to_comm.keys()),
            column_name: list(node_to_comm.values()),
        }
    )

    return df.merge(assign, on="id", how="left")


def save_community_summary(
    df_with_comm: pd.DataFrame,
    community_col: str,
    out_path: Path,
) -> pd.DataFrame:
    """Save community-level summary statistics."""
    summary = (
        df_with_comm.groupby(community_col)
        .agg(
            listings=("id", "count"),
            median_price_w=("price_w", "median"),
            mean_price_w=("price_w", "mean"),
            dominant_neighbourhood=(
                "neighbourhood_cleansed",
                lambda s: s.value_counts().index[0],
            ),
            dominant_room_type=("room_type", lambda s: s.value_counts().index[0]),
            unique_neighbourhoods=("neighbourhood_cleansed", "nunique"),
            unique_hosts=("host_id", "nunique"),
        )
        .reset_index()
        .sort_values("listings", ascending=False)
    )

    summary.to_csv(out_path, index=False)
    return summary


def make_community_figures(
    df_with_comm: pd.DataFrame,
    summary: pd.DataFrame,
    community_col: str,
    fig_dir: Path,
    prefix: str,
) -> None:
    """Create community map, community-size plot, and median-price plot."""
    fig_dir.mkdir(exist_ok=True)

    # Community map
    plot_df = df_with_comm.copy()

    if len(plot_df) > 15000:
        plot_df = plot_df.sample(15000, random_state=RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df[community_col],
        s=4,
        alpha=0.65,
    )
    ax.set_title("Detected Airbnb Communities in Toronto")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Community ID")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_community_map.png", dpi=300)
    plt.close(fig)

    # Top community sizes
    top_sizes = summary.head(20).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(top_sizes[community_col].astype(str), top_sizes["listings"])
    ax.set_title("Top 20 Detected Communities by Size")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Number of Listings")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_community_sizes.png", dpi=300)
    plt.close(fig)

    # Median price by largest communities
    top_price = summary.head(15).sort_values("median_price_w")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_price[community_col].astype(str), top_price["median_price_w"])
    ax.set_title("Median Price in Largest Detected Communities")
    ax.set_xlabel("Median Winsorized Nightly Price ($)")
    ax.set_ylabel("Community ID")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_median_price_by_community.png", dpi=300)
    plt.close(fig)


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    """Approximate adjusted R^2 for the transformed design matrix."""
    if n <= p + 1:
        return float("nan")

    return 1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1))


def evaluate_price_models(
    df_with_comm: pd.DataFrame,
    community_col: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare baseline price model with community-enhanced model.

    Target is log-transformed winsorized price.
    """
    target = "log_price_w"

    numeric_features = [
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

    categorical_base = [
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "instant_bookable",
        "host_is_superhost",
    ]

    numeric_features = [c for c in numeric_features if c in df_with_comm.columns]
    categorical_base = [c for c in categorical_base if c in df_with_comm.columns]

    work = df_with_comm[[target] + numeric_features + categorical_base + [community_col]].copy()
    work = work.dropna(subset=[target, community_col]).copy()

    for col in numeric_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median())

    for col in categorical_base + [community_col]:
        work[col] = work[col].fillna("Unknown").astype(str)

    y = work[target]
    results = []

    model_specs = [
        ("Baseline: listing + official neighbourhood", categorical_base),
        ("Expanded: baseline + network community", categorical_base + [community_col]),
    ]

    for model_name, cat_features in model_specs:
        X = work[numeric_features + cat_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", make_one_hot_encoder(), cat_features),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae_log = mean_absolute_error(y_test, pred)
        rmse_log = float(np.sqrt(mean_squared_error(y_test, pred)))

        # Dollar-scale metrics by reversing the log transform.
        y_test_dollars = np.exp(y_test)
        pred_dollars = np.exp(pred)

        mae_dollars = mean_absolute_error(y_test_dollars, pred_dollars)
        rmse_dollars = float(np.sqrt(mean_squared_error(y_test_dollars, pred_dollars)))

        transformed_train = model.named_steps["preprocessor"].transform(X_train)
        p = transformed_train.shape[1]

        results.append(
            {
                "model": model_name,
                "target": target,
                "test_r2": r2,
                "test_adjusted_r2_approx": adjusted_r2_score(r2, len(y_test), p),
                "mae_log_price": mae_log,
                "rmse_log_price": rmse_log,
                "mae_dollars_approx": mae_dollars,
                "rmse_dollars_approx": rmse_dollars,
                "test_rows": len(y_test),
                "encoded_features": p,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "price_model_results.csv", index=False)

    return results_df


def write_run_summary(
    out_dir: Path,
    graph_stats_df: pd.DataFrame,
    community_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> None:
    """Write a readable experiment summary text file."""
    lines = []
    lines.append("EECS 4414 Final Project Experiment Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Graph statistics:")
    lines.append(graph_stats_df.to_string(index=False))
    lines.append("")
    lines.append("Community detection:")
    lines.append(community_df.to_string(index=False))
    lines.append("")
    lines.append("Neighbourhood alignment:")
    lines.append(alignment_df.to_string(index=False))
    lines.append("")
    lines.append("Price model comparison:")
    lines.append(price_df.to_string(index=False))
    lines.append("")

    (out_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Toronto Airbnb graph variants and reproduce the project results."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/toronto_listings_clean.csv",
        help="Cleaned listings CSV (default: data/toronto_listings_clean.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/tables",
        help="Directory for result tables and assignments (default: results/tables).",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="results/figures",
        help="Directory for generated figures (default: results/figures).",
    )
    parser.add_argument("--radius", type=int, default=500, help="Spatial radius in metres.")
    parser.add_argument(
        "--attribute-k",
        type=int,
        default=5,
        help="Nearest neighbours used for attribute-similarity edges.",
    )
    parser.add_argument("--alpha-spatial", type=float, default=0.60)
    parser.add_argument("--alpha-host", type=float, default=0.25)
    parser.add_argument("--alpha-attribute", type=float, default=0.15)
    parser.add_argument(
        "--run-leiden",
        action="store_true",
        help="Run Leiden in addition to Louvain for every graph variant.",
    )
    parser.add_argument(
        "--clustering-sample-size",
        type=int,
        default=500,
        help="Nodes sampled for weighted clustering (default: 500).",
    )
    args = parser.parse_args()

    if args.clustering_sample_size <= 0:
        parser.error("--clustering-sample-size must be greater than zero")

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset: {csv_path}")
    df = load_clean_data(csv_path)
    print(f"Rows loaded: {len(df):,}")
    print(f"Neighbourhoods: {df['neighbourhood_cleansed'].nunique():,}")
    print(f"Hosts: {df['host_id'].nunique():,}")

    graph_stat_rows = []
    community_rows = []
    alignment_rows = []

    graph_c_louvain_assignment = None

    for config in GRAPH_CONFIGS:
        print("\n" + "=" * 70)
        print(f"Building {config.name}")

        start = time.time()
        G = build_graph_variant(
            df=df,
            config=config,
            radius_meters=args.radius,
            attribute_k=args.attribute_k,
            alpha_spatial=args.alpha_spatial,
            alpha_host=args.alpha_host,
            alpha_attribute=args.alpha_attribute,
        )
        build_seconds = time.time() - start

        stats = graph_stats(
            G,
            config.name,
            build_seconds,
            args.clustering_sample_size,
        )
        graph_stat_rows.append(stats)

        print(f"  Nodes: {stats['nodes']:,}")
        print(f"  Edges: {stats['edges']:,}")
        print(f"  Components: {stats['connected_components']:,}")
        print(f"  Largest component fraction: {stats['largest_component_fraction']:.3f}")

        print("  Running Louvain")
        start = time.time()
        communities, node_to_comm, modularity = run_louvain(G)
        run_seconds = time.time() - start

        community_rows.append(
            community_result_row(
                config.name,
                "Louvain",
                communities,
                modularity,
                run_seconds,
            )
        )

        alignment_rows.append(
            alignment_row(
                config.name,
                "Louvain",
                df,
                node_to_comm,
            )
        )

        if config.name.startswith("Graph C"):
            graph_c_louvain_assignment = make_assignment_dataframe(
                df,
                node_to_comm,
                "graph_c_louvain_community",
            )

            graph_c_louvain_assignment.to_csv(
                out_dir / "graph_c_louvain_assignments.csv",
                index=False,
            )

            graph_c_louvain_summary = save_community_summary(
                graph_c_louvain_assignment,
                "graph_c_louvain_community",
                out_dir / "graph_c_louvain_community_summary.csv",
            )

            make_community_figures(
                graph_c_louvain_assignment,
                graph_c_louvain_summary,
                "graph_c_louvain_community",
                fig_dir,
                "graph_c_louvain",
            )

        # IMPORTANT: Leiden block must be inside the graph loop so it runs for A, B, and C.
        if args.run_leiden:
            print("  Running Leiden")
            start = time.time()
            leiden_result = run_leiden_if_available(G)
            run_seconds = time.time() - start

            if leiden_result is not None:
                leiden_communities, leiden_node_to_comm, leiden_modularity = leiden_result

                community_rows.append(
                    community_result_row(
                        config.name,
                        "Leiden",
                        leiden_communities,
                        leiden_modularity,
                        run_seconds,
                    )
                )

                alignment_rows.append(
                    alignment_row(
                        config.name,
                        "Leiden",
                        df,
                        leiden_node_to_comm,
                    )
                )

                if config.name.startswith("Graph C"):
                    graph_c_leiden_assignment = make_assignment_dataframe(
                        df,
                        leiden_node_to_comm,
                        "graph_c_leiden_community",
                    )

                    graph_c_leiden_assignment.to_csv(
                        out_dir / "graph_c_leiden_assignments.csv",
                        index=False,
                    )

                    graph_c_leiden_summary = save_community_summary(
                        graph_c_leiden_assignment,
                        "graph_c_leiden_community",
                        out_dir / "graph_c_leiden_community_summary.csv",
                    )

                    make_community_figures(
                        graph_c_leiden_assignment,
                        graph_c_leiden_summary,
                        "graph_c_leiden_community",
                        fig_dir,
                        "graph_c_leiden",
                    )

    graph_stats_df = pd.DataFrame(graph_stat_rows)
    community_df = pd.DataFrame(community_rows)
    alignment_df = pd.DataFrame(alignment_rows)

    graph_stats_df.to_csv(out_dir / "graph_stats.csv", index=False)
    community_df.to_csv(out_dir / "community_results.csv", index=False)
    alignment_df.to_csv(out_dir / "alignment_results.csv", index=False)

    if graph_c_louvain_assignment is None:
        raise RuntimeError("Graph C Louvain assignment was not created.")

    print("\nEvaluating price models using Graph C Louvain communities")
    price_df = evaluate_price_models(
        graph_c_louvain_assignment,
        "graph_c_louvain_community",
        out_dir,
    )

    write_run_summary(
        out_dir,
        graph_stats_df,
        community_df,
        alignment_df,
        price_df,
    )

    print("\nDone. Main final-report outputs:")
    print(f"  {out_dir / 'graph_stats.csv'}")
    print(f"  {out_dir / 'community_results.csv'}")
    print(f"  {out_dir / 'alignment_results.csv'}")
    print(f"  {out_dir / 'price_model_results.csv'}")
    print(f"  {out_dir / 'run_summary.txt'}")
    print(f"  Figures saved in: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
