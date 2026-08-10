#!/usr/bin/env python3
# ============================================================
# EECS 4414 Final Project
# Airbnb Listing Networks: Community Detection and Price Influence
# File: generate_graph_c_leiden_figures.py
#
# Purpose:
#   Helper script for generating Graph C Leiden figures only.
#
# Input:
#   data/toronto_listings_clean.csv
#
# Output:
#   results/tables/graph_c_leiden_assignments.csv
#   results/tables/graph_c_leiden_community_summary.csv
#   results/figures/graph_c_leiden_community_map.png
#   results/figures/graph_c_leiden_community_sizes.png
#   results/figures/graph_c_leiden_median_price_by_community.png
#
# Notes:
#   Use this only when the main experiment has already been run and
#   the Leiden visualizations need to be regenerated separately.
# ============================================================

"""
Generate Graph C Leiden figures only.

Use this when the full experiment has already been run, but you also want
Leiden visualizations for the published analysis without rerunning Graph A and Graph B.

Expected outputs:
  results/tables/
    graph_c_leiden_assignments.csv
    graph_c_leiden_community_summary.csv
    graph_c_leiden_only_summary.txt

  results/figures/
    graph_c_leiden_community_map.png
    graph_c_leiden_community_sizes.png
    graph_c_leiden_median_price_by_community.png

How to run:
  python src/generate_graph_c_leiden_figures.py

Requirements:
  pip install igraph leidenalg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx


def import_experiment_functions():
    """Import reusable helpers from the canonical experiment script."""
    import airbnb_final_experiments as exp_module

    return (
        exp_module.GraphConfig,
        exp_module.load_clean_data,
        exp_module.build_graph_variant,
        exp_module.run_leiden_if_available,
        exp_module.community_result_row,
        exp_module.alignment_row,
        exp_module.make_assignment_dataframe,
        exp_module.save_community_summary,
        exp_module.make_community_figures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate only the Graph C Leiden artifacts."
    )
    parser.add_argument(
        "--csv", type=str, default="data/toronto_listings_clean.csv"
    )
    parser.add_argument("--out-dir", type=str, default="results/tables")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--radius", type=int, default=500)
    parser.add_argument("--attribute-k", type=int, default=5)
    parser.add_argument("--alpha-spatial", type=float, default=0.60)
    parser.add_argument("--alpha-host", type=float, default=0.25)
    parser.add_argument("--alpha-attribute", type=float, default=0.15)
    args = parser.parse_args()

    (
        GraphConfig,
        load_clean_data,
        build_graph_variant,
        run_leiden_if_available,
        community_result_row,
        alignment_row,
        make_assignment_dataframe,
        save_community_summary,
        make_community_figures,
    ) = import_experiment_functions()

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

    graph_c = GraphConfig(
        "Graph C: Spatial + shared host + attribute similarity",
        True,
        True,
        True,
    )

    print("\n" + "=" * 70)
    print("Building Graph C only")
    start = time.time()

    G = build_graph_variant(
        df=df,
        config=graph_c,
        radius_meters=args.radius,
        attribute_k=args.attribute_k,
        alpha_spatial=args.alpha_spatial,
        alpha_host=args.alpha_host,
        alpha_attribute=args.alpha_attribute,
    )

    build_seconds = time.time() - start

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Components: {nx.number_connected_components(G):,}")
    print(f"  Build seconds: {build_seconds:.2f}")

    print("\nRunning Leiden for Graph C")
    start = time.time()
    leiden_result = run_leiden_if_available(G)
    leiden_seconds = time.time() - start

    if leiden_result is None:
        raise RuntimeError(
            "Leiden could not be run. Install optional packages with: "
            "python -m pip install igraph leidenalg"
        )

    leiden_communities, leiden_node_to_comm, leiden_modularity = leiden_result

    community_row = community_result_row(
        graph_c.name,
        "Leiden",
        leiden_communities,
        leiden_modularity,
        leiden_seconds,
    )

    align_row = alignment_row(
        graph_c.name,
        "Leiden",
        df,
        leiden_node_to_comm,
    )

    print("\nGraph C Leiden result:")
    print(f"  Communities: {community_row['communities']}")
    print(f"  Modularity: {community_row['modularity']:.6f}")
    print(f"  Largest community: {community_row['largest_community']}")
    print(f"  Median community size: {community_row['median_community_size']}")
    print(f"  NMI vs neighbourhood: {align_row['nmi_vs_neighbourhood']:.6f}")
    print(f"  VI vs neighbourhood: {align_row['vi_vs_neighbourhood']:.6f}")
    print(f"  Leiden seconds: {leiden_seconds:.2f}")

    graph_c_leiden_assignment = make_assignment_dataframe(
        df,
        leiden_node_to_comm,
        "graph_c_leiden_community",
    )

    assignments_path = out_dir / "graph_c_leiden_assignments.csv"
    graph_c_leiden_assignment.to_csv(assignments_path, index=False)

    summary_path = out_dir / "graph_c_leiden_community_summary.csv"
    graph_c_leiden_summary = save_community_summary(
        graph_c_leiden_assignment,
        "graph_c_leiden_community",
        summary_path,
    )

    make_community_figures(
        graph_c_leiden_assignment,
        graph_c_leiden_summary,
        "graph_c_leiden_community",
        fig_dir,
        "graph_c_leiden",
    )

    summary_txt = out_dir / "graph_c_leiden_only_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "Graph C Leiden-only run summary",
                "=" * 40,
                f"CSV: {csv_path}",
                f"Nodes: {G.number_of_nodes():,}",
                f"Edges: {G.number_of_edges():,}",
                f"Connected components: {nx.number_connected_components(G):,}",
                f"Build seconds: {build_seconds:.2f}",
                "",
                "Leiden result:",
                f"Communities: {community_row['communities']}",
                f"Modularity: {community_row['modularity']:.6f}",
                f"Largest community: {community_row['largest_community']}",
                f"Median community size: {community_row['median_community_size']}",
                f"Mean community size: {community_row['mean_community_size']:.6f}",
                f"Singletons: {community_row['singletons']}",
                f"Leiden seconds: {leiden_seconds:.2f}",
                "",
                "Neighbourhood alignment:",
                f"NMI vs neighbourhood: {align_row['nmi_vs_neighbourhood']:.6f}",
                f"VI vs neighbourhood: {align_row['vi_vs_neighbourhood']:.6f}",
                "",
                "Saved files:",
                str(assignments_path),
                str(summary_path),
                str(fig_dir / "graph_c_leiden_community_map.png"),
                str(fig_dir / "graph_c_leiden_community_sizes.png"),
                str(fig_dir / "graph_c_leiden_median_price_by_community.png"),
            ]
        ),
        encoding="utf-8",
    )

    print("\nDone. Leiden files created:")
    print(f"  {assignments_path}")
    print(f"  {summary_path}")
    print(f"  {summary_txt}")
    print(f"  {fig_dir / 'graph_c_leiden_community_map.png'}")
    print(f"  {fig_dir / 'graph_c_leiden_community_sizes.png'}")
    print(f"  {fig_dir / 'graph_c_leiden_median_price_by_community.png'}")


if __name__ == "__main__":
    main()
