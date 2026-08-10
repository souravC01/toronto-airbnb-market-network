#!/usr/bin/env python3
# ============================================================
# EECS 4414 Final Project
# Airbnb Listing Networks: Community Detection and Price Influence
# File: make_alignment_plots.py
#
# Purpose:
#   Creates grouped bar charts comparing detected communities against
#   official Toronto neighbourhood labels using NMI and VI.
#
# Input:
#   results/tables/alignment_results.csv
#
# Output:
#   results/figures/alignment_nmi_comparison.png
#   results/figures/alignment_vi_comparison.png
#
# Notes:
#   NMI is better when higher. VI is better when lower.
# ============================================================

"""
Generate alignment comparison plots for NMI and VI.

This script reads results/tables/alignment_results.csv and creates:
  - results/figures/alignment_nmi_comparison.png
  - results/figures/alignment_vi_comparison.png

It compares Louvain and Leiden across:
  - Graph A
  - Graph B
  - Graph C

Usage:
  python src/make_alignment_plots.py
or
  python src/make_alignment_plots.py --csv results/tables/alignment_results.csv --out-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRAPH_ORDER = [
    "Graph A: Spatial only",
    "Graph B: Spatial + shared host",
    "Graph C: Spatial + shared host + attribute similarity",
]

GRAPH_LABELS = {
    "Graph A: Spatial only": "Graph A",
    "Graph B: Spatial + shared host": "Graph B",
    "Graph C: Spatial + shared host + attribute similarity": "Graph C",
}


def validate_input(df: pd.DataFrame) -> None:
    required_cols = {"graph", "algorithm", "nmi_vs_neighbourhood", "vi_vs_neighbourhood"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"alignment_results.csv is missing required columns: {sorted(missing)}")

    required_pairs = {(g, a) for g in GRAPH_ORDER for a in ["Louvain", "Leiden"]}
    present_pairs = set(zip(df["graph"], df["algorithm"]))
    missing_pairs = required_pairs - present_pairs
    if missing_pairs:
        raise ValueError(
            "alignment_results.csv does not contain all expected graph/algorithm pairs. "
            f"Missing: {sorted(missing_pairs)}"
        )


def add_value_labels(ax, values, offsets):
    for x, y in zip(offsets, values):
        ax.text(x, y + (0.01 if y < 1 else 0.03), f"{y:.3f}", ha="center", va="bottom", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Louvain and Leiden neighbourhood-alignment results."
    )
    parser.add_argument(
        "--csv", type=str, default="results/tables/alignment_results.csv"
    )
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_input(df)

    louvain = (
        df[df["algorithm"] == "Louvain"]
        .set_index("graph")
        .loc[GRAPH_ORDER]
        .reset_index()
    )
    leiden = (
        df[df["algorithm"] == "Leiden"]
        .set_index("graph")
        .loc[GRAPH_ORDER]
        .reset_index()
    )

    x = np.arange(len(GRAPH_ORDER))
    width = 0.35
    x_labels = [GRAPH_LABELS[g] for g in GRAPH_ORDER]

    # -----------------------------
    # NMI grouped bar chart
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, louvain["nmi_vs_neighbourhood"], width, label="Louvain")
    bars2 = ax.bar(x + width / 2, leiden["nmi_vs_neighbourhood"], width, label="Leiden")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Normalized Mutual Information (NMI)")
    ax.set_xlabel("Graph Variant")
    ax.set_title("NMI of Detected Communities vs Official Neighbourhoods")
    ax.legend()
    ax.set_ylim(0, max(df["nmi_vs_neighbourhood"]) + 0.12)

    add_value_labels(ax, louvain["nmi_vs_neighbourhood"], x - width / 2)
    add_value_labels(ax, leiden["nmi_vs_neighbourhood"], x + width / 2)

    fig.tight_layout()
    nmi_path = out_dir / "alignment_nmi_comparison.png"
    fig.savefig(nmi_path, dpi=300)
    plt.close(fig)

    # -----------------------------
    # VI grouped bar chart
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, louvain["vi_vs_neighbourhood"], width, label="Louvain")
    bars2 = ax.bar(x + width / 2, leiden["vi_vs_neighbourhood"], width, label="Leiden")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Variation of Information (VI)")
    ax.set_xlabel("Graph Variant")
    ax.set_title("VI of Detected Communities vs Official Neighbourhoods")
    ax.legend()
    ax.set_ylim(0, max(df["vi_vs_neighbourhood"]) + 0.6)

    add_value_labels(ax, louvain["vi_vs_neighbourhood"], x - width / 2)
    add_value_labels(ax, leiden["vi_vs_neighbourhood"], x + width / 2)

    fig.tight_layout()
    vi_path = out_dir / "alignment_vi_comparison.png"
    fig.savefig(vi_path, dpi=300)
    plt.close(fig)

    print("Done. Saved:")
    print(f"  {nmi_path}")
    print(f"  {vi_path}")


if __name__ == "__main__":
    main()
