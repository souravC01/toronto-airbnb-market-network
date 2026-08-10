#!/usr/bin/env python3
# ============================================================
# EECS 4414 Final Project
# Airbnb Listing Networks: Community Detection and Price Influence
# File: make_midterm_eda_plots.py
#
# Purpose:
#   Generates exploratory data analysis figures used in the
#   midterm/progress report.
#
# Input:
#   data/toronto_listings_clean.csv
#
# Output:
#   results/figures/room_type_counts.png
#   results/figures/median_price_by_room_type.png
#   results/figures/median_price_top_neighbourhoods.png
#   results/figures/spatial_log_price_scatter.png
#   results/figures/median_price_by_capacity.png
#   results/figures/host_listing_distribution.png
#
# Notes:
#   This script does not build the network. It only creates descriptive
#   plots for understanding the cleaned dataset.
# ============================================================

"""
Generate exploratory figures used in the midterm/progress report.

Input:
  data/toronto_listings_clean.csv

Outputs:
  results/figures/room_type_counts.png
  results/figures/median_price_by_room_type.png
  results/figures/median_price_top_neighbourhoods.png
  results/figures/spatial_log_price_scatter.png
  results/figures/median_price_by_capacity.png
  results/figures/host_listing_distribution.png

Example:
  python src/make_midterm_eda_plots.py --csv data/toronto_listings_clean.csv --out-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_midterm_eda_plots(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    price_col = "price_w" if "price_w" in df.columns else "price"
    log_price_col = "log_price_w" if "log_price_w" in df.columns else "log_price"

    # 1) Room type composition
    room_counts = df["room_type"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    room_counts.plot(kind="bar", ax=ax)
    ax.set_title("Listings by Room Type")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Number of Listings")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "room_type_counts.png", dpi=300)
    plt.close(fig)

    # 2) Median price by room type
    room_price = df.groupby("room_type")[price_col].median().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    room_price.plot(kind="bar", ax=ax)
    ax.set_title("Median Winsorized Price by Room Type")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Median Nightly Price ($)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "median_price_by_room_type.png", dpi=300)
    plt.close(fig)

    # 3) Median price in top 15 neighbourhoods by listing count
    top_neigh = df["neighbourhood_cleansed"].value_counts().head(15).index
    median_neigh_price = (
        df[df["neighbourhood_cleansed"].isin(top_neigh)]
        .groupby("neighbourhood_cleansed")[price_col]
        .median()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8.5, 6))
    median_neigh_price.plot(kind="barh", ax=ax)
    ax.set_title("Median Winsorized Price in Top Listing Neighbourhoods")
    ax.set_xlabel("Median Nightly Price ($)")
    ax.set_ylabel("Neighbourhood")
    fig.tight_layout()
    fig.savefig(out_dir / "median_price_top_neighbourhoods.png", dpi=300)
    plt.close(fig)

    # 4) Geographic scatter plot by log price
    sample_df = df.copy()
    if len(sample_df) > 12000:
        sample_df = sample_df.sample(12000, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        sample_df["longitude"],
        sample_df["latitude"],
        c=sample_df[log_price_col],
        s=4,
        alpha=0.55,
    )
    ax.set_title("Spatial Distribution of Listings by Log Price")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Log Price")
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_log_price_scatter.png", dpi=300)
    plt.close(fig)

    # 5) Median price by guest capacity
    acc_price = (
        df.groupby("accommodates")[price_col]
        .median()
        .reset_index()
        .sort_values("accommodates")
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(acc_price["accommodates"], acc_price[price_col], marker="o")
    ax.set_title("Median Winsorized Price by Guest Capacity")
    ax.set_xlabel("Accommodates")
    ax.set_ylabel("Median Nightly Price ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "median_price_by_capacity.png", dpi=300)
    plt.close(fig)

    # 6) Host listing count distribution
    host_counts = df.groupby("host_id").size()
    host_count_bins = pd.cut(
        host_counts,
        bins=[0, 1, 2, 5, 10, 25, 50, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-25", "26-50", "51+"],
    ).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    host_count_bins.plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Listings per Host")
    ax.set_xlabel("Listings per Host")
    ax.set_ylabel("Number of Hosts")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "host_listing_distribution.png", dpi=300)
    plt.close(fig)

    print(f"Saved midterm EDA figures to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/toronto_listings_clean.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    make_midterm_eda_plots(Path(args.csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
