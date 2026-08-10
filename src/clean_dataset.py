#!/usr/bin/env python3
# ============================================================
# EECS 4414 Final Project
# Airbnb Listing Networks: Community Detection and Price Influence
# File: clean_dataset.py
#
# Purpose:
#   Cleans the raw Inside Airbnb Toronto listings dataset and creates
#   the cleaned CSV used by the midterm and final experiments.
#
# Input:
#   data/listings1.csv.gz or another raw Inside Airbnb listings file.
#
# Output:
#   data/toronto_listings_clean.csv
#   results/figures/price_distribution.png
#   results/figures/top_neighbourhoods_by_listing_count.png
#
# Notes:
#   This script should be run first only if the cleaned dataset has not
#   already been created.
# ============================================================

"""
Clean the raw Inside Airbnb Toronto listings dataset.

This script is included for reproducibility. It creates the cleaned dataset
used by the midterm and final project experiments.

Input:
  data/listings1.csv.gz  or another raw Inside Airbnb listings CSV/CSV.GZ file

Output:
  data/toronto_listings_clean.csv
  results/figures/price_distribution.png
  results/figures/top_neighbourhoods_by_listing_count.png

Example:
  python src/clean_dataset.py --raw data/listings1.csv.gz --out data/toronto_listings_clean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEEP_COLUMNS = [
    "id",
    "host_id",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "price",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "instant_bookable",
    "host_is_superhost",
    "license",
    "calculated_host_listings_count",
    "reviews_per_month",
]


def clean_price_column(series: pd.Series) -> pd.Series:
    """Convert Airbnb price strings such as '$125.00' into numeric values."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .replace({"nan": np.nan, "": np.nan, "None": np.nan}),
        errors="coerce",
    )


def clean_dataset(raw_path: Path) -> pd.DataFrame:
    """Load and clean the raw listings file."""
    df = pd.read_csv(raw_path, low_memory=False)

    available_columns = [col for col in KEEP_COLUMNS if col in df.columns]
    cleaned = df[available_columns].copy()

    if "price" not in cleaned.columns:
        raise ValueError("The raw dataset must contain a 'price' column.")

    cleaned["price"] = clean_price_column(cleaned["price"])

    cleaned = cleaned.dropna(
        subset=["id", "host_id", "latitude", "longitude", "price"]
    )
    cleaned = cleaned.drop_duplicates(subset=["id"])
    cleaned = cleaned[cleaned["price"] > 0].copy()
    cleaned = cleaned.reset_index(drop=True)

    low, high = cleaned["price"].quantile([0.01, 0.99])
    cleaned["price_w"] = cleaned["price"].clip(low, high)
    cleaned["log_price"] = np.log(cleaned["price"])
    cleaned["log_price_w"] = np.log(cleaned["price_w"])

    numeric_columns = [
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

    for col in numeric_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_columns = [
        "room_type",
        "property_type",
        "bathrooms_text",
        "neighbourhood_cleansed",
        "instant_bookable",
        "host_is_superhost",
        "license",
    ]

    for col in categorical_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown").astype(str)

    return cleaned


def make_basic_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    """Create the two initial figures used in the midterm report."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["price_w"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Price Distribution (Winsorized)")
    axes[0].set_xlabel("Nightly Price ($)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["log_price_w"], bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_title("Log Price Distribution")
    axes[1].set_xlabel("Log Price")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(fig_dir / "price_distribution.png", dpi=300)
    plt.close(fig)

    top15 = df["neighbourhood_cleansed"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    top15.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Neighbourhoods by Listing Count")
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("Neighbourhood")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_neighbourhoods_by_listing_count.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/listings1.csv.gz")
    parser.add_argument("--out", type=str, default="data/toronto_listings_clean.csv")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    fig_dir = Path(args.fig_dir)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}. "
            "Download the Toronto listings file from Inside Airbnb and place it in data/."
        )

    cleaned = clean_dataset(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    make_basic_figures(cleaned, fig_dir)

    print("Cleaned dataset saved.")
    print(f"Rows: {len(cleaned):,}")
    print(f"Unique hosts: {cleaned['host_id'].nunique():,}")
    print(f"Neighbourhoods: {cleaned['neighbourhood_cleansed'].nunique():,}")
    print(f"Output: {out_path}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
