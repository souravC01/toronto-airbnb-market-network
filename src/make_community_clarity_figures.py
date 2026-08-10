"""
Generate additional community-detection figures that clarify the OUTCOME
of Graph C Louvain community detection.

Addresses feedback that the community result was not clear: these figures
characterise *what each detected community is* as an Airbnb market segment,
using the existing community-summary CSV (no full pipeline re-run needed).

Outputs (to results/figures/):
  community_profile_bubble.png        size vs median price, coloured by room type
  community_price_vs_spread.png       geographic spread vs median price
  community_characterisation.png      labelled table-style summary of segments
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENTIRE = "#1f77b4"
PRIVATE = "#d62728"


def room_colour(room_type: str) -> str:
    return ENTIRE if "Entire" in room_type else PRIVATE


def short_label(row: pd.Series) -> str:
    """Human-readable market-segment label for a community."""
    n = row["dominant_neighbourhood"]
    # shorten long official names for plotting
    n = (
        n.replace("Waterfront Communities-The Island", "Waterfront/Islands")
        .replace("Dovercourt-Wallace Emerson-Junction", "Dovercourt-Junction")
        .replace("Mimico (includes Humber Bay Shores)", "Mimico/Humber Bay")
        .replace("Bedford Park-Nortown", "Bedford Park")
    )
    kind = "private rooms" if "Private" in row["dominant_room_type"] else "entire homes"
    return f"C{int(row['graph_c_louvain_community'])}: {n} ({kind})"


def make_clarity_figures(summary_csv: Path, fig_dir: Path) -> None:
    """Create three portfolio-ready views of the Louvain community summary."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv).sort_values("listings", ascending=False)
    df["colour"] = df["dominant_room_type"].map(room_colour)
    df["label"] = df.apply(short_label, axis=1)

    # ---- Figure 1: bubble chart — the whole community landscape in one view ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sizes = df["listings"] / df["listings"].max() * 2500 + 40
    ax.scatter(
        df["unique_neighbourhoods"],
        df["median_price_w"],
        s=sizes,
        c=df["colour"],
        alpha=0.6,
        edgecolors="black",
        linewidths=0.6,
    )
    for _, r in df.iterrows():
        ax.annotate(
            f"C{int(r['graph_c_louvain_community'])}",
            (r["unique_neighbourhoods"], r["median_price_w"]),
            ha="center", va="center", fontsize=8, fontweight="bold",
        )
    ax.set_xlabel("Geographic spread (number of official neighbourhoods spanned)")
    ax.set_ylabel("Median winsorized nightly price ($)")
    ax.set_title(
        "Graph C Louvain communities as Airbnb market segments\n"
        "(bubble size = number of listings)"
    )
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ENTIRE,
                   markersize=11, label="Entire-home dominated"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIVATE,
                   markersize=11, label="Private-room dominated"),
    ]
    ax.legend(handles=legend, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "community_profile_bubble.png", dpi=300)
    plt.close(fig)

    # ---- Figure 2: price vs geographic spread, labelled bars ----
    d2 = df.sort_values("median_price_w")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(d2["label"], d2["median_price_w"], color=d2["colour"], alpha=0.85)
    for bar, n in zip(bars, d2["listings"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{int(n):,}", va="center", fontsize=8)
    ax.set_xlabel("Median winsorized nightly price ($)   (bar label = listings)")
    ax.set_title("What each detected community is: dominant neighbourhood, room type, and price")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "community_characterisation.png", dpi=300)
    plt.close(fig)

    # ---- Figure 3: concentration — local vs city-wide segments ----
    fig, ax = plt.subplots(figsize=(9, 6))
    # listings per neighbourhood spanned = how geographically concentrated
    concentration = df["listings"] / df["unique_neighbourhoods"]
    ax.scatter(df["listings"], concentration, s=80, c=df["colour"],
               alpha=0.7, edgecolors="black", linewidths=0.6)
    for _, r in df.iterrows():
        ax.annotate(f"C{int(r['graph_c_louvain_community'])}",
                    (r["listings"], r["listings"] / r["unique_neighbourhoods"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Community size (listings, log scale)")
    ax.set_ylabel("Listings per neighbourhood spanned (concentration)")
    ax.set_title(
        "Local boutique segments vs the broad background market\n"
        "High concentration = a tight local cluster; low = a city-wide segment"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "community_concentration.png", dpi=300)
    plt.close(fig)

    print("Wrote 3 figures to", fig_dir)
    print(df[["graph_c_louvain_community", "listings", "median_price_w",
              "unique_neighbourhoods", "dominant_neighbourhood",
              "dominant_room_type"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create interpretable Graph C Louvain community figures."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/graph_c_louvain_community_summary.csv"),
    )
    parser.add_argument(
        "--fig-dir", type=Path, default=Path("results/figures")
    )
    args = parser.parse_args()
    make_clarity_figures(args.summary, args.fig_dir)


if __name__ == "__main__":
    main()
