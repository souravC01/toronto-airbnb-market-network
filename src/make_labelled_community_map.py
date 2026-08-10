"""
Build a CLEAN, categorical, labelled Graph C Louvain community map.

The original map (results/figures/graph_c_louvain_community_map.png) colours
listings with a continuous colourbar over community IDs, which makes 16
categorical market segments look like a smooth gradient. This version uses a
distinct categorical colour per community, greys out the dominant background
community so the structured segments stand out, and annotates the largest
segments with their dominant-neighbourhood label.

Requires per-listing assignments (produced by airbnb_final_experiments.py):
  results/tables/graph_c_louvain_assignments.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd

COMM_COL = "graph_c_louvain_community"


def make_labelled_map(assignments: Path, summary: Path, fig_dir: Path) -> Path:
    """Create a categorical map with human-readable labels for large communities."""
    df = pd.read_csv(assignments)
    summ = pd.read_csv(summary).sort_values("listings", ascending=False)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # background = largest community -> grey; others get categorical colours
    background_id = int(summ.iloc[0][COMM_COL])
    structured = summ[summ[COMM_COL] != background_id]

    def short_name(raw: str) -> str:
        return (raw
                .replace("Waterfront Communities-The Island", "Waterfront/Islands")
                .replace("Dovercourt-Wallace Emerson-Junction", "Dovercourt-Junction")
                .replace("Kensington-Chinatown", "Kensington")
                .replace("Bay Street Corridor", "Bay St Corridor")
                .replace("Mimico (includes Humber Bay Shores)", "Mimico")
                .replace("York University Heights", "York U. Heights")
                .replace("Bedford Park-Nortown", "Bedford Park"))

    # label the eight largest structured segments in the legend
    legend_ids = list(structured.head(8)[COMM_COL].astype(int))
    palette = colormaps["tab10"].resampled(max(len(legend_ids), 1))
    colour_map = {cid: palette(i) for i, cid in enumerate(legend_ids)}

    fig, ax = plt.subplots(figsize=(7.5, 7))

    # 1) background market in light grey
    bg = df[df[COMM_COL] == background_id]
    ax.scatter(bg["longitude"], bg["latitude"], s=4, c="#d9d9d9", alpha=0.6)

    # 2) smaller unlabelled structured segments in a muted neutral
    other = df[(df[COMM_COL] != background_id) & (~df[COMM_COL].isin(legend_ids))]
    ax.scatter(other["longitude"], other["latitude"], s=4, c="#9e9e9e", alpha=0.5)

    # 3) the eight named segments, each in its own colour + legend entry
    summ_idx = summ.set_index(COMM_COL)
    background_price = int(summ_idx.loc[background_id, "median_price_w"])
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d9d9d9",
            markersize=8,
            label=f"C{background_id}: citywide background (${background_price})",
        )
    ]
    for cid in legend_ids:
        sub = df[df[COMM_COL] == cid]
        ax.scatter(sub["longitude"], sub["latitude"], s=10,
                   color=colour_map[cid], alpha=0.9)
        r = summ_idx.loc[cid]
        lbl = f"C{cid}: {short_name(r['dominant_neighbourhood'])} (${int(r['median_price_w'])})"
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=colour_map[cid], markersize=8,
                                  label=lbl))

    ax.set_title("Graph C Louvain communities: Toronto Airbnb market segments")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9,
              title="Largest market segments (median price)", title_fontsize=8)
    fig.tight_layout()
    out = fig_dir / "graph_c_louvain_community_map_labelled.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Wrote", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the labelled Graph C Louvain community map."
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("results/tables/graph_c_louvain_assignments.csv"),
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
    make_labelled_map(args.assignments, args.summary, args.fig_dir)


if __name__ == "__main__":
    main()
