#!/usr/bin/env python3
"""Derive every number shown in the portfolio front end from the canonical CSVs.

Why this exists
---------------
The front end used to hardcode its own copies of the community table, the
cross-validation figures, and the sensitivity grid. Those copies drifted from
``results/tables/*.csv``: the published community table disagreed with the
pipeline on listing counts, median prices, dominant neighbourhoods and
neighbourhood spans, and the published dollar-MAE levels were wrong for all
three validation schemes (with the spatial-block ordering inverted).

For a project whose headline claim is reproducibility, that class of bug has to
be impossible rather than merely unlikely. So the presentation layer now reads a
single generated artifact, and CI fails if the artifact is stale.

Usage
-----
    python scripts/export_portfolio_data.py            # write the artifact
    python scripts/export_portfolio_data.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
OUT_PATH = REPO / "portfolio" / "app" / "data" / "generated.json"

GRAPH_A = "Graph A: Spatial only"
GRAPH_B = "Graph B: Spatial + shared host"
GRAPH_C = "Graph C: Spatial + shared host + attribute similarity"

GRAPH_PRESENTATION = {
    GRAPH_A: {
        "name": "Spatial only",
        "relationships": ["Within 500 m geographic radius"],
        "takeaway": (
            "Proximity alone splits the city into many disconnected pockets that "
            "track official neighbourhood boundaries closely."
        ),
    },
    GRAPH_B: {
        "name": "Spatial + shared host",
        "relationships": [
            "Within 500 m geographic radius",
            "Same-host nearest neighbours (k=5)",
        ],
        "takeaway": (
            "Multi-listing hosts bridge nearby pockets, roughly halving the number "
            "of disconnected components."
        ),
    },
    GRAPH_C: {
        "name": "Full market network",
        "relationships": [
            "Within 500 m geographic radius",
            "Same-host nearest neighbours (k=5)",
            "Listing attribute cosine similarity (k=5)",
        ],
        "takeaway": (
            "Attribute-similarity links connect the entire city into one component "
            "and a small number of broad segments that cross administrative lines."
        ),
    },
}

SCHEME_PRESENTATION = {
    "random_5fold": {
        "id": "random",
        "label": "Random folds",
        "fullName": "Random 5-Fold Cross-Validation",
    },
    "host_grouped_5fold": {
        "id": "host",
        "label": "Host-grouped",
        "fullName": "Host-Grouped 5-Fold Cross-Validation",
    },
    "spatial_block_5fold": {
        "id": "spatial",
        "label": "Spatial blocks",
        "fullName": "Spatial-Block 5-Fold Cross-Validation",
    },
}

BASELINE_MODEL = "Baseline: listing + official neighbourhood"
EXPANDED_MODEL = "Expanded: baseline + network community"

REQUIRED_COMMUNITY_COLUMNS = {
    "listings",
    "median_price_w",
    "dominant_neighbourhood",
    "dominant_room_type",
    "unique_neighbourhoods",
}

# Emitted by the patched `save_community_summary`. Older tables predate them, so
# the room-type label degrades to a bare type instead of failing the export.
OPTIONAL_COMMUNITY_COLUMNS = {
    "dominant_room_type_share",
    "dominant_neighbourhood_share",
}


def official_neighbourhood_count() -> int:
    """Number of distinct official neighbourhoods represented in the dataset."""
    path = REPO / "data" / "toronto_listings_clean.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing cleaned dataset: {path.relative_to(REPO)}\n"
            "Run `python src/clean_dataset.py` first."
        )
    column = pd.read_csv(path, usecols=["neighbourhood_cleansed"])
    return int(column["neighbourhood_cleansed"].nunique())


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise SystemExit(
            f"Missing canonical table: {path.relative_to(REPO)}\n"
            "Run `python scripts/run_pipeline.py` first."
        )
    return pd.read_csv(path)


def signed(value: float, digits: int) -> str:
    """Format a delta with an explicit sign and a real minus glyph."""
    text = f"{abs(value):.{digits}f}"
    return f"+{text}" if value >= 0 else f"\u2212{text}"


def signed_dollars(value: float) -> str:
    text = f"${abs(value):,.2f}"
    return f"+{text}" if value >= 0 else f"\u2212{text}"


def build_graphs(graph_stats: pd.DataFrame, community: pd.DataFrame, alignment: pd.DataFrame) -> List[Dict]:
    stats = graph_stats.set_index("graph")
    louvain = community[community["algorithm"] == "Louvain"].set_index("graph")
    align = alignment[alignment["algorithm"] == "Louvain"].set_index("graph")

    graphs = []
    for graph_name in (GRAPH_A, GRAPH_B, GRAPH_C):
        row = stats.loc[graph_name]
        comm = louvain.loc[graph_name]
        ali = align.loc[graph_name]
        graphs.append(
            {
                "id": graph_name.split(":")[0].strip().removeprefix("Graph "),
                "name": GRAPH_PRESENTATION[graph_name]["name"],
                "fullName": graph_name,
                "relationships": GRAPH_PRESENTATION[graph_name]["relationships"],
                "nodes": f"{int(row['nodes']):,}",
                "edges": f"{int(row['edges']):,}",
                "components": int(row["connected_components"]),
                "largestComp": f"{float(row['largest_component_fraction']) * 100:.1f}%",
                "avgDegree": f"{float(row['average_degree']):.1f}",
                "modularity": f"{float(comm['modularity']):.4f}",
                "communities": int(comm["communities"]),
                "nmi": round(float(ali["nmi_vs_neighbourhood"]), 4),
                "vi": f"{float(ali['vi_vs_neighbourhood']):.3f}",
                "takeaway": GRAPH_PRESENTATION[graph_name]["takeaway"],
            }
        )
    return graphs


def build_algorithms(community: pd.DataFrame, alignment: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for algorithm in ("Louvain", "Leiden"):
        comm = community[
            (community["graph"] == GRAPH_C) & (community["algorithm"] == algorithm)
        ]
        ali = alignment[
            (alignment["graph"] == GRAPH_C) & (alignment["algorithm"] == algorithm)
        ]
        if comm.empty or ali.empty:
            continue
        comm_row = comm.iloc[0]
        ali_row = ali.iloc[0]
        out[algorithm] = {
            "communities": int(comm_row["communities"]),
            "modularity": round(float(comm_row["modularity"]), 4),
            "largest": f"{int(comm_row['largest_community']):,}",
            "medianSize": f"{float(comm_row['median_community_size']):g}",
            "nmi": round(float(ali_row["nmi_vs_neighbourhood"]), 4),
            "vi": round(float(ali_row["vi_vs_neighbourhood"]), 4),
            "image": f"/figures/graph_c_{algorithm.lower()}_community_map.png",
        }
    return out


def build_top_communities(summary: pd.DataFrame, limit: int = 8) -> List[Dict]:
    missing = REQUIRED_COMMUNITY_COLUMNS - set(summary.columns)
    if missing:
        raise SystemExit(
            "graph_c_louvain_community_summary.csv is missing columns "
            f"{sorted(missing)}.\nRe-run `python scripts/run_pipeline.py`."
        )
    has_room_share = "dominant_room_type_share" in summary.columns

    community_col = summary.columns[0]
    ordered = summary.sort_values("listings", ascending=False).head(limit)

    rows = []
    for _, row in ordered.iterrows():
        span = int(row["unique_neighbourhoods"])
        rows.append(
            {
                "id": f"C{int(row[community_col])}",
                "size": f"{int(row['listings']):,}",
                "price": f"${float(row['median_price_w']):,.0f}",
                "dominantNeighbourhood": str(row["dominant_neighbourhood"]),
                "roomType": (
                    f"{row['dominant_room_type']} "
                    f"({float(row['dominant_room_type_share']) * 100:.0f}%)"
                    if has_room_share
                    else str(row["dominant_room_type"])
                ),
                "span": f"{span:,}",
            }
        )
    return rows


def build_validation(summary: pd.DataFrame, deltas: pd.DataFrame) -> List[Dict]:
    indexed = summary.set_index(["validation_scheme", "model"])
    delta_indexed = deltas.set_index("validation_scheme")

    schemes = []
    for scheme, presentation in SCHEME_PRESENTATION.items():
        base = indexed.loc[(scheme, BASELINE_MODEL)]
        expanded = indexed.loc[(scheme, EXPANDED_MODEL)]
        delta = delta_indexed.loc[scheme]
        wins = int(delta["expanded_r2_wins"])
        folds = int(delta["folds"])

        schemes.append(
            {
                **presentation,
                "baseline": round(float(base["r2_mean"]), 4),
                "expanded": round(float(expanded["r2_mean"]), 4),
                "delta": signed(float(delta["r2_delta_mean"]), 4),
                "adjustedDelta": signed(float(delta["adjusted_r2_delta_mean"]), 4),
                "baseMae": f"${float(base['mae_dollars_mean']):,.2f}",
                "expandedMae": f"${float(expanded['mae_dollars_mean']):,.2f}",
                "maeDelta": signed_dollars(float(delta["mae_dollars_delta_mean"])),
                "wins": f"{wins} / {folds}",
            }
        )
    return schemes


def build_sensitivity(sensitivity: pd.DataFrame) -> List[Dict]:
    rows = []
    for _, row in sensitivity.iterrows():
        label = str(row["configuration"]).replace(" weights", "")
        rows.append(
            {
                "id": label.lower().replace(" ", "-").replace("=", "").replace("--", "-"),
                "label": label,
                "radius": f"{int(row['radius_meters'])} m",
                "k": str(int(row["attribute_k"])),
                "weights": (
                    f"{float(row['alpha_spatial']):.2f} / "
                    f"{float(row['alpha_host']):.2f} / "
                    f"{float(row['alpha_attribute']):.2f}"
                ),
                "edges": f"{int(row['edges']):,}",
                "communities": int(row["communities"]),
                "modularity": round(float(row["modularity"]), 4),
                "stability": round(float(row["nmi_vs_baseline"]), 4),
                "neighbourhood": round(float(row["nmi_vs_neighbourhood"]), 4),
            }
        )
    return rows


def build_payload() -> Dict:
    graph_stats = read_table("graph_stats.csv")
    community = read_table("community_results.csv")
    alignment = read_table("alignment_results.csv")
    louvain_summary = read_table("graph_c_louvain_community_summary.csv")
    cv_summary = read_table("price_model_cv_summary.csv")
    cv_deltas = read_table("price_model_cv_deltas.csv")
    sensitivity = read_table("parameter_sensitivity_results.csv")

    graph_c_stats = graph_stats.set_index("graph").loc[GRAPH_C]
    graph_c_louvain = community[
        (community["graph"] == GRAPH_C) & (community["algorithm"] == "Louvain")
    ].iloc[0]

    payload = {
        "_generated_by": "scripts/export_portfolio_data.py",
        "_do_not_edit": (
            "Every value here is derived from results/tables/*.csv. Edit the "
            "pipeline, re-run it, then regenerate this file. CI fails on drift."
        ),
        "_source_digests": source_digests(),
        "headline": {
            "listings": f"{int(graph_c_stats['nodes']):,}",
            # Distinct official neighbourhoods present in the cleaned dataset
            # (140). The largest community's `unique_neighbourhoods` value (114)
            # is a different quantity and must not be substituted for it.
            "officialAreas": official_neighbourhood_count(),
            "widestCommunitySpan": int(louvain_summary["unique_neighbourhoods"].max()),
            "communities": int(graph_c_louvain["communities"]),
            "components": int(graph_c_stats["connected_components"]),
            "componentsSpatialOnly": int(
                graph_stats.set_index("graph").loc[GRAPH_A, "connected_components"]
            ),
        },
        "graphs": build_graphs(graph_stats, community, alignment),
        "algorithms": build_algorithms(community, alignment),
        "topCommunities": build_top_communities(louvain_summary),
        "validationSchemes": build_validation(cv_summary, cv_deltas),
        "sensitivitySettings": build_sensitivity(sensitivity),
    }

    optional = TABLES / "community_seed_stability_summary.json"
    if optional.exists():
        payload["seedStability"] = json.loads(optional.read_text())

    optional_null = TABLES / "community_permutation_null.json"
    if optional_null.exists():
        payload["permutationNull"] = json.loads(optional_null.read_text())

    return payload


def source_digests() -> Dict[str, str]:
    digests = {}
    for path in sorted(TABLES.glob("*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        digests[path.name] = digest
    return digests


def render(payload: Dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed artifact differs from a fresh export",
    )
    args = parser.parse_args()

    rendered = render(build_payload())

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL {OUT_PATH.relative_to(REPO)} does not exist.", file=sys.stderr)
            sys.exit(1)
        if OUT_PATH.read_text() != rendered:
            print(
                f"FAIL {OUT_PATH.relative_to(REPO)} is stale.\n"
                "The portfolio no longer matches results/tables/*.csv.\n"
                "Run: python scripts/export_portfolio_data.py",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK   {OUT_PATH.relative_to(REPO)} matches results/tables/*.csv")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rendered)
    print(f"Wrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
