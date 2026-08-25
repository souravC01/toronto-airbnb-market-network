#!/usr/bin/env python3
"""Seed-stability control for Graph C community detection.

The parameter-sensitivity suite in :mod:`robustness_analysis` varies the spatial
radius, the attribute neighbour count, and the edge-weight profile. On its own
that design cannot distinguish *parameter* sensitivity from *algorithmic*
sensitivity, because Louvain is a stochastic, order-dependent heuristic and the
baseline partition is computed from a single seed.

This module supplies the missing control. It builds the baseline Graph C once,
runs Louvain under many seeds, and reports:

* the spread of community counts and modularity across seeds,
* the mean pairwise agreement between seeds (NMI and AMI),
* agreement between each seed and the canonical seed.

Interpretation rule: a parameter change is only evidence of genuine structural
sensitivity if its agreement with the baseline falls clearly *below* the
seed-to-seed agreement band reported here. Anything inside that band is noise.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

import airbnb_final_experiments as core


DEFAULT_SEEDS = tuple(range(42, 62))


def build_baseline_graph_c(df: pd.DataFrame) -> nx.Graph:
    """Build the canonical Graph C exactly as the baseline configuration does."""
    config = core.GraphConfig(
        "Graph C: Spatial + shared host + attribute similarity",
        True,
        True,
        True,
    )
    return core.build_graph_variant(
        df=df,
        config=config,
        radius_meters=500,
        attribute_k=5,
        alpha_spatial=0.60,
        alpha_host=0.25,
        alpha_attribute=0.15,
    )


def louvain_labels(graph: nx.Graph, order: Sequence, seed: int):
    """Run Louvain at one seed and return (labels in ``order``, count, modularity)."""
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    node_to_community = {
        node: cid for cid, nodes in enumerate(communities) for node in nodes
    }
    modularity = nx.community.modularity(graph, communities, weight="weight")
    labels = np.asarray([node_to_community[node] for node in order], dtype=int)
    return labels, len(communities), float(modularity)


def run_seed_stability(
    df: pd.DataFrame,
    out_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    write_assignment: bool = False,
    canonical_seed: int = core.RANDOM_STATE,
) -> Dict[str, object]:
    """Run Louvain across ``seeds`` on one fixed Graph C and summarise agreement."""
    graph = build_baseline_graph_c(df)
    order = df["id"].to_numpy()
    official = df["neighbourhood_cleansed"].astype(str).to_numpy()

    label_sets: Dict[int, np.ndarray] = {}
    rows: List[dict] = []

    for seed in seeds:
        labels, count, modularity = louvain_labels(graph, order, seed)
        label_sets[seed] = labels
        sizes = np.bincount(labels)
        rows.append(
            {
                "seed": seed,
                "communities": count,
                "modularity": modularity,
                "largest_community": int(sizes.max()),
                "median_community_size": float(np.median(sizes)),
                "nmi_vs_neighbourhood": normalized_mutual_info_score(official, labels),
                "ami_vs_neighbourhood": adjusted_mutual_info_score(official, labels),
            }
        )
        print(
            f"  seed={seed:<4} communities={count:<4} Q={modularity:.6f} "
            f"largest={int(sizes.max()):,}"
        )

    per_seed = pd.DataFrame(rows)

    if canonical_seed in label_sets:
        reference = label_sets[canonical_seed]
        per_seed["nmi_vs_canonical_seed"] = [
            normalized_mutual_info_score(reference, label_sets[seed]) for seed in per_seed["seed"]
        ]
        per_seed["ami_vs_canonical_seed"] = [
            adjusted_mutual_info_score(reference, label_sets[seed]) for seed in per_seed["seed"]
        ]

    pair_rows = []
    for left, right in itertools.combinations(sorted(label_sets), 2):
        pair_rows.append(
            {
                "seed_a": left,
                "seed_b": right,
                "nmi": normalized_mutual_info_score(label_sets[left], label_sets[right]),
                "ami": adjusted_mutual_info_score(label_sets[left], label_sets[right]),
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    summary = {
        "seeds": len(label_sets),
        "communities_min": int(per_seed["communities"].min()),
        "communities_max": int(per_seed["communities"].max()),
        "communities_median": float(per_seed["communities"].median()),
        "modularity_min": float(per_seed["modularity"].min()),
        "modularity_max": float(per_seed["modularity"].max()),
        "modularity_std": float(per_seed["modularity"].std()),
        "largest_community_min": int(per_seed["largest_community"].min()),
        "largest_community_max": int(per_seed["largest_community"].max()),
        "pairwise_nmi_mean": float(pairwise["nmi"].mean()),
        "pairwise_nmi_min": float(pairwise["nmi"].min()),
        "pairwise_nmi_p05": float(pairwise["nmi"].quantile(0.05)),
        "pairwise_ami_mean": float(pairwise["ami"].mean()),
        "pairwise_ami_min": float(pairwise["ami"].min()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if write_assignment:
        # The canonical partition is committed as an artifact rather than
        # re-derived on every run. Louvain's exact output depends on the seed AND
        # on library versions, so passing `seed=42` is not enough to make the
        # partition reproducible across environments. Every downstream table is
        # keyed to this file.
        if canonical_seed not in label_sets:
            raise ValueError(
                f"canonical_seed={canonical_seed} was not among the seeds that ran"
            )
        pd.DataFrame(
            {
                "id": df["id"].to_numpy(),
                "graph_c_louvain_community": label_sets[canonical_seed],
            }
        ).to_csv(out_dir / "graph_c_louvain_assignment.csv", index=False)
        print(
            f"\nWrote canonical assignment (seed={canonical_seed}) to "
            f"{out_dir / 'graph_c_louvain_assignment.csv'}"
        )

    per_seed.to_csv(out_dir / "community_seed_stability.csv", index=False)
    pairwise.to_csv(out_dir / "community_seed_pairwise_agreement.csv", index=False)
    (out_dir / "community_seed_stability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nSeed-stability summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(
        "\nRead the parameter-sensitivity NMI column against pairwise_nmi_p05 "
        f"({summary['pairwise_nmi_p05']:.4f}). Configurations above that value are "
        "indistinguishable from seed noise."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Louvain seed-stability control for Graph C.")
    parser.add_argument("--csv", default="data/toronto_listings_clean.csv")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--seeds", type=int, default=20, help="number of consecutive seeds from 42")
    parser.add_argument(
        "--write-assignment",
        action="store_true",
        help="also write the canonical per-listing partition (first seed) as an artifact",
    )
    args = parser.parse_args()

    df = core.load_clean_data(Path(args.csv))
    seeds = tuple(range(core.RANDOM_STATE, core.RANDOM_STATE + args.seeds))
    run_seed_stability(
        df,
        Path(args.out_dir),
        seeds=seeds,
        write_assignment=args.write_assignment,
    )


if __name__ == "__main__":
    main()
