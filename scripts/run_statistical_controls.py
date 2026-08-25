#!/usr/bin/env python3
"""Run the statistical controls that replace the two weak headline statistics.

Outputs
-------
results/tables/alignment_controls.csv
    NMI and AMI for the detected partition against (a) raw official
    neighbourhoods, (b) granularity-matched super-regions, plus a random-partition
    floor and a KMeans-geography reference at the same cluster count.

results/tables/price_model_paired_tests.csv
    Paired fold-level tests (t-interval, Wilcoxon, exact sign test, Cohen's d) for
    R-squared, MAE and RMSE under each validation scheme.

results/metrics/permutation_null_delta_r2.json
    Null distribution of the mean out-of-sample delta R-squared from the community
    feature, under a size-matched shuffled null and (optionally) a KMeans spatial
    null, with an empirical p-value.

Usage
-----
    PYTHONPATH=src python scripts/run_statistical_controls.py \
        --assignment results/tables/graph_c_louvain_assignment.csv \
        --permutations 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import airbnb_final_experiments as core  # noqa: E402
import robustness_analysis as robustness  # noqa: E402
import statistical_controls as controls  # noqa: E402

COMMUNITY_COLUMN = "graph_c_louvain_community"


def load_assignment(csv_path: Path, assignment_path: Path) -> pd.DataFrame:
    """Attach the canonical per-listing community assignment to the cleaned data.

    Community detection is *not* re-run here. Louvain's exact partition is
    seed- and version-sensitive (see `src/community_stability.py`), so the
    assignment is treated as a committed artifact and joined by listing id. This
    is what makes every downstream table reproducible.
    """
    df = core.load_clean_data(csv_path, impute_numeric=False)

    if not assignment_path.exists():
        raise SystemExit(
            f"Missing canonical assignment: {assignment_path}\n"
            "Generate it once with:\n"
            "    PYTHONPATH=src python src/community_stability.py \\\n"
            "        --csv data/toronto_listings_clean.csv \\\n"
            "        --out-dir results/tables --write-assignment"
        )

    assignment = pd.read_csv(assignment_path)
    if COMMUNITY_COLUMN not in assignment.columns:
        candidates = [c for c in assignment.columns if c != "id"]
        if len(candidates) != 1:
            raise SystemExit(
                f"Cannot infer the community column in {assignment_path.name}; "
                f"columns are {list(assignment.columns)}"
            )
        assignment = assignment.rename(columns={candidates[0]: COMMUNITY_COLUMN})

    merged = df.merge(assignment[["id", COMMUNITY_COLUMN]], on="id", how="inner")
    if len(merged) != len(df):
        print(
            f"  warning: {len(df) - len(merged):,} listings had no community "
            "assignment and were dropped"
        )
    merged[COMMUNITY_COLUMN] = merged[COMMUNITY_COLUMN].astype(str)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/toronto_listings_clean.csv")
    parser.add_argument(
        "--assignment", default="results/tables/graph_c_louvain_assignment.csv"
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument(
        "--nulls",
        default="shuffled,spatial",
        help="comma-separated subset of {shuffled,spatial}",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    tables = REPO / "results" / "tables"
    metrics = REPO / "results" / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    print("Loading data and canonical community assignment")
    df = load_assignment(REPO / args.csv, REPO / args.assignment)
    print(f"  {len(df):,} listings, {df[COMMUNITY_COLUMN].nunique()} communities")

    # ---------------------------------------------------------------- claim 2
    print("\nAlignment controls (AMI, granularity-matched ground truth, floor)")
    report = controls.alignment_report(df, df[COMMUNITY_COLUMN].to_numpy())
    for key, value in report.items():
        print(f"  {key:32s} {value}")
    pd.DataFrame([report]).to_csv(tables / "alignment_controls.csv", index=False)

    # ---------------------------------------------------------------- claim 1
    print("\nCross-validating the price models")
    cv_results, _, _ = robustness.evaluate_price_cross_validation(
        df,
        COMMUNITY_COLUMN,
        out_dir=tables / "controlled",
        fig_dir=REPO / "results" / "figures" / "controlled",
        n_splits=args.n_splits,
    )

    print("\nPaired fold-level significance tests")
    paired = controls.paired_tests_from_cv_results(cv_results)
    paired.to_csv(tables / "price_model_paired_tests.csv", index=False)
    print(
        paired[
            [
                "validation_scheme",
                "metric",
                "mean_delta",
                "ci95_low",
                "ci95_high",
                "sign_test_wins",
                "sign_test_pvalue",
            ]
        ].to_string(index=False)
    )

    print("\nPermutation null for the community feature")
    work, numeric_features, categorical_features = robustness.prepare_price_frame(
        df, COMMUNITY_COLUMN
    )
    schemes, _ = robustness.make_validation_splits(work, n_splits=args.n_splits)
    splits = schemes["host_grouped_5fold"]

    payload = {}
    for null in [n.strip() for n in args.nulls.split(",") if n.strip()]:
        print(f"  running {null} null with {args.permutations} permutations")
        payload[null] = controls.permutation_null_delta_r2(
            work,
            COMMUNITY_COLUMN,
            splits,
            numeric_features,
            categorical_features,
            robustness.make_price_pipeline,
            n_permutations=args.permutations,
            null=null,
        )
        result = payload[null]
        print(
            f"    observed delta R2 {result['observed_delta_r2']:+.5f} | "
            f"null mean {result['null_delta_r2_mean']:+.5f} "
            f"(p95 {result['null_delta_r2_p95']:+.5f}) | "
            f"p = {result['p_value']:.4f}"
        )

    payload["_note"] = (
        "Splits are host-grouped 5-fold. The observed statistic is the mean "
        "out-of-sample delta R-squared from adding the network community label. "
        "p is the empirical exceedance rate of the null distribution, with the "
        "usual +1 correction."
    )
    controls.write_json(payload, metrics / "permutation_null_delta_r2.json")

    print("\nWrote:")
    for path in (
        tables / "alignment_controls.csv",
        tables / "price_model_paired_tests.csv",
        tables / "controlled" / "price_model_cv_results.csv",
        metrics / "permutation_null_delta_r2.json",
    ):
        print(f"  {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
