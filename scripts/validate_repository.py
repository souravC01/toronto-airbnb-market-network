#!/usr/bin/env python3
"""Validate committed results, figures, and portfolio artifacts."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"

GRAPHS = {
    "Graph A: Spatial only",
    "Graph B: Spatial + shared host",
    "Graph C: Spatial + shared host + attribute similarity",
}
ALGORITHMS = {"Louvain", "Leiden"}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def read_table(name: str, errors: list[str]) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        errors.append(f"Missing table: {path.relative_to(REPO)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_pairs(df: pd.DataFrame, label: str, errors: list[str]) -> None:
    if not {"graph", "algorithm"}.issubset(df.columns):
        errors.append(f"{label} must contain graph and algorithm columns")
        return
    expected = {(graph, algorithm) for graph in GRAPHS for algorithm in ALGORITHMS}
    actual = set(zip(df["graph"], df["algorithm"]))
    require(actual == expected, f"{label} graph/algorithm coverage is inconsistent", errors)


def validate_no_frontend_drift(errors: list[str]) -> None:
    """Fail if the website's numbers no longer match ``results/tables/*.csv``.

    This is the check whose absence let the published page drift away from the
    pipeline. The previous version of this script verified that files existed and
    that tables had the expected row counts, but it never compared a single value
    rendered by the site against the CSV it was supposed to come from. As a
    result the live page showed a community table and a set of dollar-error
    figures that the pipeline had never produced.

    ``scripts/export_portfolio_data.py --check`` regenerates the JSON the page
    imports and diffs it against the committed copy, so any future pipeline
    change that is not propagated to the site fails CI instead of shipping.
    """
    exporter = REPO / "scripts" / "export_portfolio_data.py"
    if not exporter.exists():
        errors.append("Missing scripts/export_portfolio_data.py")
        return

    generated = REPO / "portfolio" / "app" / "data" / "generated.json"
    if not generated.exists():
        errors.append(
            "Missing portfolio/app/data/generated.json. Run "
            "`python scripts/export_portfolio_data.py`."
        )
        return

    completed = subprocess.run(
        [sys.executable, str(exporter), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        errors.append(
            "Website data has drifted from results/tables/*.csv. Run "
            "`python scripts/export_portfolio_data.py` and commit the result.\n"
            + "\n".join(f"      {line}" for line in detail.splitlines()[:20])
        )


def validate_no_hardcoded_numbers(errors: list[str]) -> None:
    """Fail if pipeline numbers are written by hand in the page component.

    The page must read every figure from ``generated.json``. A bare dollar amount
    with cents, or a four-decimal metric, is a pipeline output; finding one in the
    TSX means the drift has started again.
    """
    component = REPO / "portfolio" / "app" / "portfolio-experience.tsx"
    if not component.exists():
        errors.append("Missing portfolio/app/portfolio-experience.tsx")
        return

    dollars = re.compile(r"\$\d+\.\d{2}")
    metrics = re.compile(r"(?<![\w.])\d\.\d{4}(?![\w])")

    offenders: list[str] = []
    for number, line in enumerate(component.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue
        if dollars.search(stripped) or metrics.search(stripped):
            offenders.append(f"{number}: {stripped[:90]}")

    if offenders:
        errors.append(
            "portfolio-experience.tsx contains hardcoded pipeline numbers; read "
            "them from generated.json instead:\n"
            + "\n".join(f"      {entry}" for entry in offenders[:15])
        )

def validate_portfolio(errors: list[str]) -> None:
    required_files = {
        "portfolio/app/data/generated.json",
        "portfolio/app/globals.css",
        "portfolio/app/portfolio-experience.tsx",
        "portfolio/entry-client.tsx",
        "portfolio/entry-server.tsx",
        "portfolio/index.html",
        "portfolio/package.json",
        "portfolio/.node-version",
        "portfolio/pnpm-lock.yaml",
        "portfolio/pnpm-workspace.yaml",
        "portfolio/postcss.config.mjs",
        "portfolio/public/og.png",
        "portfolio/public/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf",
        "portfolio/scripts/prerender.mjs",
        "portfolio/tests/rendered-html.test.mjs",
        "portfolio/vite.config.ts",
    }
    for relative_path in sorted(required_files):
        path = REPO / relative_path
        require(path.exists(), f"Missing portfolio artifact: {relative_path}", errors)
        if path.exists() and path.is_file():
            require(path.stat().st_size > 0, f"Portfolio artifact is empty: {relative_path}", errors)

    original_report = REPO / "portfolio" / "public" / "report" / "EECS4414-Airbnb-Network-Analysis-Final-Report.pdf"
    if original_report.exists():
        require(
            hashlib.sha256(original_report.read_bytes()).hexdigest()
            == "0ad7327026e1bd875f4d61181b5c6db433a95f602d4cfc6dfbf5bea488284201",
            "Website report must be the unchanged original final-submission PDF",
            errors,
        )


def main() -> None:
    errors: list[str] = []

    graph_stats = read_table("graph_stats.csv", errors)
    communities = read_table("community_results.csv", errors)
    alignment = read_table("alignment_results.csv", errors)
    price = read_table("price_model_results.csv", errors)
    sensitivity = read_table("parameter_sensitivity_results.csv", errors)
    cv_results = read_table("price_model_cv_results.csv", errors)
    cv_summary = read_table("price_model_cv_summary.csv", errors)
    cv_deltas = read_table("price_model_cv_deltas.csv", errors)

    if not graph_stats.empty:
        require(set(graph_stats.get("graph", [])) == GRAPHS, "graph_stats.csv must contain all three graphs", errors)
    if not communities.empty:
        validate_pairs(communities, "community_results.csv", errors)
    if not alignment.empty:
        validate_pairs(alignment, "alignment_results.csv", errors)
    if not price.empty:
        require(len(price) == 2, "price_model_results.csv must contain two model rows", errors)
    if not sensitivity.empty:
        require(
            len(sensitivity) == 7,
            "parameter_sensitivity_results.csv must contain seven configurations",
            errors,
        )
        require(
            sensitivity.iloc[0].get("configuration") == "Baseline",
            "The sensitivity table must start with the baseline configuration",
            errors,
        )
    if not cv_results.empty:
        cv_columns = {"validation_scheme", "fold", "model", "host_overlap_count"}
        require(
            cv_columns.issubset(cv_results.columns),
            "price_model_cv_results.csv is missing required audit columns",
            errors,
        )
        require(
            len(cv_results) == 30,
            "price_model_cv_results.csv must contain 3 schemes x 5 folds x 2 models",
            errors,
        )
        if cv_columns.issubset(cv_results.columns):
            grouped = cv_results[
                cv_results["validation_scheme"] == "host_grouped_5fold"
            ]
            require(
                not grouped.empty and grouped["host_overlap_count"].max() == 0,
                "Host-grouped folds must have zero train/test host overlap",
                errors,
            )
    if not cv_summary.empty:
        require(len(cv_summary) == 6, "price_model_cv_summary.csv must contain six rows", errors)
    if not cv_deltas.empty:
        require(len(cv_deltas) == 3, "price_model_cv_deltas.csv must contain three rows", errors)

    required_figures = {
        "alignment_nmi_comparison.png",
        "alignment_vi_comparison.png",
        "community_characterisation.png",
        "community_concentration.png",
        "community_profile_bubble.png",
        "graph_c_louvain_community_map.png",
        "graph_c_louvain_community_map_labelled.png",
        "graph_c_leiden_community_map.png",
        "parameter_sensitivity.png",
        "price_model_cv_comparison.png",
        "price_model_cv_spatial_blocks.png",
    }
    figure_dir = REPO / "results" / "figures"
    for name in sorted(required_figures):
        require((figure_dir / name).exists(), f"Missing figure: results/figures/{name}", errors)

    validate_portfolio(errors)
    validate_no_frontend_drift(errors)
    validate_no_hardcoded_numbers(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(
        "Repository validation passed: tables, algorithm coverage, figures, "
        "and portfolio artifacts are consistent."
    )


if __name__ == "__main__":
    main()
