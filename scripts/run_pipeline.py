#!/usr/bin/env python3
"""Run the complete reproducible analysis and visualization pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, "-u", *args]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    matplotlib_cache = REPO / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    run(
        "src/airbnb_final_experiments.py",
        "--csv",
        "data/toronto_listings_clean.csv",
        "--out-dir",
        "results/tables",
        "--fig-dir",
        "results/figures",
        "--run-leiden",
    )
    run("src/make_alignment_plots.py")
    run("src/make_community_clarity_figures.py")
    run("src/make_labelled_community_map.py")
    run("src/robustness_analysis.py")
    run("scripts/validate_repository.py")

    print("\nPipeline complete. Canonical tables, robustness checks, and figures are in results/.")


if __name__ == "__main__":
    main()
