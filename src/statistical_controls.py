#!/usr/bin/env python3
"""Statistical controls for the two headline claims.

The submitted analysis established two things and reported them honestly, but it
supported each with a statistic that cannot carry the weight placed on it.

Claim 1 — "community membership does not materially improve price prediction"
    was supported by a *decline in adjusted R-squared*. That statistic is
    computed by penalising an out-of-sample R-squared with the training design
    matrix's column count. With n about 3,162 test rows and p moving from 209 to
    226, the decline is fixed in advance:

        (n-1)/(n-p-1):  3161/2952 = 1.0708  ->  3161/2935 = 1.0770
        delta_adj ~= -(1-R^2)(0.0062) + 1.077 * delta_R^2
                  ~= -(0.3754)(0.0062) + 1.077(0.0016) = -0.0006

    Any 17-level categorical produces the same value, including a column of
    random labels. The observed -0.00065 is therefore arithmetic, not evidence.

    The honest replacement is a permutation null. If the real partition beats a
    size-matched random partition, the gain is real but small, which is a
    stronger and more defensible finding than "the penalty ate it".

Claim 2 — "detected communities cross official neighbourhood boundaries"
    was supported by NMI falling from 0.749 (Graph A) to 0.518 (Graph C). But
    Graph A yields 134 clusters against 140 official labels while Graph C yields
    17. NMI is not invariant to cluster count, so a coarser partition scores
    lower against fine-grained ground truth almost by construction. The measured
    drop conflates "crosses boundaries" with "is coarser".

    The honest replacements are AMI (chance-corrected), a granularity-matched
    ground truth, and a random-partition floor so readers can see the zero point.

Both controls are cheap and both make the project's conclusions harder to attack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    r2_score,
)


# --------------------------------------------------------------------------
# Claim 2: granularity-fair alignment measurement
# --------------------------------------------------------------------------


def aggregate_neighbourhoods(
    frame: pd.DataFrame,
    n_regions: int,
    neighbourhood_column: str = "neighbourhood_cleansed",
    random_state: int = 42,
) -> np.ndarray:
    """Collapse official neighbourhoods into ``n_regions`` contiguous super-regions.

    Ward linkage over neighbourhood centroids. This produces a ground truth at
    the *same granularity* as the detected partition, so NMI against it measures
    boundary disagreement rather than a difference in cluster count.
    """
    centroids = (
        frame.groupby(neighbourhood_column)[["latitude", "longitude"]]
        .mean()
        .sort_index()
    )
    n_regions = min(n_regions, len(centroids))
    labels = AgglomerativeClustering(n_clusters=n_regions, linkage="ward").fit_predict(
        centroids.to_numpy()
    )
    mapping = dict(zip(centroids.index, labels))
    return frame[neighbourhood_column].map(mapping).to_numpy()


def random_partition(
    n_rows: int,
    sizes: Sequence[int],
    random_state: int = 42,
) -> np.ndarray:
    """A random partition with the same size distribution as an observed one."""
    rng = np.random.default_rng(random_state)
    labels = np.concatenate([np.full(size, index) for index, size in enumerate(sizes)])
    if len(labels) != n_rows:
        raise ValueError("Community sizes must sum to the number of rows")
    rng.shuffle(labels)
    return labels


def spatial_partition(
    frame: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> np.ndarray:
    """A purely geographic partition: KMeans on coordinates, ``n_clusters`` groups.

    ``n_init`` defaults to 20 for the reported reference partition. Permutation
    loops lower it, since a null draw does not need a well-optimised solution and
    the extra restarts dominate the runtime of the whole test.
    """
    coordinates = frame[["latitude", "longitude"]].to_numpy(dtype=float)
    return KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=n_init
    ).fit_predict(coordinates)


def alignment_report(
    frame: pd.DataFrame,
    detected_labels: np.ndarray,
    neighbourhood_column: str = "neighbourhood_cleansed",
    random_state: int = 42,
) -> Dict[str, float]:
    """Report chance-corrected and granularity-matched alignment, plus a floor.

    Returns NMI and AMI against the raw official neighbourhoods, against
    granularity-matched super-regions, and for a size-matched random partition.
    """
    official = frame[neighbourhood_column].astype(str).to_numpy()
    n_detected = int(len(np.unique(detected_labels)))
    n_official = int(len(np.unique(official)))

    matched = aggregate_neighbourhoods(
        frame,
        n_regions=n_detected,
        neighbourhood_column=neighbourhood_column,
        random_state=random_state,
    )

    sizes = np.bincount(
        pd.factorize(detected_labels)[0], minlength=n_detected
    ).tolist()
    shuffled = random_partition(len(frame), sizes, random_state=random_state)
    geographic = spatial_partition(frame, n_detected, random_state=random_state)

    return {
        "detected_communities": n_detected,
        "official_neighbourhoods": n_official,
        "nmi_vs_official": float(normalized_mutual_info_score(official, detected_labels)),
        "ami_vs_official": float(adjusted_mutual_info_score(official, detected_labels)),
        "nmi_vs_matched_regions": float(
            normalized_mutual_info_score(matched, detected_labels)
        ),
        "ami_vs_matched_regions": float(
            adjusted_mutual_info_score(matched, detected_labels)
        ),
        "nmi_random_floor": float(normalized_mutual_info_score(official, shuffled)),
        "ami_random_floor": float(adjusted_mutual_info_score(official, shuffled)),
        "nmi_geographic_ceiling": float(
            normalized_mutual_info_score(official, geographic)
        ),
        "ami_geographic_ceiling": float(
            adjusted_mutual_info_score(official, geographic)
        ),
    }


# --------------------------------------------------------------------------
# Claim 1: paired significance and a permutation null
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedTestResult:
    metric: str
    n_folds: int
    mean_delta: float
    std_delta: float
    ci_low: float
    ci_high: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_pvalue: Optional[float]
    sign_test_wins: int
    sign_test_pvalue: float
    cohens_d: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "n_folds": self.n_folds,
            "mean_delta": self.mean_delta,
            "std_delta": self.std_delta,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "t_statistic": self.t_statistic,
            "t_pvalue": self.t_pvalue,
            "wilcoxon_pvalue": self.wilcoxon_pvalue,
            "sign_test_wins": self.sign_test_wins,
            "sign_test_pvalue": self.sign_test_pvalue,
            "cohens_d": self.cohens_d,
        }


def paired_fold_test(
    baseline: Sequence[float],
    expanded: Sequence[float],
    metric: str,
    higher_is_better: bool = True,
) -> PairedTestResult:
    """Paired significance test over cross-validation folds.

    Reports a t-interval, a Wilcoxon signed-rank p-value, an exact sign test, and
    a standardised effect size. Five folds is a small sample, so read the sign
    test and the interval rather than leaning on any single p-value. Folds also
    share training data, which makes these tests anti-conservative; they bound
    consistency, not population significance.
    """
    baseline_array = np.asarray(baseline, dtype=float)
    expanded_array = np.asarray(expanded, dtype=float)
    differences = expanded_array - baseline_array
    if not higher_is_better:
        differences = -differences

    n = len(differences)
    mean = float(differences.mean())
    std = float(differences.std(ddof=1)) if n > 1 else 0.0
    stderr = std / np.sqrt(n) if n > 1 and std > 0 else 0.0

    if stderr > 0:
        critical = stats.t.ppf(0.975, df=n - 1)
        ci_low, ci_high = mean - critical * stderr, mean + critical * stderr
        t_statistic, t_pvalue = stats.ttest_rel(expanded_array, baseline_array)
        t_statistic = float(t_statistic)
        t_pvalue = float(t_pvalue)
    else:
        ci_low = ci_high = mean
        t_statistic, t_pvalue = float("nan"), float("nan")

    try:
        wilcoxon_pvalue = float(
            stats.wilcoxon(differences, alternative="greater", zero_method="zsplit").pvalue
        )
    except ValueError:
        wilcoxon_pvalue = None

    wins = int((differences > 0).sum())
    sign_pvalue = float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)

    return PairedTestResult(
        metric=metric,
        n_folds=n,
        mean_delta=mean,
        std_delta=std,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        t_statistic=t_statistic,
        t_pvalue=t_pvalue,
        wilcoxon_pvalue=wilcoxon_pvalue,
        sign_test_wins=wins,
        sign_test_pvalue=sign_pvalue,
        cohens_d=float(mean / std) if std > 0 else float("nan"),
    )


def paired_tests_from_cv_results(results: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`paired_fold_test` to every validation scheme in a CV table."""
    from robustness_analysis import BASELINE_MODEL, EXPANDED_MODEL

    rows: List[Dict[str, object]] = []
    for scheme, group in results.groupby("validation_scheme", sort=False):
        for metric, higher_is_better in (
            ("test_r2", True),
            ("mae_dollars_approx", False),
            ("rmse_dollars_approx", False),
        ):
            wide = group.pivot(index="fold", columns="model", values=metric)
            outcome = paired_fold_test(
                wide[BASELINE_MODEL].to_numpy(),
                wide[EXPANDED_MODEL].to_numpy(),
                metric=metric,
                higher_is_better=higher_is_better,
            ).to_dict()
            outcome["validation_scheme"] = scheme
            rows.append(outcome)

    frame = pd.DataFrame(rows)
    ordered = ["validation_scheme", "metric"] + [
        column for column in frame.columns if column not in {"validation_scheme", "metric"}
    ]
    return frame[ordered]


def permutation_null_delta_r2(
    work: pd.DataFrame,
    community_column: str,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    make_pipeline,
    n_permutations: int = 200,
    random_state: int = 42,
    null: str = "shuffled",
    progress_every: int = 25,
) -> Dict[str, object]:
    """Permutation null for the community feature's out-of-sample R-squared gain.

    Fits the baseline once per fold, then refits the expanded model
    ``n_permutations`` times with the community column replaced by a null
    partition, and returns the null distribution of mean delta R-squared.

    Parameters
    ----------
    null:
        ``"shuffled"`` reassigns the observed community labels uniformly at
        random, preserving the size distribution. This asks whether *this*
        partition beats *any* 17 arbitrary groups of the same sizes, and so
        isolates the value of the network structure from the value of simply
        adding 17 degrees of freedom.

        ``"spatial"`` replaces the labels with a KMeans partition of the
        coordinates at the same cluster count. This is the harder and more
        interesting null: it asks whether the network partition beats arbitrary
        *geography* at the same granularity.
    """
    if null not in {"shuffled", "spatial"}:
        raise ValueError("null must be 'shuffled' or 'spatial'")

    rng = np.random.default_rng(random_state)
    y = work["log_price_w"].to_numpy(dtype=float)
    base_features = list(numeric_features) + list(categorical_features)
    expanded_features = base_features + [community_column]

    observed_labels = work[community_column].to_numpy()
    n_communities = len(np.unique(observed_labels))
    sizes = np.bincount(pd.factorize(observed_labels)[0]).tolist()

    baseline_r2: List[float] = []
    observed_r2: List[float] = []
    for train_index, test_index in splits:
        pipeline = make_pipeline(numeric_features, categorical_features)
        pipeline.fit(work.iloc[train_index][base_features], y[train_index])
        baseline_r2.append(
            r2_score(y[test_index], pipeline.predict(work.iloc[test_index][base_features]))
        )

        pipeline = make_pipeline(numeric_features, list(categorical_features) + [community_column])
        pipeline.fit(work.iloc[train_index][expanded_features], y[train_index])
        observed_r2.append(
            r2_score(
                y[test_index], pipeline.predict(work.iloc[test_index][expanded_features])
            )
        )

    observed_delta = float(np.mean(observed_r2) - np.mean(baseline_r2))

    null_deltas: List[float] = []
    permuted = work.copy()
    for iteration in range(n_permutations):
        seed = int(rng.integers(0, 2**31 - 1))
        if null == "shuffled":
            labels = random_partition(len(work), sizes, random_state=seed)
        else:
            labels = spatial_partition(
                work, n_communities, random_state=seed, n_init=2
            )
        permuted[community_column] = pd.Series(labels, index=permuted.index).astype(str)

        fold_r2 = []
        for train_index, test_index in splits:
            pipeline = make_pipeline(
                numeric_features, list(categorical_features) + [community_column]
            )
            pipeline.fit(permuted.iloc[train_index][expanded_features], y[train_index])
            fold_r2.append(
                r2_score(
                    y[test_index],
                    pipeline.predict(permuted.iloc[test_index][expanded_features]),
                )
            )
        null_deltas.append(float(np.mean(fold_r2) - np.mean(baseline_r2)))

        if progress_every and (iteration + 1) % progress_every == 0:
            print(
                f"    {null} null: {iteration + 1}/{n_permutations} "
                f"(running mean {np.mean(null_deltas):+.5f})"
            )

    null_array = np.asarray(null_deltas)
    exceedances = int((null_array >= observed_delta).sum())
    p_value = (exceedances + 1) / (n_permutations + 1)
    null_std = float(null_array.std(ddof=1))

    return {
        "null": null,
        "n_permutations": n_permutations,
        "n_communities": n_communities,
        "baseline_r2_mean": float(np.mean(baseline_r2)),
        "observed_expanded_r2_mean": float(np.mean(observed_r2)),
        "observed_delta_r2": observed_delta,
        "null_delta_r2_mean": float(null_array.mean()),
        "null_delta_r2_std": null_std,
        "null_delta_r2_p05": float(np.quantile(null_array, 0.05)),
        "null_delta_r2_p95": float(np.quantile(null_array, 0.95)),
        "null_delta_r2_max": float(null_array.max()),
        "exceedances": exceedances,
        "p_value": float(p_value),
        "standardised_effect": (
            float((observed_delta - null_array.mean()) / null_std) if null_std > 0 else float("nan")
        ),
    }


def write_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
