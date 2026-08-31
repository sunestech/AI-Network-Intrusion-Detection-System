#!/usr/bin/env python3
r"""
06_train_attack_family_classifier.py

Train a second-stage, group-aware attack-family classifier for the AI-NIDS
project.

Run from the project root:
    python .\src\06_train_attack_family_classifier.py

Architecture
------------
Stage 1 (already completed):
    Binary classifier -> BENIGN or ATTACK

Stage 2 (this script):
    When Stage 1 predicts ATTACK, classify the attack into one of five
    operational families:

    1. Denial of Service
    2. Port Scan
    3. Brute Force
    4. Web Attack
    5. Other Malicious

The family model is trained only on attack records. This avoids letting the
large BENIGN population dominate attack-type learning.

Methodology
-----------
- Loads aligned processed features and generated-flow context.
- Uses scenario/IP/port/protocol/time GroupIDs.
- Keeps each GroupID in only one of training, validation, or testing.
- Caps very large families while retaining every record from smaller families.
- Compares Dummy, Logistic Regression (SGD), Random Forest, and
  HistGradientBoosting candidates.
- Selects the model with the highest validation macro F2 score.
- Refits the selected family on training + validation.
- Evaluates once on an untouched grouped test split.
- Saves the complete preprocessing/model pipeline and SOC-ready prediction
  evidence.

Raw and processed source files are never modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

# Use Matplotlib's non-interactive file backend. This avoids Tkinter GUI
# cleanup errors on Windows when tree models use parallel worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42

FEATURE_METADATA = [
    "RowID",
    "SourceFile",
    "Scenario",
    "Label",
    "AttackFamily",
    "BinaryLabel",
]

CONTEXT_COLUMNS = [
    "RowID",
    "Source IP",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]

OPERATIONAL_FAMILY_ORDER = [
    "Denial of Service",
    "Port Scan",
    "Brute Force",
    "Web Attack",
    "Other Malicious",
]


def operational_family(label: str) -> str:
    """Map normalized CICIDS labels to SOC-oriented attack families."""
    label = str(label)

    if label == "DDoS" or label.startswith("DoS "):
        return "Denial of Service"

    if label == "PortScan":
        return "Port Scan"

    if label in {"FTP-Patator", "SSH-Patator"}:
        return "Brute Force"

    if label.startswith("Web Attack - "):
        return "Web Attack"

    if label in {"Bot", "Infiltration", "Heartbleed"}:
        return "Other Malicious"

    return "Other Malicious"


def scenario_from_parquet(path: Path) -> str:
    """Read one Scenario value without loading the complete Parquet file."""
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=["Scenario"])

    if table.num_rows == 0:
        raise ValueError(f"No rows found in {path}")

    return str(table.column("Scenario")[0].as_py())


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse the dataset's mixed timestamp formats safely."""
    text = series.astype("string").str.strip()

    try:
        return pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
    except TypeError:
        return pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
        )


def build_group_id(
    context: pd.DataFrame,
    scenario: str,
    group_minutes: int,
) -> pd.Series:
    """Create a reproducible session-like grouping key."""
    timestamps = parse_timestamp(context["Timestamp"])
    buckets = (
        timestamps.dt.floor(f"{group_minutes}min")
        .astype("string")
        .fillna("NO_TIME")
    )

    def cleaned(column: str) -> pd.Series:
        return (
            context[column]
            .astype("string")
            .fillna("MISSING")
            .str.strip()
        )

    return (
        pd.Series([scenario] * len(context), dtype="string")
        + "|"
        + cleaned("Source IP")
        + "|"
        + cleaned("Destination IP")
        + "|"
        + cleaned("Destination Port")
        + "|"
        + cleaned("Protocol")
        + "|"
        + buckets
    )


def load_aligned_attack_population(
    feature_files: dict[str, Path],
    context_files: dict[str, Path],
    feature_names: list[str],
    group_minutes: int,
) -> pd.DataFrame:
    """
    Load attack-only aligned records.

    There are roughly 557,000 attack flows, so retaining the attack subset in
    memory is materially smaller than loading all 2.83 million flows.
    """
    frames: list[pd.DataFrame] = []

    print("[1/8] Loading aligned attack-only feature and context records...")

    for index, scenario in enumerate(sorted(feature_files), start=1):
        feature_path = feature_files[scenario]
        context_path = context_files[scenario]

        total_rows = int(pq.ParquetFile(feature_path).metadata.num_rows)

        print(
            f"      [{index}/{len(feature_files)}] {scenario}: "
            f"{total_rows:,} total flows"
        )

        features = pd.read_parquet(
            feature_path,
            columns=FEATURE_METADATA + feature_names,
            engine="pyarrow",
        )
        context = pd.read_parquet(
            context_path,
            columns=CONTEXT_COLUMNS,
            engine="pyarrow",
        )

        if len(features) != len(context):
            raise RuntimeError(
                f"Row-count mismatch for {scenario}: "
                f"{len(features)} vs {len(context)}"
            )

        if not np.array_equal(
            features["RowID"].to_numpy(),
            context["RowID"].to_numpy(),
        ):
            raise RuntimeError(
                f"RowID alignment failed for {scenario}."
            )

        attack_mask = features["BinaryLabel"].astype("int8").eq(1)

        if not bool(attack_mask.any()):
            del features, context
            gc.collect()
            continue

        attack_features = features.loc[attack_mask].reset_index(drop=True)
        attack_context = context.loc[attack_mask].reset_index(drop=True)

        attack_features["OperationalFamily"] = (
            attack_features["Label"]
            .astype("string")
            .map(operational_family)
            .astype("string")
        )

        attack_features["GroupID"] = build_group_id(
            attack_context,
            scenario=scenario,
            group_minutes=group_minutes,
        )

        attack_features["Source IP"] = attack_context[
            "Source IP"
        ].astype("string")
        attack_features["Destination IP"] = attack_context[
            "Destination IP"
        ].astype("string")
        attack_features["Destination Port Context"] = attack_context[
            "Destination Port"
        ]
        attack_features["Protocol Context"] = attack_context["Protocol"]
        attack_features["Timestamp"] = attack_context[
            "Timestamp"
        ].astype("string")

        frames.append(attack_features)

        print(
            f"          attack rows retained: {len(attack_features):,}"
        )

        del (
            features,
            context,
            attack_features,
            attack_context,
        )
        gc.collect()

    if not frames:
        raise RuntimeError("No attack records were found.")

    population = pd.concat(frames, ignore_index=True)
    population = population.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return population


def cap_families(
    population: pd.DataFrame,
    max_per_family: int,
) -> pd.DataFrame:
    """
    Cap dominant families while keeping all examples from smaller families.

    This is undersampling, not synthetic oversampling. No artificial network
    flows are created.
    """
    frames: list[pd.DataFrame] = []

    print("[2/8] Creating the operational family modeling population...")

    for index, family in enumerate(OPERATIONAL_FAMILY_ORDER, start=1):
        subset = population.loc[
            population["OperationalFamily"].eq(family)
        ]

        if subset.empty:
            raise RuntimeError(
                f"Operational family has no rows: {family}"
            )

        retained = min(len(subset), max_per_family)

        if retained < len(subset):
            subset = subset.sample(
                n=retained,
                replace=False,
                random_state=RANDOM_STATE + index,
            )

        frames.append(subset)

        print(
            f"      {family:<22} available={len(population.loc[population['OperationalFamily'].eq(family)]):>8,} "
            f"retained={retained:>8,}"
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return data


def all_classes_present(
    labels: pd.Series,
    indices: np.ndarray,
    expected: set[str],
) -> bool:
    return set(labels.iloc[indices].astype(str)) == expected


def find_grouped_split(
    data: pd.DataFrame,
    max_attempts: int = 100,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Find a group-isolated 64/16/20 split with every family represented in all
    three partitions.
    """
    labels = data["OperationalFamily"].astype("string")
    groups = data["GroupID"].astype("string")
    expected = set(labels.astype(str).unique())
    placeholder = np.zeros((len(data), 1), dtype=np.int8)

    for offset in range(max_attempts):
        seed = RANDOM_STATE + offset

        outer = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        for train_val_idx, test_idx in outer.split(
            placeholder,
            labels,
            groups,
        ):
            if not all_classes_present(labels, test_idx, expected):
                continue

            remaining = data.iloc[train_val_idx].reset_index()
            remaining_labels = remaining[
                "OperationalFamily"
            ].astype("string")
            remaining_groups = remaining["GroupID"].astype("string")
            remaining_placeholder = np.zeros(
                (len(remaining), 1),
                dtype=np.int8,
            )

            inner = StratifiedGroupKFold(
                n_splits=4,
                shuffle=True,
                random_state=seed + 10_000,
            )

            for train_pos, validation_pos in inner.split(
                remaining_placeholder,
                remaining_labels,
                remaining_groups,
            ):
                train_idx = remaining.iloc[train_pos][
                    "index"
                ].to_numpy(dtype=np.int64)
                validation_idx = remaining.iloc[validation_pos][
                    "index"
                ].to_numpy(dtype=np.int64)

                if not all_classes_present(
                    labels,
                    train_idx,
                    expected,
                ):
                    continue

                if not all_classes_present(
                    labels,
                    validation_idx,
                    expected,
                ):
                    continue

                split = {
                    "train": train_idx,
                    "validation": validation_idx,
                    "test": test_idx.astype(np.int64),
                }

                group_sets = {
                    name: set(
                        data.iloc[indices]["GroupID"].astype(str)
                    )
                    for name, indices in split.items()
                }

                if (
                    group_sets["train"] & group_sets["validation"]
                    or group_sets["train"] & group_sets["test"]
                    or group_sets["validation"] & group_sets["test"]
                ):
                    continue

                return split, seed

    raise RuntimeError(
        "Unable to find a grouped split with every operational family in "
        "training, validation, and testing. Try a larger --max-per-family "
        "or a different --group-minutes value."
    )


def split_distribution(
    data: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for split_name, indices in splits.items():
        subset = data.iloc[indices]

        table = (
            subset.groupby(
                ["Scenario", "OperationalFamily", "Label"],
                dropna=False,
            )
            .size()
            .rename("rows")
            .reset_index()
        )
        table.insert(0, "split", split_name)
        rows.append(table)

    return pd.concat(rows, ignore_index=True)


def build_factories(
    n_jobs: int,
    rf_trees: int,
    hgb_iterations: int,
) -> dict[str, Callable[[], Pipeline]]:
    """Return fresh model pipelines."""

    def dummy() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DummyClassifier(
                        strategy="prior",
                    ),
                ),
            ]
        )

    def logistic_sgd() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=1e-4,
                        max_iter=2_000,
                        tol=1e-3,
                        class_weight="balanced",
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=8,
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )

    def random_forest() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=rf_trees,
                        max_depth=24,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=n_jobs,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    def histogram_gradient_boosting() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=hgb_iterations,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=10,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    return {
        "Dummy Prior": dummy,
        "Logistic Regression (SGD)": logistic_sgd,
        "Random Forest": random_forest,
        "HistGradientBoosting": histogram_gradient_boosting,
    }


def fit_pipeline(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: np.ndarray,
) -> float:
    """Fit a candidate and return elapsed seconds."""
    start = time.perf_counter()

    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=labels,
        )
        pipeline.fit(
            features,
            labels,
            classifier__sample_weight=sample_weight,
        )
    else:
        pipeline.fit(features, labels)

    return time.perf_counter() - start


def classification_metrics(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    fit_seconds: float,
    prediction_seconds: float,
    training_rows: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "training_rows": training_rows,
        "evaluation_rows": len(y_true),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_throughput_flows_per_second": (
            len(y_true) / prediction_seconds
            if prediction_seconds > 0
            else float("inf")
        ),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "matthews_correlation": matthews_corrcoef(
            y_true,
            predictions,
        ),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=labels,
        display_labels=labels,
        normalize=normalize,
        values_format=".2f" if normalize else ",d",
        colorbar=False,
        xticks_rotation=30,
        ax=ax,
    )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_model_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_frame = comparison.set_index("model")[
        [
            "macro_precision",
            "macro_recall",
            "macro_f2",
            "balanced_accuracy",
        ]
    ]

    ax = plot_frame.plot(
        kind="bar",
        figsize=(12, 7),
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Candidate model")
    ax.set_title("Attack-Family Validation Model Comparison")
    ax.legend(loc="lower right")

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_family_distribution(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    counts = (
        data["OperationalFamily"]
        .value_counts()
        .reindex(OPERATIONAL_FAMILY_ORDER)
    )

    ax = counts.plot(
        kind="bar",
        figsize=(10, 6),
    )
    ax.set_ylabel("Retained attack flows")
    ax.set_xlabel("Operational attack family")
    ax.set_title("Attack-Family Modeling Distribution")

    for position, value in enumerate(counts):
        ax.text(
            position,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a group-aware second-stage attack-family classifier."
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=100_000,
        help=(
            "Maximum retained rows for each operational family. Smaller "
            "families retain every row. Default: 100000."
        ),
    )
    parser.add_argument(
        "--group-minutes",
        type=int,
        default=15,
        help="Timestamp bucket width used in GroupID. Default: 15.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Parallel workers for Random Forest/SGD. Default: up to 4.",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=150,
        help="Random Forest tree count. Default: 150.",
    )
    parser.add_argument(
        "--hgb-iterations",
        type=int,
        default=130,
        help="HistGradientBoosting iterations. Default: 130.",
    )
    args = parser.parse_args()

    if args.max_per_family < 2_000:
        print(
            "ERROR: --max-per-family must be at least 2000.",
            file=sys.stderr,
        )
        return 2

    if args.group_minutes < 1:
        print(
            "ERROR: --group-minutes must be at least 1.",
            file=sys.stderr,
        )
        return 3

    project_root = Path(__file__).resolve().parents[1]
    feature_dir = (
        project_root / "data" / "processed" / "machine_learning"
    )
    context_dir = (
        project_root / "data" / "processed" / "generated_context"
    )
    feature_manifest_path = (
        project_root / "reports" / "tables" / "feature_manifest.csv"
    )

    models_dir = project_root / "models" / "attack_family"
    reports_dir = project_root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    predictions_dir = reports_dir / "predictions"

    for directory in (
        models_dir,
        reports_dir,
        tables_dir,
        figures_dir,
        predictions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    feature_paths = sorted(feature_dir.glob("*.parquet"))
    context_paths = sorted(context_dir.glob("*.parquet"))

    if len(feature_paths) != 8 or len(context_paths) != 8:
        print(
            "ERROR: Expected eight processed feature and eight context "
            "Parquet files.",
            file=sys.stderr,
        )
        return 4

    if not feature_manifest_path.exists():
        print(
            f"ERROR: Missing feature manifest: {feature_manifest_path}",
            file=sys.stderr,
        )
        return 5

    feature_names = (
        pd.read_csv(feature_manifest_path)["feature"]
        .astype(str)
        .tolist()
    )

    feature_files = {
        scenario_from_parquet(path): path
        for path in feature_paths
    }
    context_files = {
        scenario_from_parquet(path): path
        for path in context_paths
    }

    if set(feature_files) != set(context_files):
        print(
            "ERROR: Feature/context scenario sets do not match.",
            file=sys.stderr,
        )
        return 6

    started = datetime.now(timezone.utc)

    print("AI NIDS second-stage attack-family classifier")
    print("=" * 78)
    print(f"Predictive features: {len(feature_names)}")
    print(f"Group time bucket:   {args.group_minutes} minutes")
    print()

    population = load_aligned_attack_population(
        feature_files=feature_files,
        context_files=context_files,
        feature_names=feature_names,
        group_minutes=args.group_minutes,
    )

    original_family_distribution = (
        population["OperationalFamily"]
        .value_counts()
        .reindex(OPERATIONAL_FAMILY_ORDER)
        .rename("rows")
        .reset_index()
        .rename(columns={"index": "operational_family"})
    )
    original_family_distribution.to_csv(
        tables_dir / "attack_family_original_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    data = cap_families(
        population,
        max_per_family=args.max_per_family,
    )
    del population
    gc.collect()

    save_family_distribution(
        data,
        figures_dir / "attack_family_modeling_distribution.png",
    )

    print("[3/8] Creating isolated grouped train/validation/test partitions...")
    splits, split_seed = find_grouped_split(data)

    split_distribution_frame = split_distribution(
        data,
        splits,
    )
    split_distribution_frame.to_csv(
        tables_dir / "attack_family_split_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    group_sets = {
        name: set(data.iloc[indices]["GroupID"].astype(str))
        for name, indices in splits.items()
    }
    overlap_counts = {
        "train_validation": len(
            group_sets["train"] & group_sets["validation"]
        ),
        "train_test": len(
            group_sets["train"] & group_sets["test"]
        ),
        "validation_test": len(
            group_sets["validation"] & group_sets["test"]
        ),
    }

    print(f"      Modeling rows:   {len(data):,}")
    print(f"      Training rows:   {len(splits['train']):,}")
    print(f"      Validation rows: {len(splits['validation']):,}")
    print(f"      Test rows:       {len(splits['test']):,}")
    print(f"      Unique groups:   {data['GroupID'].nunique():,}")
    print(f"      Group overlaps:  {sum(overlap_counts.values())}")

    def X(indices: np.ndarray) -> pd.DataFrame:
        return data.iloc[indices][feature_names].reset_index(drop=True)

    def y(indices: np.ndarray) -> np.ndarray:
        return (
            data.iloc[indices]["OperationalFamily"]
            .astype(str)
            .to_numpy()
        )

    X_train = X(splits["train"])
    y_train = y(splits["train"])
    X_validation = X(splits["validation"])
    y_validation = y(splits["validation"])
    X_test = X(splits["test"])
    y_test = y(splits["test"])

    factories = build_factories(
        n_jobs=args.n_jobs,
        rf_trees=args.rf_trees,
        hgb_iterations=args.hgb_iterations,
    )

    print("[4/8] Comparing attack-family classifiers...")

    candidate_rows: list[dict[str, Any]] = []

    for position, (model_name, factory) in enumerate(
        factories.items(),
        start=1,
    ):
        print(
            f"      [{position}/{len(factories)}] Fitting {model_name}..."
        )

        pipeline = factory()
        fit_seconds = fit_pipeline(
            model_name,
            pipeline,
            X_train,
            y_train,
        )

        prediction_start = time.perf_counter()
        validation_predictions = pipeline.predict(
            X_validation
        )
        prediction_seconds = (
            time.perf_counter() - prediction_start
        )

        metrics = classification_metrics(
            model_name=model_name,
            y_true=y_validation,
            predictions=validation_predictions,
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
            training_rows=len(y_train),
        )
        candidate_rows.append(metrics)

        save_confusion_matrix(
            y_validation,
            validation_predictions,
            labels=OPERATIONAL_FAMILY_ORDER,
            title=f"{model_name}: Attack-Family Validation",
            output_path=(
                figures_dir
                / f"attack_family_validation_confusion_{re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')}.png"
            ),
        )

        print(
            f"          macro recall={metrics['macro_recall']:.4f} | "
            f"macro F2={metrics['macro_f2']:.4f} | "
            f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
        )

        del pipeline
        gc.collect()

    comparison = pd.DataFrame(candidate_rows).sort_values(
        ["macro_f2", "macro_recall", "macro_f1"],
        ascending=False,
    )
    comparison.to_csv(
        tables_dir / "attack_family_validation_model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    save_model_comparison(
        comparison,
        figures_dir / "attack_family_validation_model_comparison.png",
    )

    selected_model_name = str(comparison.iloc[0]["model"])

    print(
        "[5/8] Selected validation winner by macro F2: "
        f"{selected_model_name}"
    )

    print("[6/8] Refitting on grouped training + validation rows...")

    X_final_train = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )
    y_final_train = np.concatenate(
        [y_train, y_validation]
    )

    selected_pipeline = factories[selected_model_name]()
    final_fit_seconds = fit_pipeline(
        selected_model_name,
        selected_pipeline,
        X_final_train,
        y_final_train,
    )

    print("[7/8] Evaluating once on the untouched grouped test set...")

    prediction_start = time.perf_counter()
    test_predictions = selected_pipeline.predict(X_test)
    test_probabilities = selected_pipeline.predict_proba(
        X_test
    )
    test_prediction_seconds = (
        time.perf_counter() - prediction_start
    )

    final_metrics = classification_metrics(
        model_name=selected_model_name,
        y_true=y_test,
        predictions=test_predictions,
        fit_seconds=final_fit_seconds,
        prediction_seconds=test_prediction_seconds,
        training_rows=len(y_final_train),
    )

    classifier_classes = list(
        selected_pipeline.named_steps["classifier"].classes_
    )
    confidence = test_probabilities.max(axis=1)

    test_metadata = data.iloc[splits["test"]][
        [
            "RowID",
            "SourceFile",
            "Scenario",
            "GroupID",
            "Label",
            "AttackFamily",
            "OperationalFamily",
            "Source IP",
            "Destination IP",
            "Destination Port Context",
            "Protocol Context",
            "Timestamp",
        ]
    ].reset_index(drop=True)

    test_metadata["PredictedOperationalFamily"] = (
        test_predictions
    )
    test_metadata["PredictionConfidence"] = confidence
    test_metadata["Correct"] = (
        test_metadata["OperationalFamily"].astype(str).to_numpy()
        == test_predictions
    )

    for class_index, class_name in enumerate(classifier_classes):
        test_metadata[
            f"Probability::{class_name}"
        ] = test_probabilities[:, class_index]

    test_metadata.to_parquet(
        predictions_dir / "attack_family_test_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    test_metadata.to_csv(
        predictions_dir / "attack_family_test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=OPERATIONAL_FAMILY_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()
    report_frame.to_csv(
        tables_dir / "attack_family_final_classification_report.csv",
        encoding="utf-8",
    )

    report_text = classification_report(
        y_test,
        test_predictions,
        labels=OPERATIONAL_FAMILY_ORDER,
        digits=6,
        zero_division=0,
    )
    (
        reports_dir / "attack_family_final_classification_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    confusion = confusion_matrix(
        y_test,
        test_predictions,
        labels=OPERATIONAL_FAMILY_ORDER,
    )
    pd.DataFrame(
        confusion,
        index=[
            f"Actual {label}"
            for label in OPERATIONAL_FAMILY_ORDER
        ],
        columns=[
            f"Predicted {label}"
            for label in OPERATIONAL_FAMILY_ORDER
        ],
    ).to_csv(
        tables_dir / "attack_family_final_confusion_matrix.csv",
        encoding="utf-8",
    )

    save_confusion_matrix(
        y_test,
        test_predictions,
        labels=OPERATIONAL_FAMILY_ORDER,
        title=f"Selected Attack-Family Model: {selected_model_name}",
        output_path=(
            figures_dir
            / "attack_family_final_test_confusion_matrix.png"
        ),
    )

    save_confusion_matrix(
        y_test,
        test_predictions,
        labels=OPERATIONAL_FAMILY_ORDER,
        title="Attack-Family Test Recall by Actual Class",
        output_path=(
            figures_dir
            / "attack_family_final_test_confusion_matrix_normalized.png"
        ),
        normalize="true",
    )

    print("[8/8] Saving the family-model artifact and evidence...")

    model_artifact = {
        "pipeline": selected_pipeline,
        "model_name": selected_model_name,
        "feature_names": feature_names,
        "classes": classifier_classes,
        "operational_family_mapping": {
            "DDoS and DoS variants": "Denial of Service",
            "PortScan": "Port Scan",
            "FTP-Patator and SSH-Patator": "Brute Force",
            "Web Attack labels": "Web Attack",
            "Bot, Infiltration, Heartbleed": "Other Malicious",
        },
        "stage": (
            "Second-stage classifier. Run only when the binary detector "
            "classifies a flow as ATTACK."
        ),
        "group_definition": (
            "Scenario + Source IP + Destination IP + Destination Port + "
            f"Protocol + {args.group_minutes}-minute timestamp bucket"
        ),
        "validation_selection_metric": "macro_f2",
        "final_test_metrics": final_metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    model_path = (
        models_dir / "best_attack_family_pipeline.joblib"
    )
    joblib.dump(
        model_artifact,
        model_path,
        compress=3,
    )

    summary = {
        "model_family_selected": selected_model_name,
        "operational_families": OPERATIONAL_FAMILY_ORDER,
        "source_attack_rows": int(
            original_family_distribution["rows"].sum()
        ),
        "modeling_rows_after_family_caps": len(data),
        "maximum_rows_per_family": args.max_per_family,
        "training_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "unique_groups": int(data["GroupID"].nunique()),
        "group_overlap_counts": overlap_counts,
        "group_minutes": args.group_minutes,
        "split_random_state": split_seed,
        "feature_count": len(feature_names),
        "final_test_metrics": final_metrics,
        "experiment_started_utc": started.isoformat(),
        "experiment_completed_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "limitations": [
            (
                "The operational families combine several original labels. "
                "The model should not be presented as a precise malware or "
                "exploit attribution engine."
            ),
            (
                "Other Malicious combines Bot, Infiltration, and Heartbleed "
                "because the last two classes contain extremely few records."
            ),
            (
                "All data comes from one historical benchmark environment. "
                "External traffic validation is still required."
            ),
        ],
    }

    with (
        reports_dir / "attack_family_model_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    text_lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "SECOND-STAGE ATTACK-FAMILY CLASSIFIER SUMMARY",
        "=" * 80,
        "",
        "Two-stage architecture:",
        "  Stage 1: Binary detector -> BENIGN or ATTACK",
        "  Stage 2: If ATTACK, classify the operational attack family",
        "",
        f"Selected family model: {selected_model_name}",
        f"Predictive features: {len(feature_names)}",
        f"Source attack flows: "
        f"{int(original_family_distribution['rows'].sum()):,}",
        f"Modeling rows after family caps: {len(data):,}",
        f"Training rows: {len(splits['train']):,}",
        f"Validation rows: {len(splits['validation']):,}",
        f"Test rows: {len(splits['test']):,}",
        f"Unique session-like groups: {data['GroupID'].nunique():,}",
        f"Group overlap across splits: {sum(overlap_counts.values())}",
        "",
        "Operational families:",
        "  Denial of Service",
        "  Port Scan",
        "  Brute Force",
        "  Web Attack",
        "  Other Malicious",
        "",
        "Untouched grouped test performance:",
        f"  Accuracy:            {final_metrics['accuracy']:.6f}",
        f"  Balanced accuracy:   {final_metrics['balanced_accuracy']:.6f}",
        f"  Macro precision:     {final_metrics['macro_precision']:.6f}",
        f"  Macro recall:        {final_metrics['macro_recall']:.6f}",
        f"  Macro F1-score:      {final_metrics['macro_f1']:.6f}",
        f"  Macro F2-score:      {final_metrics['macro_f2']:.6f}",
        f"  Weighted F1-score:   {final_metrics['weighted_f1']:.6f}",
        f"  Matthews correlation:{final_metrics['matthews_correlation']:.6f}",
        "",
        "Important interpretation:",
        "  This model classifies broad operational families after the binary",
        "  detector has already identified a flow as malicious. It does not",
        "  prove exact malware attribution or zero-day recognition.",
        "",
        f"Saved model: {model_path}",
        "",
        "Attack-family training completed successfully.",
    ]

    summary_path = (
        reports_dir / "attack_family_model_summary.txt"
    )
    summary_path.write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Attack-family classifier finished.")
    print(f"Selected model: {selected_model_name}")
    print(
        f"Macro recall:  {final_metrics['macro_recall']:.4%}"
    )
    print(f"Macro F2:      {final_metrics['macro_f2']:.4%}")
    print(f"Model artifact:{model_path}")
    print(f"Summary:       {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
