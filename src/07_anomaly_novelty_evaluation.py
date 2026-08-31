#!/usr/bin/env python3
r"""
07_anomaly_novelty_evaluation.py

Benign-trained anomaly detection and novel-scenario evaluation for the AI-NIDS
project.

Run from the project root:
    python .\src\07_anomaly_novelty_evaluation.py

Purpose
-------
The supervised binary detector is excellent on known attack patterns but Step 4
showed weak recognition when an entire attack scenario was excluded from
training. This script evaluates a different security question:

    Can a detector trained only on BENIGN traffic flag flows that are unlike
    previously observed normal behavior?

Methodology
-----------
1. Build a deterministic, scenario-proportional BENIGN sample pool.
2. Perform eight leave-one-scenario-out folds.
3. For each fold:
   - exclude the complete held-out scenario;
   - fit SimpleImputer + IsolationForest on BENIGN rows from the other seven;
   - calibrate the anomaly threshold using BENIGN validation rows only;
   - target a configurable validation false-positive rate;
   - evaluate every valid flow in the held-out scenario.
4. Aggregate all scenario-holdout predictions.
5. Train a final benign-only anomaly artifact across all scenarios for the
   future Streamlit dashboard.

Important
---------
An anomaly is not proof of malicious activity. It means the flow is unusual
relative to the benign training population. The model is intended to complement,
not replace, the supervised binary and attack-family models.

The raw and processed source datasets are never modified.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

METADATA_COLUMNS = [
    "RowID",
    "SourceFile",
    "Scenario",
    "Label",
    "AttackFamily",
    "BinaryLabel",
]


def scenario_from_parquet(path: Path) -> str:
    """Read one Scenario value without loading the complete file."""
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=["Scenario"])

    if table.num_rows == 0:
        raise ValueError(f"No rows found in {path}")

    return str(table.column("Scenario")[0].as_py())


def safe_name(value: str) -> str:
    """Make a scenario name safe for Windows filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def allocate_quotas(
    counts: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """Allocate an exact proportional sample total across scenarios."""
    available = sum(counts.values())
    if available <= 0:
        raise ValueError("No benign rows were found.")

    target_total = min(target_total, available)
    raw = {
        scenario: target_total * count / available
        for scenario, count in counts.items()
    }
    quotas = {
        scenario: int(math.floor(value))
        for scenario, value in raw.items()
    }

    remainder = target_total - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda scenario: raw[scenario] - quotas[scenario],
        reverse=True,
    )

    for scenario in ranked[:remainder]:
        quotas[scenario] += 1

    return quotas


def count_benign_rows(
    scenario_files: dict[str, Path],
) -> dict[str, int]:
    """Count BENIGN records while reading only the one-byte target column."""
    counts: dict[str, int] = {}

    print("[1/8] Counting benign rows by scenario...")

    for index, scenario in enumerate(sorted(scenario_files), start=1):
        labels = pd.read_parquet(
            scenario_files[scenario],
            columns=["BinaryLabel"],
            engine="pyarrow",
        )["BinaryLabel"]

        count = int(labels.astype("int8").eq(0).sum())
        counts[scenario] = count

        print(
            f"      [{index}/{len(scenario_files)}] "
            f"{scenario}: {count:,} benign rows"
        )

        del labels
        gc.collect()

    return counts


def build_benign_pool(
    scenario_files: dict[str, Path],
    feature_names: list[str],
    benign_counts: dict[str, int],
    pool_size: int,
) -> dict[str, pd.DataFrame]:
    """Create one reusable, deterministic benign sample for every scenario."""
    quotas = allocate_quotas(benign_counts, pool_size)
    pool: dict[str, pd.DataFrame] = {}

    print("[2/8] Building the reusable benign-only modeling pool...")
    print(f"      Available benign rows: {sum(benign_counts.values()):,}")
    print(f"      Retained pool rows:    {sum(quotas.values()):,}")

    selected_columns = feature_names + ["BinaryLabel"]

    for index, scenario in enumerate(sorted(scenario_files), start=1):
        frame = pd.read_parquet(
            scenario_files[scenario],
            columns=selected_columns,
            engine="pyarrow",
        )
        benign = frame.loc[
            frame["BinaryLabel"].astype("int8").eq(0)
        ].reset_index(drop=True)

        quota = quotas[scenario]
        if quota < len(benign):
            benign = benign.sample(
                n=quota,
                replace=False,
                random_state=RANDOM_STATE + index,
            )

        pool[scenario] = benign.reset_index(drop=True)

        print(
            f"      [{index}/{len(scenario_files)}] "
            f"{scenario}: retained {len(benign):,}"
        )

        del frame, benign
        gc.collect()

    return pool


def split_benign_training_calibration(
    pool: dict[str, pd.DataFrame],
    excluded_scenario: str | None,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split every included scenario separately so both training and calibration
    contain benign traffic from all available scenarios.
    """
    train_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []

    for index, scenario in enumerate(sorted(pool), start=1):
        if scenario == excluded_scenario:
            continue

        frame = pool[scenario]
        if len(frame) < 2:
            train_frames.append(frame)
            continue

        train, calibration = train_test_split(
            frame,
            test_size=calibration_fraction,
            random_state=RANDOM_STATE + index,
            shuffle=True,
        )

        train_frames.append(train)
        calibration_frames.append(calibration)

    if not train_frames or not calibration_frames:
        raise RuntimeError(
            "Unable to create benign training and calibration populations."
        )

    training = pd.concat(
        train_frames,
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    calibration = pd.concat(
        calibration_frames,
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=RANDOM_STATE + 1,
    ).reset_index(drop=True)

    return training, calibration


def build_pipeline(
    training_rows: int,
    trees: int,
    max_samples: int,
    n_jobs: int,
) -> Pipeline:
    """Create a fresh benign-only Isolation Forest pipeline."""
    effective_max_samples = min(max_samples, training_rows)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "detector",
                IsolationForest(
                    n_estimators=trees,
                    max_samples=effective_max_samples,
                    max_features=1.0,
                    contamination="auto",
                    bootstrap=False,
                    n_jobs=n_jobs,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def anomaly_scores(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Return a score where larger values mean more anomalous.

    IsolationForest.score_samples uses the opposite orientation: larger means
    more normal. Negating it creates an intuitive anomaly-risk direction.
    """
    transformed = pipeline.named_steps["imputer"].transform(features)
    normality = pipeline.named_steps["detector"].score_samples(transformed)
    return -normality.astype(float)


def calculate_metrics(
    scenario: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    training_rows: int,
    calibration_rows: int,
    fit_seconds: float,
    prediction_seconds: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    benign = int(tn + fp)
    attack = int(tp + fn)
    both_classes = benign > 0 and attack > 0

    return {
        "holdout_scenario": scenario,
        "training_benign_rows": training_rows,
        "calibration_benign_rows": calibration_rows,
        "test_rows": len(y_true),
        "actual_benign": benign,
        "actual_attack": attack,
        "anomaly_threshold": threshold,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_throughput_flows_per_second": (
            len(y_true) / prediction_seconds
            if prediction_seconds > 0
            else float("inf")
        ),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, predictions)
            if both_classes
            else float("nan")
        ),
        "precision_attack": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_attack": (
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
            if attack
            else float("nan")
        ),
        "f1_attack": (
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
            if attack
            else float("nan")
        ),
        "f2_attack": (
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                pos_label=1,
                zero_division=0,
            )
            if attack
            else float("nan")
        ),
        "specificity": tn / benign if benign else float("nan"),
        "false_positive_rate": fp / benign if benign else float("nan"),
        "false_negative_rate": fn / attack if attack else float("nan"),
        "roc_auc": (
            roc_auc_score(y_true, scores)
            if both_classes
            else float("nan")
        ),
        "pr_auc": (
            average_precision_score(y_true, scores)
            if both_classes
            else float("nan")
        ),
        "matthews_correlation": (
            matthews_corrcoef(y_true, predictions)
            if both_classes
            else float("nan")
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "errors": int(fp + fn),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    predictions: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["BENIGN", "ANOMALOUS"],
        values_format=",d",
        colorbar=False,
        ax=ax,
    )
    display.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scenario_figures(
    metrics: pd.DataFrame,
    figures_dir: Path,
) -> None:
    indexed = metrics.set_index("holdout_scenario")

    error_rates = indexed[
        ["false_positive_rate", "false_negative_rate"]
    ]
    ax = error_rates.plot(kind="bar", figsize=(13, 7))
    ax.set_ylabel("Rate")
    ax.set_xlabel("Completely held-out traffic scenario")
    ax.set_title("Benign-Trained Anomaly Detector: Scenario Error Rates")
    ax.legend(["False-positive rate", "False-negative rate"])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_scenario_error_rates.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    recall_frame = metrics.loc[
        metrics["actual_attack"] > 0,
        ["holdout_scenario", "recall_attack"],
    ].set_index("holdout_scenario")

    ax = recall_frame.plot(kind="bar", legend=False, figsize=(12, 7))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly detection rate on attacks")
    ax.set_xlabel("Completely held-out traffic scenario")
    ax.set_title("Benign-Trained Anomaly Detector: Attack Recall")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_scenario_attack_recall.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_attack_label_figure(
    label_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    attacks = label_frame.loc[
        label_frame["actual_class"].eq("ATTACK")
    ].copy()

    if attacks.empty:
        return

    aggregate = (
        attacks.groupby("label", as_index=False)
        .agg(
            rows=("rows", "sum"),
            flagged_rows=("flagged_rows", "sum"),
        )
    )
    aggregate["detection_rate"] = (
        aggregate["flagged_rows"] / aggregate["rows"]
    )
    aggregate = aggregate.sort_values("detection_rate")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(
        aggregate["label"],
        aggregate["detection_rate"],
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Anomaly detection rate")
    ax.set_ylabel("Attack label")
    ax.set_title("Benign-Trained Anomaly Detection by Attack Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_aggregate_curves(
    y_true: np.ndarray,
    scores: np.ndarray,
    figures_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"Anomaly detector (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Aggregate Novel-Scenario Anomaly ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_aggregate_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        scores,
    )
    pr_auc = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall_values,
        precision_values,
        label=f"Anomaly detector (AP={pr_auc:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Aggregate Novel-Scenario Anomaly Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_aggregate_precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_supervised_comparison(
    anomaly_metrics: dict[str, Any],
    reports_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows = [
        {
            "evaluation": "Benign-trained anomaly scenario holdout",
            "accuracy": anomaly_metrics["accuracy"],
            "precision_attack": anomaly_metrics["precision_attack"],
            "recall_attack": anomaly_metrics["recall_attack"],
            "f2_attack": anomaly_metrics["f2_attack"],
            "pr_auc": anomaly_metrics["pr_auc"],
            "false_positive_rate": anomaly_metrics["false_positive_rate"],
            "false_negative_rate": anomaly_metrics["false_negative_rate"],
        }
    ]

    supervised_path = reports_dir / "scenario_holdout_summary.json"
    if supervised_path.exists():
        supervised = json.loads(
            supervised_path.read_text(encoding="utf-8")
        )["aggregate_metrics"]

        rows.insert(
            0,
            {
                "evaluation": "Supervised complete scenario holdout",
                "accuracy": supervised["accuracy"],
                "precision_attack": supervised["precision_attack"],
                "recall_attack": supervised["recall_attack"],
                "f2_attack": supervised["f2_attack"],
                "pr_auc": supervised["pr_auc"],
                "false_positive_rate": supervised[
                    "false_positive_rate"
                ],
                "false_negative_rate": supervised[
                    "false_negative_rate"
                ],
            },
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        tables_dir / "novel_scenario_detector_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    plot_frame = comparison.set_index("evaluation")[
        [
            "precision_attack",
            "recall_attack",
            "f2_attack",
            "pr_auc",
        ]
    ]
    ax = plot_frame.plot(kind="bar", figsize=(11, 7))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Novel-scenario detector")
    ax.set_title("Supervised Versus Benign-Trained Novel-Scenario Detection")
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "novel_scenario_detector_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a benign-trained Isolation Forest on unseen NIDS scenarios."
    )
    parser.add_argument(
        "--benign-pool-size",
        type=int,
        default=240_000,
        help=(
            "Total reusable benign sample across all scenarios. "
            "Default: 240000."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help="Benign fraction reserved for threshold calibration. Default: 0.20.",
    )
    parser.add_argument(
        "--target-validation-fpr",
        type=float,
        default=0.01,
        help="Target benign calibration false-positive rate. Default: 0.01.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=200,
        help="Isolation Forest tree count. Default: 200.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Rows used per Isolation Forest tree. Default: 10000.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Held-out Parquet prediction batch size. Default: 50000.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Parallel Isolation Forest workers. Default: up to 4.",
    )
    parser.add_argument(
        "--alert-preview-limit",
        type=int,
        default=5_000,
        help="Maximum saved alert-preview rows per scenario. Default: 5000.",
    )
    args = parser.parse_args()

    if args.benign_pool_size < 50_000:
        print(
            "ERROR: --benign-pool-size must be at least 50000.",
            file=sys.stderr,
        )
        return 2

    if not 0 < args.calibration_fraction < 0.5:
        print(
            "ERROR: --calibration-fraction must be between 0 and 0.5.",
            file=sys.stderr,
        )
        return 3

    if not 0 < args.target_validation_fpr < 0.20:
        print(
            "ERROR: --target-validation-fpr must be between 0 and 0.20.",
            file=sys.stderr,
        )
        return 4

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = (
        project_root / "data" / "processed" / "machine_learning"
    )
    feature_manifest_path = (
        project_root / "reports" / "tables" / "feature_manifest.csv"
    )

    models_dir = project_root / "models" / "anomaly"
    reports_dir = project_root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    preview_dir = reports_dir / "predictions" / "anomaly_alert_previews"

    for directory in (
        models_dir,
        reports_dir,
        tables_dir,
        figures_dir,
        preview_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(processed_dir.glob("*.parquet"))
    if len(parquet_paths) != 8:
        print(
            f"ERROR: Expected 8 processed Parquet files, "
            f"found {len(parquet_paths)}.",
            file=sys.stderr,
        )
        return 5

    if not feature_manifest_path.exists():
        print(
            f"ERROR: Missing feature manifest: {feature_manifest_path}",
            file=sys.stderr,
        )
        return 6

    feature_names = (
        pd.read_csv(feature_manifest_path)["feature"]
        .astype(str)
        .tolist()
    )

    scenario_files = {
        scenario_from_parquet(path): path
        for path in parquet_paths
    }

    if len(scenario_files) != 8:
        print(
            "ERROR: The eight processed files did not produce eight "
            "unique scenario names.",
            file=sys.stderr,
        )
        return 7

    started = datetime.now(timezone.utc)

    print("AI NIDS benign-trained anomaly evaluation")
    print("=" * 78)
    print(f"Features:                {len(feature_names)}")
    print(f"Target calibration FPR:  {args.target_validation_fpr:.2%}")
    print(f"Isolation Forest trees:  {args.trees}")
    print()

    benign_counts = count_benign_rows(scenario_files)
    benign_pool = build_benign_pool(
        scenario_files=scenario_files,
        feature_names=feature_names,
        benign_counts=benign_counts,
        pool_size=args.benign_pool_size,
    )

    print("[3/8] Running eight complete scenario-holdout anomaly folds...")

    scenario_metric_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    aggregate_y: list[np.ndarray] = []
    aggregate_scores: list[np.ndarray] = []

    for fold_number, holdout_scenario in enumerate(
        sorted(scenario_files),
        start=1,
    ):
        print(
            f"      [{fold_number}/{len(scenario_files)}] "
            f"Holding out: {holdout_scenario}"
        )

        benign_train, benign_calibration = (
            split_benign_training_calibration(
                pool=benign_pool,
                excluded_scenario=holdout_scenario,
                calibration_fraction=args.calibration_fraction,
            )
        )

        X_train = benign_train[feature_names]
        X_calibration = benign_calibration[feature_names]

        pipeline = build_pipeline(
            training_rows=len(X_train),
            trees=args.trees,
            max_samples=args.max_samples,
            n_jobs=args.n_jobs,
        )

        fit_start = time.perf_counter()
        pipeline.fit(X_train)
        fit_seconds = time.perf_counter() - fit_start

        calibration_scores = anomaly_scores(
            pipeline,
            X_calibration,
        )
        threshold = float(
            np.quantile(
                calibration_scores,
                1.0 - args.target_validation_fpr,
            )
        )

        calibration_flag_rate = float(
            (calibration_scores >= threshold).mean()
        )

        parquet_file = pq.ParquetFile(
            scenario_files[holdout_scenario]
        )
        columns = feature_names + METADATA_COLUMNS

        fold_y_parts: list[np.ndarray] = []
        fold_score_parts: list[np.ndarray] = []
        label_counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"rows": 0, "flagged_rows": 0, "actual_attack": 0}
        )
        preview_frames: list[pd.DataFrame] = []
        preview_rows = 0

        prediction_start = time.perf_counter()

        for batch in parquet_file.iter_batches(
            batch_size=args.batch_size,
            columns=columns,
        ):
            frame = batch.to_pandas()

            y_batch = (
                frame["BinaryLabel"]
                .astype("int8")
                .to_numpy()
            )
            score_batch = anomaly_scores(
                pipeline,
                frame[feature_names],
            )
            prediction_batch = (
                score_batch >= threshold
            ).astype("int8")

            fold_y_parts.append(y_batch)
            fold_score_parts.append(score_batch)

            label_text = frame["Label"].astype(str)
            for label in label_text.unique():
                mask = label_text.eq(label).to_numpy()
                bucket = label_counts[str(label)]
                bucket["rows"] += int(mask.sum())
                bucket["flagged_rows"] += int(
                    prediction_batch[mask].sum()
                )
                bucket["actual_attack"] = int(
                    y_batch[mask][0]
                )

            if (
                preview_rows < args.alert_preview_limit
                and bool(prediction_batch.any())
            ):
                alert_mask = prediction_batch.astype(bool)
                alert_frame = frame.loc[
                    alert_mask,
                    METADATA_COLUMNS,
                ].copy()
                alert_frame["AnomalyScore"] = score_batch[alert_mask]
                alert_frame["AnomalyThreshold"] = threshold
                alert_frame["FlaggedAnomaly"] = True
                alert_frame["CorrectBinaryFlag"] = (
                    y_batch[alert_mask] == 1
                )

                remaining = (
                    args.alert_preview_limit - preview_rows
                )
                alert_frame = alert_frame.head(remaining)
                preview_frames.append(alert_frame)
                preview_rows += len(alert_frame)

            del frame
            gc.collect()

        prediction_seconds = (
            time.perf_counter() - prediction_start
        )

        y_fold = np.concatenate(fold_y_parts)
        scores_fold = np.concatenate(fold_score_parts)
        predictions_fold = (
            scores_fold >= threshold
        ).astype("int8")

        metrics = calculate_metrics(
            scenario=holdout_scenario,
            y_true=y_fold,
            scores=scores_fold,
            threshold=threshold,
            training_rows=len(X_train),
            calibration_rows=len(X_calibration),
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
        )
        metrics["calibration_observed_fpr"] = calibration_flag_rate
        scenario_metric_rows.append(metrics)

        for label, counts in sorted(label_counts.items()):
            label_rows.append(
                {
                    "holdout_scenario": holdout_scenario,
                    "label": label,
                    "actual_class": (
                        "ATTACK"
                        if counts["actual_attack"] == 1
                        else "BENIGN"
                    ),
                    "rows": counts["rows"],
                    "flagged_rows": counts["flagged_rows"],
                    "flag_rate": (
                        counts["flagged_rows"] / counts["rows"]
                        if counts["rows"]
                        else float("nan")
                    ),
                }
            )

        if preview_frames:
            pd.concat(
                preview_frames,
                ignore_index=True,
            ).to_csv(
                preview_dir
                / f"{safe_name(holdout_scenario)}_anomaly_alert_preview.csv",
                index=False,
                encoding="utf-8",
            )

        save_confusion_matrix(
            y_fold,
            predictions_fold,
            title=f"Anomaly Scenario Holdout: {holdout_scenario}",
            output_path=(
                figures_dir
                / f"anomaly_confusion_matrix_{safe_name(holdout_scenario)}.png"
            ),
        )

        aggregate_y.append(y_fold)
        aggregate_scores.append(scores_fold)

        print(
            f"          test={len(y_fold):,} | "
            f"attacks={metrics['actual_attack']:,} | "
            f"attack recall="
            f"{metrics['recall_attack']:.4f}"
            if metrics["actual_attack"] > 0
            else (
                f"          test={len(y_fold):,} | "
                "attacks=0 | attack recall=N/A"
            )
        )
        print(
            f"          held-out benign FPR="
            f"{metrics['false_positive_rate']:.4f} | "
            f"calibration FPR={calibration_flag_rate:.4f}"
        )

        del (
            benign_train,
            benign_calibration,
            X_train,
            X_calibration,
            pipeline,
            calibration_scores,
            fold_y_parts,
            fold_score_parts,
            y_fold,
            scores_fold,
            predictions_fold,
            preview_frames,
        )
        gc.collect()

    print("[4/8] Calculating aggregate novel-scenario anomaly metrics...")

    metrics_frame = pd.DataFrame(scenario_metric_rows)
    label_frame = pd.DataFrame(label_rows)
    y_all = np.concatenate(aggregate_y)
    scores_all = np.concatenate(aggregate_scores)

    # Each fold has a different calibrated threshold, so aggregate predictions
    # must be reconstructed from per-scenario thresholds.
    aggregate_predictions: list[np.ndarray] = []
    cursor = 0
    for row in scenario_metric_rows:
        rows = int(row["test_rows"])
        threshold = float(row["anomaly_threshold"])
        aggregate_predictions.append(
            (
                scores_all[cursor : cursor + rows] >= threshold
            ).astype("int8")
        )
        cursor += rows

    predictions_all = np.concatenate(aggregate_predictions)
    tn, fp, fn, tp = confusion_matrix(
        y_all,
        predictions_all,
        labels=[0, 1],
    ).ravel()

    benign_total = int(tn + fp)
    attack_total = int(tp + fn)

    aggregate_metrics = {
        "test_rows": len(y_all),
        "actual_benign": benign_total,
        "actual_attack": attack_total,
        "accuracy": accuracy_score(y_all, predictions_all),
        "balanced_accuracy": balanced_accuracy_score(
            y_all,
            predictions_all,
        ),
        "precision_attack": precision_score(
            y_all,
            predictions_all,
            pos_label=1,
            zero_division=0,
        ),
        "recall_attack": recall_score(
            y_all,
            predictions_all,
            pos_label=1,
            zero_division=0,
        ),
        "f1_attack": f1_score(
            y_all,
            predictions_all,
            pos_label=1,
            zero_division=0,
        ),
        "f2_attack": fbeta_score(
            y_all,
            predictions_all,
            beta=2,
            pos_label=1,
            zero_division=0,
        ),
        "specificity": tn / benign_total if benign_total else float("nan"),
        "false_positive_rate": fp / benign_total if benign_total else float("nan"),
        "false_negative_rate": fn / attack_total if attack_total else float("nan"),
        "roc_auc": roc_auc_score(y_all, scores_all),
        "pr_auc": average_precision_score(y_all, scores_all),
        "matthews_correlation": matthews_corrcoef(
            y_all,
            predictions_all,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "errors": int(fp + fn),
    }

    print("[5/8] Writing anomaly-evaluation tables and figures...")

    metrics_frame.to_csv(
        tables_dir / "anomaly_scenario_holdout_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    label_frame.to_csv(
        tables_dir / "anomaly_attack_label_detection.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame([aggregate_metrics]).to_csv(
        tables_dir / "anomaly_aggregate_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    save_scenario_figures(
        metrics_frame,
        figures_dir,
    )
    save_attack_label_figure(
        label_frame,
        figures_dir / "anomaly_attack_label_detection.png",
    )
    save_confusion_matrix(
        y_all,
        predictions_all,
        title="Aggregate Benign-Trained Novel-Scenario Anomaly Detection",
        output_path=(
            figures_dir / "anomaly_aggregate_confusion_matrix.png"
        ),
    )
    save_aggregate_curves(
        y_all,
        scores_all,
        figures_dir,
    )
    save_supervised_comparison(
        aggregate_metrics,
        reports_dir,
        tables_dir,
        figures_dir,
    )

    print("[6/8] Training the final dashboard anomaly artifact...")

    final_train, final_calibration = (
        split_benign_training_calibration(
            pool=benign_pool,
            excluded_scenario=None,
            calibration_fraction=args.calibration_fraction,
        )
    )
    X_final_train = final_train[feature_names]
    X_final_calibration = final_calibration[feature_names]

    final_pipeline = build_pipeline(
        training_rows=len(X_final_train),
        trees=args.trees,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
    )
    final_pipeline.fit(X_final_train)

    final_calibration_scores = anomaly_scores(
        final_pipeline,
        X_final_calibration,
    )
    final_threshold = float(
        np.quantile(
            final_calibration_scores,
            1.0 - args.target_validation_fpr,
        )
    )
    final_calibration_fpr = float(
        (final_calibration_scores >= final_threshold).mean()
    )

    anomaly_artifact = {
        "pipeline": final_pipeline,
        "feature_names": feature_names,
        "anomaly_threshold": final_threshold,
        "score_direction": (
            "Higher anomaly score means more unusual. Score is the negative "
            "IsolationForest score_samples output."
        ),
        "training_population": "BENIGN flows only",
        "target_validation_false_positive_rate": (
            args.target_validation_fpr
        ),
        "observed_calibration_false_positive_rate": (
            final_calibration_fpr
        ),
        "training_benign_rows": len(X_final_train),
        "calibration_benign_rows": len(X_final_calibration),
        "label_meaning": {
            0: "Within learned benign profile",
            1: "Anomalous relative to benign profile",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "warning": (
            "An anomaly flag is not proof of malicious activity. It should be "
            "combined with the supervised detector, threat intelligence, and "
            "analyst review."
        ),
    }

    model_path = (
        models_dir / "benign_isolation_forest_pipeline.joblib"
    )
    joblib.dump(
        anomaly_artifact,
        model_path,
        compress=3,
    )

    print("[7/8] Saving the anomaly experiment summary...")

    summary = {
        "model": "IsolationForest",
        "feature_count": len(feature_names),
        "benign_pool_size_requested": args.benign_pool_size,
        "target_validation_false_positive_rate": (
            args.target_validation_fpr
        ),
        "scenario_holdout_metrics": aggregate_metrics,
        "final_artifact_threshold": final_threshold,
        "final_artifact_calibration_false_positive_rate": (
            final_calibration_fpr
        ),
        "experiment_started_utc": started.isoformat(),
        "experiment_completed_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "interpretation": (
            "The anomaly detector learns benign behavior only. Scenario-holdout "
            "attack recall measures whether unseen attack flows look unusual, "
            "while held-out benign FPR measures sensitivity to benign "
            "distribution shift."
        ),
        "limitations": [
            (
                "Isolation Forest detects unusual feature combinations; it "
                "does not identify the cause or prove that a flow is malicious."
            ),
            (
                "A low calibration FPR can become much higher on a held-out "
                "network scenario if normal traffic changes."
            ),
            (
                "All data still originates from one historical benchmark. "
                "External and current traffic validation remains necessary."
            ),
        ],
    }

    with (
        reports_dir / "anomaly_model_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "BENIGN-TRAINED ANOMALY DETECTOR SUMMARY",
        "=" * 78,
        "",
        "Model: Isolation Forest",
        f"Predictive features: {len(feature_names)}",
        f"Reusable benign pool: {sum(len(frame) for frame in benign_pool.values()):,}",
        f"Target calibration FPR: {args.target_validation_fpr:.4%}",
        "",
        "Aggregate complete-scenario holdout performance:",
        f"  Rows evaluated:        {aggregate_metrics['test_rows']:,}",
        f"  Accuracy:              {aggregate_metrics['accuracy']:.6f}",
        f"  Balanced accuracy:     {aggregate_metrics['balanced_accuracy']:.6f}",
        f"  Attack precision:      {aggregate_metrics['precision_attack']:.6f}",
        f"  Attack recall:         {aggregate_metrics['recall_attack']:.6f}",
        f"  Attack F1-score:       {aggregate_metrics['f1_attack']:.6f}",
        f"  Attack F2-score:       {aggregate_metrics['f2_attack']:.6f}",
        f"  Specificity:           {aggregate_metrics['specificity']:.6f}",
        f"  False-positive rate:   {aggregate_metrics['false_positive_rate']:.6f}",
        f"  False-negative rate:   {aggregate_metrics['false_negative_rate']:.6f}",
        f"  ROC-AUC:               {aggregate_metrics['roc_auc']:.6f}",
        f"  PR-AUC:                {aggregate_metrics['pr_auc']:.6f}",
        "",
        "Aggregate confusion matrix:",
        f"  True negatives:  {aggregate_metrics['true_negative']:,}",
        f"  False positives: {aggregate_metrics['false_positive']:,}",
        f"  False negatives: {aggregate_metrics['false_negative']:,}",
        f"  True positives:  {aggregate_metrics['true_positive']:,}",
        "",
        "Final dashboard artifact:",
        f"  Model path: {model_path}",
        f"  Threshold: {final_threshold:.6f}",
        f"  Calibration FPR: {final_calibration_fpr:.4%}",
        "",
        "Interpretation:",
        "  This detector flags flows that differ from its learned benign",
        "  profile. It is intended to complement the supervised binary and",
        "  attack-family classifiers, especially for possible novel patterns.",
        "",
        "Warning:",
        "  An anomaly is not proof of an intrusion. Benign operational changes",
        "  can also generate anomaly alerts.",
        "",
        "Anomaly evaluation completed successfully.",
    ]

    summary_path = reports_dir / "anomaly_model_summary.txt"
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("[8/8] Anomaly artifacts and reports saved.")
    print()
    print("[COMPLETE] Benign-trained anomaly evaluation finished.")
    print(
        f"Aggregate attack recall: "
        f"{aggregate_metrics['recall_attack']:.4%}"
    )
    print(
        f"Aggregate held-out benign FPR: "
        f"{aggregate_metrics['false_positive_rate']:.4%}"
    )
    print(f"Aggregate PR-AUC: {aggregate_metrics['pr_auc']:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
