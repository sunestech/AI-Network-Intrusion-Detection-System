#!/usr/bin/env python3
r"""
04_scenario_holdout_evaluation.py

Leakage-aware scenario-holdout evaluation for the AI-NIDS binary detector.

Run from the project root:
    python .\src\04_scenario_holdout_evaluation.py

Why this evaluation exists
--------------------------
The Step 3 baseline used a random, stratified flow-level split. That is useful
and reproducible, but related flows from the same traffic scenario can occur in
both training and testing data. This script performs a stronger evaluation:

- Each of the eight traffic files is held out in turn.
- A fresh copy of the validation-selected pipeline is trained using sampled
  flows from the other seven scenarios only.
- The model is tested on every valid row in the completely held-out scenario.
- Metrics are reported per scenario, per attack label, and in aggregate.

The final aggregate therefore predicts each processed flow with a model that
was not trained on that flow's source scenario.

The script does not modify raw or processed source datasets and does not
replace the production model artifact created in Step 3.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
METADATA_COLUMNS = [
    "RowID",
    "SourceFile",
    "Scenario",
    "Label",
    "AttackFamily",
    "BinaryLabel",
]


def safe_name(value: str) -> str:
    """Convert a scenario name into a safe filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parquet_scenario(path: Path) -> str:
    """Read the first Scenario value without loading the complete Parquet file."""
    parquet_file = pq.ParquetFile(path)
    first_group = parquet_file.read_row_group(0, columns=["Scenario"])
    values = first_group.column("Scenario")
    if len(values) == 0:
        raise ValueError(f"No rows found in {path}")
    return str(values[0].as_py())


def allocate_integer_quotas(
    row_counts: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """Allocate an exact proportional sample across scenarios."""
    total_rows = sum(row_counts.values())
    target_total = min(target_total, total_rows)

    raw = {
        scenario: target_total * count / total_rows
        for scenario, count in row_counts.items()
    }
    quotas = {
        scenario: int(math.floor(value))
        for scenario, value in raw.items()
    }

    remainder = target_total - sum(quotas.values())
    ranked = sorted(
        row_counts,
        key=lambda scenario: raw[scenario] - quotas[scenario],
        reverse=True,
    )

    for scenario in ranked[:remainder]:
        quotas[scenario] += 1

    return quotas


def build_global_training_pool(
    scenario_files: dict[str, Path],
    feature_names: list[str],
    pool_size: int,
) -> dict[str, pd.DataFrame]:
    """
    Build one deterministic, scenario-proportional sample pool.

    Each holdout fold excludes the held-out scenario from this pool. Reading and
    sampling each file once keeps the eight-fold experiment practical.
    """
    row_counts = {
        scenario: int(pq.ParquetFile(path).metadata.num_rows)
        for scenario, path in scenario_files.items()
    }
    quotas = allocate_integer_quotas(row_counts, pool_size)

    selected_columns = feature_names + ["BinaryLabel"]
    pool: dict[str, pd.DataFrame] = {}

    print("[1/6] Building the reusable scenario-proportional training pool...")
    print(f"      Available rows: {sum(row_counts.values()):,}")
    print(f"      Pool rows:      {sum(quotas.values()):,}")

    for index, scenario in enumerate(sorted(scenario_files), start=1):
        path = scenario_files[scenario]
        quota = quotas[scenario]

        print(
            f"      [{index}/{len(scenario_files)}] {scenario}: "
            f"reading {row_counts[scenario]:,}, retaining {quota:,}"
        )

        frame = pd.read_parquet(
            path,
            columns=selected_columns,
            engine="pyarrow",
        )

        if quota < len(frame):
            frame = frame.sample(
                n=quota,
                replace=False,
                random_state=RANDOM_STATE + index,
            )

        pool[scenario] = frame.reset_index(drop=True)
        del frame
        gc.collect()

    return pool


def get_attack_probabilities(
    pipeline: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return P(ATTACK) regardless of class-column order."""
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.named_steps["classifier"].classes_)

    if 1 not in classes:
        return np.zeros(len(features), dtype=float)

    return probabilities[:, classes.index(1)].astype(float)


def fit_fresh_pipeline(
    template_pipeline: Any,
    features: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[Any, float]:
    """Clone and fit a fresh pipeline without reusing fitted Step 3 state."""
    pipeline = clone(template_pipeline)
    classifier = pipeline.named_steps["classifier"]

    start = time.perf_counter()

    if isinstance(classifier, HistGradientBoostingClassifier):
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

    return pipeline, time.perf_counter() - start


def metric_value(
    numerator: int,
    denominator: int,
) -> float:
    return numerator / denominator if denominator else float("nan")


def calculate_metrics(
    holdout_scenario: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    training_rows: int,
    fit_seconds: float,
    prediction_seconds: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    actual_benign = int(tn + fp)
    actual_attack = int(tp + fn)
    both_classes = actual_benign > 0 and actual_attack > 0

    precision_attack = precision_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    recall_attack = (
        recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )
        if actual_attack
        else float("nan")
    )
    f1_attack = (
        f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )
        if actual_attack
        else float("nan")
    )
    f2_attack = (
        fbeta_score(
            y_true,
            predictions,
            beta=2,
            pos_label=1,
            zero_division=0,
        )
        if actual_attack
        else float("nan")
    )

    throughput = (
        len(y_true) / prediction_seconds
        if prediction_seconds > 0
        else float("inf")
    )

    return {
        "holdout_scenario": holdout_scenario,
        "training_rows": training_rows,
        "test_rows": len(y_true),
        "actual_benign": actual_benign,
        "actual_attack": actual_attack,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_throughput_flows_per_second": throughput,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, predictions)
            if both_classes
            else float("nan")
        ),
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "f2_attack": f2_attack,
        "specificity": metric_value(int(tn), actual_benign),
        "false_positive_rate": metric_value(int(fp), actual_benign),
        "false_negative_rate": metric_value(int(fn), actual_attack),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if both_classes
            else float("nan")
        ),
        "pr_auc": (
            average_precision_score(y_true, probabilities)
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
        display_labels=["BENIGN", "ATTACK"],
        values_format=",d",
        colorbar=False,
        ax=ax,
    )
    display.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def label_detection_rows(
    metadata: pd.DataFrame,
    holdout_scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create per-original-label and per-family detection summaries."""
    label_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for label, group in metadata.groupby("Label", dropna=False):
        actual_attack = int(group["BinaryLabel"].iloc[0]) == 1
        predicted_attack_count = int(group["PredictedBinaryLabel"].sum())
        total = len(group)

        label_rows.append(
            {
                "holdout_scenario": holdout_scenario,
                "label": str(label),
                "actual_class": "ATTACK" if actual_attack else "BENIGN",
                "rows": total,
                "predicted_attack_rows": predicted_attack_count,
                "attack_prediction_rate": predicted_attack_count / total,
                "detection_rate_if_attack": (
                    predicted_attack_count / total
                    if actual_attack
                    else float("nan")
                ),
                "false_positive_rate_if_benign": (
                    predicted_attack_count / total
                    if not actual_attack
                    else float("nan")
                ),
            }
        )

    for family, group in metadata.groupby("AttackFamily", dropna=False):
        actual_attack = int(group["BinaryLabel"].iloc[0]) == 1
        predicted_attack_count = int(group["PredictedBinaryLabel"].sum())
        total = len(group)

        family_rows.append(
            {
                "holdout_scenario": holdout_scenario,
                "attack_family": str(family),
                "actual_class": "ATTACK" if actual_attack else "BENIGN",
                "rows": total,
                "predicted_attack_rows": predicted_attack_count,
                "attack_prediction_rate": predicted_attack_count / total,
                "detection_rate_if_attack": (
                    predicted_attack_count / total
                    if actual_attack
                    else float("nan")
                ),
                "false_positive_rate_if_benign": (
                    predicted_attack_count / total
                    if not actual_attack
                    else float("nan")
                ),
            }
        )

    return label_rows, family_rows


def save_aggregate_figures(
    metrics: pd.DataFrame,
    label_detection: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    figures_dir: Path,
) -> None:
    """Create scenario-level and aggregate generalization figures."""
    indexed = metrics.set_index("holdout_scenario")

    error_rates = indexed[
        ["false_positive_rate", "false_negative_rate"]
    ].copy()
    ax = error_rates.plot(kind="bar", figsize=(13, 7))
    ax.set_ylabel("Rate")
    ax.set_xlabel("Completely held-out traffic scenario")
    ax.set_title("Scenario-Holdout Binary NIDS Error Rates")
    ax.legend(["False-positive rate", "False-negative rate"])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "scenario_holdout_error_rates.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    attack_recall = metrics.loc[
        metrics["actual_attack"] > 0,
        ["holdout_scenario", "recall_attack"],
    ].set_index("holdout_scenario")

    ax = attack_recall.plot(kind="bar", legend=False, figsize=(12, 7))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack recall")
    ax.set_xlabel("Completely held-out traffic scenario")
    ax.set_title("Scenario-Holdout Attack Recall")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "scenario_holdout_attack_recall.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    attack_labels = label_detection.loc[
        label_detection["actual_class"] == "ATTACK",
        ["label", "rows", "detection_rate_if_attack"],
    ].copy()

    if not attack_labels.empty:
        attack_labels = (
            attack_labels.groupby("label", as_index=False)
            .agg(
                rows=("rows", "sum"),
                detection_rate_if_attack=(
                    "detection_rate_if_attack",
                    "mean",
                ),
            )
            .sort_values("detection_rate_if_attack")
        )

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh(
            attack_labels["label"],
            attack_labels["detection_rate_if_attack"],
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Detection rate")
        ax.set_ylabel("Attack label in held-out scenario")
        ax.set_title("Unseen-Scenario Detection Rate by Attack Label")
        fig.tight_layout()
        fig.savefig(
            figures_dir / "scenario_holdout_attack_label_detection.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    predictions = (probabilities >= threshold).astype("int8")
    save_confusion_matrix(
        y_true,
        predictions,
        "Aggregate Scenario-Holdout Binary NIDS Confusion Matrix",
        figures_dir / "scenario_holdout_aggregate_confusion_matrix.png",
    )

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"Scenario holdout (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Aggregate Scenario-Holdout ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "scenario_holdout_aggregate_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    pr_auc = average_precision_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall_values,
        precision_values,
        label=f"Scenario holdout (AP={pr_auc:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Aggregate Scenario-Holdout Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "scenario_holdout_aggregate_precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected binary NIDS model with complete scenario holdouts."
    )
    parser.add_argument(
        "--training-pool-size",
        type=int,
        default=300_000,
        help=(
            "Total reusable sample pool drawn across all eight scenarios. "
            "Each fold excludes the held-out scenario. Default: 300000."
        ),
    )
    args = parser.parse_args()

    if args.training_pool_size < 50_000:
        print(
            "ERROR: --training-pool-size must be at least 50000.",
            file=sys.stderr,
        )
        return 2

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed" / "machine_learning"
    model_path = (
        project_root
        / "models"
        / "binary"
        / "best_binary_nids_pipeline.joblib"
    )

    reports_dir = project_root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    predictions_dir = reports_dir / "predictions" / "scenario_holdout_errors"

    for directory in (tables_dir, figures_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if len(parquet_files) != 8:
        print(
            f"ERROR: Expected 8 processed Parquet files, found {len(parquet_files)}.",
            file=sys.stderr,
        )
        return 3

    if not model_path.exists():
        print(
            f"ERROR: Missing Step 3 model artifact: {model_path}",
            file=sys.stderr,
        )
        return 4

    artifact = joblib.load(model_path)
    template_pipeline = artifact["pipeline"]
    model_name = str(artifact["model_name"])
    feature_names = list(artifact["feature_names"])
    threshold = float(artifact.get("threshold", 0.50))

    scenario_files = {
        parquet_scenario(path): path
        for path in parquet_files
    }

    if len(scenario_files) != 8:
        print(
            "ERROR: Scenario names were not unique across the eight Parquet files.",
            file=sys.stderr,
        )
        return 5

    started = datetime.now(timezone.utc)

    print("AI NIDS scenario-holdout evaluation")
    print("=" * 76)
    print(f"Selected Step 3 model family: {model_name}")
    print(f"Predictive features:          {len(feature_names)}")
    print(f"Decision threshold:           {threshold:.2f}")
    print()

    training_pool = build_global_training_pool(
        scenario_files=scenario_files,
        feature_names=feature_names,
        pool_size=args.training_pool_size,
    )

    print("[2/6] Running eight complete scenario-holdout folds...")

    metric_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    aggregate_true: list[np.ndarray] = []
    aggregate_probabilities: list[np.ndarray] = []

    for fold_number, holdout_scenario in enumerate(
        sorted(scenario_files),
        start=1,
    ):
        print(
            f"      [{fold_number}/{len(scenario_files)}] "
            f"Holding out: {holdout_scenario}"
        )

        training_frames = [
            frame
            for scenario, frame in training_pool.items()
            if scenario != holdout_scenario
        ]
        training = pd.concat(
            training_frames,
            ignore_index=True,
        ).sample(
            frac=1.0,
            random_state=RANDOM_STATE + fold_number,
        ).reset_index(drop=True)

        X_train = training[feature_names]
        y_train = training["BinaryLabel"].astype("int8").to_numpy()

        test_path = scenario_files[holdout_scenario]
        test = pd.read_parquet(
            test_path,
            columns=feature_names + METADATA_COLUMNS,
            engine="pyarrow",
        )

        X_test = test[feature_names]
        y_test = test["BinaryLabel"].astype("int8").to_numpy()

        pipeline, fit_seconds = fit_fresh_pipeline(
            template_pipeline,
            X_train,
            y_train,
        )

        prediction_start = time.perf_counter()
        probabilities = get_attack_probabilities(
            pipeline,
            X_test,
        )
        prediction_seconds = time.perf_counter() - prediction_start
        predictions = (probabilities >= threshold).astype("int8")

        metrics = calculate_metrics(
            holdout_scenario=holdout_scenario,
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
            training_rows=len(y_train),
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
        )
        metric_rows.append(metrics)

        result_metadata = test[METADATA_COLUMNS].copy()
        result_metadata["PredictedBinaryLabel"] = predictions
        result_metadata["PredictedClass"] = pd.Series(predictions).map(
            {0: "BENIGN", 1: "ATTACK"}
        )
        result_metadata["AttackProbability"] = probabilities
        result_metadata["Correct"] = (
            result_metadata["BinaryLabel"].to_numpy() == predictions
        )

        current_label_rows, current_family_rows = label_detection_rows(
            result_metadata,
            holdout_scenario,
        )
        label_rows.extend(current_label_rows)
        family_rows.extend(current_family_rows)

        errors = result_metadata.loc[~result_metadata["Correct"]].copy()
        errors.to_parquet(
            predictions_dir / f"{safe_name(holdout_scenario)}_errors.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        errors.head(5_000).to_csv(
            predictions_dir / f"{safe_name(holdout_scenario)}_errors_preview.csv",
            index=False,
            encoding="utf-8",
        )

        save_confusion_matrix(
            y_test,
            predictions,
            title=f"Scenario Holdout: {holdout_scenario}",
            output_path=(
                figures_dir
                / f"scenario_holdout_confusion_matrix_{safe_name(holdout_scenario)}.png"
            ),
        )

        aggregate_true.append(y_test.copy())
        aggregate_probabilities.append(probabilities.copy())

        print(
            f"          test={len(y_test):,} | "
            f"attacks={metrics['actual_attack']:,} | "
            f"recall={metrics['recall_attack']:.4f}"
            if metrics["actual_attack"] > 0
            else (
                f"          test={len(y_test):,} | "
                "attacks=0 | recall=N/A"
            )
        )
        print(
            f"          FPR={metrics['false_positive_rate']:.6f} | "
            f"errors={metrics['errors']:,}"
        )

        del (
            training,
            X_train,
            y_train,
            test,
            X_test,
            y_test,
            pipeline,
            result_metadata,
            errors,
        )
        gc.collect()

    print("[3/6] Calculating aggregate scenario-holdout performance...")

    metrics_frame = pd.DataFrame(metric_rows)
    label_frame = pd.DataFrame(label_rows)
    family_frame = pd.DataFrame(family_rows)

    aggregate_y = np.concatenate(aggregate_true)
    aggregate_prob = np.concatenate(aggregate_probabilities)

    aggregate_metrics = calculate_metrics(
        holdout_scenario="ALL_SCENARIOS_AGGREGATED",
        y_true=aggregate_y,
        probabilities=aggregate_prob,
        threshold=threshold,
        training_rows=0,
        fit_seconds=float(metrics_frame["fit_seconds"].sum()),
        prediction_seconds=float(
            metrics_frame["prediction_seconds"].sum()
        ),
    )

    print("[4/6] Writing scenario, label, and family tables...")

    metrics_frame.to_csv(
        tables_dir / "scenario_holdout_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    label_frame.to_csv(
        tables_dir / "scenario_holdout_attack_label_detection.csv",
        index=False,
        encoding="utf-8",
    )
    family_frame.to_csv(
        tables_dir / "scenario_holdout_attack_family_detection.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame([aggregate_metrics]).to_csv(
        tables_dir / "scenario_holdout_aggregate_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    print("[5/6] Creating generalization figures...")

    save_aggregate_figures(
        metrics=metrics_frame,
        label_detection=label_frame,
        y_true=aggregate_y,
        probabilities=aggregate_prob,
        threshold=threshold,
        figures_dir=figures_dir,
    )

    print("[6/6] Saving the scenario-holdout summary...")

    summary = {
        "model_family_evaluated": model_name,
        "feature_count": len(feature_names),
        "decision_threshold": threshold,
        "training_pool_size_requested": args.training_pool_size,
        "scenarios_evaluated": sorted(scenario_files),
        "evaluation_method": (
            "Eight-fold leave-one-scenario-out evaluation. Each fold trained a "
            "fresh clone of the Step 3 selected pipeline on the reusable sample "
            "pool after excluding the complete held-out scenario, then tested "
            "on every valid row in that scenario."
        ),
        "aggregate_metrics": aggregate_metrics,
        "experiment_started_utc": started.isoformat(),
        "experiment_completed_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            (
                "Scenario holdout is stronger than a random flow-level split, "
                "but all data still originates from the same historical dataset."
            ),
            (
                "A scenario with no attacks cannot produce attack recall or "
                "ROC/PR-AUC by itself; it remains useful for false-positive analysis."
            ),
            (
                "Very rare attack labels have unstable estimates because they "
                "contain only a small number of examples."
            ),
        ],
    }

    with (
        reports_dir / "scenario_holdout_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "SCENARIO-HOLDOUT GENERALIZATION SUMMARY",
        "=" * 76,
        "",
        f"Model family evaluated: {model_name}",
        f"Predictive features: {len(feature_names)}",
        f"Decision threshold: {threshold:.2f}",
        f"Reusable training pool: {args.training_pool_size:,} rows",
        f"Complete scenarios held out: {len(scenario_files)}",
        "",
        "Aggregate performance across all scenario holdouts:",
        f"  Rows evaluated:         {aggregate_metrics['test_rows']:,}",
        f"  Accuracy:               {aggregate_metrics['accuracy']:.6f}",
        f"  Balanced accuracy:      {aggregate_metrics['balanced_accuracy']:.6f}",
        f"  Attack precision:       {aggregate_metrics['precision_attack']:.6f}",
        f"  Attack recall:          {aggregate_metrics['recall_attack']:.6f}",
        f"  Attack F1-score:        {aggregate_metrics['f1_attack']:.6f}",
        f"  Attack F2-score:        {aggregate_metrics['f2_attack']:.6f}",
        f"  Specificity:            {aggregate_metrics['specificity']:.6f}",
        f"  False-positive rate:    {aggregate_metrics['false_positive_rate']:.6f}",
        f"  False-negative rate:    {aggregate_metrics['false_negative_rate']:.6f}",
        f"  ROC-AUC:                {aggregate_metrics['roc_auc']:.6f}",
        f"  PR-AUC:                 {aggregate_metrics['pr_auc']:.6f}",
        "",
        "Aggregate confusion matrix:",
        f"  True negatives:  {aggregate_metrics['true_negative']:,}",
        f"  False positives: {aggregate_metrics['false_positive']:,}",
        f"  False negatives: {aggregate_metrics['false_negative']:,}",
        f"  True positives:  {aggregate_metrics['true_positive']:,}",
        "",
        "Interpretation:",
        "  Each row was predicted by a fresh model that did not train on that",
        "  row's source traffic scenario. This is a stronger generalization test",
        "  than the Step 3 random, stratified flow-level baseline.",
        "",
        "Important limitations:",
        "  - All scenarios still come from one historical dataset.",
        "  - Some scenarios contain no attacks or very few rare attacks.",
        "  - This evaluation does not replace testing on newer external traffic.",
        "",
        "Reports created:",
        f"  {tables_dir / 'scenario_holdout_metrics.csv'}",
        f"  {tables_dir / 'scenario_holdout_attack_label_detection.csv'}",
        f"  {tables_dir / 'scenario_holdout_attack_family_detection.csv'}",
        f"  {tables_dir / 'scenario_holdout_aggregate_metrics.csv'}",
        f"  {figures_dir / 'scenario_holdout_aggregate_confusion_matrix.png'}",
        f"  {figures_dir / 'scenario_holdout_error_rates.png'}",
        f"  {figures_dir / 'scenario_holdout_attack_recall.png'}",
        "",
        "Scenario-holdout evaluation completed successfully.",
    ]

    summary_path = reports_dir / "scenario_holdout_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("[COMPLETE] Scenario-holdout evaluation finished.")
    print(
        f"Aggregate attack recall: "
        f"{aggregate_metrics['recall_attack']:.4%}"
    )
    print(
        f"Aggregate false-positive rate: "
        f"{aggregate_metrics['false_positive_rate']:.4%}"
    )
    print(f"Aggregate PR-AUC: {aggregate_metrics['pr_auc']:.6f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
