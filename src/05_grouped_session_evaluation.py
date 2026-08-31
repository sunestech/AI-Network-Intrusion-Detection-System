#!/usr/bin/env python3
r"""
05_grouped_session_evaluation.py

Group-aware known-attack evaluation for the AI-NIDS binary detector.

Run from the project root:
    python .\src\05_grouped_session_evaluation.py

Purpose
-------
Step 3 used a random flow-level split and Step 4 held out complete scenario
files. Step 4 is a deliberately harsh, zero-shot stress test because each file
is strongly tied to one or more attack classes; holding out the file often
removes that attack class from training entirely.

This script adds the missing middle ground:

- It samples aligned feature and generated-flow context records.
- It creates a session-like GroupID using scenario, source IP, destination IP,
  destination port, protocol, and a time bucket.
- It uses StratifiedGroupKFold to keep every GroupID in exactly one split.
- All traffic scenarios can still appear in training, validation, and testing,
  so this evaluates new sessions of known traffic families rather than
  zero-shot recognition of an attack class never seen in training.
- It selects a decision threshold on validation data only, refits the selected
  Step 3 model family on training + validation, and evaluates once on the
  untouched grouped test set.

The raw and processed source data are never modified.
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
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
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
from sklearn.model_selection import StratifiedGroupKFold
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


def scenario_from_parquet(path: Path) -> str:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=["Scenario"])
    if table.num_rows == 0:
        raise ValueError(f"No rows found in {path}")
    return str(table.column("Scenario")[0].as_py())


def allocate_quotas(
    row_counts: dict[str, int],
    target_total: int,
) -> dict[str, int]:
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


def parse_timestamp(series: pd.Series) -> pd.Series:
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
    timestamps = parse_timestamp(context["Timestamp"])
    buckets = (
        timestamps.dt.floor(f"{group_minutes}min")
        .astype("string")
        .fillna("NO_TIME")
    )

    def normalized_text(column: str) -> pd.Series:
        return (
            context[column]
            .astype("string")
            .fillna("MISSING")
            .str.strip()
        )

    return (
        pd.Series([scenario] * len(context), dtype="string")
        + "|"
        + normalized_text("Source IP")
        + "|"
        + normalized_text("Destination IP")
        + "|"
        + normalized_text("Destination Port")
        + "|"
        + normalized_text("Protocol")
        + "|"
        + buckets
    )


def build_aligned_sample(
    feature_files: dict[str, Path],
    context_files: dict[str, Path],
    feature_names: list[str],
    sample_size: int,
    group_minutes: int,
) -> pd.DataFrame:
    row_counts = {
        scenario: int(pq.ParquetFile(path).metadata.num_rows)
        for scenario, path in feature_files.items()
    }
    quotas = allocate_quotas(row_counts, sample_size)

    frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(RANDOM_STATE)

    print("[1/7] Building an aligned feature + context modeling sample...")
    print(f"      Available rows: {sum(row_counts.values()):,}")
    print(f"      Requested rows: {sum(quotas.values()):,}")

    for index, scenario in enumerate(sorted(feature_files), start=1):
        feature_path = feature_files[scenario]
        context_path = context_files[scenario]
        row_count = row_counts[scenario]
        quota = quotas[scenario]

        print(
            f"      [{index}/{len(feature_files)}] {scenario}: "
            f"reading {row_count:,}, retaining {quota:,}"
        )

        feature_frame = pd.read_parquet(
            feature_path,
            columns=FEATURE_METADATA + feature_names,
            engine="pyarrow",
        )
        context_frame = pd.read_parquet(
            context_path,
            columns=CONTEXT_COLUMNS,
            engine="pyarrow",
        )

        if len(feature_frame) != len(context_frame):
            raise RuntimeError(
                f"Row-count mismatch for {scenario}: "
                f"{len(feature_frame)} vs {len(context_frame)}"
            )

        if not np.array_equal(
            feature_frame["RowID"].to_numpy(),
            context_frame["RowID"].to_numpy(),
        ):
            raise RuntimeError(
                f"RowID alignment failed for {scenario}."
            )

        if quota < row_count:
            positions = np.sort(
                rng.choice(
                    row_count,
                    size=quota,
                    replace=False,
                )
            )
            sampled_features = feature_frame.iloc[positions].reset_index(drop=True)
            sampled_context = context_frame.iloc[positions].reset_index(drop=True)
        else:
            sampled_features = feature_frame.reset_index(drop=True)
            sampled_context = context_frame.reset_index(drop=True)

        sampled_features["GroupID"] = build_group_id(
            sampled_context,
            scenario=scenario,
            group_minutes=group_minutes,
        )
        sampled_features["Source IP"] = sampled_context["Source IP"].astype("string")
        sampled_features["Destination IP"] = sampled_context[
            "Destination IP"
        ].astype("string")
        sampled_features["Destination Port Context"] = sampled_context[
            "Destination Port"
        ]
        sampled_features["Protocol Context"] = sampled_context["Protocol"]
        sampled_features["Timestamp"] = sampled_context["Timestamp"].astype("string")

        frames.append(sampled_features)

        del (
            feature_frame,
            context_frame,
            sampled_features,
            sampled_context,
        )
        gc.collect()

    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return sample


def make_strata(data: pd.DataFrame) -> pd.Series:
    """
    Balance the grouped split across scenario and binary class.

    Each scenario remains represented in the three splits when enough groups
    exist, while GroupID prevents session-like groups from crossing boundaries.
    """
    return (
        data["Scenario"].astype("string")
        + "::"
        + data["BinaryLabel"].astype("string")
    )


def grouped_split(
    data: pd.DataFrame,
) -> dict[str, np.ndarray]:
    strata = make_strata(data)
    groups = data["GroupID"].astype("string")
    placeholder = np.zeros((len(data), 1), dtype=np.int8)

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(
        outer.split(
            placeholder,
            strata,
            groups,
        )
    )

    remaining = data.iloc[train_val_idx].reset_index()
    remaining_strata = make_strata(remaining)
    remaining_groups = remaining["GroupID"].astype("string")
    remaining_placeholder = np.zeros(
        (len(remaining), 1),
        dtype=np.int8,
    )

    inner = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )
    inner_train_pos, validation_pos = next(
        inner.split(
            remaining_placeholder,
            remaining_strata,
            remaining_groups,
        )
    )

    train_idx = remaining.iloc[inner_train_pos]["index"].to_numpy(dtype=np.int64)
    validation_idx = remaining.iloc[validation_pos]["index"].to_numpy(dtype=np.int64)

    return {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx.astype(np.int64),
    }


def validate_group_isolation(
    data: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> dict[str, int]:
    group_sets = {
        name: set(data.iloc[indices]["GroupID"].astype(str))
        for name, indices in splits.items()
    }

    overlaps = {
        "train_validation_overlap": len(
            group_sets["train"] & group_sets["validation"]
        ),
        "train_test_overlap": len(
            group_sets["train"] & group_sets["test"]
        ),
        "validation_test_overlap": len(
            group_sets["validation"] & group_sets["test"]
        ),
    }

    if any(overlaps.values()):
        raise RuntimeError(
            f"Group isolation failed: {overlaps}"
        )

    return overlaps


def fit_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    labels: np.ndarray,
) -> float:
    classifier = pipeline.named_steps["classifier"]
    start = time.perf_counter()

    if isinstance(classifier, HistGradientBoostingClassifier):
        weights = compute_sample_weight(
            class_weight="balanced",
            y=labels,
        )
        pipeline.fit(
            features,
            labels,
            classifier__sample_weight=weights,
        )
    else:
        pipeline.fit(features, labels)

    return time.perf_counter() - start


def get_attack_probability(
    pipeline: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.named_steps["classifier"].classes_)

    if 1 not in classes:
        return np.zeros(len(features), dtype=float)

    return probabilities[:, classes.index(1)].astype(float)


def metric_components(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    benign = int(tn + fp)
    attack = int(tp + fn)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision_attack": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_attack": recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f1_attack": f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f2_attack": fbeta_score(
            y_true,
            predictions,
            beta=2,
            pos_label=1,
            zero_division=0,
        ),
        "specificity": tn / benign if benign else float("nan"),
        "false_positive_rate": fp / benign if benign else float("nan"),
        "false_negative_rate": fn / attack if attack else float("nan"),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "matthews_correlation": matthews_corrcoef(
            y_true,
            predictions,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "errors": int(fp + fn),
    }


def select_threshold(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    max_fpr: float,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(
                    probabilities,
                    np.linspace(0.01, 0.99, 99),
                ),
            ]
        )
    )

    for threshold in thresholds:
        metrics = metric_components(
            y_validation,
            probabilities,
            float(threshold),
        )
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    eligible = frame.loc[
        frame["false_positive_rate"] <= max_fpr
    ].copy()

    if eligible.empty:
        selected = frame.sort_values(
            ["f2_attack", "recall_attack", "precision_attack"],
            ascending=False,
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["f2_attack", "recall_attack", "precision_attack"],
            ascending=False,
        ).iloc[0]

    return float(selected["threshold"]), frame


def split_distribution(
    data: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for split_name, indices in splits.items():
        subset = data.iloc[indices]
        table = (
            subset.groupby(
                [
                    "Scenario",
                    "Label",
                    "AttackFamily",
                    "BinaryLabel",
                ],
                dropna=False,
            )
            .size()
            .rename("rows")
            .reset_index()
        )
        table.insert(0, "split", split_name)
        rows.append(table)

    return pd.concat(rows, ignore_index=True)


def label_coverage(
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    pivot = distribution.pivot_table(
        index=["Label", "AttackFamily", "BinaryLabel"],
        columns="split",
        values="rows",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for column in ("train", "validation", "test"):
        if column not in pivot.columns:
            pivot[column] = 0

    pivot["seen_in_training"] = pivot["train"] > 0
    pivot["represented_in_all_splits"] = (
        (pivot["train"] > 0)
        & (pivot["validation"] > 0)
        & (pivot["test"] > 0)
    )

    return pivot


def save_confusion_matrix(
    y_true: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    title: str,
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


def save_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    figures_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"Grouped test (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Group-Aware Known-Attack ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "grouped_session_test_roc_curve.png",
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
        label=f"Grouped test (AP={pr_auc:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Group-Aware Known-Attack Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "grouped_session_test_precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_probability_histogram(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        probabilities[y_true == 0],
        bins=50,
        alpha=0.65,
        label="BENIGN",
    )
    ax.hist(
        probabilities[y_true == 1],
        bins=50,
        alpha=0.65,
        label="ATTACK",
    )
    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Selected threshold={threshold:.3f}",
    )
    ax.set_xlabel("Predicted attack probability")
    ax.set_ylabel("Flow count")
    ax.set_title("Group-Aware Test Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = [
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f2_attack",
        "pr_auc",
    ]
    plot_frame = comparison.set_index("evaluation")[metrics]
    ax = plot_frame.plot(kind="bar", figsize=(12, 7))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation design")
    ax.set_title("Binary NIDS Evaluation Design Comparison")
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a group-aware, known-attack binary NIDS evaluation."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=400_000,
        help="Aligned feature/context sample size. Default: 400000.",
    )
    parser.add_argument(
        "--group-minutes",
        type=int,
        default=15,
        help="Time-bucket width used in GroupID. Default: 15.",
    )
    parser.add_argument(
        "--max-validation-fpr",
        type=float,
        default=0.01,
        help=(
            "Maximum validation false-positive rate allowed during threshold "
            "selection. Default: 0.01."
        ),
    )
    args = parser.parse_args()

    if args.sample_size < 50_000:
        print("ERROR: --sample-size must be at least 50000.", file=sys.stderr)
        return 2

    if args.group_minutes < 1:
        print("ERROR: --group-minutes must be at least 1.", file=sys.stderr)
        return 3

    project_root = Path(__file__).resolve().parents[1]
    feature_dir = project_root / "data" / "processed" / "machine_learning"
    context_dir = project_root / "data" / "processed" / "generated_context"
    artifact_path = (
        project_root
        / "models"
        / "binary"
        / "best_binary_nids_pipeline.joblib"
    )

    reports_dir = project_root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    predictions_dir = reports_dir / "predictions"
    models_dir = project_root / "models" / "binary"

    for directory in (
        reports_dir,
        tables_dir,
        figures_dir,
        predictions_dir,
        models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        print(
            f"ERROR: Missing Step 3 artifact: {artifact_path}",
            file=sys.stderr,
        )
        return 4

    artifact = joblib.load(artifact_path)
    template_pipeline = artifact["pipeline"]
    model_name = str(artifact["model_name"])
    feature_names = list(artifact["feature_names"])

    feature_paths = sorted(feature_dir.glob("*.parquet"))
    context_paths = sorted(context_dir.glob("*.parquet"))

    if len(feature_paths) != 8 or len(context_paths) != 8:
        print(
            "ERROR: Expected eight feature and eight context Parquet files.",
            file=sys.stderr,
        )
        return 5

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
            "ERROR: Feature and context scenario sets do not match.",
            file=sys.stderr,
        )
        return 6

    started = datetime.now(timezone.utc)

    print("AI NIDS group-aware known-attack evaluation")
    print("=" * 76)
    print(f"Model family:      {model_name}")
    print(f"Features:          {len(feature_names)}")
    print(f"Group time bucket: {args.group_minutes} minutes")
    print()

    data = build_aligned_sample(
        feature_files=feature_files,
        context_files=context_files,
        feature_names=feature_names,
        sample_size=args.sample_size,
        group_minutes=args.group_minutes,
    )

    print("[2/7] Creating isolated training, validation, and test groups...")
    splits = grouped_split(data)
    overlaps = validate_group_isolation(data, splits)

    distribution = split_distribution(data, splits)
    coverage = label_coverage(distribution)

    distribution.to_csv(
        tables_dir / "grouped_session_split_distribution.csv",
        index=False,
        encoding="utf-8",
    )
    coverage.to_csv(
        tables_dir / "grouped_session_label_coverage.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"      Training rows:   {len(splits['train']):,}")
    print(f"      Validation rows: {len(splits['validation']):,}")
    print(f"      Test rows:       {len(splits['test']):,}")
    print(
        f"      Unique groups:   "
        f"{data['GroupID'].nunique():,}"
    )
    print(
        f"      Group overlaps:  "
        f"{sum(overlaps.values())}"
    )

    X_train = data.iloc[splits["train"]][feature_names].reset_index(drop=True)
    y_train = (
        data.iloc[splits["train"]]["BinaryLabel"]
        .astype("int8")
        .to_numpy()
    )

    X_validation = data.iloc[splits["validation"]][
        feature_names
    ].reset_index(drop=True)
    y_validation = (
        data.iloc[splits["validation"]]["BinaryLabel"]
        .astype("int8")
        .to_numpy()
    )

    X_test = data.iloc[splits["test"]][feature_names].reset_index(drop=True)
    y_test = (
        data.iloc[splits["test"]]["BinaryLabel"]
        .astype("int8")
        .to_numpy()
    )

    print("[3/7] Fitting the Step 3 selected model family on grouped training data...")
    initial_pipeline = clone(template_pipeline)
    initial_fit_seconds = fit_pipeline(
        initial_pipeline,
        X_train,
        y_train,
    )

    validation_probability = get_attack_probability(
        initial_pipeline,
        X_validation,
    )

    print("[4/7] Selecting the decision threshold on validation data only...")
    selected_threshold, threshold_table = select_threshold(
        y_validation,
        validation_probability,
        max_fpr=args.max_validation_fpr,
    )
    threshold_table.to_csv(
        tables_dir / "grouped_session_threshold_sweep.csv",
        index=False,
        encoding="utf-8",
    )

    validation_metrics = metric_components(
        y_validation,
        validation_probability,
        selected_threshold,
    )

    print(
        f"      Selected threshold: {selected_threshold:.4f}"
    )
    print(
        f"      Validation attack recall: "
        f"{validation_metrics['recall_attack']:.4%}"
    )
    print(
        f"      Validation FPR: "
        f"{validation_metrics['false_positive_rate']:.4%}"
    )

    print("[5/7] Refitting on grouped training + validation data...")
    X_final_train = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )
    y_final_train = np.concatenate(
        [y_train, y_validation]
    )

    final_pipeline = clone(template_pipeline)
    final_fit_seconds = fit_pipeline(
        final_pipeline,
        X_final_train,
        y_final_train,
    )

    print("[6/7] Evaluating once on the untouched grouped test set...")
    prediction_start = time.perf_counter()
    test_probability = get_attack_probability(
        final_pipeline,
        X_test,
    )
    prediction_seconds = time.perf_counter() - prediction_start

    test_metrics = metric_components(
        y_test,
        test_probability,
        selected_threshold,
    )
    test_predictions = (
        test_probability >= selected_threshold
    ).astype("int8")

    test_metadata = data.iloc[splits["test"]][
        [
            "RowID",
            "SourceFile",
            "Scenario",
            "GroupID",
            "Label",
            "AttackFamily",
            "BinaryLabel",
            "Source IP",
            "Destination IP",
            "Destination Port Context",
            "Protocol Context",
            "Timestamp",
        ]
    ].reset_index(drop=True)

    test_metadata["PredictedBinaryLabel"] = test_predictions
    test_metadata["PredictedClass"] = pd.Series(test_predictions).map(
        {0: "BENIGN", 1: "ATTACK"}
    )
    test_metadata["AttackProbability"] = test_probability
    test_metadata["Correct"] = (
        test_metadata["BinaryLabel"].to_numpy()
        == test_predictions
    )

    test_metadata.to_parquet(
        predictions_dir / "grouped_session_test_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    test_metadata.to_csv(
        predictions_dir / "grouped_session_test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    print("[7/7] Saving evidence and the grouped-evaluation model artifact...")

    save_confusion_matrix(
        y_test,
        test_predictions,
        figures_dir / "grouped_session_test_confusion_matrix.png",
        title=f"Group-Aware Known-Attack Test: {model_name}",
    )
    save_curves(
        y_test,
        test_probability,
        figures_dir,
    )
    save_probability_histogram(
        y_test,
        test_probability,
        selected_threshold,
        figures_dir / "grouped_session_test_probability_histogram.png",
    )

    random_metrics_path = (
        reports_dir / "binary_final_test_metrics.json"
    )
    scenario_metrics_path = (
        reports_dir / "scenario_holdout_summary.json"
    )

    comparison_rows: list[dict[str, Any]] = []

    if random_metrics_path.exists():
        random_metrics = json.loads(
            random_metrics_path.read_text(encoding="utf-8")
        )
        comparison_rows.append(
            {
                "evaluation": "Random flow split",
                "accuracy": random_metrics["accuracy"],
                "precision_attack": random_metrics["precision_attack"],
                "recall_attack": random_metrics["recall_attack"],
                "f2_attack": random_metrics["f2_attack"],
                "pr_auc": random_metrics["pr_auc"],
                "false_positive_rate": random_metrics["false_positive_rate"],
                "false_negative_rate": random_metrics["false_negative_rate"],
            }
        )

    comparison_rows.append(
        {
            "evaluation": "Grouped known-attack split",
            "accuracy": test_metrics["accuracy"],
            "precision_attack": test_metrics["precision_attack"],
            "recall_attack": test_metrics["recall_attack"],
            "f2_attack": test_metrics["f2_attack"],
            "pr_auc": test_metrics["pr_auc"],
            "false_positive_rate": test_metrics["false_positive_rate"],
            "false_negative_rate": test_metrics["false_negative_rate"],
        }
    )

    if scenario_metrics_path.exists():
        scenario_summary = json.loads(
            scenario_metrics_path.read_text(encoding="utf-8")
        )
        scenario_metrics = scenario_summary["aggregate_metrics"]
        comparison_rows.append(
            {
                "evaluation": "Complete scenario holdout",
                "accuracy": scenario_metrics["accuracy"],
                "precision_attack": scenario_metrics["precision_attack"],
                "recall_attack": scenario_metrics["recall_attack"],
                "f2_attack": scenario_metrics["f2_attack"],
                "pr_auc": scenario_metrics["pr_auc"],
                "false_positive_rate": scenario_metrics[
                    "false_positive_rate"
                ],
                "false_negative_rate": scenario_metrics[
                    "false_negative_rate"
                ],
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        tables_dir / "binary_evaluation_design_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    save_comparison_figure(
        comparison,
        figures_dir / "binary_evaluation_design_comparison.png",
    )

    grouped_artifact = {
        "pipeline": final_pipeline,
        "model_name": model_name,
        "feature_names": feature_names,
        "threshold": selected_threshold,
        "group_definition": (
            "Scenario + Source IP + Destination IP + Destination Port + "
            f"Protocol + {args.group_minutes}-minute timestamp bucket"
        ),
        "label_mapping": {0: "BENIGN", 1: "ATTACK"},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    grouped_model_path = (
        models_dir / "grouped_known_attack_nids_pipeline.joblib"
    )
    joblib.dump(
        grouped_artifact,
        grouped_model_path,
        compress=3,
    )

    labels_missing_from_training = coverage.loc[
        (coverage["test"] > 0)
        & (~coverage["seen_in_training"]),
        "Label",
    ].astype(str).tolist()

    summary = {
        "model_family": model_name,
        "sample_rows": len(data),
        "training_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "unique_groups": int(data["GroupID"].nunique()),
        "group_minutes": args.group_minutes,
        "group_overlap_counts": overlaps,
        "selected_threshold": selected_threshold,
        "maximum_validation_fpr_constraint": args.max_validation_fpr,
        "initial_fit_seconds": initial_fit_seconds,
        "final_fit_seconds": final_fit_seconds,
        "test_prediction_seconds": prediction_seconds,
        "labels_present_in_test_but_absent_from_training": (
            labels_missing_from_training
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "experiment_started_utc": started.isoformat(),
        "experiment_completed_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "This grouped evaluation tests unseen session-like groups while "
            "allowing known traffic scenarios and attack families to remain "
            "represented across the data partitions. It is more leakage-aware "
            "than a random flow split but less severe than complete scenario "
            "holdout, which often removes an entire attack class from training."
        ),
    }

    with (
        reports_dir / "grouped_session_evaluation_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    text_lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "GROUP-AWARE KNOWN-ATTACK EVALUATION SUMMARY",
        "=" * 78,
        "",
        f"Model family: {model_name}",
        f"Aligned sample rows: {len(data):,}",
        f"Training rows: {len(splits['train']):,}",
        f"Validation rows: {len(splits['validation']):,}",
        f"Test rows: {len(splits['test']):,}",
        f"Unique session-like groups: {data['GroupID'].nunique():,}",
        f"Group time bucket: {args.group_minutes} minutes",
        f"Group overlap across splits: {sum(overlaps.values())}",
        "",
        f"Validation-selected threshold: {selected_threshold:.6f}",
        f"Validation FPR constraint: {args.max_validation_fpr:.4%}",
        "",
        "Untouched grouped test performance:",
        f"  Accuracy:             {test_metrics['accuracy']:.6f}",
        f"  Balanced accuracy:    {test_metrics['balanced_accuracy']:.6f}",
        f"  Attack precision:     {test_metrics['precision_attack']:.6f}",
        f"  Attack recall:        {test_metrics['recall_attack']:.6f}",
        f"  Attack F1-score:      {test_metrics['f1_attack']:.6f}",
        f"  Attack F2-score:      {test_metrics['f2_attack']:.6f}",
        f"  Specificity:          {test_metrics['specificity']:.6f}",
        f"  False-positive rate:  {test_metrics['false_positive_rate']:.6f}",
        f"  False-negative rate:  {test_metrics['false_negative_rate']:.6f}",
        f"  ROC-AUC:              {test_metrics['roc_auc']:.6f}",
        f"  PR-AUC:               {test_metrics['pr_auc']:.6f}",
        "",
        "Confusion matrix:",
        f"  True negatives:  {test_metrics['true_negative']:,}",
        f"  False positives: {test_metrics['false_positive']:,}",
        f"  False negatives: {test_metrics['false_negative']:,}",
        f"  True positives:  {test_metrics['true_positive']:,}",
        "",
        "Label coverage warning:",
        (
            "  Test labels absent from training: "
            + (
                ", ".join(labels_missing_from_training)
                if labels_missing_from_training
                else "None"
            )
        ),
        "",
        "Interpretation:",
        "  No session-like GroupID appears in more than one split. Unlike the",
        "  complete scenario holdout, this design allows known attack families",
        "  to remain represented while testing new communication/time groups.",
        "",
        f"Saved grouped model: {grouped_model_path}",
        "",
        "Grouped known-attack evaluation completed successfully.",
    ]

    summary_path = reports_dir / "grouped_session_evaluation_summary.txt"
    summary_path.write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Group-aware known-attack evaluation finished.")
    print(f"Attack recall: {test_metrics['recall_attack']:.4%}")
    print(f"Attack F2:     {test_metrics['f2_attack']:.4%}")
    print(f"FPR:           {test_metrics['false_positive_rate']:.4%}")
    print(f"PR-AUC:        {test_metrics['pr_auc']:.6f}")
    print(f"Summary:       {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
