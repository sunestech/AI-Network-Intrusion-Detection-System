#!/usr/bin/env python3
r"""
03_train_binary_baselines.py

Scalable binary intrusion-detection baseline training for the AI-NIDS project.

Run from the project root:
    python .\src\03_train_binary_baselines.py

Default experiment:
- Draws a deterministic 400,000-row scenario-proportional sample from the
  2,830,743 processed flow records.
- Uses 64% for training, 16% for model validation, and 20% for the final test.
- Fits all preprocessing on training data only.
- Compares:
    1. Dummy prior classifier
    2. Scalable logistic regression via SGD
    3. Random Forest
    4. Histogram Gradient Boosting
- Selects the candidate with the highest attack-class F2 score on validation.
- Refits that selected model on train + validation and evaluates it once on the
  untouched test set.
- Saves the complete preprocessing/model pipeline and reporting evidence.

The script does not modify the raw or processed source datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
THRESHOLD = 0.50

METADATA_COLUMNS = ["RowID", "SourceFile", "Scenario"]
TARGET_COLUMNS = ["BinaryLabel", "Label", "AttackFamily"]


@dataclass
class CandidateResult:
    model: str
    training_rows: int
    validation_rows: int
    fit_seconds: float
    prediction_seconds: float
    prediction_throughput_flows_per_second: float
    accuracy: float
    balanced_accuracy: float
    precision_attack: float
    recall_attack: float
    f1_attack: float
    f2_attack: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float
    pr_auc: float
    matthews_correlation: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def allocate_integer_quotas(
    weights: dict[Path, int],
    target_total: int,
) -> dict[Path, int]:
    """Allocate an exact integer sample total proportionally across files."""
    available_total = sum(weights.values())
    if available_total <= 0:
        raise ValueError("The processed Parquet files contain no rows.")

    target_total = min(target_total, available_total)

    raw = {
        path: target_total * count / available_total
        for path, count in weights.items()
    }
    quotas = {path: int(math.floor(value)) for path, value in raw.items()}

    remainder = target_total - sum(quotas.values())
    ranked = sorted(
        weights,
        key=lambda path: raw[path] - quotas[path],
        reverse=True,
    )

    for path in ranked[:remainder]:
        quotas[path] += 1

    return quotas


def load_scenario_proportional_sample(
    parquet_files: list[Path],
    feature_names: list[str],
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Read one processed Parquet file at a time, draw a deterministic proportional
    sample, release the full file, and concatenate only the retained rows.
    """
    row_counts = {
        path: int(pq.ParquetFile(path).metadata.num_rows)
        for path in parquet_files
    }
    total_rows = sum(row_counts.values())
    quotas = allocate_integer_quotas(row_counts, sample_size)

    selected_columns = (
        METADATA_COLUMNS
        + feature_names
        + TARGET_COLUMNS
    )

    sampled_frames: list[pd.DataFrame] = []

    print("[1/7] Building a deterministic scenario-proportional modeling sample...")
    print(f"      Available processed rows: {total_rows:,}")
    print(f"      Requested sample rows:    {min(sample_size, total_rows):,}")

    for index, path in enumerate(parquet_files, start=1):
        quota = quotas[path]
        print(
            f"      [{index}/{len(parquet_files)}] {path.name}: "
            f"reading {row_counts[path]:,}, retaining {quota:,}"
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
                random_state=random_state + index,
            )

        sampled_frames.append(frame)
        del frame
        gc.collect()

    sample = pd.concat(sampled_frames, ignore_index=True)
    sample = sample.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    if len(sample) != min(sample_size, total_rows):
        raise RuntimeError(
            f"Expected {min(sample_size, total_rows):,} sampled rows, "
            f"but created {len(sample):,}."
        )

    return sample


def split_train_validation_test(
    data: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Create a 64/16/20 stratified split.

    First, hold out 20% for final testing. Then split the remaining 80% so that
    20% of it (16% of the original sample) becomes validation data.
    """
    all_indices = np.arange(len(data), dtype=np.int64)
    y_all = data["BinaryLabel"].astype("int8").to_numpy()

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all[train_val_idx],
    )

    def x(indices: np.ndarray) -> pd.DataFrame:
        return data.iloc[indices][feature_names].reset_index(drop=True)

    def y(indices: np.ndarray) -> np.ndarray:
        return (
            data.iloc[indices]["BinaryLabel"]
            .astype("int8")
            .to_numpy()
        )

    return {
        "X_train": x(train_idx),
        "y_train": y(train_idx),
        "X_validation": x(validation_idx),
        "y_validation": y(validation_idx),
        "X_test": x(test_idx),
        "y_test": y(test_idx),
        "test_metadata": data.iloc[test_idx][
            METADATA_COLUMNS + ["Label", "AttackFamily", "BinaryLabel"]
        ].reset_index(drop=True),
        "train_indices": train_idx,
        "validation_indices": validation_idx,
        "test_indices": test_idx,
    }


def build_candidate_factories(
    feature_names: list[str],
    n_jobs: int,
    rf_trees: int,
    hgb_iterations: int,
) -> dict[str, Callable[[], Pipeline]]:
    """Return fresh model pipelines so final refitting cannot reuse test-fitted state."""

    def dummy() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="prior")),
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
                        max_iter=1_500,
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
                        max_depth=20,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        bootstrap=True,
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
    X: pd.DataFrame,
    y: np.ndarray,
) -> float:
    """Fit a pipeline and return elapsed seconds."""
    start = time.perf_counter()

    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
        pipeline.fit(
            X,
            y,
            classifier__sample_weight=sample_weight,
        )
    else:
        pipeline.fit(X, y)

    return time.perf_counter() - start


def get_attack_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    probabilities = pipeline.predict_proba(X)

    classes = list(pipeline.named_steps["classifier"].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)

    attack_index = classes.index(1)
    return probabilities[:, attack_index].astype(float)


def calculate_binary_metrics(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fit_seconds: float,
    prediction_seconds: float,
    training_rows: int,
) -> CandidateResult:
    predictions = (probabilities >= THRESHOLD).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    both_classes = len(np.unique(y_true)) == 2
    roc_auc = (
        roc_auc_score(y_true, probabilities)
        if both_classes
        else float("nan")
    )
    pr_auc = (
        average_precision_score(y_true, probabilities)
        if both_classes
        else float("nan")
    )

    throughput = (
        len(y_true) / prediction_seconds
        if prediction_seconds > 0
        else float("inf")
    )

    return CandidateResult(
        model=model_name,
        training_rows=training_rows,
        validation_rows=len(y_true),
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
        prediction_throughput_flows_per_second=throughput,
        accuracy=accuracy_score(y_true, predictions),
        balanced_accuracy=balanced_accuracy_score(y_true, predictions),
        precision_attack=precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        recall_attack=recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        f1_attack=f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        f2_attack=fbeta_score(
            y_true,
            predictions,
            beta=2,
            pos_label=1,
            zero_division=0,
        ),
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        matthews_correlation=matthews_corrcoef(y_true, predictions),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def save_confusion_matrix_figure(
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
        ax=ax,
        colorbar=False,
    )
    display.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_validation_curve_figures(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (fpr, tpr, auc_value) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Binary NIDS Validation ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_validation_roc_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (recall_values, precision_values, ap_value) in pr_data.items():
        ax.plot(
            recall_values,
            precision_values,
            label=f"{name} (AP={ap_value:.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Binary NIDS Validation Precision–Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_validation_precision_recall_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_model_comparison_figure(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_frame = comparison.set_index("model")[
        ["recall_attack", "precision_attack", "f2_attack", "pr_auc"]
    ]
    ax = plot_frame.plot(kind="bar", figsize=(12, 7))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Candidate model")
    ax.set_title("Binary NIDS Validation Model Comparison")
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_random_forest_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    classifier = pipeline.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    table.to_csv(
        tables_dir / "binary_random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )

    top = table.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random Forest importance")
    ax.set_ylabel("Network-flow feature")
    ax.set_title("Top 20 Binary NIDS Random Forest Features")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_random_forest_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_final_curve_figures(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    figures_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"Selected model (AUC={auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random reference")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Selected Binary NIDS Model: Final Test ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_final_test_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    ap_value = average_precision_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall_values,
        precision_values,
        label=f"Selected model (AP={ap_value:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Selected Binary NIDS Model: Final Test Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_final_test_precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare scalable binary NIDS baselines."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=400_000,
        help="Total processed rows used for the baseline experiment. Default: 400000.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Parallel Random Forest/SGD workers. Default: up to 4.",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=120,
        help="Random Forest tree count. Default: 120.",
    )
    parser.add_argument(
        "--hgb-iterations",
        type=int,
        default=100,
        help="Histogram Gradient Boosting iterations. Default: 100.",
    )
    args = parser.parse_args()

    if args.sample_size < 10_000:
        print("ERROR: --sample-size must be at least 10000.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed" / "machine_learning"
    feature_manifest_path = (
        project_root / "reports" / "tables" / "feature_manifest.csv"
    )

    models_dir = project_root / "models" / "binary"
    figures_dir = project_root / "reports" / "figures"
    tables_dir = project_root / "reports" / "tables"
    predictions_dir = project_root / "reports" / "predictions"

    for directory in (models_dir, figures_dir, tables_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if len(parquet_files) != 8:
        print(
            f"ERROR: Expected 8 processed machine-learning Parquet files, "
            f"found {len(parquet_files)}.",
            file=sys.stderr,
        )
        return 3

    if not feature_manifest_path.exists():
        print(
            f"ERROR: Missing feature manifest: {feature_manifest_path}",
            file=sys.stderr,
        )
        return 4

    feature_manifest = pd.read_csv(feature_manifest_path)
    feature_names = feature_manifest["feature"].astype(str).tolist()

    if len(feature_names) != 77:
        print(
            f"ERROR: Expected 77 predictive features, found {len(feature_names)}.",
            file=sys.stderr,
        )
        return 5

    experiment_started = datetime.now(timezone.utc)

    sample = load_scenario_proportional_sample(
        parquet_files=parquet_files,
        feature_names=feature_names,
        sample_size=args.sample_size,
        random_state=RANDOM_STATE,
    )

    sample_distribution = (
        sample.groupby(["Scenario", "BinaryLabel"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    sample_distribution["class_name"] = sample_distribution["BinaryLabel"].map(
        {0: "BENIGN", 1: "ATTACK"}
    )
    sample_distribution.to_csv(
        tables_dir / "binary_modeling_sample_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    print("[2/7] Creating a stratified train/validation/test split...")
    splits = split_train_validation_test(sample, feature_names)

    print(f"      Training rows:   {len(splits['y_train']):,}")
    print(f"      Validation rows: {len(splits['y_validation']):,}")
    print(f"      Test rows:       {len(splits['y_test']):,}")

    factories = build_candidate_factories(
        feature_names=feature_names,
        n_jobs=args.n_jobs,
        rf_trees=args.rf_trees,
        hgb_iterations=args.hgb_iterations,
    )

    print("[3/7] Comparing binary intrusion-detection candidates...")
    candidate_results: list[CandidateResult] = []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for position, (model_name, factory) in enumerate(factories.items(), start=1):
        print(f"      [{position}/{len(factories)}] Fitting {model_name}...")

        pipeline = factory()
        fit_seconds = fit_pipeline(
            model_name,
            pipeline,
            splits["X_train"],
            splits["y_train"],
        )

        prediction_start = time.perf_counter()
        validation_probabilities = get_attack_probabilities(
            pipeline,
            splits["X_validation"],
        )
        prediction_seconds = time.perf_counter() - prediction_start

        result = calculate_binary_metrics(
            model_name=model_name,
            y_true=splits["y_validation"],
            probabilities=validation_probabilities,
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
            training_rows=len(splits["y_train"]),
        )
        candidate_results.append(result)

        fpr, tpr, _ = roc_curve(
            splits["y_validation"],
            validation_probabilities,
        )
        roc_data[model_name] = (fpr, tpr, result.roc_auc)

        precision_values, recall_values, _ = precision_recall_curve(
            splits["y_validation"],
            validation_probabilities,
        )
        pr_data[model_name] = (
            recall_values,
            precision_values,
            result.pr_auc,
        )

        validation_predictions = (
            validation_probabilities >= THRESHOLD
        ).astype("int8")
        save_confusion_matrix_figure(
            splits["y_validation"],
            validation_predictions,
            title=f"{model_name}: Validation Confusion Matrix",
            output_path=(
                figures_dir
                / f"binary_validation_confusion_matrix_{slugify(model_name)}.png"
            ),
        )

        if model_name == "Random Forest":
            save_random_forest_importance(
                pipeline,
                feature_names,
                tables_dir,
                figures_dir,
            )

        print(
            f"          recall={result.recall_attack:.4f} | "
            f"precision={result.precision_attack:.4f} | "
            f"F2={result.f2_attack:.4f} | "
            f"PR-AUC={result.pr_auc:.4f}"
        )

        del pipeline
        gc.collect()

    comparison = pd.DataFrame(
        [result.as_dict() for result in candidate_results]
    ).sort_values(
        ["f2_attack", "recall_attack", "pr_auc"],
        ascending=False,
    )
    comparison.to_csv(
        tables_dir / "binary_validation_model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    save_validation_curve_figures(roc_data, pr_data, figures_dir)
    save_model_comparison_figure(
        comparison,
        figures_dir / "binary_validation_model_comparison.png",
    )

    selected_model_name = str(comparison.iloc[0]["model"])
    print(
        "[4/7] Selected validation winner by attack F2 score: "
        f"{selected_model_name}"
    )

    print("[5/7] Refitting the selected pipeline on training + validation rows...")
    X_train_final = pd.concat(
        [splits["X_train"], splits["X_validation"]],
        ignore_index=True,
    )
    y_train_final = np.concatenate(
        [splits["y_train"], splits["y_validation"]]
    )

    selected_pipeline = factories[selected_model_name]()
    final_fit_seconds = fit_pipeline(
        selected_model_name,
        selected_pipeline,
        X_train_final,
        y_train_final,
    )

    print("[6/7] Evaluating once on the untouched final test split...")
    test_prediction_start = time.perf_counter()
    test_probabilities = get_attack_probabilities(
        selected_pipeline,
        splits["X_test"],
    )
    test_prediction_seconds = time.perf_counter() - test_prediction_start

    final_result = calculate_binary_metrics(
        model_name=selected_model_name,
        y_true=splits["y_test"],
        probabilities=test_probabilities,
        fit_seconds=final_fit_seconds,
        prediction_seconds=test_prediction_seconds,
        training_rows=len(y_train_final),
    )

    test_predictions = (test_probabilities >= THRESHOLD).astype("int8")
    test_metadata = splits["test_metadata"].copy()
    test_metadata["PredictedBinaryLabel"] = test_predictions
    test_metadata["PredictedClass"] = pd.Series(test_predictions).map(
        {0: "BENIGN", 1: "ATTACK"}
    )
    test_metadata["AttackProbability"] = test_probabilities
    test_metadata["Correct"] = (
        test_metadata["BinaryLabel"].to_numpy() == test_predictions
    )

    test_metadata.to_parquet(
        predictions_dir / "binary_final_test_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    # A compact CSV is convenient for spreadsheet review.
    test_metadata.to_csv(
        predictions_dir / "binary_final_test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    final_metrics = final_result.as_dict()
    final_metrics.update(
        {
            "selected_by": "highest validation attack-class F2 score",
            "threshold": THRESHOLD,
            "sample_rows": len(sample),
            "training_rows_after_refit": len(y_train_final),
            "final_test_rows": len(splits["y_test"]),
            "feature_count": len(feature_names),
            "random_state": RANDOM_STATE,
            "sample_size_requested": args.sample_size,
            "rf_trees": args.rf_trees,
            "hgb_iterations": args.hgb_iterations,
            "n_jobs": args.n_jobs,
            "experiment_started_utc": experiment_started.isoformat(),
            "experiment_completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    with (
        project_root / "reports" / "binary_final_test_metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    classification_text = classification_report(
        splits["y_test"],
        test_predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        digits=6,
        zero_division=0,
    )
    (
        project_root / "reports" / "binary_final_classification_report.txt"
    ).write_text(classification_text, encoding="utf-8")

    confusion = confusion_matrix(
        splits["y_test"],
        test_predictions,
        labels=[0, 1],
    )
    pd.DataFrame(
        confusion,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"],
    ).to_csv(
        tables_dir / "binary_final_confusion_matrix.csv",
        encoding="utf-8",
    )

    save_confusion_matrix_figure(
        splits["y_test"],
        test_predictions,
        title=f"Selected Binary NIDS Model: {selected_model_name}",
        output_path=figures_dir / "binary_final_test_confusion_matrix.png",
    )
    save_final_curve_figures(
        splits["y_test"],
        test_probabilities,
        figures_dir,
    )

    artifact = {
        "pipeline": selected_pipeline,
        "model_name": selected_model_name,
        "feature_names": feature_names,
        "threshold": THRESHOLD,
        "label_mapping": {0: "BENIGN", 1: "ATTACK"},
        "validation_selection_metric": "attack_f2",
        "final_test_metrics": final_metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    model_path = models_dir / "best_binary_nids_pipeline.joblib"
    joblib.dump(artifact, model_path, compress=3)

    print("[7/7] Saving the model artifact and reporting evidence...")

    summary_lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "BINARY BASELINE MODEL SUMMARY",
        "=" * 76,
        "",
        f"Modeling sample rows: {len(sample):,}",
        f"Training rows: {len(splits['y_train']):,}",
        f"Validation rows: {len(splits['y_validation']):,}",
        f"Final test rows: {len(splits['y_test']):,}",
        f"Predictive features: {len(feature_names)}",
        "",
        "Validation-selected model:",
        f"  {selected_model_name}",
        f"  Selection metric: attack-class F2 score",
        "",
        "Untouched final test performance:",
        f"  Accuracy:              {final_result.accuracy:.6f}",
        f"  Balanced accuracy:     {final_result.balanced_accuracy:.6f}",
        f"  Attack precision:      {final_result.precision_attack:.6f}",
        f"  Attack recall:         {final_result.recall_attack:.6f}",
        f"  Attack F1-score:       {final_result.f1_attack:.6f}",
        f"  Attack F2-score:       {final_result.f2_attack:.6f}",
        f"  Specificity:           {final_result.specificity:.6f}",
        f"  False-positive rate:   {final_result.false_positive_rate:.6f}",
        f"  False-negative rate:   {final_result.false_negative_rate:.6f}",
        f"  ROC-AUC:               {final_result.roc_auc:.6f}",
        f"  PR-AUC:                {final_result.pr_auc:.6f}",
        f"  Matthews correlation:  {final_result.matthews_correlation:.6f}",
        "",
        "Final test confusion matrix:",
        f"  True negatives:  {final_result.true_negative:,}",
        f"  False positives: {final_result.false_positive:,}",
        f"  False negatives: {final_result.false_negative:,}",
        f"  True positives:  {final_result.true_positive:,}",
        "",
        f"Saved model: {model_path}",
        "",
        "Important methodological notes:",
        "  - The imputer was fitted inside each pipeline using training data only.",
        "  - The final test split was not used to choose the model.",
        "  - This is a random, stratified flow-level baseline.",
        "  - A later scenario/day-aware evaluation is still required to test",
        "    generalization under stronger leakage controls.",
        "",
        "Binary baseline training completed successfully.",
    ]

    summary_path = project_root / "reports" / "binary_model_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Binary baseline experiment finished.")
    print(f"Selected model: {selected_model_name}")
    print(f"Attack recall:  {final_result.recall_attack:.4%}")
    print(f"Attack F2:      {final_result.f2_attack:.4%}")
    print(f"PR-AUC:         {final_result.pr_auc:.6f}")
    print(f"Model artifact: {model_path}")
    print(f"Summary:        {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
