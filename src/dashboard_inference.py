#!/usr/bin/env python3
r"""
Reusable inference utilities for the AI-NIDS Streamlit dashboard and batch CLI.

This module loads three independently evaluated artifacts:

1. Group-aware binary detector:
   BENIGN versus ATTACK

2. Second-stage family classifier:
   Denial of Service, Port Scan, Brute Force, Web Attack, Other Malicious

3. Benign-trained anomaly detector:
   Unusual relative to the learned benign profile

The fusion severity is an analyst-triage heuristic. It is not itself a fourth
machine-learning model and must not be presented as independently benchmarked.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BINARY_PRIMARY = Path(
    "models/binary/grouped_known_attack_nids_pipeline.joblib"
)
BINARY_FALLBACK = Path(
    "models/binary/best_binary_nids_pipeline.joblib"
)
FAMILY_MODEL = Path(
    "models/attack_family/best_attack_family_pipeline.joblib"
)
ANOMALY_MODEL = Path(
    "models/anomaly/benign_isolation_forest_pipeline.joblib"
)
MITRE_MAPPING = Path("config/mitre_attack_mapping.json")

OPTIONAL_CONTEXT_COLUMNS = [
    "Flow ID",
    "RowID",
    "SourceFile",
    "Scenario",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Context Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
    "AttackFamily",
    "BinaryLabel",
]


@dataclass(frozen=True)
class ModelBundle:
    project_root: Path
    binary_artifact: dict[str, Any]
    family_artifact: dict[str, Any]
    anomaly_artifact: dict[str, Any]
    feature_names: list[str]
    mitre_mapping: dict[str, Any]
    binary_path: Path
    family_path: Path
    anomaly_path: Path


def _load_artifact(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} artifact was not found: {path}"
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(
            f"{label} artifact has an unexpected structure: {path}"
        )
    return artifact


def load_model_bundle(project_root: str | Path) -> ModelBundle:
    root = Path(project_root).resolve()

    binary_path = root / BINARY_PRIMARY
    if not binary_path.exists():
        binary_path = root / BINARY_FALLBACK

    family_path = root / FAMILY_MODEL
    anomaly_path = root / ANOMALY_MODEL

    binary_artifact = _load_artifact(
        binary_path,
        "Binary detector",
    )
    family_artifact = _load_artifact(
        family_path,
        "Attack-family classifier",
    )
    anomaly_artifact = _load_artifact(
        anomaly_path,
        "Anomaly detector",
    )

    binary_features = list(binary_artifact.get("feature_names", []))
    family_features = list(family_artifact.get("feature_names", []))
    anomaly_features = list(anomaly_artifact.get("feature_names", []))

    if not binary_features:
        raise ValueError(
            "The binary artifact does not contain feature_names."
        )

    if family_features != binary_features:
        raise ValueError(
            "Binary and family artifacts do not use the same feature order."
        )

    if anomaly_features != binary_features:
        raise ValueError(
            "Binary and anomaly artifacts do not use the same feature order."
        )

    mapping_path = root / MITRE_MAPPING
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"MITRE mapping file was not found: {mapping_path}"
        )

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    return ModelBundle(
        project_root=root,
        binary_artifact=binary_artifact,
        family_artifact=family_artifact,
        anomaly_artifact=anomaly_artifact,
        feature_names=binary_features,
        mitre_mapping=mapping,
        binary_path=binary_path,
        family_path=family_path,
        anomaly_path=anomaly_path,
    )


def _strip_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in cleaned.columns
    ]

    duplicates = cleaned.columns[
        cleaned.columns.duplicated()
    ].tolist()
    if duplicates:
        raise ValueError(
            "Duplicate column names remain after trimming whitespace: "
            + ", ".join(sorted(set(duplicates)))
        )

    return cleaned


def prepare_features(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    Validate and coerce a dashboard input frame.

    Returns:
        numeric_features,
        data_quality_summary,
        input_frame_with_clean_column_names
    """
    cleaned = _strip_column_names(frame)

    missing = [
        feature
        for feature in feature_names
        if feature not in cleaned.columns
    ]
    if missing:
        preview = ", ".join(missing[:15])
        remainder = len(missing) - min(len(missing), 15)
        suffix = (
            f" (+{remainder} more)"
            if remainder > 0
            else ""
        )
        raise ValueError(
            "The uploaded file is not model-compatible. "
            f"Missing required features: {preview}{suffix}"
        )

    numeric = pd.DataFrame(index=cleaned.index)
    original_non_null = 0
    newly_coerced_missing = 0
    infinity_values = 0

    for feature in feature_names:
        source = cleaned[feature]
        original_non_null += int(source.notna().sum())

        converted = pd.to_numeric(
            source,
            errors="coerce",
        )

        numeric_values = converted.to_numpy(
            dtype="float64",
            copy=False,
        )
        infinite_mask = np.isinf(numeric_values)
        infinity_values += int(infinite_mask.sum())

        converted = converted.mask(infinite_mask)
        numeric[feature] = converted.astype("float32")

        newly_coerced_missing += int(
            converted.isna().sum()
            - source.isna().sum()
            - infinite_mask.sum()
        )

    quality = {
        "rows_received": len(cleaned),
        "required_feature_count": len(feature_names),
        "missing_feature_cells_after_coercion": int(
            numeric.isna().sum().sum()
        ),
        "infinity_values_converted_to_missing": infinity_values,
        "non_numeric_values_coerced_to_missing": max(
            0,
            newly_coerced_missing,
        ),
        "original_non_null_feature_cells": original_non_null,
    }

    return numeric, quality, cleaned


def _positive_class_probability(
    pipeline: Any,
    features: pd.DataFrame,
    positive_class: Any,
) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)

    if positive_class not in classes:
        return np.zeros(len(features), dtype=float)

    return probabilities[:, classes.index(positive_class)].astype(float)


def _anomaly_score(
    anomaly_artifact: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    pipeline = anomaly_artifact["pipeline"]
    threshold = float(
        anomaly_artifact["anomaly_threshold"]
    )

    imputer = pipeline.named_steps["imputer"]
    detector = pipeline.named_steps["detector"]

    transformed = imputer.transform(features)
    score = -detector.score_samples(transformed)

    return score.astype(float), threshold


def _severity(
    known_attack: bool,
    anomaly: bool,
) -> tuple[str, str]:
    if known_attack and anomaly:
        return (
            "Critical",
            "Known attack pattern and anomalous relative to benign behavior",
        )
    if known_attack:
        return (
            "High",
            "Known supervised attack pattern",
        )
    if anomaly:
        return (
            "Medium",
            "Unusual relative to the learned benign profile",
        )
    return (
        "Low",
        "No known attack pattern and no anomaly flag",
    )


def _mapping_for_result(
    family: str,
    known_attack: bool,
    anomaly: bool,
    mapping: dict[str, Any],
) -> dict[str, str]:
    if known_attack:
        record = mapping.get(
            family,
            mapping["Other Malicious"],
        )
    elif anomaly:
        record = mapping["Anomalous / Unmapped"]
    else:
        record = mapping["BENIGN"]

    techniques = record.get("techniques", [])
    technique_ids = "; ".join(
        item.get("id", "")
        for item in techniques
        if item.get("id")
    )
    technique_names = "; ".join(
        item.get("name", "")
        for item in techniques
        if item.get("name")
    )
    technique_urls = "; ".join(
        item.get("url", "")
        for item in techniques
        if item.get("url")
    )

    return {
        "MITRETactic": record.get("tactic", ""),
        "MITRETechniqueID": technique_ids,
        "MITRETechniqueName": technique_names,
        "MITREReference": technique_urls,
        "MITREMappingNote": record.get("note", ""),
    }


def analyze_dataframe(
    frame: pd.DataFrame,
    bundle: ModelBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, quality, cleaned = prepare_features(
        frame,
        bundle.feature_names,
    )

    binary_artifact = bundle.binary_artifact
    binary_pipeline = binary_artifact["pipeline"]
    binary_threshold = float(
        binary_artifact.get("threshold", 0.50)
    )

    known_probability = _positive_class_probability(
        binary_pipeline,
        features,
        positive_class=1,
    )
    known_attack = (
        known_probability >= binary_threshold
    )

    family_prediction = np.full(
        len(cleaned),
        "Not run",
        dtype=object,
    )
    family_confidence = np.full(
        len(cleaned),
        np.nan,
        dtype=float,
    )

    attack_positions = np.flatnonzero(known_attack)
    if len(attack_positions):
        family_pipeline = bundle.family_artifact["pipeline"]
        family_features = features.iloc[attack_positions]

        predicted_family = family_pipeline.predict(
            family_features
        )
        predicted_probability = family_pipeline.predict_proba(
            family_features
        )

        family_prediction[attack_positions] = predicted_family
        family_confidence[attack_positions] = (
            predicted_probability.max(axis=1)
        )

    anomaly_score, anomaly_threshold = _anomaly_score(
        bundle.anomaly_artifact,
        features,
    )
    anomaly_flag = anomaly_score >= anomaly_threshold
    anomaly_margin = anomaly_score - anomaly_threshold

    severity: list[str] = []
    reason: list[str] = []
    mitre_rows: list[dict[str, str]] = []

    for known, anomalous, family in zip(
        known_attack,
        anomaly_flag,
        family_prediction,
    ):
        current_severity, current_reason = _severity(
            bool(known),
            bool(anomalous),
        )
        severity.append(current_severity)
        reason.append(current_reason)
        mitre_rows.append(
            _mapping_for_result(
                family=str(family),
                known_attack=bool(known),
                anomaly=bool(anomalous),
                mapping=bundle.mitre_mapping,
            )
        )

    result = cleaned.copy()
    result["KnownAttackProbability"] = known_probability
    result["BinaryDecisionThreshold"] = binary_threshold
    result["BinaryPrediction"] = np.where(
        known_attack,
        "ATTACK",
        "BENIGN",
    )
    result["PredictedAttackFamily"] = family_prediction
    result["FamilyConfidence"] = family_confidence
    result["AnomalyScore"] = anomaly_score
    result["AnomalyThreshold"] = anomaly_threshold
    result["AnomalyMargin"] = anomaly_margin
    result["AnomalyFlag"] = anomaly_flag
    result["AlertSeverity"] = severity
    result["AlertReason"] = reason

    mitre_frame = pd.DataFrame(mitre_rows)
    for column in mitre_frame.columns:
        result[column] = mitre_frame[column].to_numpy()

    if "BinaryLabel" in result.columns:
        actual_binary = pd.to_numeric(
            result["BinaryLabel"],
            errors="coerce",
        )
        result["BinaryCorrect"] = (
            actual_binary
            == known_attack.astype(int)
        )

    context_columns = [
        column
        for column in OPTIONAL_CONTEXT_COLUMNS
        if column in result.columns
    ]
    prediction_columns = [
        "KnownAttackProbability",
        "BinaryDecisionThreshold",
        "BinaryPrediction",
        "PredictedAttackFamily",
        "FamilyConfidence",
        "AnomalyScore",
        "AnomalyThreshold",
        "AnomalyMargin",
        "AnomalyFlag",
        "AlertSeverity",
        "AlertReason",
        "MITRETactic",
        "MITRETechniqueID",
        "MITRETechniqueName",
        "MITREReference",
        "MITREMappingNote",
    ]
    if "BinaryCorrect" in result.columns:
        prediction_columns.append("BinaryCorrect")

    quality["binary_model"] = str(bundle.binary_path)
    quality["binary_threshold"] = binary_threshold
    quality["family_model"] = str(bundle.family_path)
    quality["anomaly_model"] = str(bundle.anomaly_path)
    quality["anomaly_threshold"] = anomaly_threshold
    quality["known_attack_alerts"] = int(
        known_attack.sum()
    )
    quality["anomaly_alerts"] = int(
        anomaly_flag.sum()
    )
    quality["critical_or_high_alerts"] = int(
        pd.Series(severity).isin(
            ["Critical", "High"]
        ).sum()
    )

    ordered = (
        context_columns
        + [
            column
            for column in prediction_columns
            if column not in context_columns
        ]
        + [
            column
            for column in result.columns
            if column not in context_columns
            and column not in prediction_columns
        ]
    )
    result = result.loc[:, ordered]

    return result, quality


def alert_view(result: pd.DataFrame) -> pd.DataFrame:
    """Return a compact SOC-oriented alert table."""
    preferred = [
        "AlertSeverity",
        "AlertReason",
        "KnownAttackProbability",
        "BinaryPrediction",
        "PredictedAttackFamily",
        "FamilyConfidence",
        "AnomalyScore",
        "AnomalyMargin",
        "AnomalyFlag",
        "MITRETactic",
        "MITRETechniqueID",
        "MITRETechniqueName",
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Context Destination Port",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Scenario",
        "Label",
        "BinaryLabel",
        "BinaryCorrect",
    ]

    columns = [
        column
        for column in preferred
        if column in result.columns
    ]
    return result.loc[:, columns]
