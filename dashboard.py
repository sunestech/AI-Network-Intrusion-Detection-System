#!/usr/bin/env python3
r"""
AI-Powered Network Intrusion Detection System — Streamlit SOC Dashboard

Run from the project root:
    streamlit run .\dashboard.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_inference import (  # noqa: E402
    alert_view,
    analyze_dataframe,
    load_model_bundle,
)
from csv_validation import validate_csv_frame  # noqa: E402


st.set_page_config(
    page_title="AI NIDS SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
)


SEVERITY_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
]


def inject_global_styles() -> None:
    """Apply readable typography and wider spacing to the Streamlit interface."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            font-size: 1.04rem;
            line-height: 1.62;
        }
        [data-testid="stWidgetLabel"] p {
            font-size: 1rem;
            font-weight: 650;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important;
            justify-content: space-between !important;
            gap: clamp(2rem, 4vw, 4.5rem) !important;
            padding-top: .45rem;
            padding-bottom: .45rem;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            flex: 1 1 0 !important;
            width: auto !important;
            min-width: 0 !important;
            max-width: none !important;
            min-height: 3.4rem;
            justify-content: center !important;
            text-align: center !important;
            padding-left: 1.35rem !important;
            padding-right: 1.35rem !important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] p,
        [data-testid="stTabs"] [role="tab"] p {
            width: 100%;
            margin: 0;
            font-size: 1.13rem !important;
            font-weight: 720 !important;
            letter-spacing: .01em;
            text-align: center !important;
            white-space: nowrap;
        }
        @media (max-width: 980px) {
            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: .75rem !important;
            }
            [data-testid="stTabs"] button[data-baseweb="tab"] {
                padding-left: .45rem !important;
                padding-right: .45rem !important;
            }
            [data-testid="stTabs"] [role="tab"] p {
                font-size: 1rem !important;
                white-space: normal;
            }
        }
        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNav"] p {
            font-size: 1.02rem !important;
        }
        .nids-hero {
            border: 1px solid rgba(112, 178, 255, .30);
            border-radius: 20px;
            padding: 1.65rem 1.8rem;
            margin-bottom: 1.15rem;
            background: linear-gradient(
                135deg,
                rgba(27, 77, 132, .35),
                rgba(17, 25, 39, .82)
            );
        }
        .nids-hero h1 {
            margin: 0 0 .55rem 0;
            font-size: 2.9rem;
            line-height: 1.12;
        }
        .nids-hero p {
            margin: 0;
            max-width: 960px;
            color: rgba(255,255,255,.77);
            font-size: 1.12rem;
            line-height: 1.55;
        }
        .nids-section-note {
            border-left: 4px solid #5aa9ff;
            padding: .72rem 1rem;
            margin: .35rem 0 1.25rem 0;
            background: rgba(49, 111, 181, .12);
            border-radius: 0 10px 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_hero(required_features: int) -> None:
    st.markdown(
        """
        <div class="nids-hero">
            <h1>🛡️ AI-Powered Network Intrusion Detection System</h1>
            <p>
                Review CICIDS/CICFlowMeter-style network flows with supervised
                attack detection, attack-family enrichment, benign-trained anomaly
                detection, MITRE ATT&amp;CK context, and dynamic cross-flow correlation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview = st.columns(4)
    overview[0].metric("Required flow features", f"{required_features}")
    overview[1].metric("Machine-learning stages", "3")
    overview[2].metric("Cross-flow correlation", "Dynamic")
    overview[3].metric("Analyst view", "SOC + MITRE")

    st.markdown(
        """
        <div class="nids-section-note">
            Start in <strong>Analyze flows</strong>. The dashboard validates the CSV
            before inference and shares the current input with the Cross Flow
            Correlation page so source-level behavior can be recalculated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    navigation = st.columns(2)
    with navigation[0]:
        if hasattr(st, "page_link"):
            st.page_link(
                "pages/4_Cross_Flow_Correlation.py",
                label="Open dynamic cross-flow correlation",
                icon="🔗",
            )
    with navigation[1]:
        if hasattr(st, "page_link"):
            st.page_link(
                "pages/5_About_The_Project.py",
                label="Read about the project",
                icon="ℹ️",
            )


def clear_current_input_state() -> None:
    """Remove the selected input and any cross-flow result derived from it."""
    for key in (
        "nids_current_input",
        "nids_current_source_key",
        "nids_current_source_label",
        "nids_current_validation",
        "nids_cross_flow_result_key",
        "nids_cross_flow_summary",
        "nids_cross_flow_alerts",
    ):
        st.session_state.pop(key, None)


def publish_current_input(
    frame: pd.DataFrame,
    source_key: str,
    source_label: str,
    validation: dict[str, Any],
) -> None:
    """Share the current input with other Streamlit pages in this session."""
    prior_key = st.session_state.get("nids_current_source_key")
    if prior_key != source_key:
        for key in (
            "nids_cross_flow_result_key",
            "nids_cross_flow_summary",
            "nids_cross_flow_alerts",
        ):
            st.session_state.pop(key, None)

    st.session_state["nids_current_input"] = frame.copy()
    st.session_state["nids_current_source_key"] = source_key
    st.session_state["nids_current_source_label"] = source_label
    st.session_state["nids_current_validation"] = validation


def validation_check_table(report: dict[str, Any]) -> pd.DataFrame:
    missing_features = report.get("missing_required_features", [])
    unusable_features = report.get("unusable_required_features", [])
    duplicate_columns = report.get("duplicate_columns", [])
    missing_context = report.get("missing_cross_flow_context", [])
    invalid_context = report.get("invalid_context_counts", {})
    invalid_context_total = sum(int(value) for value in invalid_context.values())

    feature_detail = (
        f"{int(report.get('present_required_features', 0))}/"
        f"{int(report.get('required_feature_count', 0))} required features present"
    )
    if missing_features:
        feature_detail += f"; {len(missing_features)} missing"
    if unusable_features:
        feature_detail += f"; {len(unusable_features)} contain no usable numbers"

    context_detail = (
        "Source IP, destination IP, timestamp, destination port, and SYN count are usable"
        if report.get("cross_flow_ready")
        else (
            f"Missing: {', '.join(missing_context)}"
            if missing_context
            else f"Invalid or blank context values: {invalid_context_total:,}"
        )
    )

    return pd.DataFrame(
        [
            {
                "Validation check": "Data rows",
                "Status": "PASS" if int(report.get("rows", 0)) > 0 else "FAIL",
                "Detail": f"{int(report.get('rows', 0)):,} rows loaded",
            },
            {
                "Validation check": "Column-name integrity",
                "Status": "PASS" if not duplicate_columns else "FAIL",
                "Detail": (
                    "No duplicate names after BOM/whitespace cleanup"
                    if not duplicate_columns
                    else f"Duplicates: {', '.join(duplicate_columns)}"
                ),
            },
            {
                "Validation check": "Trained model schema",
                "Status": (
                    "PASS"
                    if not missing_features and not unusable_features
                    else "FAIL"
                ),
                "Detail": feature_detail,
            },
            {
                "Validation check": "Numeric feature quality",
                "Status": (
                    "PASS"
                    if int(report.get("missing_feature_cells_after_coercion", 0)) == 0
                    else "WARNING"
                ),
                "Detail": (
                    f"{int(report.get('missing_feature_cells_after_coercion', 0)):,} "
                    "feature cells require training-time imputation"
                ),
            },
            {
                "Validation check": "Dynamic cross-flow context",
                "Status": "PASS" if report.get("cross_flow_ready") else "NOT READY",
                "Detail": context_detail,
            },
        ]
    )


def display_csv_validation(report: dict[str, Any]) -> None:
    st.subheader("CSV upload validation")

    status = str(report.get("status", "FAIL"))
    if status == "PASS":
        st.success("PASS: The CSV is compatible with the trained model pipeline.")
    elif status == "PASS WITH WARNINGS":
        st.warning(
            "PASS WITH WARNINGS: The CSV can be analyzed, but the notes below "
            "should be reviewed."
        )
    else:
        st.error(
            "FAIL: The CSV is not ready for model inference. Correct the failed "
            "checks before running the analysis."
        )

    metrics = st.columns(5)
    metrics[0].metric("Rows", f"{int(report.get('rows', 0)):,}")
    metrics[1].metric("Columns", f"{int(report.get('columns', 0)):,}")
    metrics[2].metric(
        "Required features",
        (
            f"{int(report.get('present_required_features', 0))}/"
            f"{int(report.get('required_feature_count', 0))}"
        ),
    )
    metrics[3].metric(
        "Rows with feature gaps",
        f"{int(report.get('rows_with_feature_gaps', 0)):,}",
    )
    metrics[4].metric(
        "Cross-flow ready",
        "Yes" if report.get("cross_flow_ready") else "No",
    )

    st.dataframe(
        validation_check_table(report),
        use_container_width=True,
        hide_index=True,
    )

    errors = list(report.get("errors", []))
    warnings = list(report.get("warnings", []))
    if errors:
        for message in errors:
            st.error(message)
    if warnings:
        for message in warnings:
            st.warning(message)

    with st.expander("Detailed validation report", expanded=False):
        missing = list(report.get("missing_required_features", []))
        unusable = list(report.get("unusable_required_features", []))
        duplicates = list(report.get("duplicate_columns", []))
        extras = list(report.get("extra_columns", []))

        if missing:
            st.write("**Missing required features**")
            st.code("\n".join(missing))
        if unusable:
            st.write("**Required features with no usable numeric values**")
            st.code("\n".join(unusable))
        if duplicates:
            st.write("**Duplicate normalized columns**")
            st.code("\n".join(duplicates))

        st.write("**Resolved cross-flow context**")
        st.json(report.get("context_mapping", {}))
        st.write(
            f"Additional context or label columns preserved: {len(extras):,}."
        )


@st.cache_resource(show_spinner=False)
def cached_models(project_root: str):
    return load_model_bundle(project_root)


@st.cache_data(show_spinner=False)
def load_demo(path: str, modified_ns: int) -> pd.DataFrame:
    # modified_ns is part of the cache key, so replacing the demo CSV
    # forces Streamlit to reload it instead of returning an older copy.
    _ = modified_ns
    return pd.read_csv(path, low_memory=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_percent(
    value: Any,
    digits: int = 2,
) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "N/A"


def clear_analysis_state() -> None:
    """Remove analysis results that belong to a previously selected file."""
    for key in (
        "nids_result",
        "nids_quality",
        "nids_result_source_key",
        "nids_result_source_label",
        "nids_result_rows",
    ):
        st.session_state.pop(key, None)


def synchronize_analysis_source(current_source_key: str) -> None:
    """
    Prevent an old prediction ledger from appearing below a newly selected file.

    This also clears results created by the earlier dashboard version, which
    did not store a source key.
    """
    has_result = (
        "nids_result" in st.session_state
        or "nids_quality" in st.session_state
    )
    stored_source_key = st.session_state.get("nids_result_source_key")

    if has_result and stored_source_key != current_source_key:
        clear_analysis_state()


def display_project_evidence() -> None:
    grouped = read_json(
        PROJECT_ROOT
        / "reports"
        / "grouped_session_evaluation_summary.json"
    )
    scenario = read_json(
        PROJECT_ROOT
        / "reports"
        / "scenario_holdout_summary.json"
    )
    family = read_json(
        PROJECT_ROOT
        / "reports"
        / "attack_family_model_summary.json"
    )
    anomaly = read_json(
        PROJECT_ROOT
        / "reports"
        / "anomaly_model_summary.json"
    )

    st.subheader("Validated model evidence")

    columns = st.columns(4)

    grouped_metrics = grouped.get("test_metrics", {})
    columns[0].metric(
        "Known-attack recall",
        metric_percent(
            grouped_metrics.get("recall_attack")
        ),
        help=(
            "Group-aware binary test. No session-like GroupID "
            "crossed the split boundary."
        ),
    )
    columns[0].caption(
        "Grouped binary detector"
    )

    family_metrics = family.get(
        "final_test_metrics",
        {},
    )
    columns[1].metric(
        "Family macro recall",
        metric_percent(
            family_metrics.get("macro_recall")
        ),
        help=(
            "Conditional second-stage classification on flows "
            "already known to be malicious."
        ),
    )
    columns[1].caption(
        "Attack-family classifier"
    )

    scenario_metrics = scenario.get(
        "aggregate_metrics",
        {},
    )
    columns[2].metric(
        "Novel-scenario supervised recall",
        metric_percent(
            scenario_metrics.get("recall_attack")
        ),
        help=(
            "Complete scenario holdout. Often removes an entire "
            "attack class from supervised training."
        ),
    )
    columns[2].caption(
        "Zero-shot supervised stress test"
    )

    anomaly_metrics = anomaly.get(
        "scenario_holdout_metrics",
        {},
    )
    columns[3].metric(
        "Novel-scenario anomaly recall",
        metric_percent(
            anomaly_metrics.get("recall_attack")
        ),
        help=(
            "Isolation Forest trained only on benign behavior "
            "from the other scenarios."
        ),
    )
    columns[3].caption(
        "Benign-trained anomaly detector"
    )

    comparison_rows = []
    if grouped_metrics:
        comparison_rows.append(
            {
                "Evaluation": "Grouped known-attack binary",
                "Accuracy": grouped_metrics.get("accuracy"),
                "Attack precision": grouped_metrics.get(
                    "precision_attack"
                ),
                "Attack recall": grouped_metrics.get(
                    "recall_attack"
                ),
                "F2": grouped_metrics.get("f2_attack"),
                "PR-AUC": grouped_metrics.get("pr_auc"),
                "FPR": grouped_metrics.get(
                    "false_positive_rate"
                ),
            }
        )
    if scenario_metrics:
        comparison_rows.append(
            {
                "Evaluation": "Complete scenario supervised",
                "Accuracy": scenario_metrics.get("accuracy"),
                "Attack precision": scenario_metrics.get(
                    "precision_attack"
                ),
                "Attack recall": scenario_metrics.get(
                    "recall_attack"
                ),
                "F2": scenario_metrics.get("f2_attack"),
                "PR-AUC": scenario_metrics.get("pr_auc"),
                "FPR": scenario_metrics.get(
                    "false_positive_rate"
                ),
            }
        )
    if anomaly_metrics:
        comparison_rows.append(
            {
                "Evaluation": "Complete scenario anomaly",
                "Accuracy": anomaly_metrics.get("accuracy"),
                "Attack precision": anomaly_metrics.get(
                    "precision_attack"
                ),
                "Attack recall": anomaly_metrics.get(
                    "recall_attack"
                ),
                "F2": anomaly_metrics.get("f2_attack"),
                "PR-AUC": anomaly_metrics.get("pr_auc"),
                "FPR": anomaly_metrics.get(
                    "false_positive_rate"
                ),
            }
        )

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison.style.format(
                {
                    "Accuracy": "{:.4%}",
                    "Attack precision": "{:.4%}",
                    "Attack recall": "{:.4%}",
                    "F2": "{:.4%}",
                    "PR-AUC": "{:.6f}",
                    "FPR": "{:.4%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def severity_chart(result: pd.DataFrame) -> None:
    counts = (
        result["AlertSeverity"]
        .value_counts()
        .reindex(SEVERITY_ORDER, fill_value=0)
        .rename("Flows")
        .to_frame()
    )
    st.bar_chart(counts)


def family_chart(result: pd.DataFrame) -> None:
    family = result.loc[
        result["BinaryPrediction"].eq("ATTACK"),
        "PredictedAttackFamily",
    ]
    if family.empty:
        st.info("No supervised attack-family predictions were produced.")
        return

    counts = (
        family.value_counts()
        .rename("Flows")
        .to_frame()
    )
    st.bar_chart(counts)


def source_chart(result: pd.DataFrame) -> None:
    source_column = next(
        (
            candidate
            for candidate in ("SourceIP", "Source IP", "source_ip", "src_ip")
            if candidate in result.columns
        ),
        None,
    )
    if not source_column:
        st.info(
            "Source IP context was not included in the uploaded file."
        )
        return

    alerts = result.loc[
        result["AlertSeverity"].isin(
            ["Critical", "High", "Medium"]
        )
    ]
    if alerts.empty:
        st.info("No alerting flows were available.")
        return

    counts = (
        alerts[source_column]
        .astype("string")
        .fillna("MISSING")
        .value_counts()
        .head(15)
        .rename("Alerts")
        .to_frame()
    )
    st.bar_chart(counts)


def mitre_chart(result: pd.DataFrame) -> None:
    values = result.loc[
        result["MITRETechniqueID"].astype(str).ne(""),
        "MITRETechniqueID",
    ]
    if values.empty:
        st.info("No MITRE-mapped attack-family results were produced.")
        return

    counts = (
        values.value_counts()
        .rename("Flows")
        .to_frame()
    )
    st.bar_chart(counts)


def display_analysis(result: pd.DataFrame, quality: dict[str, Any]) -> None:
    total = len(result)
    known = int(
        result["BinaryPrediction"].eq("ATTACK").sum()
    )
    anomalous = int(result["AnomalyFlag"].sum())
    critical_high = int(
        result["AlertSeverity"].isin(
            ["Critical", "High"]
        ).sum()
    )
    medium = int(
        result["AlertSeverity"].eq("Medium").sum()
    )

    columns = st.columns(5)
    columns[0].metric("Flows analyzed", f"{total:,}")
    columns[1].metric("Known attacks", f"{known:,}")
    columns[2].metric("Anomaly flags", f"{anomalous:,}")
    columns[3].metric(
        "Critical / High",
        f"{critical_high:,}",
    )
    columns[4].metric(
        "Anomaly-only",
        f"{medium:,}",
    )

    if quality.get(
        "missing_feature_cells_after_coercion",
        0,
    ):
        st.warning(
            "The model pipeline imputed "
            f"{quality['missing_feature_cells_after_coercion']:,} "
            "missing feature cells using the medians learned during "
            "model training."
        )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Alert severity")
        severity_chart(result)
    with chart_columns[1]:
        st.subheader("Predicted attack families")
        family_chart(result)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Top alerting source IPs")
        source_chart(result)
    with chart_columns[1]:
        st.subheader("MITRE ATT&CK enrichment")
        mitre_chart(result)

    st.subheader("SOC alert queue")
    queue = alert_view(result)
    queue = queue.loc[
        queue["AlertSeverity"].isin(
            ["Critical", "High", "Medium"]
        )
    ]

    if queue.empty:
        st.success(
            "No supervised or anomaly alerts were produced for this file."
        )
    else:
        queue = queue.copy()
        queue["_SeverityOrder"] = pd.Categorical(
            queue["AlertSeverity"],
            categories=SEVERITY_ORDER,
            ordered=True,
        )
        queue = queue.sort_values(
            [
                "_SeverityOrder",
                "KnownAttackProbability",
                "AnomalyMargin",
            ],
            ascending=[True, False, False],
        ).drop(columns="_SeverityOrder")

        st.dataframe(
            queue.head(5_000),
            use_container_width=True,
            hide_index=True,
        )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download complete prediction ledger",
        data=csv_bytes,
        file_name="ai_nids_prediction_ledger.csv",
        mime="text/csv",
    )

    st.subheader("Inspect one flow")
    selected_position = st.number_input(
        "Row position",
        min_value=0,
        max_value=max(0, len(result) - 1),
        value=0,
        step=1,
    )
    selected = result.iloc[int(selected_position)]

    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Binary result",
        str(selected["BinaryPrediction"]),
        metric_percent(
            selected["KnownAttackProbability"]
        ),
    )
    summary_columns[1].metric(
        "Family",
        str(selected["PredictedAttackFamily"]),
        (
            metric_percent(selected["FamilyConfidence"])
            if pd.notna(selected["FamilyConfidence"])
            else "Not run"
        ),
    )
    summary_columns[2].metric(
        "Anomaly",
        "FLAGGED" if bool(selected["AnomalyFlag"]) else "No",
        f"margin {float(selected['AnomalyMargin']):.4f}",
    )
    summary_columns[3].metric(
        "Severity",
        str(selected["AlertSeverity"]),
    )

    st.write(
        {
            "Alert reason": selected["AlertReason"],
            "MITRE tactic": selected["MITRETactic"],
            "MITRE technique ID": selected["MITRETechniqueID"],
            "MITRE technique": selected["MITRETechniqueName"],
            "Mapping note": selected["MITREMappingNote"],
        }
    )

    with st.expander("Show all fields for the selected flow"):
        st.dataframe(
            selected.rename("Value").to_frame(),
            use_container_width=True,
        )


def main() -> None:
    inject_global_styles()

    try:
        bundle = cached_models(str(PROJECT_ROOT))
    except Exception as exc:
        st.error(
            "The trained artifacts could not be loaded. "
            f"Details: {exc}"
        )
        st.stop()

    render_landing_hero(len(bundle.feature_names))

    st.warning(
        "This dashboard accepts CICIDS/CICFlowMeter-style flow records with "
        "the same 77 features used during training. Raw packets and PCAP files "
        "must be converted to compatible flow records before upload."
    )

    with st.sidebar:
        st.header("Loaded artifacts")
        st.caption(f"Binary: {bundle.binary_path.name}")
        st.caption(f"Family: {bundle.family_path.name}")
        st.caption(f"Anomaly: {bundle.anomaly_path.name}")
        st.metric("Required features", len(bundle.feature_names))
        st.metric(
            "Binary threshold",
            f"{float(bundle.binary_artifact.get('threshold', 0.5)):.3f}",
        )
        st.metric(
            "Anomaly threshold",
            f"{float(bundle.anomaly_artifact['anomaly_threshold']):.6f}",
        )

        current_label = st.session_state.get("nids_current_source_label")
        if current_label:
            st.divider()
            st.subheader("Shared dashboard input")
            st.caption(str(current_label))
            current_frame = st.session_state.get("nids_current_input")
            if isinstance(current_frame, pd.DataFrame):
                st.metric("Shared rows", f"{len(current_frame):,}")

    tabs = st.tabs(
        [
            "Analyze flows",
            "Model evidence",
            "MITRE ATT&CK",
            "Architecture & limitations",
        ]
    )

    with tabs[0]:
        st.subheader("Choose a compatible flow dataset")
        st.caption(
            "The selected file is validated before inference. The same current "
            "input is also shared with the dynamic Cross Flow Correlation page."
        )

        demo_path = PROJECT_ROOT / "data" / "dashboard_demo.csv"
        source_options = ["Upload CSV"]
        if demo_path.exists():
            source_options.insert(0, "Use local demo CSV")

        source = st.radio("Data source", source_options, horizontal=True)

        maximum_rows = st.number_input(
            "Maximum rows to load in the interactive dashboard",
            min_value=100,
            max_value=200_000,
            value=50_000,
            step=1_000,
            help=(
                "Use the batch CLI for larger files. The dashboard is intended "
                "for interactive SOC review."
            ),
        )

        input_frame: pd.DataFrame | None = None
        validation: dict[str, Any] | None = None
        current_source_key: str | None = None
        current_source_label: str | None = None

        if source == "Use local demo CSV":
            demo_modified_ns = demo_path.stat().st_mtime_ns
            input_frame = load_demo(
                str(demo_path),
                demo_modified_ns,
            ).head(int(maximum_rows))

            current_source_key = (
                f"demo|{demo_path.resolve()}|{demo_modified_ns}|"
                f"{int(maximum_rows)}|{len(input_frame)}"
            )
            current_source_label = f"{demo_path.name} (local demo)"
            st.success(
                f"Loaded {len(input_frame):,} rows from {demo_path.name}."
            )
        else:
            uploaded = st.file_uploader(
                "Upload a CSV containing all 77 trained flow features",
                type=["csv"],
            )
            if uploaded is not None:
                try:
                    uploaded_bytes = uploaded.getvalue()
                    digest = hashlib.sha256(uploaded_bytes).hexdigest()
                    input_frame = pd.read_csv(
                        io.BytesIO(uploaded_bytes),
                        nrows=int(maximum_rows),
                        low_memory=False,
                    )
                    current_source_key = (
                        f"upload|{uploaded.name}|{len(uploaded_bytes)}|"
                        f"{digest}|{int(maximum_rows)}|{len(input_frame)}"
                    )
                    current_source_label = f"{uploaded.name} (uploaded CSV)"
                    st.success(
                        f"Loaded {len(input_frame):,} rows from {uploaded.name}."
                    )
                except Exception as exc:
                    clear_analysis_state()
                    clear_current_input_state()
                    st.error(f"Could not read the CSV: {exc}")

        if input_frame is not None:
            try:
                input_frame, validation = validate_csv_frame(
                    input_frame,
                    bundle.feature_names,
                )
            except Exception as exc:
                clear_analysis_state()
                clear_current_input_state()
                st.error(f"CSV validation failed: {exc}")
                input_frame = None

        if (
            input_frame is not None
            and validation is not None
            and current_source_key is not None
            and current_source_label is not None
        ):
            synchronize_analysis_source(current_source_key)
            publish_current_input(
                input_frame,
                current_source_key,
                current_source_label,
                validation,
            )
        elif source == "Upload CSV":
            clear_analysis_state()
            clear_current_input_state()

        if input_frame is not None and validation is not None:
            display_csv_validation(validation)

            st.subheader("Input preview")
            st.dataframe(
                input_frame.head(20),
                use_container_width=True,
                hide_index=True,
            )

            run_analysis = st.button(
                "Run hybrid NIDS analysis",
                type="primary",
                disabled=not bool(validation.get("model_ready")),
                help=(
                    "The button is enabled only when all trained model features "
                    "are present and usable."
                ),
            )
            if run_analysis:
                with st.spinner("Analyzing network flows..."):
                    try:
                        result, quality = analyze_dataframe(input_frame, bundle)
                        st.session_state["nids_result"] = result
                        st.session_state["nids_quality"] = quality
                        st.session_state["nids_result_source_key"] = current_source_key
                        st.session_state["nids_result_source_label"] = current_source_label
                        st.session_state["nids_result_rows"] = len(input_frame)
                    except Exception as exc:
                        clear_analysis_state()
                        st.error(f"Analysis failed: {exc}")

            if validation.get("cross_flow_ready") and hasattr(st, "page_link"):
                st.page_link(
                    "pages/4_Cross_Flow_Correlation.py",
                    label="Continue with this file in dynamic cross-flow correlation",
                    icon="🔗",
                )
            elif not validation.get("cross_flow_ready"):
                st.info(
                    "Model inference is separate from cross-flow readiness. Add "
                    "source IP, destination IP, timestamp, destination-port, and "
                    "SYN-count context to use the correlation page."
                )

        result = st.session_state.get("nids_result")
        quality = st.session_state.get("nids_quality")
        stored_source_key = st.session_state.get("nids_result_source_key")

        if (
            isinstance(result, pd.DataFrame)
            and isinstance(quality, dict)
            and current_source_key is not None
            and stored_source_key == current_source_key
        ):
            analyzed_label = st.session_state.get(
                "nids_result_source_label",
                "selected input",
            )
            analyzed_rows = int(
                st.session_state.get("nids_result_rows", len(result))
            )
            st.info(
                f"Current analysis source: {analyzed_label}. "
                f"Rows analyzed: {analyzed_rows:,}."
            )
            display_analysis(result, quality)

    with tabs[1]:
        display_project_evidence()
        st.info(
            "The dashboard severity fusion has not been independently "
            "benchmarked as a fourth model. The binary, family, and anomaly "
            "components retain their separate validated metrics."
        )

    with tabs[2]:
        st.subheader("Operational MITRE ATT&CK enrichment")
        mapping_rows = []
        for family, record in bundle.mitre_mapping.items():
            techniques = record.get("techniques", [])
            if techniques:
                for technique in techniques:
                    mapping_rows.append(
                        {
                            "Dashboard category": family,
                            "Tactic": record.get("tactic", ""),
                            "Technique ID": technique.get("id", ""),
                            "Technique": technique.get("name", ""),
                            "Reference": technique.get("url", ""),
                            "Mapping note": record.get("note", ""),
                        }
                    )
            else:
                mapping_rows.append(
                    {
                        "Dashboard category": family,
                        "Tactic": record.get("tactic", ""),
                        "Technique ID": "",
                        "Technique": "",
                        "Reference": "",
                        "Mapping note": record.get("note", ""),
                    }
                )

        st.dataframe(
            pd.DataFrame(mapping_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.warning(
            "The mapping is family-level enrichment, not definitive attribution. "
            "Original labels, packet evidence, signatures, asset context, and "
            "analyst review are still required."
        )

    with tabs[3]:
        st.subheader("Decision architecture")
        st.code(
            """
Flow record with 77 trained features
        |
        +--> Group-aware binary model
        |       +--> BENIGN
        |       +--> ATTACK --> family classifier
        |
        +--> Benign-trained Isolation Forest
        |       +--> within benign profile
        |       +--> anomalous
        |
        +--> Dynamic source-level rolling correlation
                +--> source IP and destination context
                +--> SYN-bearing flow count
                +--> unique destination-port count
                +--> configurable time window
                +--> PORT_SCAN / T1595

Triage fusion
        Critical = supervised ATTACK + anomaly
        High     = supervised ATTACK only
        Medium   = anomaly only
        Low      = neither
            """.strip()
        )

        st.markdown(
            """
### Boundaries of the system

- The dashboard requires **flow-level features**, not a raw packet or URL.
- A supervised attack result means the flow resembles learned attack patterns.
- An anomaly result means the flow differs from the learned benign profile.
- An anomaly is **not proof** of malicious activity.
- The attack-family model runs only after the binary stage predicts `ATTACK`.
- Dynamic cross-flow correlation requires source IP, destination IP, timestamp,
  destination port, and SYN-count context.
- Family-level MITRE ATT&CK enrichment is a starting point for analyst triage.
- The project has not yet been validated on current external enterprise traffic.
- A real deployment still requires feature extraction, schema monitoring,
  retraining, access control, secure logging, alert suppression, and analyst
  feedback.
            """
        )


if __name__ == "__main__":
    main()
