#!/usr/bin/env python3
r"""
AI-Powered Network Intrusion Detection System — Streamlit SOC Dashboard

Run from the project root:
    streamlit run .\dashboard.py
"""

from __future__ import annotations

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


@st.cache_resource(show_spinner=False)
def cached_models(project_root: str):
    return load_model_bundle(project_root)


@st.cache_data(show_spinner=False)
def load_demo(path: str) -> pd.DataFrame:
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
    source_column = (
        "Source IP"
        if "Source IP" in result.columns
        else None
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
    st.title("🛡️ AI-Powered Network Intrusion Detection System")
    st.caption(
        "Known-pattern classification + attack-family enrichment + "
        "benign-trained anomaly detection"
    )

    st.warning(
        "This dashboard analyzes CICIDS/CICFlowMeter-style flow records "
        "containing the same 77 features used during training. It does not "
        "accept raw packets or PCAP files directly."
    )

    try:
        bundle = cached_models(str(PROJECT_ROOT))
    except Exception as exc:
        st.error(
            "The trained artifacts could not be loaded. "
            f"Details: {exc}"
        )
        st.stop()

    with st.sidebar:
        st.header("Loaded artifacts")
        st.caption(f"Binary: {bundle.binary_path.name}")
        st.caption(f"Family: {bundle.family_path.name}")
        st.caption(f"Anomaly: {bundle.anomaly_path.name}")
        st.metric(
            "Required features",
            len(bundle.feature_names),
        )
        st.metric(
            "Binary threshold",
            f"{float(bundle.binary_artifact.get('threshold', 0.5)):.3f}",
        )
        st.metric(
            "Anomaly threshold",
            f"{float(bundle.anomaly_artifact['anomaly_threshold']):.6f}",
        )

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

        demo_path = PROJECT_ROOT / "data" / "dashboard_demo.csv"
        source_options = ["Upload CSV"]
        if demo_path.exists():
            source_options.insert(0, "Use local demo CSV")

        source = st.radio(
            "Data source",
            source_options,
            horizontal=True,
        )

        maximum_rows = st.number_input(
            "Maximum rows to load in the interactive dashboard",
            min_value=100,
            max_value=200_000,
            value=50_000,
            step=1_000,
            help=(
                "Use the batch CLI for larger files. The dashboard is "
                "intended for interactive SOC review."
            ),
        )

        input_frame = None

        if source == "Use local demo CSV":
            input_frame = load_demo(str(demo_path)).head(
                int(maximum_rows)
            )
            st.success(
                f"Loaded {len(input_frame):,} rows from "
                f"{demo_path.name}."
            )
        else:
            uploaded = st.file_uploader(
                "Upload a CSV containing all 77 trained flow features",
                type=["csv"],
            )
            if uploaded is not None:
                try:
                    input_frame = pd.read_csv(
                        uploaded,
                        nrows=int(maximum_rows),
                        low_memory=False,
                    )
                    st.success(
                        f"Loaded {len(input_frame):,} rows from "
                        f"{uploaded.name}."
                    )
                except Exception as exc:
                    st.error(f"Could not read the CSV: {exc}")

        if input_frame is not None:
            st.write("Input preview")
            st.dataframe(
                input_frame.head(20),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Run hybrid NIDS analysis",
                type="primary",
            ):
                with st.spinner("Analyzing network flows..."):
                    try:
                        result, quality = analyze_dataframe(
                            input_frame,
                            bundle,
                        )
                        st.session_state["nids_result"] = result
                        st.session_state["nids_quality"] = quality
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

        result = st.session_state.get("nids_result")
        quality = st.session_state.get("nids_quality")
        if isinstance(result, pd.DataFrame) and isinstance(
            quality,
            dict,
        ):
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
            "The mapping is family-level enrichment, not definitive "
            "attribution. Original labels, packet evidence, signatures, "
            "asset context, and analyst review are still required."
        )

    with tabs[3]:
        st.subheader("Decision architecture")
        st.code(
            """
Flow with 77 trained features
        |
        +--> Group-aware binary model
        |       +--> BENIGN
        |       +--> ATTACK --> family classifier
        |
        +--> Benign-trained Isolation Forest
                +--> within benign profile
                +--> anomalous

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
- Family-level MITRE ATT&CK enrichment is a starting point for analyst triage.
- The project has not yet been validated on current external enterprise traffic.
- A real deployment still requires feature extraction, schema monitoring,
  retraining, access control, secure logging, alert suppression, and analyst
  feedback.
            """
        )


if __name__ == "__main__":
    main()
