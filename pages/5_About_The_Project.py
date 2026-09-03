"""About page for the AI-NIDS Streamlit portfolio project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


def find_project_root(start: Path) -> Path:
    candidates: Iterable[Path] = (start, *start.parents)
    for candidate in candidates:
        if (
            (candidate / "reports").is_dir()
            and (candidate / "data").is_dir()
            and (candidate / "models").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate the AI-NIDS project root. Place this page in the "
        "project's pages directory beside dashboard.py."
    )


ROOT = find_project_root(Path(__file__).resolve().parent)

st.set_page_config(
    page_title="About the AI-NIDS Project",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-size: 1.06rem;
        line-height: 1.65;
    }
    [data-testid="stSidebarNav"] span,
    [data-testid="stSidebarNav"] p {
        font-size: 1.02rem !important;
    }
    .about-hero {
        border: 1px solid rgba(112, 178, 255, .30);
        border-radius: 20px;
        padding: 1.75rem 1.9rem;
        margin-bottom: 1.25rem;
        background: linear-gradient(135deg, rgba(27, 77, 132, .35), rgba(17, 25, 39, .82));
    }
    .about-hero h1 {
        margin: 0 0 .55rem 0;
        font-size: 2.75rem;
        line-height: 1.12;
    }
    .about-hero p {
        margin: 0;
        max-width: 980px;
        color: rgba(255,255,255,.77);
        font-size: 1.12rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_percent(value: Any, fallback: str = "N/A") -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


st.markdown(
    """
    <div class="about-hero">
        <h1>ℹ️ About the AI-NIDS Project</h1>
        <p>
            A portfolio-grade network intrusion detection prototype that combines
            machine learning, anomaly detection, Suricata evidence, MITRE ATT&amp;CK
            enrichment, and source-level behavioral correlation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Project purpose")
st.write(
    "The project was built to show how several detection methods can work "
    "together. A compatible network-flow CSV is checked against the trained "
    "77-feature schema, scored by the saved models, enriched for analyst review, "
    "and optionally examined across many related flows from the same source."
)

capabilities = st.columns(4)
capabilities[0].metric("Trained flow features", "77")
capabilities[1].metric("Machine-learning components", "3")
capabilities[2].metric("Behavioral correlation", "Source level")
capabilities[3].metric("Rule-based layer", "Suricata")

st.subheader("How the system works")
st.code(
    """
CICIDS/CICFlowMeter-style flow CSV
        |
        +--> CSV schema and data-quality validation
        |
        +--> Group-aware binary detector
        |       +--> BENIGN
        |       +--> ATTACK
        |              +--> Attack-family classifier
        |
        +--> Benign-trained Isolation Forest
        |       +--> within learned benign profile
        |       +--> anomalous
        |
        +--> Dynamic cross-flow correlation
        |       +--> source IP
        |       +--> destination IP and port
        |       +--> SYN-bearing flows
        |       +--> rolling time window
        |       +--> PORT_SCAN / T1595
        |
        +--> SOC queue, severity, MITRE ATT&CK, and downloadable ledgers

Suricata operates as a separate rule-based detection layer against captured traffic.
    """.strip()
)

st.subheader("Detection components")
components = pd.DataFrame(
    [
        {
            "Component": "Binary detector",
            "Role": "Classifies each flow as BENIGN or ATTACK based on learned patterns.",
        },
        {
            "Component": "Attack-family classifier",
            "Role": "Runs after an ATTACK decision and assigns a broad operational family.",
        },
        {
            "Component": "Isolation Forest",
            "Role": "Flags flows that differ from the learned benign profile.",
        },
        {
            "Component": "Cross-flow correlation",
            "Role": "Groups related flows from one source and detects scanning behavior across a rolling window.",
        },
        {
            "Component": "Suricata",
            "Role": "Provides a separate signature- and rule-based view of the same laboratory traffic.",
        },
        {
            "Component": "MITRE ATT&CK enrichment",
            "Role": "Adds analyst-facing tactic and technique context without claiming attribution.",
        },
    ]
)
st.dataframe(components, use_container_width=True, hide_index=True)

grouped = read_json(ROOT / "reports" / "grouped_session_evaluation_summary.json")
scenario = read_json(ROOT / "reports" / "scenario_holdout_summary.json")
family = read_json(ROOT / "reports" / "attack_family_model_summary.json")
anomaly = read_json(ROOT / "reports" / "anomaly_model_summary.json")

st.subheader("Validated model evidence")
metrics = st.columns(4)
metrics[0].metric(
    "Known-attack recall",
    metric_percent(grouped.get("test_metrics", {}).get("recall_attack"), "99.92%"),
)
metrics[1].metric(
    "Family macro recall",
    metric_percent(family.get("final_test_metrics", {}).get("macro_recall"), "98.68%"),
)
metrics[2].metric(
    "Novel-scenario supervised recall",
    metric_percent(scenario.get("aggregate_metrics", {}).get("recall_attack"), "15.45%"),
)
metrics[3].metric(
    "Novel-scenario anomaly recall",
    metric_percent(anomaly.get("scenario_holdout_metrics", {}).get("recall_attack"), "35.74%"),
)

st.info(
    "The large gap between familiar-pattern performance and complete-scenario "
    "performance is an important result. The project keeps that limitation "
    "visible instead of treating the strongest internal score as proof of "
    "production or zero-day performance."
)

st.subheader("Live-lab finding")
st.write(
    "In the controlled laboratory test, the per-flow supervised model and the "
    "Isolation Forest both classified all 502 rate-limited port-scan flows as "
    "non-alerting. A tuned Suricata threshold produced notifications, and the "
    "source-level correlation layer produced one de-duplicated PORT_SCAN alert. "
    "This showed why related flows must sometimes be evaluated together."
)

finding_columns = st.columns(3)
finding_columns[0].metric("Controlled scan flows", "502")
finding_columns[1].metric("Tuned Suricata notifications", "24")
finding_columns[2].metric("Cross-flow alerts", "1")

st.subheader("Scope and limitations")
st.markdown(
    """
- The dashboard accepts flow-level CSV records, not raw PCAP files.
- The training data comes from one historical dataset family.
- The attack-family stage depends on the binary detector first predicting `ATTACK`.
- An anomaly is a behavioral deviation, not proof of malicious activity.
- Cross-flow and tuned Suricata thresholds are laboratory settings.
- MITRE ATT&CK mappings support triage and do not establish attribution.
- Production use would require broader benign validation, secure deployment,
  drift monitoring, suppression, deduplication, continuous feature extraction,
  and analyst feedback.
    """
)

st.warning(
    "All attack simulation must remain inside systems you own or are explicitly "
    "authorized to test. The controlled scan used in this project was limited to "
    "an isolated VMware laboratory."
)

st.subheader("Project owner")
st.write("Prepared by **Sunday Esumike** — August 2026.")

if hasattr(st, "page_link"):
    navigation = st.columns(2)
    with navigation[0]:
        st.page_link("dashboard.py", label="Return to Analyze flows", icon="📊")
    with navigation[1]:
        st.page_link(
            "pages/4_Cross_Flow_Correlation.py",
            label="Open dynamic cross-flow correlation",
            icon="🔗",
        )
