from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


def find_project_root(start: Path) -> Path:
    """Locate the AI-NIDS project root from this page's file location."""
    candidates: Iterable[Path] = (start, *start.parents)

    for candidate in candidates:
        if (
            (candidate / "reports").is_dir()
            and (candidate / "data").is_dir()
            and (candidate / "models").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the AI-NIDS project root. "
        "Place this file in a pages directory beside the main Streamlit entrypoint."
    )


def read_csv_required(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        st.error(f"Missing {description}: `{path}`")
        st.stop()

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read {description}: {exc}")
        st.stop()

    if frame.empty:
        st.error(f"{description} is empty: `{path}`")
        st.stop()

    return frame


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def count_nonempty_lines(path: Path) -> int | None:
    if not path.is_file():
        return None

    try:
        return sum(
            1
            for line in path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            if line.strip()
        )
    except OSError:
        return None


def find_scenario_row(
    frame: pd.DataFrame,
    token: str,
) -> pd.Series | None:
    if "Scenario" not in frame.columns:
        return None

    matches = frame[
        frame["Scenario"]
        .astype(str)
        .str.contains(token, case=False, regex=False, na=False)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


ROOT = find_project_root(Path(__file__).resolve().parent)

SUMMARY_PATH = (
    ROOT
    / "reports"
    / "tables"
    / "live_lab_cross_flow_summary.csv"
)
ALERT_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "live_lab"
    / "cross_flow_scan_alerts.csv"
)
COMPARISON_PATH = (
    ROOT
    / "reports"
    / "tables"
    / "final_live_lab_detection_comparison.csv"
)
SURICATA_DIR = ROOT / "reports" / "evidence" / "suricata"

st.set_page_config(
    page_title="Cross-Flow Correlation",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Cross-Flow Behavioral Correlation")
st.caption(
    "Source-level scan detection that supplements the trained "
    "per-flow supervised and anomaly models."
)

st.info(
    "Detection condition: at least 20 SYN-bearing flows to at least "
    "20 unique destination ports from one source within a rolling "
    "10-second window."
)

summary = read_csv_required(
    SUMMARY_PATH,
    "cross-flow summary",
)
alerts = read_csv_optional(ALERT_PATH)
comparison = read_csv_optional(COMPARISON_PATH)

benign = find_scenario_row(summary, "benign")
scan = find_scenario_row(summary, "portscan")

if benign is None or scan is None:
    st.error(
        "The summary must contain both a benign scenario and a "
        "portscan scenario."
    )
    st.stop()

benign_flag = as_bool(benign.get("CrossFlowScanFlag", False))
scan_flag = as_bool(scan.get("CrossFlowScanFlag", False))

status_left, status_right = st.columns(2)

with status_left:
    st.subheader("Benign baseline")
    if benign_flag:
        st.error("Unexpected result: benign traffic was flagged.")
    else:
        st.success("NO_SCAN_PATTERN — benign traffic was not flagged.")

    benign_metrics = st.columns(3)
    benign_metrics[0].metric(
        "Flows",
        int(float(benign.get("TotalFlows", 0))),
    )
    benign_metrics[1].metric(
        "Unique ports",
        int(float(benign.get("TotalUniqueDestinationPorts", 0))),
    )
    benign_metrics[2].metric(
        "Maximum SYN flows/window",
        int(float(benign.get("MaxSYNFlowsInWindow", 0))),
    )

with status_right:
    st.subheader("Controlled port scan")
    if scan_flag:
        st.error("PORT_SCAN — High-severity source-level detection.")
    else:
        st.warning("No scan pattern was detected.")

    scan_metrics = st.columns(3)
    scan_metrics[0].metric(
        "Flows",
        int(float(scan.get("TotalFlows", 0))),
    )
    scan_metrics[1].metric(
        "Unique ports",
        int(float(scan.get("TotalUniqueDestinationPorts", 0))),
    )
    scan_metrics[2].metric(
        "Maximum ports/window",
        int(float(scan.get("MaxUniqueDestinationPortsInWindow", 0))),
    )

st.divider()

st.subheader("Source-level scenario evidence")

preferred_summary_columns = [
    "Scenario",
    "GroundTruth",
    "SourceIP",
    "FirstTimestamp",
    "LastTimestamp",
    "DurationSeconds",
    "TotalFlows",
    "TotalUniqueDestinationPorts",
    "MaxFlowsInWindow",
    "MaxUniqueDestinationPortsInWindow",
    "MaxSYNFlowsInWindow",
    "CrossFlowScanFlag",
    "Detection",
]

summary_columns = [
    column
    for column in preferred_summary_columns
    if column in summary.columns
]

st.dataframe(
    summary[summary_columns],
    use_container_width=True,
    hide_index=True,
)

chart_columns = [
    column
    for column in [
        "Scenario",
        "TotalUniqueDestinationPorts",
        "MaxUniqueDestinationPortsInWindow",
        "MaxSYNFlowsInWindow",
    ]
    if column in summary.columns
]

if len(chart_columns) >= 2:
    chart_frame = summary[chart_columns].copy()
    chart_frame = chart_frame.set_index("Scenario")
    st.bar_chart(chart_frame)

st.divider()

st.subheader("SOC correlation alert")

if alerts.empty:
    st.warning("No cross-flow alert ledger was found or it is empty.")
else:
    alert = alerts.iloc[0]

    alert_metrics = st.columns(5)
    alert_metrics[0].metric(
        "Severity",
        str(alert.get("Severity", "")),
    )
    alert_metrics[1].metric(
        "Detection",
        str(alert.get("Detection", "")),
    )
    alert_metrics[2].metric(
        "Source",
        str(alert.get("SourceIP", "")),
    )
    alert_metrics[3].metric(
        "Unique ports/window",
        str(alert.get("WindowUniqueDestinationPorts", "")),
    )
    alert_metrics[4].metric(
        "SYN flows/window",
        str(alert.get("WindowSYNFlows", "")),
    )

    st.error(
        f"MITRE ATT&CK: {alert.get('MITRETactic', 'Reconnaissance')} — "
        f"{alert.get('MITRETechniqueID', 'T1595')} "
        f"{alert.get('MITRETechniqueName', 'Active Scanning')}"
    )

    st.dataframe(
        alerts,
        use_container_width=True,
        hide_index=True,
    )

    reason = str(alert.get("Reason", "")).strip()
    if reason:
        st.write("**Alert reason:**", reason)

st.divider()

st.subheader("Suricata evidence")

suricata_files = {
    "Benign records": SURICATA_DIR / "benign_alerts.json",
    "Default scan alerts": (
        SURICATA_DIR / "portscan_default_alerts.json"
    ),
    "Tuned scan alerts": (
        SURICATA_DIR / "portscan_tuned_alerts.json"
    ),
    "ICMP validation records": (
        SURICATA_DIR / "icmp_test_alerts.json"
    ),
}

suricata_counts = {
    label: count_nonempty_lines(path)
    for label, path in suricata_files.items()
}

suricata_metric_columns = st.columns(4)

for column, (label, count) in zip(
    suricata_metric_columns,
    suricata_counts.items(),
):
    column.metric(label, "Missing" if count is None else count)

if all(count is not None for count in suricata_counts.values()):
    st.caption(
        "The tuned alert count represents repeated threshold notifications "
        "for one controlled scan scenario, not separate attacks."
    )
else:
    st.info(
        "Some Suricata evidence files are not present in "
        "`reports/evidence/suricata`."
    )

st.divider()

st.subheader("Final detection-layer comparison")

if comparison.empty:
    st.info(
        "The consolidated comparison table is missing. Generate "
        "`reports/tables/final_live_lab_detection_comparison.csv` "
        "before final reporting."
    )
else:
    preferred_comparison_columns = [
        "Scenario",
        "GroundTruth",
        "DetectionLayer",
        "AnalystFacingDetection",
        "EventCount",
        "Result",
        "Severity",
        "MITRETactic",
        "MITRETechniqueID",
        "MITRETechniqueName",
        "Interpretation",
    ]

    comparison_columns = [
        column
        for column in preferred_comparison_columns
        if column in comparison.columns
    ]

    st.dataframe(
        comparison[comparison_columns],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Updated decision architecture")

st.code(
    """
Flow record with 77 trained features
    |
    +--> Supervised binary detector
    |       +--> BENIGN
    |       +--> ATTACK --> attack-family classifier
    |
    +--> Benign-trained Isolation Forest
    |       +--> within learned benign profile
    |       +--> anomalous
    |
    +--> Source-level rolling correlation
            +--> source IP
            +--> SYN-bearing flow count
            +--> unique destination-port count
            +--> 10-second window
            +--> PORT_SCAN / T1595
    """.strip()
)

st.warning(
    "The cross-flow layer is a deterministic behavioral control. "
    "It supplements the trained ML models and must not be reported "
    "as retrained ML performance. The tested rule is lab-specific "
    "and still requires validation on broader benign and enterprise traffic."
)
