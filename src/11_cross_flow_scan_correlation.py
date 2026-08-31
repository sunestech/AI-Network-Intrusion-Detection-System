from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = [
    PROJECT_ROOT
    / "data"
    / "live_lab"
    / "model_ready"
    / "benign_baseline_dashboard.csv",
    PROJECT_ROOT
    / "data"
    / "live_lab"
    / "model_ready"
    / "portscan_rate_limited_dashboard.csv",
]

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
PREDICTION_DIR = PROJECT_ROOT / "reports" / "predictions" / "live_lab"
TEXT_REPORT = PROJECT_ROOT / "reports" / "live_lab_cross_flow_summary.txt"


COLUMN_ALIASES = {
    "source_ip": ["SourceIP", "Source IP", "src_ip"],
    "destination_ip": ["DestinationIP", "Destination IP", "dst_ip"],
    "timestamp": ["Timestamp", "timestamp"],
    "destination_port": ["Destination Port", "DestinationPort", "dst_port"],
    "syn_count": ["SYN Flag Count", "syn_flag_cnt"],
    "rst_count": ["RST Flag Count", "rst_flag_cnt"],
    "ack_count": ["ACK Flag Count", "ack_flag_cnt"],
    "backward_packets": [
        "Total Backward Packets",
        "Total Bwd Packets",
        "tot_bwd_pkts",
    ],
    "scenario": ["Scenario", "scenario"],
    "ground_truth": ["GroundTruth", "Ground Truth", "ground_truth"],
    "row_id": ["RowID", "Row ID", "row_id"],
}


@dataclass(frozen=True)
class DetectorConfig:
    window_seconds: float
    minimum_unique_ports: int
    minimum_syn_flows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate individual flow records into source-level scan behavior. "
            "This supplements, but does not replace, the trained per-flow ML models."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help=(
            "Dashboard-compatible flow CSV. Repeat for multiple files. "
            "Defaults to the benign and port-scan live-lab CSVs."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Sliding correlation window in seconds (default: 10).",
    )
    parser.add_argument(
        "--minimum-unique-ports",
        type=int,
        default=20,
        help="Minimum unique destination ports in a window (default: 20).",
    )
    parser.add_argument(
        "--minimum-syn-flows",
        type=int,
        default=20,
        help="Minimum SYN-bearing flows in a window (default: 20).",
    )
    return parser.parse_args()


def resolve_column(frame: pd.DataFrame, logical_name: str) -> str | None:
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in frame.columns:
            return candidate
    return None


def require_column(frame: pd.DataFrame, logical_name: str) -> str:
    column = resolve_column(frame, logical_name)
    if column is None:
        raise ValueError(
            f"Missing required {logical_name!r} column. "
            f"Accepted names: {COLUMN_ALIASES[logical_name]}"
        )
    return column


def numeric_series(
    frame: pd.DataFrame,
    logical_name: str,
    default: float = 0.0,
) -> pd.Series:
    column = resolve_column(frame, logical_name)
    if column is None:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def text_series(
    frame: pd.DataFrame,
    logical_name: str,
    default: str,
) -> pd.Series:
    column = resolve_column(frame, logical_name)
    if column is None:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def prepare_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()

    source_column = require_column(frame, "source_ip")
    destination_column = require_column(frame, "destination_ip")
    timestamp_column = require_column(frame, "timestamp")
    port_column = require_column(frame, "destination_port")

    prepared = pd.DataFrame(index=frame.index)
    prepared["SourceIP"] = frame[source_column].astype(str)
    prepared["DestinationIP"] = frame[destination_column].astype(str)
    prepared["Timestamp"] = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
        utc=True,
    )
    prepared["DestinationPort"] = pd.to_numeric(
        frame[port_column],
        errors="coerce",
    )
    prepared["SYNCount"] = numeric_series(frame, "syn_count")
    prepared["RSTCount"] = numeric_series(frame, "rst_count")
    prepared["ACKCount"] = numeric_series(frame, "ack_count")
    prepared["BackwardPackets"] = numeric_series(frame, "backward_packets")
    prepared["Scenario"] = text_series(
        frame,
        "scenario",
        path.stem,
    )
    prepared["GroundTruth"] = text_series(
        frame,
        "ground_truth",
        "UNKNOWN",
    )

    row_id_column = resolve_column(frame, "row_id")
    if row_id_column is None:
        prepared["RowID"] = range(len(prepared))
    else:
        prepared["RowID"] = frame[row_id_column]

    invalid_timestamps = int(prepared["Timestamp"].isna().sum())
    invalid_ports = int(prepared["DestinationPort"].isna().sum())

    if invalid_timestamps:
        raise ValueError(
            f"{path.name}: {invalid_timestamps} rows have invalid timestamps."
        )
    if invalid_ports:
        raise ValueError(
            f"{path.name}: {invalid_ports} rows have invalid destination ports."
        )

    prepared["DestinationPort"] = prepared["DestinationPort"].astype(int)
    prepared["InputFile"] = path.name
    return prepared.sort_values(
        ["SourceIP", "Timestamp", "RowID"],
        kind="stable",
    ).reset_index(drop=True)


def correlate_source(
    source_frame: pd.DataFrame,
    config: DetectorConfig,
) -> tuple[dict[str, object], dict[str, object] | None]:
    source_frame = source_frame.sort_values(
        ["Timestamp", "RowID"],
        kind="stable",
    ).reset_index(drop=True)

    window: deque[dict[str, object]] = deque()
    port_counts: Counter[int] = Counter()
    endpoint_counts: Counter[tuple[str, int]] = Counter()

    max_flow_count = 0
    max_unique_ports = 0
    max_unique_endpoints = 0
    max_syn_flows = 0
    max_rst_flows = 0
    max_no_response_flows = 0

    current_syn_flows = 0
    current_rst_flows = 0
    current_no_response_flows = 0
    first_alert: dict[str, object] | None = None

    for row in source_frame.itertuples(index=False):
        timestamp = row.Timestamp
        entry = {
            "Timestamp": timestamp,
            "DestinationIP": row.DestinationIP,
            "DestinationPort": int(row.DestinationPort),
            "SYNFlow": int(float(row.SYNCount) > 0),
            "RSTFlow": int(float(row.RSTCount) > 0),
            "NoResponseFlow": int(float(row.BackwardPackets) <= 0),
            "RowID": row.RowID,
        }

        window.append(entry)
        port_counts[entry["DestinationPort"]] += 1
        endpoint = (
            str(entry["DestinationIP"]),
            int(entry["DestinationPort"]),
        )
        endpoint_counts[endpoint] += 1
        current_syn_flows += int(entry["SYNFlow"])
        current_rst_flows += int(entry["RSTFlow"])
        current_no_response_flows += int(entry["NoResponseFlow"])

        cutoff = timestamp - pd.Timedelta(seconds=config.window_seconds)

        while window and window[0]["Timestamp"] < cutoff:
            old = window.popleft()

            old_port = int(old["DestinationPort"])
            port_counts[old_port] -= 1
            if port_counts[old_port] <= 0:
                del port_counts[old_port]

            old_endpoint = (
                str(old["DestinationIP"]),
                int(old["DestinationPort"]),
            )
            endpoint_counts[old_endpoint] -= 1
            if endpoint_counts[old_endpoint] <= 0:
                del endpoint_counts[old_endpoint]

            current_syn_flows -= int(old["SYNFlow"])
            current_rst_flows -= int(old["RSTFlow"])
            current_no_response_flows -= int(old["NoResponseFlow"])

        flow_count = len(window)
        unique_ports = len(port_counts)
        unique_endpoints = len(endpoint_counts)

        max_flow_count = max(max_flow_count, flow_count)
        max_unique_ports = max(max_unique_ports, unique_ports)
        max_unique_endpoints = max(
            max_unique_endpoints,
            unique_endpoints,
        )
        max_syn_flows = max(max_syn_flows, current_syn_flows)
        max_rst_flows = max(max_rst_flows, current_rst_flows)
        max_no_response_flows = max(
            max_no_response_flows,
            current_no_response_flows,
        )

        if (
            first_alert is None
            and unique_ports >= config.minimum_unique_ports
            and current_syn_flows >= config.minimum_syn_flows
        ):
            first_alert = {
                "AlertTimestamp": timestamp.isoformat(),
                "SourceIP": row.SourceIP,
                "DestinationIP": row.DestinationIP,
                "WindowSeconds": config.window_seconds,
                "WindowFlowCount": flow_count,
                "WindowUniqueDestinationPorts": unique_ports,
                "WindowUniqueDestinationEndpoints": unique_endpoints,
                "WindowSYNFlows": current_syn_flows,
                "WindowRSTFlows": current_rst_flows,
                "WindowNoResponseFlows": current_no_response_flows,
                "TriggerDestinationPort": int(row.DestinationPort),
                "TriggerRowID": row.RowID,
                "Detector": "Cross-flow behavioral correlation",
                "Detection": "PORT_SCAN",
                "Severity": "High",
                "MITRETactic": "Reconnaissance",
                "MITRETechniqueID": "T1595",
                "MITRETechniqueName": "Active Scanning",
                "Reason": (
                    f"At least {config.minimum_syn_flows} SYN-bearing flows "
                    f"reached at least {config.minimum_unique_ports} unique "
                    f"destination ports within {config.window_seconds:g} seconds."
                ),
            }

    first_timestamp = source_frame["Timestamp"].min()
    last_timestamp = source_frame["Timestamp"].max()
    duration_seconds = max(
        float((last_timestamp - first_timestamp).total_seconds()),
        0.0,
    )
    total_rows = len(source_frame)
    average_rate = (
        total_rows / duration_seconds
        if duration_seconds > 0
        else float(total_rows)
    )

    total_unique_ports = int(
        source_frame["DestinationPort"].nunique()
    )
    total_unique_endpoints = int(
        source_frame[
            ["DestinationIP", "DestinationPort"]
        ].drop_duplicates().shape[0]
    )
    total_syn_flows = int((source_frame["SYNCount"] > 0).sum())
    total_rst_flows = int((source_frame["RSTCount"] > 0).sum())
    total_no_response_flows = int(
        (source_frame["BackwardPackets"] <= 0).sum()
    )

    summary = {
        "InputFile": source_frame["InputFile"].iloc[0],
        "Scenario": source_frame["Scenario"].iloc[0],
        "GroundTruth": source_frame["GroundTruth"].iloc[0],
        "SourceIP": source_frame["SourceIP"].iloc[0],
        "FirstTimestamp": first_timestamp.isoformat(),
        "LastTimestamp": last_timestamp.isoformat(),
        "DurationSeconds": duration_seconds,
        "TotalFlows": total_rows,
        "TotalUniqueDestinationPorts": total_unique_ports,
        "TotalUniqueDestinationEndpoints": total_unique_endpoints,
        "TotalSYNFlows": total_syn_flows,
        "TotalRSTFlows": total_rst_flows,
        "TotalNoResponseFlows": total_no_response_flows,
        "AverageFlowsPerSecond": average_rate,
        "WindowSeconds": config.window_seconds,
        "MaxFlowsInWindow": max_flow_count,
        "MaxUniqueDestinationPortsInWindow": max_unique_ports,
        "MaxUniqueDestinationEndpointsInWindow": max_unique_endpoints,
        "MaxSYNFlowsInWindow": max_syn_flows,
        "MaxRSTFlowsInWindow": max_rst_flows,
        "MaxNoResponseFlowsInWindow": max_no_response_flows,
        "MinimumUniquePortsThreshold": config.minimum_unique_ports,
        "MinimumSYNFlowsThreshold": config.minimum_syn_flows,
        "CrossFlowScanFlag": first_alert is not None,
        "Detection": (
            "PORT_SCAN"
            if first_alert is not None
            else "NO_SCAN_PATTERN"
        ),
        "MITREGroundTruth": (
            "T1595 - Active Scanning"
            if first_alert is not None
            else ""
        ),
    }

    if first_alert is not None:
        first_alert.update(
            {
                "InputFile": source_frame["InputFile"].iloc[0],
                "Scenario": source_frame["Scenario"].iloc[0],
                "GroundTruth": source_frame["GroundTruth"].iloc[0],
            }
        )

    return summary, first_alert


def process_inputs(
    paths: Iterable[Path],
    config: DetectorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    alerts: list[dict[str, object]] = []

    for path in paths:
        frame = prepare_frame(path)

        for _, source_frame in frame.groupby(
            "SourceIP",
            sort=True,
            dropna=False,
        ):
            summary, alert = correlate_source(
                source_frame.reset_index(drop=True),
                config,
            )
            summaries.append(summary)
            if alert is not None:
                alerts.append(alert)

    return pd.DataFrame(summaries), pd.DataFrame(alerts)


def write_text_report(
    summary_frame: pd.DataFrame,
    alert_frame: pd.DataFrame,
    config: DetectorConfig,
) -> None:
    lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "LIVE-LAB CROSS-FLOW CORRELATION SUMMARY",
        "=" * 78,
        "",
        "Detector type: deterministic source-level behavioral correlation",
        (
            "Detection condition: "
            f">= {config.minimum_syn_flows} SYN-bearing flows and "
            f">= {config.minimum_unique_ports} unique destination ports "
            f"within {config.window_seconds:g} seconds"
        ),
        "",
        "Important:",
        "  This layer supplements the trained per-flow ML models.",
        "  It is not a retrained classifier and does not alter saved ML thresholds.",
        "",
    ]

    for row in summary_frame.itertuples(index=False):
        lines.extend(
            [
                f"Scenario: {row.Scenario}",
                f"  Ground truth: {row.GroundTruth}",
                f"  Source IP: {row.SourceIP}",
                f"  Total flows: {row.TotalFlows}",
                (
                    "  Total unique destination ports: "
                    f"{row.TotalUniqueDestinationPorts}"
                ),
                (
                    "  Maximum unique ports in one window: "
                    f"{row.MaxUniqueDestinationPortsInWindow}"
                ),
                (
                    "  Maximum SYN-bearing flows in one window: "
                    f"{row.MaxSYNFlowsInWindow}"
                ),
                f"  Cross-flow scan flag: {row.CrossFlowScanFlag}",
                f"  Detection: {row.Detection}",
                "",
            ]
        )

    lines.extend(
        [
            f"Alerts emitted: {len(alert_frame)}",
            f"Summary CSV: {TABLE_DIR / 'live_lab_cross_flow_summary.csv'}",
            (
                "Alert CSV: "
                f"{PREDICTION_DIR / 'cross_flow_scan_alerts.csv'}"
            ),
            "",
        ]
    )

    TEXT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = DetectorConfig(
        window_seconds=args.window_seconds,
        minimum_unique_ports=args.minimum_unique_ports,
        minimum_syn_flows=args.minimum_syn_flows,
    )

    input_paths = args.input or DEFAULT_INPUTS
    input_paths = [path.expanduser().resolve() for path in input_paths]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    summary_frame, alert_frame = process_inputs(
        input_paths,
        config,
    )

    summary_path = TABLE_DIR / "live_lab_cross_flow_summary.csv"
    alert_path = PREDICTION_DIR / "cross_flow_scan_alerts.csv"

    summary_frame.to_csv(summary_path, index=False)

    if alert_frame.empty:
        pd.DataFrame(
            columns=[
                "AlertTimestamp",
                "SourceIP",
                "DestinationIP",
                "WindowSeconds",
                "WindowFlowCount",
                "WindowUniqueDestinationPorts",
                "WindowSYNFlows",
                "Detector",
                "Detection",
                "Severity",
                "MITRETactic",
                "MITRETechniqueID",
                "MITRETechniqueName",
                "Reason",
                "InputFile",
                "Scenario",
                "GroundTruth",
            ]
        ).to_csv(alert_path, index=False)
    else:
        alert_frame.to_csv(alert_path, index=False)

    write_text_report(summary_frame, alert_frame, config)

    print("AI NIDS cross-flow scan correlation")
    print("=" * 78)
    print(
        "Rule: "
        f">= {config.minimum_syn_flows} SYN-bearing flows and "
        f">= {config.minimum_unique_ports} unique destination ports "
        f"within {config.window_seconds:g} seconds"
    )
    print()

    display_columns = [
        "Scenario",
        "GroundTruth",
        "SourceIP",
        "TotalFlows",
        "TotalUniqueDestinationPorts",
        "MaxUniqueDestinationPortsInWindow",
        "MaxSYNFlowsInWindow",
        "CrossFlowScanFlag",
        "Detection",
    ]
    print(summary_frame[display_columns].to_string(index=False))
    print()
    print(f"Alerts emitted: {len(alert_frame)}")
    print(f"Summary CSV:   {summary_path}")
    print(f"Alert CSV:     {alert_path}")
    print(f"Text report:   {TEXT_REPORT}")

    if not alert_frame.empty:
        print()
        print("First correlation alert:")
        print(
            alert_frame[
                [
                    "AlertTimestamp",
                    "SourceIP",
                    "DestinationIP",
                    "WindowUniqueDestinationPorts",
                    "WindowSYNFlows",
                    "Detection",
                    "MITRETechniqueID",
                ]
            ]
            .head(1)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
