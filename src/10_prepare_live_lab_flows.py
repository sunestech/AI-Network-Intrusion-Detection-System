from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "live_lab" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "live_lab" / "model_ready"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
FEATURE_MANIFEST = REPORT_DIR / "feature_manifest.csv"

SCENARIOS = {
    "benign_baseline_raw.csv": {
        "scenario": "benign_baseline",
        "ground_truth": "BENIGN",
    },
    "portscan_rate_limited_raw.csv": {
        "scenario": "portscan_rate_limited",
        "ground_truth": "PORT SCAN",
    },
}

# Exact mapping from the 77-feature CICIDS2017 training schema to the
# names emitted by the Python cicflowmeter 0.5.0 extractor.
COLUMN_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    # CICIDS2017 uses the historical heading "CWE Flag Count" for CWR.
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}

# The Python extractor stores packet-time differences in seconds, whereas
# the CICFlowMeter/CICIDS training schema uses microseconds for time features.
TIME_FEATURES = {
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
}
SECONDS_TO_MICROSECONDS = 1_000_000.0

CONTEXT_COLUMNS = {
    "src_ip": "SourceIP",
    "dst_ip": "DestinationIP",
    "src_port": "SourcePort",
    "protocol": "ProtocolNumber",
    "timestamp": "Timestamp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_manifest() -> list[str]:
    if not FEATURE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Feature manifest not found: {FEATURE_MANIFEST}"
        )

    manifest = pd.read_csv(FEATURE_MANIFEST)
    required_columns = {"position", "feature"}
    missing_columns = required_columns.difference(manifest.columns)

    if missing_columns:
        raise ValueError(
            "Feature manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    manifest["position"] = pd.to_numeric(
        manifest["position"],
        errors="raise",
    )
    manifest = manifest.sort_values("position")
    feature_names = (
        manifest["feature"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(feature_names) != 77:
        raise ValueError(
            f"Expected 77 trained features; found {len(feature_names)}."
        )

    if len(set(feature_names)) != 77:
        raise ValueError("Feature manifest contains duplicate feature names.")

    mapped_features = set(COLUMN_MAP)
    manifest_features = set(feature_names)

    if mapped_features != manifest_features:
        missing_from_map = sorted(manifest_features - mapped_features)
        unexpected_in_map = sorted(mapped_features - manifest_features)
        raise ValueError(
            "Mapping does not match the trained feature manifest.\n"
            f"Missing from mapping: {missing_from_map}\n"
            f"Unexpected in mapping: {unexpected_in_map}"
        )

    return feature_names


def prepare_one_file(
    input_path: Path,
    feature_names: list[str],
    scenario: str,
    ground_truth: str,
) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw flow CSV not found: {input_path}")

    raw = pd.read_csv(input_path)
    raw.columns = raw.columns.astype(str).str.strip()

    required_raw_columns = set(COLUMN_MAP.values()) | set(CONTEXT_COLUMNS)
    missing_raw_columns = sorted(required_raw_columns - set(raw.columns))

    if missing_raw_columns:
        raise ValueError(
            f"{input_path.name} is missing required raw columns: "
            + ", ".join(missing_raw_columns)
        )

    model_columns: dict[str, pd.Series] = {}
    coercion_failures = 0

    for feature in feature_names:
        raw_column = COLUMN_MAP[feature]
        original = raw[raw_column]
        numeric = pd.to_numeric(original, errors="coerce")

        coercion_failures += int(
            ((original.notna()) & (numeric.isna())).sum()
        )

        if feature in TIME_FEATURES:
            numeric = numeric * SECONDS_TO_MICROSECONDS

        model_columns[feature] = numeric

    model_frame = pd.DataFrame(
        model_columns,
        columns=feature_names,
    )
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan)
    model_frame = model_frame.astype("float32")

    if list(model_frame.columns) != feature_names:
        raise RuntimeError("Model feature order does not match the manifest.")

    context = raw[list(CONTEXT_COLUMNS)].rename(
        columns=CONTEXT_COLUMNS
    )
    context.insert(0, "RowID", np.arange(len(context), dtype=np.int64))
    context.insert(1, "Scenario", scenario)
    context.insert(2, "GroundTruth", ground_truth)

    dashboard_frame = pd.concat(
        [
            context.reset_index(drop=True),
            model_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_output = OUTPUT_DIR / f"{scenario}_model77.csv"
    dashboard_output = OUTPUT_DIR / f"{scenario}_dashboard.csv"

    model_frame.to_csv(model_output, index=False)
    dashboard_frame.to_csv(dashboard_output, index=False)

    missing_cells = int(model_frame.isna().sum().sum())
    rows_with_missing = int(model_frame.isna().any(axis=1).sum())

    return {
        "scenario": scenario,
        "ground_truth": ground_truth,
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "model_rows": int(len(model_frame)),
        "model_features": int(len(model_frame.columns)),
        "exact_feature_order_match": (
            list(model_frame.columns) == feature_names
        ),
        "time_features_scaled": len(TIME_FEATURES),
        "time_scale_factor": int(SECONDS_TO_MICROSECONDS),
        "numeric_coercion_failures": int(coercion_failures),
        "missing_model_cells": missing_cells,
        "rows_with_missing_model_values": rows_with_missing,
        "model_output": str(model_output),
        "model_output_sha256": sha256(model_output),
        "dashboard_output": str(dashboard_output),
        "dashboard_output_sha256": sha256(dashboard_output),
    }


def main() -> None:
    feature_names = load_feature_manifest()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    mapping_report = pd.DataFrame(
        {
            "position": range(1, len(feature_names) + 1),
            "feature": feature_names,
            "raw_column": [COLUMN_MAP[name] for name in feature_names],
            "transform": [
                "seconds_to_microseconds"
                if name in TIME_FEATURES
                else "numeric_identity"
                for name in feature_names
            ],
        }
    )
    mapping_path = REPORT_DIR / "live_lab_feature_mapping.csv"
    mapping_report.to_csv(mapping_path, index=False)

    summaries: list[dict[str, object]] = []

    print("AI NIDS live-lab feature compatibility preparation")
    print("=" * 78)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Raw flow folder:  {RAW_DIR}")
    print(f"Feature manifest: {FEATURE_MANIFEST}")
    print(f"Trained features: {len(feature_names)}")
    print()

    for filename, metadata in SCENARIOS.items():
        input_path = RAW_DIR / filename
        print(f"[PROCESSING] {filename}")

        summary = prepare_one_file(
            input_path=input_path,
            feature_names=feature_names,
            scenario=metadata["scenario"],
            ground_truth=metadata["ground_truth"],
        )
        summaries.append(summary)

        print(f"  Rows:                  {summary['model_rows']}")
        print(f"  Raw columns:           {summary['raw_columns']}")
        print(f"  Model features:        {summary['model_features']}")
        print(
            "  Exact feature order:   "
            f"{summary['exact_feature_order_match']}"
        )
        print(
            "  Time fields converted: "
            f"{summary['time_features_scaled']}"
        )
        print(
            "  Missing model cells:   "
            f"{summary['missing_model_cells']}"
        )
        print(f"  Dashboard CSV:         {summary['dashboard_output']}")
        print()

    summary_frame = pd.DataFrame(summaries)
    compatibility_path = (
        REPORT_DIR / "live_lab_feature_compatibility.csv"
    )
    compatibility_json = (
        REPORT_DIR / "live_lab_feature_compatibility.json"
    )

    summary_frame.to_csv(compatibility_path, index=False)
    compatibility_json.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    limitation_path = (
        PROJECT_ROOT
        / "reports"
        / "live_lab_extractor_limitations.txt"
    )
    limitation_path.parent.mkdir(parents=True, exist_ok=True)
    limitation_path.write_text(
        "AI NIDS LIVE-LAB EXTRACTOR LIMITATIONS\n"
        "======================================\n"
        "1. The laboratory CSVs were generated by the community Python "
        "cicflowmeter 0.5.0 implementation, not the original Java "
        "CICFlowMeter used to create CICIDS2017.\n"
        "2. The Python extractor emits packet-time differences in seconds. "
        "This preparation step multiplies the 23 trained time features by "
        "1,000,000 to align them with the CICFlowMeter microsecond schema.\n"
        "3. The extractor exposes cwr_flag_count, mapped to the historical "
        "CICIDS heading 'CWE Flag Count'. Its implementation should be "
        "treated as an extractor-specific limitation.\n"
        "4. The extractor duplicates several convenience fields, including "
        "segment-size averages and subflow totals. Predictions are therefore "
        "an external compatibility experiment, not a native CICIDS2017 "
        "benchmark result.\n"
        "5. The model pipeline retains responsibility for imputing any "
        "missing numeric values using parameters learned during training.\n",
        encoding="utf-8",
    )

    print("[COMPLETE] Live-lab feature preparation finished.")
    print(f"Compatibility CSV: {compatibility_path}")
    print(f"Mapping CSV:       {mapping_path}")
    print(f"Limitations:       {limitation_path}")
    print(f"Model-ready files: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
