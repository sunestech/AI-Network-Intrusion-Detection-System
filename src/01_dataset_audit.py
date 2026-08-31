#!/usr/bin/env python3
"""
01_dataset_audit.py

Chunked, reproducible intake audit for the AI-Based Network Intrusion
Detection System project.

Run from the project root:
    python .\src\01_dataset_audit.py

The script:
- discovers the 8 machine-learning CSVs and 8 generated-flow CSVs;
- reads them in chunks to avoid loading millions of rows into RAM;
- normalizes headers and attack labels;
- counts rows, empty rows, missing cells, and infinite values;
- detects duplicate header names;
- compares the 79-column and 85-column collections;
- creates CSV/JSON/TXT reports and class-distribution figures.

It never modifies the original source CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


LABEL_CANDIDATES = ("label", "class", "target", "attack", "attack_cat", "category")
INFINITY_TOKENS = {"Infinity", "+Infinity", "-Infinity", "inf", "+inf", "-inf", "INF", "-INF"}

CONTEXT_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Protocol",
    "Timestamp",
}

CANONICAL_LABELS = {
    "benign": "BENIGN",
    "bot": "Bot",
    "ddos": "DDoS",
    "dos goldeneye": "DoS GoldenEye",
    "dos hulk": "DoS Hulk",
    "dos slowhttptest": "DoS Slowhttptest",
    "dos slowloris": "DoS slowloris",
    "ftp-patator": "FTP-Patator",
    "heartbleed": "Heartbleed",
    "infiltration": "Infiltration",
    "portscan": "PortScan",
    "ssh-patator": "SSH-Patator",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - sql injection": "Web Attack - SQL Injection",
    "web attack - xss": "Web Attack - XSS",
}


def normalize_column(name: Any) -> str:
    """Remove BOM characters and trim surrounding whitespace."""
    return str(name).replace("\ufeff", "").strip()


def make_unique(columns: Iterable[str]) -> list[str]:
    """Create deterministic unique column names while preserving order."""
    seen: dict[str, int] = {}
    result: list[str] = []

    for column in columns:
        count = seen.get(column, 0)
        result.append(column if count == 0 else f"{column}.{count}")
        seen[column] = count + 1

    return result


def normalize_label(value: Any) -> str:
    """Repair encoding artifacts and standardize known CICIDS label spelling."""
    if pd.isna(value):
        return "<MISSING>"

    text = str(value).strip()
    text = (
        text.replace("\ufffd", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Repair variants where the replacement character disappeared without a dash.
    text = re.sub(
        r"^Web Attack\s+(Brute Force|XSS|Sql Injection|SQL Injection)$",
        r"Web Attack - \1",
        text,
        flags=re.IGNORECASE,
    )

    lowered = text.lower()
    if lowered == "web attack - sql injection":
        return "Web Attack - SQL Injection"

    return CANONICAL_LABELS.get(lowered, text)


def find_label_column(columns: Iterable[str]) -> str | None:
    """Find the target column after header normalization."""
    column_list = list(columns)
    lower_map = {column.lower(): column for column in column_list}

    for candidate in LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in column_list:
        lowered = column.lower()
        if lowered.endswith("label") or "attack label" in lowered:
            return column

    return None


def raw_header(path: Path) -> list[str]:
    """Read the CSV header without loading any data rows."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        return [normalize_column(value) for value in next(reader)]


def scenario_key(filename: str) -> str:
    """Create a stable key for matching corresponding files across collections."""
    value = filename.lower()
    value = re.sub(r"\(\d+\)", "", value)
    value = value.replace(".pcap_iscx.csv", "")
    value = value.replace(".csv", "")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


def schema_sha256(columns: Iterable[str]) -> str:
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_infinite_values(chunk: pd.DataFrame) -> int:
    """Count both numeric infinities and textual Infinity tokens."""
    total = 0

    for column in chunk.columns:
        series = chunk[column]

        if is_numeric_dtype(series):
            values = series.to_numpy(dtype=float, copy=False)
            total += int(np.isinf(values).sum())
        else:
            total += int(series.isin(INFINITY_TOKENS).sum())

    return total


@dataclass
class FileAudit:
    collection: str
    file: str
    full_path: str
    scenario_key: str
    size_mb: float
    physical_rows: int = 0
    valid_rows: int = 0
    empty_rows: int = 0
    columns: int = 0
    unique_columns: int = 0
    predictive_columns_before_cleaning: int = 0
    duplicate_header_names: str = ""
    label_column: str = ""
    missing_cells: int = 0
    infinite_values: int = 0
    schema_sha256: str = ""
    status: str = "OK"
    column_list: list[str] = field(default_factory=list, repr=False)
    label_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def row(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "file": self.file,
            "scenario_key": self.scenario_key,
            "size_mb": self.size_mb,
            "physical_rows": self.physical_rows,
            "valid_rows": self.valid_rows,
            "empty_rows": self.empty_rows,
            "columns": self.columns,
            "unique_columns": self.unique_columns,
            "predictive_columns_before_cleaning": self.predictive_columns_before_cleaning,
            "duplicate_header_names": self.duplicate_header_names,
            "label_column": self.label_column,
            "missing_cells": self.missing_cells,
            "infinite_values": self.infinite_values,
            "schema_sha256": self.schema_sha256,
            "status": self.status,
        }


def audit_file(path: Path, collection: str, chunk_size: int) -> FileAudit:
    """Audit one CSV safely in chunks."""
    audit = FileAudit(
        collection=collection,
        file=path.name,
        full_path=str(path),
        scenario_key=scenario_key(path.name),
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
    )

    try:
        raw_columns = raw_header(path)
    except Exception as exc:
        audit.status = f"HEADER_ERROR: {exc}"
        return audit

    duplicate_names = [
        name for name, count in Counter(raw_columns).items() if count > 1
    ]
    unique_columns = make_unique(raw_columns)
    label_column = find_label_column(unique_columns)

    audit.columns = len(raw_columns)
    audit.unique_columns = len(set(raw_columns))
    audit.duplicate_header_names = "; ".join(duplicate_names)
    audit.label_column = label_column or ""
    audit.predictive_columns_before_cleaning = (
        len(raw_columns) - (1 if label_column else 0)
    )
    audit.column_list = unique_columns
    audit.schema_sha256 = schema_sha256(unique_columns)

    label_counts: defaultdict[str, int] = defaultdict(int)

    try:
        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk.columns = [normalize_column(column) for column in chunk.columns]

            # Pandas normally mangles duplicate headers with ".1". Enforce deterministic
            # names if a future version behaves differently.
            chunk.columns = make_unique(chunk.columns)

            physical_rows = len(chunk)
            empty_mask = chunk.isna().all(axis=1)
            empty_rows = int(empty_mask.sum())
            valid_chunk = chunk.loc[~empty_mask]

            audit.physical_rows += physical_rows
            audit.empty_rows += empty_rows
            audit.valid_rows += len(valid_chunk)
            audit.missing_cells += int(valid_chunk.isna().sum().sum())
            audit.infinite_values += count_infinite_values(valid_chunk)

            if label_column and label_column in valid_chunk.columns:
                normalized = valid_chunk[label_column].map(normalize_label)
                for label, count in normalized.value_counts(dropna=False).items():
                    label_counts[str(label)] += int(count)

            print(
                f"    {collection:<18} {path.name:<65} "
                f"{audit.physical_rows:>10,} rows",
                end="\r",
                flush=True,
            )

        print(" " * 130, end="\r")

    except Exception as exc:
        audit.status = f"READ_ERROR: {exc}"

    audit.label_counts = dict(sorted(label_counts.items()))
    return audit


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def create_schema_presence(
    audits: list[FileAudit],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for collection in sorted({audit.collection for audit in audits}):
        collection_audits = [
            audit for audit in audits
            if audit.collection == collection and audit.status == "OK"
        ]
        all_columns = sorted(
            {column for audit in collection_audits for column in audit.column_list}
        )

        for column in all_columns:
            rows.append(
                {
                    "collection": collection,
                    "column": column,
                    "files_present": sum(
                        column in audit.column_list for audit in collection_audits
                    ),
                    "total_files": len(collection_audits),
                    "present_in_all_files": all(
                        column in audit.column_list for audit in collection_audits
                    ),
                }
            )

    write_csv(output_path, rows)


def pair_comparison(audits: list[FileAudit]) -> list[dict[str, Any]]:
    ml = {
        audit.scenario_key: audit
        for audit in audits
        if audit.collection == "machine_learning"
    }
    generated = {
        audit.scenario_key: audit
        for audit in audits
        if audit.collection == "generated_flows"
    }

    rows: list[dict[str, Any]] = []

    for key in sorted(set(ml) | set(generated)):
        left = ml.get(key)
        right = generated.get(key)

        left_columns = set(left.column_list) if left else set()
        right_columns = set(right.column_list) if right else set()

        rows.append(
            {
                "scenario_key": key,
                "machine_learning_file": left.file if left else "",
                "generated_flow_file": right.file if right else "",
                "machine_learning_valid_rows": left.valid_rows if left else "",
                "generated_flow_valid_rows": right.valid_rows if right else "",
                "valid_row_difference": (
                    right.valid_rows - left.valid_rows if left and right else ""
                ),
                "machine_learning_columns": left.columns if left else "",
                "generated_flow_columns": right.columns if right else "",
                "common_columns": len(left_columns & right_columns),
                "generated_extra_columns": "; ".join(
                    sorted(right_columns - left_columns)
                ),
                "machine_learning_only_columns": "; ".join(
                    sorted(left_columns - right_columns)
                ),
                "label_counts_match": (
                    left.label_counts == right.label_counts if left and right else ""
                ),
            }
        )

    return rows


def aggregate_labels(
    audits: list[FileAudit],
    collection: str,
) -> list[dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)

    for audit in audits:
        if audit.collection != collection:
            continue
        for label, count in audit.label_counts.items():
            counts[label] += count

    total = sum(counts.values())
    return [
        {
            "collection": collection,
            "label": label,
            "count": count,
            "percentage": (count / total * 100) if total else 0.0,
        }
        for label, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def create_label_figures(
    ml_label_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    if not ml_label_rows:
        return

    labels = [row["label"] for row in ml_label_rows]
    counts = [int(row["count"]) for row in ml_label_rows]

    # Multiclass chart: logarithmic x-axis keeps rare classes visible.
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(len(labels))
    ax.barh(positions, counts)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Flow records (log scale)")
    ax.set_ylabel("Traffic label")
    ax.set_title("Machine-Learning Collection: Class Distribution")

    for position, count in zip(positions, counts):
        ax.text(count * 1.03, position, f"{count:,}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "multiclass_label_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    benign = sum(
        int(row["count"])
        for row in ml_label_rows
        if str(row["label"]).upper() == "BENIGN"
    )
    total = sum(counts)
    attack = total - benign

    fig, ax = plt.subplots(figsize=(8, 5))
    binary_labels = ["BENIGN", "ATTACK"]
    binary_counts = [benign, attack]
    bars = ax.bar(binary_labels, binary_counts)
    ax.set_ylabel("Flow records")
    ax.set_title("Machine-Learning Collection: Binary Distribution")

    for bar, count in zip(bars, binary_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        figures_dir / "binary_label_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CICIDS-style NIDS CSV collections without loading them all into RAM."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows loaded into memory at once. Default: 50000.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    ml_dir = project_root / "data" / "raw" / "machine_learning"
    generated_dir = project_root / "data" / "raw" / "generated_flows"
    reports_dir = project_root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    collections = {
        "machine_learning": sorted(ml_dir.rglob("*.csv")),
        "generated_flows": sorted(generated_dir.rglob("*.csv")),
    }

    print("AI NIDS dataset audit")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Machine-learning CSVs found: {len(collections['machine_learning'])}")
    print(f"Generated-flow CSVs found: {len(collections['generated_flows'])}")
    print()

    if not collections["machine_learning"]:
        print(f"ERROR: No machine-learning CSVs found under {ml_dir}", file=sys.stderr)
        return 2

    if not collections["generated_flows"]:
        print(f"ERROR: No generated-flow CSVs found under {generated_dir}", file=sys.stderr)
        return 3

    audits: list[FileAudit] = []

    total_files = sum(len(files) for files in collections.values())
    file_number = 0

    for collection, files in collections.items():
        for path in files:
            file_number += 1
            print(f"[{file_number}/{total_files}] Auditing {collection}: {path.name}")
            audits.append(
                audit_file(
                    path=path,
                    collection=collection,
                    chunk_size=max(1_000, args.chunk_size),
                )
            )

    inventory_rows = [audit.row() for audit in audits]
    write_csv(tables_dir / "dataset_inventory.csv", inventory_rows)

    per_file_label_rows: list[dict[str, Any]] = []
    for audit in audits:
        for label, count in audit.label_counts.items():
            per_file_label_rows.append(
                {
                    "collection": audit.collection,
                    "file": audit.file,
                    "label": label,
                    "count": count,
                    "percentage_within_file": (
                        count / audit.valid_rows * 100 if audit.valid_rows else 0.0
                    ),
                }
            )
    write_csv(tables_dir / "label_counts_by_file.csv", per_file_label_rows)

    ml_label_rows = aggregate_labels(audits, "machine_learning")
    generated_label_rows = aggregate_labels(audits, "generated_flows")
    write_csv(tables_dir / "machine_learning_label_distribution.csv", ml_label_rows)
    write_csv(tables_dir / "generated_flow_label_distribution.csv", generated_label_rows)

    create_schema_presence(
        audits,
        tables_dir / "schema_presence.csv",
    )

    pair_rows = pair_comparison(audits)
    write_csv(tables_dir / "collection_pair_comparison.csv", pair_rows)

    create_label_figures(ml_label_rows, figures_dir)

    collection_summary: dict[str, Any] = {}
    for collection in collections:
        relevant = [audit for audit in audits if audit.collection == collection]
        collection_summary[collection] = {
            "files": len(relevant),
            "successful_files": sum(audit.status == "OK" for audit in relevant),
            "physical_rows": sum(audit.physical_rows for audit in relevant),
            "valid_rows": sum(audit.valid_rows for audit in relevant),
            "empty_rows": sum(audit.empty_rows for audit in relevant),
            "missing_cells": sum(audit.missing_cells for audit in relevant),
            "infinite_values": sum(audit.infinite_values for audit in relevant),
            "total_size_mb": round(sum(audit.size_mb for audit in relevant), 2),
        }

    ml_counts = {
        row["label"]: int(row["count"])
        for row in ml_label_rows
    }
    ml_total = sum(ml_counts.values())
    ml_benign = ml_counts.get("BENIGN", 0)
    ml_attack = ml_total - ml_benign

    generated_extra_columns = sorted(
        {
            column
            for row in pair_rows
            for column in str(row["generated_extra_columns"]).split("; ")
            if column
        }
    )

    summary = {
        "project_root": str(project_root),
        "chunk_size": args.chunk_size,
        "collections": collection_summary,
        "machine_learning_binary_distribution": {
            "BENIGN": ml_benign,
            "ATTACK": ml_attack,
            "BENIGN_percentage": (ml_benign / ml_total * 100) if ml_total else 0.0,
            "ATTACK_percentage": (ml_attack / ml_total * 100) if ml_total else 0.0,
        },
        "machine_learning_distinct_labels": len(ml_counts),
        "generated_context_columns_detected": generated_extra_columns,
        "expected_context_columns": sorted(CONTEXT_COLUMNS),
        "important_notes": [
            "The source CSV files were read only; they were not modified.",
            "Completely empty rows are excluded from valid-row and label counts.",
            "Generated-flow files should not be treated as an independent test set when they contain the same underlying flows as the machine-learning collection.",
            "Overall accuracy alone is insufficient because rare attack classes are heavily imbalanced.",
        ],
    }

    with (reports_dir / "dataset_audit_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    text_lines = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "DATASET AUDIT SUMMARY",
        "=" * 72,
        "",
    ]

    for collection, values in collection_summary.items():
        text_lines.extend(
            [
                f"Collection: {collection}",
                f"  Files: {values['files']}",
                f"  Valid rows: {values['valid_rows']:,}",
                f"  Empty rows: {values['empty_rows']:,}",
                f"  Missing cells: {values['missing_cells']:,}",
                f"  Infinite values: {values['infinite_values']:,}",
                f"  Total size: {values['total_size_mb']:,.2f} MB",
                "",
            ]
        )

    text_lines.extend(
        [
            "Machine-learning binary distribution:",
            f"  BENIGN: {ml_benign:,}",
            f"  ATTACK: {ml_attack:,}",
            "",
            "Generated-flow context columns detected:",
            "  " + ", ".join(generated_extra_columns),
            "",
            "Reports created:",
            f"  {tables_dir / 'dataset_inventory.csv'}",
            f"  {tables_dir / 'label_counts_by_file.csv'}",
            f"  {tables_dir / 'machine_learning_label_distribution.csv'}",
            f"  {tables_dir / 'generated_flow_label_distribution.csv'}",
            f"  {tables_dir / 'schema_presence.csv'}",
            f"  {tables_dir / 'collection_pair_comparison.csv'}",
            f"  {figures_dir / 'multiclass_label_distribution.png'}",
            f"  {figures_dir / 'binary_label_distribution.png'}",
            "",
            "Audit completed successfully.",
        ]
    )

    (reports_dir / "dataset_audit_summary.txt").write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Dataset audit finished.")
    print(f"Summary: {reports_dir / 'dataset_audit_summary.txt'}")
    print(f"Inventory: {tables_dir / 'dataset_inventory.csv'}")
    print(f"Label table: {tables_dir / 'machine_learning_label_distribution.csv'}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
