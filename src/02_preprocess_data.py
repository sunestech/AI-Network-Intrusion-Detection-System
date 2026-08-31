#!/usr/bin/env python3
r"""
02_preprocess_data.py

Memory-safe preprocessing for the AI-Based Network Intrusion Detection System.

Run from the project root:
    python .\src\02_preprocess_data.py

What this script does
---------------------
1. Reads the eight 79-column machine-learning CSVs in chunks.
2. Reads the eight 85-column generated-flow CSVs in chunks.
3. Drops completely empty rows.
4. Normalizes column names and attack labels.
5. Verifies and removes duplicated feature columns such as
   "Fwd Header Length.1" when they match the original column.
6. Converts feature columns to numeric float32 values.
7. Replaces positive/negative infinity with missing values.
8. Preserves missing values for a train-only imputer in the later model pipeline.
9. Creates:
   - normalized multiclass labels,
   - binary labels (0=BENIGN, 1=ATTACK),
   - operational attack-family labels.
10. Adds a stable RowID to corresponding files.
11. Writes compressed Parquet files for faster model training.
12. Creates compact generated-flow context Parquet files for the SOC dashboard.
13. Verifies row-order alignment between corresponding machine-learning and
    generated-flow files using hashes of shared columns.

The raw source CSV files are read only and are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LABEL_CANDIDATES = ("label", "class", "target", "attack", "attack_cat", "category")
INFINITY_TOKENS = {
    "Infinity",
    "+Infinity",
    "-Infinity",
    "inf",
    "+inf",
    "-inf",
    "INF",
    "+INF",
    "-INF",
}

GENERATED_CONTEXT_COLUMNS = (
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
)

ALIGNMENT_COLUMNS = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Label",
)

CANONICAL_LABELS = {
    "benign": "BENIGN",
    "bot": "Bot",
    "ddos": "DDoS",
    "dos goldeneye": "DoS GoldenEye",
    "dos hulk": "DoS Hulk",
    "dos slowhttptest": "DoS Slowhttptest",
    "dos slowloris": "DoS slowloris",
    "ftp-patator": "FTP-Patator",
    "ftp - patator": "FTP-Patator",
    "heartbleed": "Heartbleed",
    "infiltration": "Infiltration",
    "portscan": "PortScan",
    "ssh-patator": "SSH-Patator",
    "ssh - patator": "SSH-Patator",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - sql injection": "Web Attack - SQL Injection",
    "web attack - xss": "Web Attack - XSS",
}

ATTACK_FAMILY_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "Heartbleed": "Exploit",
    "PortScan": "Port Scan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - SQL Injection": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
}


def normalize_column(name: Any) -> str:
    """Remove a BOM and trim surrounding whitespace."""
    return str(name).replace("\ufeff", "").strip()


def make_unique(columns: Iterable[str]) -> list[str]:
    """Make duplicate column names deterministic while preserving order."""
    seen: dict[str, int] = {}
    unique: list[str] = []

    for column in columns:
        count = seen.get(column, 0)
        unique.append(column if count == 0 else f"{column}.{count}")
        seen[column] = count + 1

    return unique


def normalize_label(value: Any) -> str:
    """Repair encoding artifacts and standardize known CICIDS labels."""
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

    text = re.sub(
        r"^Web Attack\s+(Brute Force|XSS|Sql Injection|SQL Injection)$",
        r"Web Attack - \1",
        text,
        flags=re.IGNORECASE,
    )

    return CANONICAL_LABELS.get(text.lower(), text)


def attack_family(label: str) -> str:
    """Map a normalized attack label to a broader operational family."""
    return ATTACK_FAMILY_MAP.get(label, "Other Attack")


def find_label_column(columns: Iterable[str]) -> str | None:
    """Locate the target column after header normalization."""
    column_list = list(columns)
    lower_map = {column.lower(): column for column in column_list}

    for candidate in LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in column_list:
        if column.lower().endswith("label"):
            return column

    return None


def scenario_key(filename: str) -> str:
    """Create a stable matching key from a CICIDS filename."""
    value = filename.lower()
    value = re.sub(r"\(\d+\)", "", value)
    value = value.replace(".pcap_iscx.csv", "")
    value = value.replace(".csv", "")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def safe_stem(filename: str) -> str:
    """Create a filesystem-safe output stem."""
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem


def clean_duplicate_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Remove pandas-mangled duplicate columns only when they exactly match
    their base column in the current chunk.

    Returns:
        cleaned_frame, dropped_columns, mismatched_duplicate_columns
    """
    dropped: list[str] = []
    mismatched: list[str] = []

    for column in list(frame.columns):
        match = re.match(r"^(.*)\.(\d+)$", column)
        if not match:
            continue

        base = match.group(1)
        if base not in frame.columns:
            continue

        left = frame[base]
        right = frame[column]

        equal_mask = left.eq(right) | (left.isna() & right.isna())
        if bool(equal_mask.all()):
            dropped.append(column)
        else:
            mismatched.append(column)

    if dropped:
        frame = frame.drop(columns=dropped)

    return frame, dropped, mismatched


def numeric_alignment_frame(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Build a canonical subset used to verify cross-collection row alignment."""
    result = pd.DataFrame(index=frame.index)

    for column in ALIGNMENT_COLUMNS:
        actual = label_column if column == "Label" else column

        if actual not in frame.columns:
            result[column] = np.nan if column != "Label" else "<MISSING>"
            continue

        if column == "Label":
            result[column] = frame[actual].map(normalize_label).astype("string")
        else:
            result[column] = pd.to_numeric(
                frame[actual],
                errors="coerce",
            ).astype("float64")

    return result


def update_alignment_hash(
    digest: "hashlib._Hash",
    frame: pd.DataFrame,
    label_column: str,
) -> None:
    """Update a streaming SHA-256 hash using shared columns in row order."""
    canonical = numeric_alignment_frame(frame, label_column)
    row_hashes = pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())


def replace_infinity_and_convert(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, int, int]:
    """
    Convert feature columns to float32 and replace +/- infinity with NaN.

    Returns:
        frame, infinity_values_replaced, non_numeric_values_coerced
    """
    infinity_replaced = 0
    non_numeric_coerced = 0

    for column in feature_columns:
        original = frame[column]

        if not pd.api.types.is_numeric_dtype(original):
            stripped = original.astype("string").str.strip()
            textual_inf = stripped.isin(INFINITY_TOKENS)
            infinity_replaced += int(textual_inf.sum())

            converted = pd.to_numeric(
                stripped.mask(textual_inf),
                errors="coerce",
            )

            newly_missing = (
                converted.isna()
                & original.notna()
                & ~textual_inf
            )
            non_numeric_coerced += int(newly_missing.sum())
        else:
            converted = pd.to_numeric(original, errors="coerce")

        numeric_values = converted.to_numpy(dtype="float64", copy=False)
        numeric_inf = np.isinf(numeric_values)
        infinity_replaced += int(numeric_inf.sum())

        converted = converted.mask(numeric_inf)
        frame[column] = converted.astype("float32")

    return frame, infinity_replaced, non_numeric_coerced


def write_parquet_chunk(
    writer: pq.ParquetWriter | None,
    output_path: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter:
    """Append one pandas chunk to a compressed Parquet file."""
    table = pa.Table.from_pandas(frame, preserve_index=False)

    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            output_path,
            table.schema,
            compression="snappy",
            use_dictionary=True,
        )
    elif table.schema != writer.schema:
        table = table.cast(writer.schema)

    writer.write_table(table)
    return writer


@dataclass
class ProcessingSummary:
    collection: str
    source_file: str
    scenario: str
    output_file: str
    input_rows: int = 0
    valid_rows: int = 0
    empty_rows_dropped: int = 0
    input_columns: int = 0
    output_columns: int = 0
    feature_count: int = 0
    duplicate_columns_dropped: str = ""
    duplicate_columns_mismatched: str = ""
    missing_feature_values_after_cleaning: int = 0
    infinity_values_replaced: int = 0
    non_numeric_values_coerced: int = 0
    alignment_sha256: str = ""
    status: str = "OK"


def process_machine_learning_file(
    source_path: Path,
    output_path: Path,
    chunk_size: int,
    expected_features: list[str] | None,
    preview_frames: list[pd.DataFrame],
) -> tuple[ProcessingSummary, list[str]]:
    """Clean one 79-column machine-learning CSV and write Parquet."""
    summary = ProcessingSummary(
        collection="machine_learning",
        source_file=source_path.name,
        scenario=scenario_key(source_path.name),
        output_file=str(output_path),
    )

    writer: pq.ParquetWriter | None = None
    row_id = 0
    digest = hashlib.sha256()
    dropped_all: set[str] = set()
    mismatched_all: set[str] = set()
    discovered_features: list[str] | None = None

    try:
        reader = pd.read_csv(
            source_path,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="error",
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            chunk.columns = make_unique(
                [normalize_column(column) for column in chunk.columns]
            )

            summary.input_rows += len(chunk)
            summary.input_columns = max(summary.input_columns, len(chunk.columns))

            empty_mask = chunk.isna().all(axis=1)
            summary.empty_rows_dropped += int(empty_mask.sum())
            chunk = chunk.loc[~empty_mask].copy()

            if chunk.empty:
                continue

            label_column = find_label_column(chunk.columns)
            if not label_column:
                raise ValueError("No label column was found.")

            update_alignment_hash(digest, chunk, label_column)

            chunk, dropped, mismatched = clean_duplicate_features(chunk)
            dropped_all.update(dropped)
            mismatched_all.update(mismatched)

            if mismatched:
                raise ValueError(
                    "Duplicate-looking columns did not match their base columns: "
                    + ", ".join(sorted(mismatched))
                )

            labels_raw = chunk[label_column].astype("string").str.strip()
            labels = labels_raw.map(normalize_label).astype("string")
            binary = labels.ne("BENIGN").astype("int8")
            families = labels.map(attack_family).astype("string")

            feature_columns = [
                column for column in chunk.columns if column != label_column
            ]

            if discovered_features is None:
                discovered_features = feature_columns

                if expected_features is not None and feature_columns != expected_features:
                    raise ValueError(
                        "Feature schema differs from the first machine-learning file."
                    )
            elif feature_columns != discovered_features:
                raise ValueError("Feature order changed between chunks.")

            chunk, infinity_count, coerced_count = replace_infinity_and_convert(
                chunk,
                feature_columns,
            )

            summary.infinity_values_replaced += infinity_count
            summary.non_numeric_values_coerced += coerced_count
            summary.missing_feature_values_after_cleaning += int(
                chunk[feature_columns].isna().sum().sum()
            )

            valid_count = len(chunk)
            output = pd.DataFrame(
                {
                    "RowID": np.arange(
                        row_id,
                        row_id + valid_count,
                        dtype=np.int64,
                    ),
                    "SourceFile": pd.Series(
                        [source_path.name] * valid_count,
                        dtype="string",
                    ),
                    "Scenario": pd.Series(
                        [summary.scenario] * valid_count,
                        dtype="string",
                    ),
                }
            )

            for column in feature_columns:
                output[column] = chunk[column].reset_index(drop=True)

            output["LabelRaw"] = labels_raw.reset_index(drop=True).astype("string")
            output["Label"] = labels.reset_index(drop=True).astype("string")
            output["BinaryLabel"] = binary.reset_index(drop=True).astype("int8")
            output["AttackFamily"] = families.reset_index(drop=True).astype("string")

            writer = write_parquet_chunk(writer, output_path, output)

            if len(preview_frames) < 8:
                preview_frames.append(output.head(20).copy())

            row_id += valid_count
            summary.valid_rows += valid_count
            summary.output_columns = len(output.columns)
            summary.feature_count = len(feature_columns)

            print(
                f"    {source_path.name}: {summary.valid_rows:,} valid rows",
                end="\r",
                flush=True,
            )

        print(" " * 120, end="\r")

        if discovered_features is None:
            raise ValueError("No valid rows were found.")

        summary.duplicate_columns_dropped = "; ".join(sorted(dropped_all))
        summary.duplicate_columns_mismatched = "; ".join(sorted(mismatched_all))
        summary.alignment_sha256 = digest.hexdigest()
        return summary, discovered_features

    except Exception as exc:
        summary.status = f"ERROR: {exc}"
        raise

    finally:
        if writer is not None:
            writer.close()


def process_generated_context_file(
    source_path: Path,
    output_path: Path,
    chunk_size: int,
) -> ProcessingSummary:
    """Create a compact context dataset from one 85-column generated-flow CSV."""
    summary = ProcessingSummary(
        collection="generated_flows",
        source_file=source_path.name,
        scenario=scenario_key(source_path.name),
        output_file=str(output_path),
    )

    writer: pq.ParquetWriter | None = None
    row_id = 0
    digest = hashlib.sha256()

    try:
        reader = pd.read_csv(
            source_path,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
            on_bad_lines="error",
        )

        for chunk in reader:
            chunk.columns = make_unique(
                [normalize_column(column) for column in chunk.columns]
            )

            summary.input_rows += len(chunk)
            summary.input_columns = max(summary.input_columns, len(chunk.columns))

            empty_mask = chunk.isna().all(axis=1)
            summary.empty_rows_dropped += int(empty_mask.sum())
            chunk = chunk.loc[~empty_mask].copy()

            if chunk.empty:
                continue

            label_column = find_label_column(chunk.columns)
            if not label_column:
                raise ValueError("No label column was found.")

            update_alignment_hash(digest, chunk, label_column)

            labels_raw = chunk[label_column].astype("string").str.strip()
            labels = labels_raw.map(normalize_label).astype("string")
            binary = labels.ne("BENIGN").astype("int8")
            families = labels.map(attack_family).astype("string")

            valid_count = len(chunk)
            output = pd.DataFrame(
                {
                    "RowID": np.arange(
                        row_id,
                        row_id + valid_count,
                        dtype=np.int64,
                    ),
                    "SourceFile": pd.Series(
                        [source_path.name] * valid_count,
                        dtype="string",
                    ),
                    "Scenario": pd.Series(
                        [summary.scenario] * valid_count,
                        dtype="string",
                    ),
                }
            )

            for column in GENERATED_CONTEXT_COLUMNS:
                if column not in chunk.columns:
                    if column in {"Source Port", "Destination Port", "Protocol"}:
                        output[column] = pd.Series(
                            [np.nan] * valid_count,
                            dtype="float64",
                        )
                    else:
                        output[column] = pd.Series(
                            [pd.NA] * valid_count,
                            dtype="string",
                        )
                    continue

                if column in {"Source Port", "Destination Port", "Protocol"}:
                    output[column] = pd.to_numeric(
                        chunk[column],
                        errors="coerce",
                    ).reset_index(drop=True).astype("float64")
                else:
                    output[column] = (
                        chunk[column]
                        .reset_index(drop=True)
                        .astype("string")
                    )

            output["LabelRaw"] = labels_raw.reset_index(drop=True).astype("string")
            output["Label"] = labels.reset_index(drop=True).astype("string")
            output["BinaryLabel"] = binary.reset_index(drop=True).astype("int8")
            output["AttackFamily"] = families.reset_index(drop=True).astype("string")

            writer = write_parquet_chunk(writer, output_path, output)

            row_id += valid_count
            summary.valid_rows += valid_count
            summary.output_columns = len(output.columns)

            print(
                f"    {source_path.name}: {summary.valid_rows:,} valid rows",
                end="\r",
                flush=True,
            )

        print(" " * 120, end="\r")

        summary.alignment_sha256 = digest.hexdigest()
        return summary

    except Exception as exc:
        summary.status = f"ERROR: {exc}"
        raise

    finally:
        if writer is not None:
            writer.close()


def pair_summaries(
    summaries: list[ProcessingSummary],
) -> list[dict[str, Any]]:
    ml = {
        item.scenario: item
        for item in summaries
        if item.collection == "machine_learning"
    }
    generated = {
        item.scenario: item
        for item in summaries
        if item.collection == "generated_flows"
    }

    rows: list[dict[str, Any]] = []

    for scenario in sorted(set(ml) | set(generated)):
        left = ml.get(scenario)
        right = generated.get(scenario)

        rows.append(
            {
                "scenario": scenario,
                "machine_learning_file": left.source_file if left else "",
                "generated_flow_file": right.source_file if right else "",
                "machine_learning_rows": left.valid_rows if left else "",
                "generated_flow_rows": right.valid_rows if right else "",
                "row_counts_match": (
                    left.valid_rows == right.valid_rows if left and right else False
                ),
                "alignment_hash_match": (
                    left.alignment_sha256 == right.alignment_sha256
                    if left and right
                    else False
                ),
                "machine_learning_alignment_sha256": (
                    left.alignment_sha256 if left else ""
                ),
                "generated_flow_alignment_sha256": (
                    right.alignment_sha256 if right else ""
                ),
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the CICIDS NIDS datasets into compressed Parquet files."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows read into memory at once. Default: 50000.",
    )
    parser.add_argument(
        "--skip-generated-context",
        action="store_true",
        help="Process only the machine-learning CSVs.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    ml_input = project_root / "data" / "raw" / "machine_learning"
    generated_input = project_root / "data" / "raw" / "generated_flows"

    processed_root = project_root / "data" / "processed"
    ml_output = processed_root / "machine_learning"
    generated_output = processed_root / "generated_context"

    reports_root = project_root / "reports"
    tables_dir = reports_root / "tables"

    ml_files = sorted(ml_input.rglob("*.csv"))
    generated_files = sorted(generated_input.rglob("*.csv"))

    print("AI NIDS preprocessing")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Machine-learning CSVs: {len(ml_files)}")
    print(f"Generated-flow CSVs: {len(generated_files)}")
    print()

    if len(ml_files) != 8:
        print(
            f"ERROR: Expected 8 machine-learning CSVs, found {len(ml_files)}.",
            file=sys.stderr,
        )
        return 2

    if not args.skip_generated_context and len(generated_files) != 8:
        print(
            f"ERROR: Expected 8 generated-flow CSVs, found {len(generated_files)}.",
            file=sys.stderr,
        )
        return 3

    # Clear only generated outputs from prior runs. Raw data is never touched.
    for directory in (ml_output, generated_output):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    summaries: list[ProcessingSummary] = []
    expected_features: list[str] | None = None
    preview_frames: list[pd.DataFrame] = []

    print("[1/3] Processing machine-learning feature files...")
    for index, source_path in enumerate(ml_files, start=1):
        output_path = ml_output / f"{safe_stem(source_path.name)}.parquet"
        print(f"  [{index}/{len(ml_files)}] {source_path.name}")

        summary, features = process_machine_learning_file(
            source_path=source_path,
            output_path=output_path,
            chunk_size=max(1_000, args.chunk_size),
            expected_features=expected_features,
            preview_frames=preview_frames,
        )

        if expected_features is None:
            expected_features = features

        summaries.append(summary)

    if expected_features is None:
        print("ERROR: No feature schema was produced.", file=sys.stderr)
        return 4

    print("[2/3] Processing generated-flow dashboard context...")
    if args.skip_generated_context:
        print("  Skipped by command-line option.")
    else:
        for index, source_path in enumerate(generated_files, start=1):
            output_path = generated_output / f"{safe_stem(source_path.name)}.parquet"
            print(f"  [{index}/{len(generated_files)}] {source_path.name}")

            summaries.append(
                process_generated_context_file(
                    source_path=source_path,
                    output_path=output_path,
                    chunk_size=max(1_000, args.chunk_size),
                )
            )

    print("[3/3] Writing manifests and verification reports...")
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [asdict(item) for item in summaries]
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "preprocessing_file_summary.csv",
        index=False,
        encoding="utf-8",
    )

    feature_manifest = pd.DataFrame(
        {
            "position": np.arange(1, len(expected_features) + 1),
            "feature": expected_features,
            "dtype": "float32",
        }
    )
    feature_manifest.to_csv(
        tables_dir / "feature_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    pair_rows = pair_summaries(summaries)
    pd.DataFrame(pair_rows).to_csv(
        tables_dir / "preprocessing_alignment_check.csv",
        index=False,
        encoding="utf-8",
    )

    if preview_frames:
        preview = pd.concat(preview_frames, ignore_index=True)
        preview.to_csv(
            processed_root / "preprocessing_preview.csv",
            index=False,
            encoding="utf-8",
        )

    ml_summaries = [
        item for item in summaries if item.collection == "machine_learning"
    ]
    generated_summaries = [
        item for item in summaries if item.collection == "generated_flows"
    ]

    all_alignment_matches = bool(pair_rows) and all(
        row["row_counts_match"] and row["alignment_hash_match"]
        for row in pair_rows
    )

    report = {
        "project_root": str(project_root),
        "machine_learning_files_processed": len(ml_summaries),
        "generated_flow_files_processed": len(generated_summaries),
        "machine_learning_valid_rows": sum(
            item.valid_rows for item in ml_summaries
        ),
        "generated_context_valid_rows": sum(
            item.valid_rows for item in generated_summaries
        ),
        "generated_empty_rows_dropped": sum(
            item.empty_rows_dropped for item in generated_summaries
        ),
        "feature_count_after_duplicate_removal": len(expected_features),
        "duplicate_feature_columns_removed": sorted(
            {
                column
                for item in ml_summaries
                for column in item.duplicate_columns_dropped.split("; ")
                if column
            }
        ),
        "infinity_values_replaced_with_nan": sum(
            item.infinity_values_replaced for item in ml_summaries
        ),
        "non_numeric_feature_values_coerced_to_nan": sum(
            item.non_numeric_values_coerced for item in ml_summaries
        ),
        "missing_feature_values_preserved_for_train_only_imputation": sum(
            item.missing_feature_values_after_cleaning for item in ml_summaries
        ),
        "all_collection_pairs_have_matching_rows_and_alignment_hashes": (
            all_alignment_matches
        ),
        "targets_created": {
            "Label": "normalized original multiclass label",
            "BinaryLabel": "0=BENIGN, 1=ATTACK",
            "AttackFamily": sorted(set(ATTACK_FAMILY_MAP.values())),
        },
        "important_methodology_note": (
            "Missing values were not globally imputed. A SimpleImputer will be "
            "fit only on training data inside the later scikit-learn pipeline "
            "to avoid test-data leakage."
        ),
    }

    with (reports_root / "preprocessing_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)

    text = [
        "AI-BASED NETWORK INTRUSION DETECTION SYSTEM",
        "PREPROCESSING SUMMARY",
        "=" * 72,
        "",
        f"Machine-learning files processed: {len(ml_summaries)}",
        f"Machine-learning valid rows: {report['machine_learning_valid_rows']:,}",
        f"Generated-flow files processed: {len(generated_summaries)}",
        f"Generated context valid rows: {report['generated_context_valid_rows']:,}",
        f"Generated empty rows dropped: {report['generated_empty_rows_dropped']:,}",
        "",
        f"Predictive features after duplicate removal: {len(expected_features)}",
        "Duplicate feature columns removed: "
        + (
            ", ".join(report["duplicate_feature_columns_removed"])
            if report["duplicate_feature_columns_removed"]
            else "None"
        ),
        f"Infinity values replaced with NaN: "
        f"{report['infinity_values_replaced_with_nan']:,}",
        f"Non-numeric feature values coerced to NaN: "
        f"{report['non_numeric_feature_values_coerced_to_nan']:,}",
        f"Missing feature values retained for train-only imputation: "
        f"{report['missing_feature_values_preserved_for_train_only_imputation']:,}",
        "",
        "Targets created:",
        "  Label: normalized 15-class target",
        "  BinaryLabel: 0=BENIGN, 1=ATTACK",
        "  AttackFamily: operational family target",
        "",
        "Machine-learning/generated-flow alignment:",
        f"  All row counts and shared-column hashes match: "
        f"{all_alignment_matches}",
        "",
        "Important:",
        "  The raw CSV files were not modified.",
        "  Missing values remain missing until the training pipeline fits an",
        "  imputer using training data only.",
        "",
        "Output locations:",
        f"  {ml_output}",
        f"  {generated_output}",
        f"  {tables_dir / 'preprocessing_file_summary.csv'}",
        f"  {tables_dir / 'feature_manifest.csv'}",
        f"  {tables_dir / 'preprocessing_alignment_check.csv'}",
        f"  {reports_root / 'preprocessing_summary.json'}",
        "",
        "Preprocessing completed successfully.",
    ]

    (reports_root / "preprocessing_summary.txt").write_text(
        "\n".join(text),
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Preprocessing finished.")
    print(f"Summary: {reports_root / 'preprocessing_summary.txt'}")
    print(f"Feature manifest: {tables_dir / 'feature_manifest.csv'}")
    print(f"Alignment check: {tables_dir / 'preprocessing_alignment_check.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
