#!/usr/bin/env python3
r"""
Create a compact, aligned demonstration CSV for the Streamlit dashboard.

Run from the project root:
    python .\src\08_create_dashboard_demo.py

The script selects a mixture of benign and attack rows from every scenario,
preserves all 77 trained features, and adds generated-flow context such as
Flow ID, IP addresses, ports, protocol, and timestamp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RANDOM_STATE = 42

FEATURE_METADATA = [
    "RowID",
    "SourceFile",
    "Scenario",
    "Label",
    "AttackFamily",
    "BinaryLabel",
]

CONTEXT_COLUMNS = [
    "RowID",
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]


def scenario_from_parquet(path: Path) -> str:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=["Scenario"])
    return str(table.column("Scenario")[0].as_py())


def read_positions(
    path: Path,
    columns: list[str],
    positions: np.ndarray,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """
    Read only selected row positions without loading the complete Parquet file.
    """
    parquet_file = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    offset = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        rows = batch.num_rows
        lower = np.searchsorted(
            positions,
            offset,
            side="left",
        )
        upper = np.searchsorted(
            positions,
            offset + rows,
            side="left",
        )

        if upper > lower:
            local_positions = (
                positions[lower:upper] - offset
            )
            frame = batch.to_pandas().iloc[
                local_positions
            ]
            frames.append(frame)

        offset += rows

        if offset > positions[-1]:
            break

    if not frames:
        raise RuntimeError(
            f"No selected rows could be read from {path}"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def choose_positions(
    labels: np.ndarray,
    rows_per_scenario: int,
    rng: np.random.Generator,
) -> np.ndarray:
    benign = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)

    attack_target = min(
        len(attack),
        rows_per_scenario // 2,
    )
    benign_target = min(
        len(benign),
        rows_per_scenario - attack_target,
    )

    remaining = rows_per_scenario - (
        attack_target + benign_target
    )

    if remaining > 0:
        extra_attack = min(
            remaining,
            len(attack) - attack_target,
        )
        attack_target += extra_attack
        remaining -= extra_attack

    if remaining > 0:
        benign_target += min(
            remaining,
            len(benign) - benign_target,
        )

    selected = []

    if benign_target:
        selected.append(
            rng.choice(
                benign,
                size=benign_target,
                replace=False,
            )
        )

    if attack_target:
        selected.append(
            rng.choice(
                attack,
                size=attack_target,
                replace=False,
            )
        )

    return np.sort(np.concatenate(selected))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a compact Streamlit demonstration CSV."
    )
    parser.add_argument(
        "--rows-per-scenario",
        type=int,
        default=250,
        help="Rows retained from each scenario. Default: 250.",
    )
    parser.add_argument(
        "--output",
        default="data/dashboard_demo.csv",
        help="Output CSV path relative to the project root.",
    )
    args = parser.parse_args()

    if args.rows_per_scenario < 20:
        raise SystemExit(
            "--rows-per-scenario must be at least 20."
        )

    project_root = Path(__file__).resolve().parents[1]
    feature_dir = (
        project_root / "data" / "processed" / "machine_learning"
    )
    context_dir = (
        project_root / "data" / "processed" / "generated_context"
    )
    feature_manifest = (
        project_root
        / "reports"
        / "tables"
        / "feature_manifest.csv"
    )

    features = (
        pd.read_csv(feature_manifest)["feature"]
        .astype(str)
        .tolist()
    )

    feature_files = {
        scenario_from_parquet(path): path
        for path in feature_dir.glob("*.parquet")
    }
    context_files = {
        scenario_from_parquet(path): path
        for path in context_dir.glob("*.parquet")
    }

    if set(feature_files) != set(context_files):
        raise RuntimeError(
            "Feature and context scenario sets do not match."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    output_frames: list[pd.DataFrame] = []

    print("Creating aligned Streamlit demonstration CSV...")

    for index, scenario in enumerate(
        sorted(feature_files),
        start=1,
    ):
        feature_path = feature_files[scenario]
        context_path = context_files[scenario]

        labels = pd.read_parquet(
            feature_path,
            columns=["BinaryLabel"],
            engine="pyarrow",
        )["BinaryLabel"].astype("int8").to_numpy()

        positions = choose_positions(
            labels,
            rows_per_scenario=args.rows_per_scenario,
            rng=rng,
        )

        feature_frame = read_positions(
            feature_path,
            FEATURE_METADATA + features,
            positions,
        )
        context_frame = read_positions(
            context_path,
            CONTEXT_COLUMNS,
            positions,
        )

        if not np.array_equal(
            feature_frame["RowID"].to_numpy(),
            context_frame["RowID"].to_numpy(),
        ):
            raise RuntimeError(
                f"RowID alignment failed for {scenario}."
            )

        context_frame = context_frame.rename(
            columns={
                "Destination Port": "Context Destination Port",
            }
        )

        context_to_add = [
            column
            for column in context_frame.columns
            if column != "RowID"
        ]

        combined = pd.concat(
            [
                feature_frame.reset_index(drop=True),
                context_frame[
                    context_to_add
                ].reset_index(drop=True),
            ],
            axis=1,
        )
        output_frames.append(combined)

        benign_count = int(
            combined["BinaryLabel"].eq(0).sum()
        )
        attack_count = int(
            combined["BinaryLabel"].eq(1).sum()
        )
        print(
            f"  [{index}/8] {scenario}: "
            f"{len(combined):,} rows "
            f"({benign_count:,} benign, {attack_count:,} attack)"
        )

    output = pd.concat(
        output_frames,
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    )

    output_path = project_root / args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print("[COMPLETE] Dashboard demo CSV created.")
    print(f"Rows: {len(output):,}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
