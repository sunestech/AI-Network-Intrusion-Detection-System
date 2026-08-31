#!/usr/bin/env python3
r"""
Run the three-model NIDS inference stack on a large compatible CSV in chunks.

Example:
    python .\src\08_batch_predict.py `
        --input .\data\dashboard_demo.csv `
        --output .\reports\predictions\dashboard_batch_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_inference import (  # noqa: E402
    analyze_dataframe,
    load_model_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run hybrid NIDS inference on a compatible feature CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing all 77 required features.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output prediction-ledger CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows analyzed at a time. Default: 50000.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(
            f"ERROR: Input file does not exist: {input_path}",
            file=sys.stderr,
        )
        return 2

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if output_path.exists():
        output_path.unlink()

    bundle = load_model_bundle(PROJECT_ROOT)

    first_chunk = True
    total_rows = 0
    total_known = 0
    total_anomaly = 0

    for chunk_number, frame in enumerate(
        pd.read_csv(
            input_path,
            chunksize=args.chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        result, quality = analyze_dataframe(
            frame,
            bundle,
        )

        result.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8",
        )

        first_chunk = False
        total_rows += len(result)
        total_known += quality["known_attack_alerts"]
        total_anomaly += quality["anomaly_alerts"]

        print(
            f"Chunk {chunk_number}: "
            f"{len(result):,} rows | "
            f"known attacks={quality['known_attack_alerts']:,} | "
            f"anomalies={quality['anomaly_alerts']:,}"
        )

    print()
    print("[COMPLETE] Batch inference finished.")
    print(f"Rows analyzed: {total_rows:,}")
    print(f"Known attack alerts: {total_known:,}")
    print(f"Anomaly flags: {total_anomaly:,}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
