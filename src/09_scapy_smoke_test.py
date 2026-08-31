from __future__ import annotations

import argparse
from pathlib import Path

from scapy.all import PcapWriter, get_if_list, sniff


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a small number of packets and save them to a PCAP file."
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="Display the capture interfaces available to Scapy and exit.",
    )
    parser.add_argument(
        "--iface",
        default=None,
        help="Exact interface name to capture from. The default lets Scapy choose.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of packets to capture. Default: 20.",
    )
    parser.add_argument(
        "--output",
        default="reports/advanced_layer/captures/scapy_smoke_test.pcap",
        help="PCAP output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    interfaces = get_if_list()

    if args.list_interfaces:
        print("Interfaces available to Scapy:")
        for interface in interfaces:
            print(f"  - {interface}")
        return

    if args.count < 1:
        raise SystemExit("--count must be at least 1.")

    if args.iface and args.iface not in interfaces:
        print(f"Interface not found: {args.iface}")
        print("\nAvailable interfaces:")
        for interface in interfaces:
            print(f"  - {interface}")
        raise SystemExit(2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PcapWriter(
        str(output_path),
        append=False,
        sync=True,
    )

    captured = 0

    def process_packet(packet) -> None:
        nonlocal captured
        captured += 1
        writer.write(packet)
        print(f"[{captured:03d}] {packet.summary()}")

    print("Starting the Scapy capture test.")
    print(f"Interface: {args.iface or 'Scapy default interface'}")
    print(f"Packet count: {args.count}")
    print("Open a normal webpage or generate ordinary network traffic.")
    print("Press Ctrl+C to stop early.\n")

    sniff_arguments = {
        "prn": process_packet,
        "count": args.count,
        "store": False,
    }

    if args.iface:
        sniff_arguments["iface"] = args.iface

    try:
        sniff(**sniff_arguments)
    except KeyboardInterrupt:
        print("\nCapture stopped by the user.")
    finally:
        writer.close()

    print()
    print(f"Packets captured: {captured}")
    print(f"PCAP saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
