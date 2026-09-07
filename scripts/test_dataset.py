#!/usr/bin/env python3
"""
Test scanner against external dataset.
Validates that all contracts can be scanned without errors.
"""
import argparse
import base64
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasmscan.scanner import SecurityScanner


def load_bytecode(contract_dir: Path) -> Optional[bytes]:
    """Load bytecode from contract directory."""
    b64_file = contract_dir / "bytecode.boc.b64"
    if b64_file.exists():
        try:
            content = b64_file.read_text().strip()
            return base64.b64decode(content)
        except Exception as e:
            return None
    return None


def scan_contract(scanner: SecurityScanner, boc_data: bytes, contract_id: str) -> Tuple[bool, str, List[str]]:
    """
    Scan a single contract.

    Returns:
        (success, error_message, findings)
    """
    try:
        vulns = scanner.scan_boc_bytes(boc_data)
        findings = [f"[{v.severity}] {v.detector}: {v.message[:80]}" for v in vulns]
        return True, "", findings
    except Exception as e:
        return False, str(e), []


def main():
    parser = argparse.ArgumentParser(description="Test scanner against dataset")
    parser.add_argument("dataset_path", help="Path to dataset sources directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of contracts (0=all)")
    parser.add_argument("--network", choices=["mainnet", "testnet", "all"], default="all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--show-errors", action="store_true", help="Show error details")
    parser.add_argument("--show-findings", action="store_true", help="Show vulnerability findings")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Collect contract directories
    contract_dirs = []
    networks = ["mainnet", "testnet"] if args.network == "all" else [args.network]

    for network in networks:
        network_dir = dataset_path / network
        if network_dir.exists():
            for item in network_dir.iterdir():
                if item.is_dir() and (item / "bytecode.boc.b64").exists():
                    contract_dirs.append((network, item))

    if args.limit > 0:
        contract_dirs = contract_dirs[:args.limit]

    print(f"Found {len(contract_dirs)} contracts to scan")
    print("=" * 60)

    scanner = SecurityScanner()

    # Statistics
    total = len(contract_dirs)
    success = 0
    failed = 0
    error_types: Counter = Counter()
    finding_counts: Counter = Counter()

    start_time = time.time()

    for i, (network, contract_dir) in enumerate(contract_dirs):
        contract_id = f"{network}/{contract_dir.name}"

        if args.verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Progress: {i + 1}/{total} ({rate:.1f} contracts/sec)")

        boc_data = load_bytecode(contract_dir)
        if boc_data is None:
            failed += 1
            error_types["load_error"] += 1
            if args.show_errors:
                print(f"  LOAD ERROR: {contract_id}")
            continue

        ok, error, findings = scan_contract(scanner, boc_data, contract_id)

        if ok:
            success += 1
            for finding in findings:
                finding_counts[finding.split("]")[0] + "]"] += 1
            if args.show_findings and findings:
                print(f"  {contract_id}: {len(findings)} findings")
                for f in findings[:3]:
                    print(f"    {f}")
        else:
            failed += 1
            # Categorize error
            if "Invalid opcode" in error or "Unknown opcode" in error:
                error_types["unknown_opcode"] += 1
            elif "recursion" in error.lower():
                error_types["recursion"] += 1
            elif "timeout" in error.lower():
                error_types["timeout"] += 1
            else:
                error_types["other"] += 1

            if args.show_errors:
                print(f"  SCAN ERROR: {contract_id}")
                print(f"    {error[:200]}")

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Results:")
    print(f"  Total:   {total}")
    print(f"  Success: {success} ({100*success/total:.1f}%)")
    print(f"  Failed:  {failed} ({100*failed/total:.1f}%)")
    print(f"  Time:    {elapsed:.1f}s ({total/elapsed:.1f} contracts/sec)")

    if error_types:
        print(f"\nError breakdown:")
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count}")

    if finding_counts:
        print(f"\nFinding severity distribution:")
        for severity, count in finding_counts.most_common():
            print(f"  {severity}: {count}")

    # Exit with error if any failures
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
