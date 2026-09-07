#!/usr/bin/env python3
"""
Example usage of tasmscan Python API.
"""
from pathlib import Path
import sys

from pytoniq_core import Cell
from tasmscan.disassembler import parse_boc, decompile_cell
from tasmscan.analyzer import ProgramAnalyzer
from tasmscan.detectors import default_detectors


def scan_contract_file(boc_file: str):
    """Scan a BOC file for vulnerabilities."""
    print(f"Scanning: {boc_file}")
    print("=" * 80)

    # Load BOC
    with open(boc_file, 'rb') as f:
        boc_data = f.read()

    cell = parse_boc(boc_data)

    # Disassemble
    instructions = decompile_cell(cell)
    print(f"✓ Disassembled {len(instructions)} instructions")

    # Analyze
    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(instructions, cell)
    print(f"✓ Built analysis facts:")
    print(f"  - Instructions: {len(facts.instructions)}")
    print(f"  - Events: {len(facts.events)}")
    print(f"  - Basic blocks: {len(facts.basic_blocks)}")
    print(f"  - CFG edges: {len(facts.cfg_edges)}")

    # Run detectors
    print(f"\n✓ Running detectors...")
    detectors = default_detectors()
    vulnerabilities = []

    for detector in detectors:
        findings = detector.detect(facts)
        if findings:
            print(f"  - {detector.name}: {len(findings)} findings")
        vulnerabilities.extend(findings)

    # Display results
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {len(vulnerabilities)} vulnerabilities found")
    print("=" * 80)

    if not vulnerabilities:
        print("✅ No vulnerabilities detected!")
        return

    # Group by severity
    by_severity = {}
    for vuln in vulnerabilities:
        by_severity.setdefault(vuln.severity, []).append(vuln)

    for severity in ['critical', 'high', 'medium', 'low']:
        if severity not in by_severity:
            continue

        vulns = by_severity[severity]
        print(f"\n🔴 [{severity.upper()}] ({len(vulns)} issues)")
        print("-" * 80)

        for vuln in vulns:
            print(f"\n  Issue: {vuln.message}")
            print(f"  Detector: {vuln.detector}")
            if vuln.remediation:
                print(f"  Fix: {vuln.remediation}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python example.py <contract.boc>")
        sys.exit(1)

    scan_contract_file(sys.argv[1])
