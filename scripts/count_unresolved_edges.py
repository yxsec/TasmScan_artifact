#!/usr/bin/env python3
"""Quick script to count unresolved CFG edges across N contracts."""

import sys
import base64
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasmscan.boc_utils import load_boc_file
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.disassembler import decompile_cell
from tasmscan.disassembler import parse_boc

BENCHMARK_DIR = Path(os.environ.get("TASMSCAN_BENCHMARK_DIR", "ton_benchmark/sources"))


def find_contracts(max_n: int = 100):
    contracts = []
    network_dir = BENCHMARK_DIR / "mainnet"
    for contract_dir in sorted(network_dir.iterdir()):
        if not contract_dir.is_dir():
            continue
        bytecode_file = contract_dir / "bytecode.boc.b64"
        if bytecode_file.exists():
            contracts.append(bytecode_file)
        if len(contracts) >= max_n:
            break
    return contracts


def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    contracts = find_contracts(max_n)
    print(f"Scanning {len(contracts)} contracts...")

    analyzer = ProgramAnalyzer()

    total_edges = Counter()       # kind -> count
    unresolved_edges = Counter()  # kind -> count with target=None
    unresolved_opcodes = Counter()
    total_contracts = 0
    errors = 0

    for i, path in enumerate(contracts):
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(contracts)}]...")
        try:
            boc_data = load_boc_file(path)
            cell = parse_boc(boc_data)
            instructions = decompile_cell(cell)
            facts = analyzer.analyze(instructions, cell)

            for edge in facts.cfg_edges:
                total_edges[edge.kind] += 1
                if edge.target is None:
                    unresolved_edges[edge.kind] += 1
                    src_idx = edge.source
                    if src_idx is not None and src_idx < len(facts.instructions):
                        opcode = facts.instructions[src_idx].instruction.opcode
                        unresolved_opcodes[opcode] += 1

            total_contracts += 1
        except Exception as e:
            errors += 1

    print(f"\n{'='*70}")
    print(f"UNRESOLVED EDGE STATISTICS ({total_contracts} contracts, {errors} errors)")
    print(f"{'='*70}")

    print(f"\n{'Kind':<15} {'Unresolved':>12} {'Total':>12} {'%':>8}")
    print("-" * 50)
    all_kinds = sorted(set(list(total_edges.keys()) + list(unresolved_edges.keys())))
    grand_total = 0
    grand_unresolved = 0
    for kind in all_kinds:
        t = total_edges[kind]
        u = unresolved_edges[kind]
        pct = 100 * u / t if t > 0 else 0
        print(f"{kind:<15} {u:>12,} {t:>12,} {pct:>7.1f}%")
        grand_total += t
        grand_unresolved += u

    print("-" * 50)
    pct = 100 * grand_unresolved / grand_total if grand_total > 0 else 0
    print(f"{'TOTAL':<15} {grand_unresolved:>12,} {grand_total:>12,} {pct:>7.1f}%")

    if unresolved_opcodes:
        print(f"\nTop unresolved opcodes:")
        for opcode, count in unresolved_opcodes.most_common(15):
            print(f"  {opcode:<25} {count:>8,}")


if __name__ == "__main__":
    main()
