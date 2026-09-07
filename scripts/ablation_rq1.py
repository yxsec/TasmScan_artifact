#!/usr/bin/env python3
"""
RQ1 Ablation Study: measure per-edge and per-contract resolution rates
for each incremental technique configuration.

Configurations (cumulative, matching Table tab:rq1):
  Stage 0: Baseline (intracontinuation simulation + known-prefix fallback)
  Stage 1: + intercontinuation propagation
  Stage 2: + reachability pruning
  Stage 3: + savelist propagation (full)
"""

import json
import argparse
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasmscan.disassembler import parse_boc, decompile_cell
from tasmscan.boc_utils import detect_and_decode_boc
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.ir.ir_builder import IRBuilder

BENCHMARK_DIR = Path(os.environ.get("TASMSCAN_BENCHMARK_DIR", "ton_benchmark/sources"))

# Ablation stages: each dict contains the flags that differ from all-disabled baseline.
# Flags are cumulative — each stage adds one technique on top of the previous stage.
STAGES = [
    {
        "name": "Baseline (intracontinuation simulation + known-prefix)",
        "flags": {
            "enable_known_prefix_fallback": True,
            "enable_entry_shape_propagation": False,
            "enable_ctrl_reg_propagation": False,
            "enable_varargs_depth_fix": False,
            "enable_reachability_pruning": False,
        },
    },
    {
        "name": "+ intercontinuation propagation",
        "flags": {
            "enable_known_prefix_fallback": True,
            "enable_entry_shape_propagation": True,
            "enable_ctrl_reg_propagation": True,
            "enable_varargs_depth_fix": True,
            "enable_reachability_pruning": False,
        },
    },
    {
        "name": "+ reachability pruning",
        "flags": {
            "enable_known_prefix_fallback": True,
            "enable_entry_shape_propagation": True,
            "enable_ctrl_reg_propagation": True,
            "enable_varargs_depth_fix": True,
            "enable_reachability_pruning": True,
        },
    },
    {
        "name": "+ savelist propagation (full)",
        "flags": {
            "enable_known_prefix_fallback": True,
            "enable_entry_shape_propagation": True,
            "enable_ctrl_reg_propagation": True,
            "enable_varargs_depth_fix": True,
            "enable_reachability_pruning": True,
        },
        "savelist": True,
    },
]


def _analyze_one(args: Tuple[str, Dict[str, bool]]) -> Dict:
    """Analyze a single contract with the given ablation flags."""
    boc_path, flags = args
    try:
        text_data = Path(boc_path).read_text("utf-8").strip()
        boc_bytes = detect_and_decode_boc(text_data)
        cell = parse_boc(boc_bytes)
        instructions = decompile_cell(cell)

        analyzer = ProgramAnalyzer()
        analysis_flags = {k: v for k, v in flags.items() if k != "savelist"}
        facts = analyzer.analyze(instructions, cell, **analysis_flags)

        # Savelist propagation is a TASIR/linking phase rather than a
        # ProgramAnalyzer flag.  Build the module here so the final stage
        # exercises the same implementation used by the scanner.
        savelist_enabled = bool(flags.get("savelist", False))
        tasir_module = IRBuilder().build_tasir(
            facts, skip_savelists=not savelist_enabled
        )

        total_edges = 0
        truly_unresolved = 0
        for edge in facts.cfg_edges:
            total_edges += 1
            if edge.target is None and edge.kind in ("branch", "call", "jump"):
                truly_unresolved += 1

        return {
            "ok": True,
            "total_edges": total_edges,
            "truly_unresolved": truly_unresolved,
            "savelist_enabled": savelist_enabled,
            "savelist_registers": sum(
                len(desc.savelist.registers)
                for desc in tasir_module.cross_function_continuations.values()
            ),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_stage(stage_idx: int, contracts: List[str], workers: int) -> Dict:
    """Run one ablation stage across all contracts."""
    stage = STAGES[stage_idx]
    flags = stage["flags"]

    args_list = [(c, flags) for c in contracts]
    results = []
    with multiprocessing.Pool(workers) as pool:
        for r in pool.imap_unordered(_analyze_one, args_list, chunksize=16):
            results.append(r)

    valid = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]

    total_edges = sum(r["total_edges"] for r in valid)
    truly_unresolved = sum(r["truly_unresolved"] for r in valid)
    contracts_with_unresolved = sum(1 for r in valid if r["truly_unresolved"] > 0)
    contracts_fully_resolved = sum(1 for r in valid if r["truly_unresolved"] == 0)

    return {
        "stage": stage_idx,
        "name": stage["name"],
        "contracts_analyzed": len(valid),
        "errors": len(errors),
        "total_edges": total_edges,
        "truly_unresolved": truly_unresolved,
        "edge_resolution_rate": round(100 * (total_edges - truly_unresolved) / total_edges, 4) if total_edges else 0,
        "contracts_fully_resolved": contracts_fully_resolved,
        "contracts_with_unresolved": contracts_with_unresolved,
        "contract_resolution_rate": round(100 * contracts_fully_resolved / len(valid), 2) if valid else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the RQ1 continuation-resolution ablation.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Directory containing bytecode.boc.b64 files (not bundled).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TASMSCAN_RESULTS_DIR", "/tmp/tasmscan-results")),
        help="External directory for the JSON report.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    print("Searching for contracts...")
    contracts = []
    for root, dirs, files in os.walk(str(args.benchmark_dir)):
        for f in files:
            if f == "bytecode.boc.b64":
                contracts.append(os.path.join(root, f))
    contracts.sort()
    print(f"Found {len(contracts)} contracts\n")

    workers = max(1, args.workers)
    all_results = []

    for stage_idx in range(len(STAGES)):
        stage_name = STAGES[stage_idx]["name"]
        print(f"Stage {stage_idx}: {stage_name} ...")
        t0 = time.time()
        result = run_stage(stage_idx, contracts, workers)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        all_results.append(result)
        print(f"  Done in {elapsed:.1f}s — "
              f"unresolved edges: {result['truly_unresolved']:,} / {result['total_edges']:,}, "
              f"contracts affected: {result['contracts_with_unresolved']:,} / {result['contracts_analyzed']:,}")

    # Print summary table
    print("\n" + "=" * 100)
    print("RQ1 ABLATION RESULTS")
    print("=" * 100)
    print(f"{'Stage':<50} {'Unresolved':>12} {'Edge Rate':>10} {'Contracts':>16} {'Ctr Rate':>10}")
    print(f"{'':50} {'Edges':>12} {'(%)':>10} {'Affected':>16} {'(%)':>10}")
    print("-" * 100)

    prev_unresolved = None
    for r in all_results:
        delta = ""
        if prev_unresolved is not None and prev_unresolved > 0:
            reduction = (prev_unresolved - r["truly_unresolved"]) / prev_unresolved * 100
            delta = f" (-{reduction:.1f}%)"
        elif prev_unresolved == 0:
            delta = ""

        edge_rate = f"{r['edge_resolution_rate']:.4f}%" if r['total_edges'] else "N/A"
        ctr_rate = f"{r['contract_resolution_rate']:.2f}%"
        ctr_affected = f"{r['contracts_with_unresolved']:,} / {r['contracts_analyzed']:,}"

        print(f"{r['name']:<50} {r['truly_unresolved']:>12,}{delta:>0} {edge_rate:>10} {ctr_affected:>16} {ctr_rate:>10}")
        prev_unresolved = r["truly_unresolved"]

    print("=" * 100)

    # Save JSON
    output_path = args.output_dir / "ablation_rq1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
