#!/usr/bin/env python3
"""Validate CFG construction correctness across benchmark contracts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tasmscan.analyzer.constants import (
    CALL_OPCODES,
    MAIN_CONTEXT,
    TERMINATORS,
)
from tasmscan.boc_utils import detect_and_decode_boc
from tasmscan.disassembler import decompile_cell, parse_boc
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


ALLOWED_EDGE_KINDS = {
    "fallthrough",
    "branch",
    "jump",
    "call",
    "call_cont",
    "return_cont",
    "call_return",
    "guard_throw",
    "guard_return",
}

UNRESOLVED_EDGE_KINDS = {"branch", "jump", "call"}
DESIGN_NULL_EDGE_KINDS = {"guard_throw", "guard_return"}


def _iter_boc_files(benchmark_root: Path) -> List[Path]:
    return sorted(benchmark_root.glob("**/bytecode.boc.b64"))


def _load_metadata(contract_dir: Path) -> Dict[str, Any]:
    metadata_path = contract_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        obj = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _context_of_fact(fact) -> str:
    return fact.continuation_id if fact.continuation_id is not None else MAIN_CONTEXT


def _next_index_map_by_context(instructions: Sequence[Any]) -> Dict[int, int]:
    by_ctx: Dict[str, List[int]] = defaultdict(list)
    for fact in instructions:
        by_ctx[_context_of_fact(fact)].append(fact.index)
    for indices in by_ctx.values():
        indices.sort()

    out: Dict[int, int] = {}
    for indices in by_ctx.values():
        for i in range(len(indices) - 1):
            out[indices[i]] = indices[i + 1]
    return out


def _validate_facts_cfg(facts) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def add_error(code: str, **data: Any) -> None:
        errors.append({"code": code, **data})

    def add_warning(code: str, **data: Any) -> None:
        warnings.append({"code": code, **data})

    instr_by_index: Dict[int, Any] = {f.index: f for f in facts.instructions}
    next_idx_by_ctx = _next_index_map_by_context(facts.instructions)

    seen_edges: Set[Tuple[int, Optional[int], str]] = set()
    duplicate_edges = 0

    unresolved_sources: Set[int] = set()

    for edge in facts.cfg_edges:
        if edge.kind not in ALLOWED_EDGE_KINDS:
            add_error("edge_kind_invalid", kind=edge.kind, source=edge.source, target=edge.target)

        if edge.source not in instr_by_index:
            add_error("edge_source_missing_instruction", source=edge.source, kind=edge.kind)
            continue

        src_fact = instr_by_index[edge.source]
        src_ctx = _context_of_fact(src_fact)
        src_opcode = src_fact.opcode.upper()

        if edge.target is not None:
            if edge.target not in instr_by_index:
                add_error(
                    "edge_target_missing_instruction",
                    source=edge.source,
                    target=edge.target,
                    kind=edge.kind,
                )
                continue

            dst_fact = instr_by_index[edge.target]
            dst_ctx = _context_of_fact(dst_fact)

            if edge.kind == "fallthrough":
                expected = next_idx_by_ctx.get(edge.source)
                if expected != edge.target:
                    add_error(
                        "fallthrough_not_next_instruction",
                        source=edge.source,
                        target=edge.target,
                        expected_target=expected,
                    )
                if src_ctx != dst_ctx:
                    add_error(
                        "fallthrough_cross_context",
                        source=edge.source,
                        target=edge.target,
                        source_context=src_ctx,
                        target_context=dst_ctx,
                    )
                if src_opcode in TERMINATORS:
                    add_error("terminator_has_fallthrough", source=edge.source, opcode=src_opcode)

            if edge.kind == "call_return":
                expected = next_idx_by_ctx.get(edge.source)
                if expected != edge.target:
                    add_error(
                        "call_return_not_next_instruction",
                        source=edge.source,
                        target=edge.target,
                        expected_target=expected,
                    )
                if src_opcode not in CALL_OPCODES:
                    add_error(
                        "call_return_source_not_call_opcode",
                        source=edge.source,
                        opcode=src_opcode,
                    )
                if src_ctx != dst_ctx:
                    add_error(
                        "call_return_cross_context",
                        source=edge.source,
                        target=edge.target,
                        source_context=src_ctx,
                        target_context=dst_ctx,
                    )

            if edge.kind in DESIGN_NULL_EDGE_KINDS:
                add_error("design_null_edge_has_target", kind=edge.kind, source=edge.source, target=edge.target)

        else:
            if edge.kind in UNRESOLVED_EDGE_KINDS:
                unresolved_sources.add(edge.source)
            elif edge.kind in DESIGN_NULL_EDGE_KINDS:
                pass
            else:
                add_error("non_nullable_edge_has_null_target", kind=edge.kind, source=edge.source)

        key = (edge.source, edge.target, edge.kind)
        if key in seen_edges:
            duplicate_edges += 1
        seen_edges.add(key)

    # Check that terminators do not have fallthrough edges.
    fallthrough_sources = {e.source for e in facts.cfg_edges if e.kind == "fallthrough"}
    for idx, fact in instr_by_index.items():
        if fact.opcode.upper() in TERMINATORS and idx in fallthrough_sources:
            add_error("terminator_has_fallthrough", source=idx, opcode=fact.opcode.upper())

    # Validate basic-block mapping and successors.
    block_by_id = {b.id: b for b in facts.basic_blocks}
    instr_to_block: Dict[int, int] = {}
    for block in facts.basic_blocks:
        for idx in block.instruction_indices:
            if idx in instr_to_block:
                add_error(
                    "instruction_in_multiple_blocks",
                    instruction_index=idx,
                    block_id=block.id,
                    previous_block_id=instr_to_block[idx],
                )
            instr_to_block[idx] = block.id
            fact = instr_by_index.get(idx)
            if fact is None:
                add_error("block_contains_unknown_instruction", block_id=block.id, instruction_index=idx)
                continue
            expected_context = _context_of_fact(fact)
            if block.context != expected_context:
                add_error(
                    "block_context_mismatch",
                    block_id=block.id,
                    block_context=block.context,
                    instruction_index=idx,
                    instruction_context=expected_context,
                )

    expected_successors: Dict[int, Set[int]] = defaultdict(set)
    expected_unknown_successor_blocks: Set[int] = set()
    for edge in facts.cfg_edges:
        src_block = instr_to_block.get(edge.source)
        if src_block is None:
            continue
        if edge.target is None:
            if edge.kind in UNRESOLVED_EDGE_KINDS:
                expected_unknown_successor_blocks.add(src_block)
            continue
        dst_block = instr_to_block.get(edge.target)
        if dst_block is None:
            continue
        # CFG builder includes self-loop block successors only for branch/jump edges.
        if src_block != dst_block:
            expected_successors[src_block].add(dst_block)
        elif edge.kind in {"branch", "jump"}:
            expected_successors[src_block].add(dst_block)

    for block in facts.basic_blocks:
        actual = set(block.successors)
        expected = expected_successors.get(block.id, set())
        missing = expected - actual
        extra = actual - expected
        if missing:
            add_error(
                "block_missing_successor",
                block_id=block.id,
                missing_successors=sorted(missing),
            )
        if extra:
            add_error(
                "block_extra_successor",
                block_id=block.id,
                extra_successors=sorted(extra),
            )

        should_have_unknown = block.id in expected_unknown_successor_blocks
        if should_have_unknown and not block.has_unknown_successor:
            add_error("block_missing_unknown_successor_flag", block_id=block.id)
        if (not should_have_unknown) and block.has_unknown_successor:
            add_warning("block_unknown_successor_flag_without_unresolved_edge", block_id=block.id)

    # Entry point block IDs must exist.
    valid_block_ids = set(block_by_id.keys())
    for ep in facts.entry_points:
        block_id = ep.get("block_id")
        if block_id not in valid_block_ids:
            add_error("entry_point_block_missing", entry_point=ep)

    truly_unresolved = sum(
        1 for edge in facts.cfg_edges if edge.target is None and edge.kind in UNRESOLVED_EDGE_KINDS
    )
    design_null = sum(
        1 for edge in facts.cfg_edges if edge.target is None and edge.kind in DESIGN_NULL_EDGE_KINDS
    )

    return {
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
        "duplicate_edges": duplicate_edges,
        "truly_unresolved_edges": truly_unresolved,
        "design_null_edges": design_null,
        "unresolved_source_count": len(unresolved_sources),
    }


def _analyze_contract(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "contract_name": path.parent.name,
        "error": None,
        "cfg_status": None,
        "instruction_count": 0,
        "edge_count": 0,
        "block_count": 0,
        "entry_point_count": 0,
        "truly_unresolved_edges": 0,
        "design_null_edges": 0,
        "validator_error_count": 0,
        "validator_warning_count": 0,
        "validator_errors": [],
        "validator_warnings": [],
        "validator_duplicate_edges": 0,
        "contract_address": None,
        "network": None,
    }

    metadata = _load_metadata(path.parent)
    result["contract_address"] = metadata.get("contract_address")
    result["network"] = metadata.get("network")

    try:
        text = path.read_text(encoding="utf-8").strip()
        boc = detect_and_decode_boc(text)
        cell = parse_boc(boc)
        instructions = decompile_cell(cell)
        facts = ProgramAnalyzer().analyze(instructions, cell)

        result["instruction_count"] = len(facts.instructions)
        result["edge_count"] = len(facts.cfg_edges)
        result["block_count"] = len(facts.basic_blocks)
        result["entry_point_count"] = len(facts.entry_points)

        validation = _validate_facts_cfg(facts)
        result["truly_unresolved_edges"] = validation["truly_unresolved_edges"]
        result["design_null_edges"] = validation["design_null_edges"]
        result["validator_error_count"] = validation["error_count"]
        result["validator_warning_count"] = validation["warning_count"]
        result["validator_errors"] = validation["errors"]
        result["validator_warnings"] = validation["warnings"]
        result["validator_duplicate_edges"] = validation["duplicate_edges"]

        if validation["error_count"] > 0:
            result["cfg_status"] = "invalid"
        elif validation["truly_unresolved_edges"] > 0:
            result["cfg_status"] = "valid_with_unresolved"
        else:
            result["cfg_status"] = "valid"

    except Exception as exc:  # pragma: no cover - benchmark runtime surface
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["cfg_status"] = "analysis_failed"

    return result


def _aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [r for r in results if not r.get("error")]
    failed_rows = [r for r in results if r.get("error")]

    status_counter = Counter(r.get("cfg_status") for r in valid_rows)
    network_counter = Counter((r.get("network") or "unknown") for r in valid_rows)
    error_code_counter = Counter()
    warning_code_counter = Counter()

    total_truly_unresolved = 0
    total_design_null = 0
    total_validator_errors = 0
    total_validator_warnings = 0

    for row in valid_rows:
        total_truly_unresolved += int(row.get("truly_unresolved_edges", 0) or 0)
        total_design_null += int(row.get("design_null_edges", 0) or 0)
        total_validator_errors += int(row.get("validator_error_count", 0) or 0)
        total_validator_warnings += int(row.get("validator_warning_count", 0) or 0)
        for err in row.get("validator_errors", []):
            if isinstance(err, dict):
                code = err.get("code")
                if code:
                    error_code_counter[str(code)] += 1
        for warn in row.get("validator_warnings", []):
            if isinstance(warn, dict):
                code = warn.get("code")
                if code:
                    warning_code_counter[str(code)] += 1

    top_invalid = sorted(
        (r for r in valid_rows if r.get("cfg_status") == "invalid"),
        key=lambda r: int(r.get("validator_error_count", 0) or 0),
        reverse=True,
    )[:30]
    top_unresolved = sorted(
        (r for r in valid_rows if int(r.get("truly_unresolved_edges", 0) or 0) > 0),
        key=lambda r: int(r.get("truly_unresolved_edges", 0) or 0),
        reverse=True,
    )[:50]

    return {
        "summary": {
            "total_contracts": len(results),
            "analysis_failed": len(failed_rows),
            "analyzed_contracts": len(valid_rows),
            "status_distribution": dict(status_counter),
            "network_distribution": dict(network_counter),
            "total_truly_unresolved_edges": total_truly_unresolved,
            "total_design_null_edges": total_design_null,
            "total_validator_errors": total_validator_errors,
            "total_validator_warnings": total_validator_warnings,
            "top_error_codes": dict(error_code_counter.most_common(20)),
            "top_warning_codes": dict(warning_code_counter.most_common(20)),
        },
        "top_invalid_contracts": top_invalid,
        "top_unresolved_contracts": top_unresolved,
        "contracts": list(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CFG correctness across benchmark contracts.")
    parser.add_argument(
        "--benchmark-root",
        default="ton_benchmark/sources",
        help="Benchmark root directory containing bytecode.boc.b64 files.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker processes.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max contracts to analyze (0 = all).",
    )
    parser.add_argument(
        "--output",
        default="tmp/cfg_correctness_report.full.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    boc_files = _iter_boc_files(benchmark_root)
    if args.limit and args.limit > 0:
        boc_files = boc_files[: args.limit]

    print(f"[cfg-validate] benchmark_root={benchmark_root}")
    print(f"[cfg-validate] contracts={len(boc_files)} workers={args.workers}")

    results: List[Dict[str, Any]] = []
    with multiprocessing.Pool(processes=max(1, int(args.workers))) as pool:
        done = 0
        for row in pool.imap_unordered(_analyze_contract, boc_files, chunksize=16):
            results.append(row)
            done += 1
            if done % 200 == 0 or done == len(boc_files):
                print(f"[cfg-validate] progress {done}/{len(boc_files)}")

    report = _aggregate(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    print(f"[cfg-validate] report={out_path}")
    print(
        "[cfg-validate] analyzed="
        f"{summary['analyzed_contracts']} failed={summary['analysis_failed']} "
        f"valid={summary['status_distribution'].get('valid', 0)} "
        f"valid_with_unresolved={summary['status_distribution'].get('valid_with_unresolved', 0)} "
        f"invalid={summary['status_distribution'].get('invalid', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
