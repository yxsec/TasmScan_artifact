#!/usr/bin/env python3
"""
Main CLI for tasmscan.

Thin CLI layer over SecurityScanner — all scanning logic lives in scanner.py.
"""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .scanner import SecurityScanner, AnalysisIncompleteError, _filter_detectors
from .detectors import default_detectors
from .detectors.registry import DETECTOR_CLASSES
from .config import AnalysisMode


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='tasmscan',
        description='tasmscan - Independent TVM bytecode security scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tasmscan contract.boc                           # Scan contract
  tasmscan contract.boc --json                    # Output JSON
  tasmscan contract.boc --config rules.json       # Use custom config
  tasmscan contract.boc --stats --verbose         # Show detailed statistics
  tasmscan --list-detectors                       # List all detectors

Note: If filename starts with '-', use:
  tasmscan -- -1_hash.boc                # Use -- separator
  tasmscan ./-1_hash.boc                 # Use relative path
        """
    )

    parser.add_argument('boc_file', nargs='?', help='Path to BOC file')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--config', '-c', metavar='FILE',
                        help='Load detector configuration from file (JSON/YAML)')
    parser.add_argument('--list-detectors', action='store_true',
                        help='List available detectors and exit')
    parser.add_argument('--detectors', help='Comma-separated list of detectors to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output with detailed analysis')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics about the contract')
    parser.add_argument('--path-insensitive', action='store_true',
                        help='Disable path-sensitive dataflow (faster, less precise)')
    parser.add_argument('--analysis-mode', choices=[m.value for m in AnalysisMode],
                        help='Preset analysis mode for path-sensitive dataflow')
    parser.add_argument('--max-paths', type=int,
                        help='Maximum CFG paths to explore (path-sensitive only)')
    parser.add_argument('--max-loop-unroll', type=int,
                        help='Maximum loop unroll depth (path-sensitive only)')
    parser.add_argument('--max-analysis-depth', type=int,
                        help='Maximum analysis depth (path-sensitive only)')
    parser.add_argument('--max-block-visits', type=int,
                        help='Maximum visits per basic block (path-sensitive only)')
    parser.add_argument('--max-worklist-size', type=int,
                        help='Maximum worklist size (path-sensitive only)')
    tasir_group = parser.add_mutually_exclusive_group()
    tasir_group.add_argument(
        '--use-tasir',
        dest='use_tasir',
        action='store_true',
        help='Use TASIR for analysis (default)'
    )
    tasir_group.add_argument(
        '--no-tasir',
        dest='use_tasir',
        action='store_false',
        help='Disable TASIR (use legacy analysis)'
    )
    parser.set_defaults(use_tasir=True)
    solver_ir_group = parser.add_mutually_exclusive_group()
    solver_ir_group.add_argument(
        '--use-solver-ir',
        dest='use_solver_ir',
        action='store_true',
        help='Lower TASIR to Solver IR after TASIR build (disabled by default)'
    )
    solver_ir_group.add_argument(
        '--no-solver-ir',
        dest='use_solver_ir',
        action='store_false',
        help='Disable Solver IR lowering step'
    )
    parser.set_defaults(use_solver_ir=False)
    parser.add_argument(
        '--include-quality',
        action='store_true',
        default=False,
        help='Include code quality detectors (e.g., stack_underflow) in analysis'
    )
    parser.add_argument(
        '--strict-completeness',
        action='store_true',
        default=False,
        help='Fail closed when analysis is incomplete'
    )
    return parser


def _print_completeness_warnings(analysis_info: dict) -> None:
    """Print completeness warnings to stderr."""
    completeness = analysis_info.get("completeness", {})
    metadata = analysis_info.get("metadata", {})

    if completeness.get("analysis_incomplete"):
        reason_text = completeness.get("summary", "analysis incomplete")
        print(f"⚠️  Dataflow analysis incomplete: {reason_text}.", file=sys.stderr)
    if metadata.get("continuation_extraction_failed"):
        error_detail = metadata.get("continuation_extraction_error")
        msg = "⚠️  Continuation extraction failed; analysis may be incomplete."
        if error_detail:
            msg = f"{msg} ({error_detail})"
        print(msg, file=sys.stderr)
    if metadata.get("continuation_calls_detected"):
        print(
            "⚠️  Continuations detected; unresolved continuation targets "
            "are analyzed conservatively.",
            file=sys.stderr,
        )


def _print_stats(facts, verbose: bool) -> None:
    """Print contract statistics."""
    print("\n" + "=" * 80)
    print("CONTRACT STATISTICS")
    print("=" * 80)
    print(f"Instructions: {len(facts.instructions)}")
    print(f"Basic blocks: {len(facts.basic_blocks)}")
    print(f"CFG edges: {len(facts.cfg_edges)}")
    print(f"Events: {len(facts.events)}")
    print(f"  - Accept: {len(facts.events_of('accept'))}")
    print(f"  - Send: {len(facts.events_of('send'))}")
    print(f"  - Sender read: {len(facts.events_of('sender_read'))}")
    print(f"  - Guards: {len(facts.events_of('guard'))}")
    print(f"Call sites: {len(facts.call_sites)}")
    if verbose:
        print("\nTop opcodes:")
        opcode_counts: dict = {}
        for inst in facts.instructions:
            opcode_counts[inst.opcode] = opcode_counts.get(inst.opcode, 0) + 1
        for opcode, count in sorted(
            opcode_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {opcode:<25s}: {count:3d}")


def _format_text_output(vulnerabilities: list, facts) -> None:
    """Format vulnerability results as text."""
    print(f"Instructions: {len(facts.instructions)}")
    print(f"Basic blocks: {len(facts.basic_blocks)}")
    print()

    if not vulnerabilities:
        print("✅ No vulnerabilities detected.")
        return

    import re
    from collections import OrderedDict

    def _message_template(msg: str) -> str:
        """Normalize message by replacing variable parts with placeholders."""
        t = re.sub(r'\b(instruction|index|at index)\s+\d+', r'\1 N', msg)
        t = re.sub(r'\b(Block|block|cont_|method)\s*\w+', r'\1 X', t)
        t = re.sub(r'\bmethod\s+\d+', 'method N', t)
        t = re.sub(r'\bmin_height=[-\d]+', 'min_height=N', t)
        t = re.sub(r'\bdelta~[-\d]+', 'delta~N', t)
        t = re.sub(r'\bheight becomes [-\d]+', 'height becomes N', t)
        t = re.sub(
            r'\b\d+ (read operation|consecutive|LDREF operation|element|iteration)',
            r'N \1', t,
        )
        t = re.sub(r'\bmode \d+\s*\(0x[0-9a-f]+\)', 'mode N', t)
        t = re.sub(r'net stack consumption of \d+', 'net stack consumption of N', t)
        return t

    groups: OrderedDict[tuple, list] = OrderedDict()
    for vuln in vulnerabilities:
        key = (vuln.severity, vuln.detector, _message_template(vuln.message))
        groups.setdefault(key, []).append(vuln)

    unique_count = len(groups)
    print(
        f"⚠️  Detected {unique_count} unique issue(s) "
        f"({len(vulnerabilities)} total occurrences):"
    )
    print()

    for severity in ['critical', 'high', 'medium', 'low']:
        severity_groups = [(k, v) for k, v in groups.items() if k[0] == severity]
        if not severity_groups:
            continue

        print(f"[{severity.upper()}]")
        for (_, detector, _tmpl), vulns in severity_groups:
            count_str = f" (×{len(vulns)})" if len(vulns) > 1 else ""
            print(f"  • {vulns[0].message}{count_str}")
            print(f"    Detector: {detector}")
            if vulns[0].remediation:
                print(f"    Fix: {vulns[0].remediation}")
            locations = []
            for v in vulns:
                parts = []
                if v.instruction:
                    parts.append(f"#{v.instruction.index}")
                    if hasattr(v.instruction, 'opcode'):
                        parts.append(v.instruction.opcode)
                    if v.instruction.continuation_id:
                        parts.append(f"in {v.instruction.continuation_id}")
                if parts:
                    locations.append(" ".join(parts))
            if locations:
                shown = locations[:8]
                remaining = len(locations) - 8
                loc_str = ", ".join(shown)
                if remaining > 0:
                    loc_str += f", ... +{remaining} more"
                print(f"    At: {loc_str}")
            print()


def main():
    """Main CLI function."""
    parser = _build_parser()
    args = parser.parse_args()

    # --list-detectors: print and exit
    if args.list_detectors:
        from .detectors import list_detector_metadata
        print("Available detectors:")
        for meta in list_detector_metadata():
            category = meta.get('category', 'security')
            cat_tag = f"[{category}]" if category != 'security' else ""
            print(
                f"  {meta['name']:<30s} [{meta['severity']:<8s}] "
                f"{cat_tag} {meta['description']}"
            )
        print("\nNote: Code quality detectors require --include-quality flag to run.")
        return

    if not args.boc_file:
        parser.print_help()
        print("\nError: BOC file is required", file=sys.stderr)
        sys.exit(1)

    # Validate file path
    boc_path = Path(args.boc_file).resolve()
    if not boc_path.is_file():
        print(f"Error: Invalid file path: {args.boc_file}", file=sys.stderr)
        sys.exit(1)

    # Create scanner with CLI options
    path_overrides = {
        "max_paths": args.max_paths,
        "max_loop_unroll": args.max_loop_unroll,
        "max_analysis_depth": args.max_analysis_depth,
        "max_block_visits": args.max_block_visits,
        "max_worklist_size": args.max_worklist_size,
    }
    scanner = SecurityScanner(
        enable_all_detectors=False,
        path_sensitive=not args.path_insensitive,
        analysis_mode=args.analysis_mode,
        path_config=path_overrides,
        use_tasir=args.use_tasir,
        use_solver_ir=args.use_solver_ir,
        strict_completeness=args.strict_completeness,
    )

    # Configure detectors
    if args.config:
        try:
            from .detectors import load_config_file, load_detectors_from_config
            config = load_config_file(args.config)
            scanner.detectors = load_detectors_from_config(config)
            if args.verbose:
                print(f"Loaded {len(scanner.detectors)} detectors from config: {args.config}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        include_quality = getattr(args, 'include_quality', False)
        scanner.detectors = default_detectors(include_code_quality=include_quality)

    # Filter by detector names if specified
    if args.detectors:
        requested = [n.strip() for n in args.detectors.split(',') if n.strip()]
        try:
            _filter_detectors(scanner, requested)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Scan ---
    try:
        vulnerabilities = scanner.scan_file(str(boc_path))
    except AnalysisIncompleteError:
        # Strict completeness rejection
        analysis_info = scanner.get_last_analysis_info()
        if args.json:
            output = {
                "version": __version__,
                "file": args.boc_file,
                "scan_status": "scan_failed",
                "scan_failed_reason": "analysis_incomplete",
                "vulnerabilities": [],
                "summary": {
                    "total": 0, "critical": 0,
                    "high": 0, "medium": 0, "low": 0,
                },
                "analysis": analysis_info,
            }
            print(json.dumps(output, indent=2))
        else:
            reason = analysis_info.get("completeness", {}).get(
                "summary", "analysis incomplete"
            )
            print(
                f"Error: strict completeness enabled; analysis incomplete: {reason}",
                file=sys.stderr,
            )
        sys.exit(2)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # --- Post-scan output ---
    analysis_info = scanner.get_last_analysis_info()
    facts = scanner.last_facts

    # Completeness warnings
    _print_completeness_warnings(analysis_info)

    # Statistics
    if args.stats and facts is not None:
        _print_stats(facts, args.verbose)

    # Results
    if args.json:
        output = {
            "version": __version__,
            "file": args.boc_file,
            "vulnerabilities": [v.to_dict() for v in vulnerabilities],
            "summary": {
                "total": len(vulnerabilities),
                "critical": sum(1 for v in vulnerabilities if v.severity == "critical"),
                "high": sum(1 for v in vulnerabilities if v.severity == "high"),
                "medium": sum(1 for v in vulnerabilities if v.severity == "medium"),
                "low": sum(1 for v in vulnerabilities if v.severity == "low"),
            },
        }
        if analysis_info:
            output["analysis"] = analysis_info
        print(json.dumps(output, indent=2))
    else:
        print(f"Scanning: {args.boc_file}")
        if facts is not None:
            _format_text_output(vulnerabilities, facts)


if __name__ == '__main__':
    main()
