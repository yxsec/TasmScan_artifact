#!/usr/bin/env python3
"""
Validate tasmscan against a dataset of TON contracts.

Usage:
    python scripts/validate_dataset.py /path/to/dataset [--limit N] [--verbose]
"""
import argparse
import json
import sys
import traceback
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytoniq_core import Cell
from tasmscan.disassembler import parse_boc, decompile_cell
from tasmscan.analyzer import ProgramAnalyzer
from tasmscan.detectors import default_detectors
from tasmscan.boc_utils import load_boc_file
from tasmscan.scanner import SecurityScanner


@dataclass
class AnalysisResult:
    """Result of analyzing a single contract."""
    contract_id: str
    success: bool
    error: Optional[str] = None
    error_stage: Optional[str] = None
    instruction_count: int = 0
    block_count: int = 0
    vulnerability_count: int = 0
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    analysis_time_ms: float = 0
    metadata: Optional[Dict[str, Any]] = None
    analysis_mode: str = "primary"
    timed_out_primary: bool = False


class ContractTimeoutError(TimeoutError):
    """Raised when single-contract analysis exceeds wall-clock budget."""


@contextmanager
def _contract_timeout(seconds: int):
    """Apply SIGALRM timeout for a single contract analysis task."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _on_timeout(signum, frame):  # type: ignore[unused-argument]
        raise ContractTimeoutError(f"Timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _serialize_scanner_vulnerabilities(vulnerabilities: List[Any]) -> List[Dict[str, Any]]:
    """Normalize scanner vulnerability objects into JSON-serializable dicts."""
    serialized: List[Dict[str, Any]] = []
    for vuln in vulnerabilities:
        serialized.append(
            {
                "detector": str(getattr(vuln, "detector", "unknown")),
                "severity": str(getattr(vuln, "severity", "unknown")),
                "message": str(getattr(vuln, "message", "")),
            }
        )
    return serialized


def _run_timeout_fallback_no_tasir(
    contract_id: str,
    bytecode_file: Path,
    metadata: Optional[Dict[str, Any]],
    timeout_seconds: int,
) -> AnalysisResult:
    """
    Retry timed-out contract with conservative scanner mode:
    path-insensitive, no TASIR, strict completeness disabled.
    """
    start_time = time.time()
    try:
        scanner = SecurityScanner(
            path_sensitive=False,
            use_tasir=False,
            strict_completeness=False,
        )
        with _contract_timeout(timeout_seconds):
            vulnerabilities = scanner.scan_file(str(bytecode_file))
        info = scanner.get_last_analysis_info() or {}
        info_metadata = info.get("metadata", {}) if isinstance(info, dict) else {}
        serialized_vulns = _serialize_scanner_vulnerabilities(vulnerabilities)
        elapsed_ms = (time.time() - start_time) * 1000
        return AnalysisResult(
            contract_id=contract_id,
            success=True,
            instruction_count=int(info_metadata.get("instruction_count") or 0),
            block_count=int(info_metadata.get("basic_block_count") or 0),
            vulnerability_count=len(serialized_vulns),
            vulnerabilities=serialized_vulns,
            analysis_time_ms=elapsed_ms,
            metadata=metadata,
            analysis_mode="fallback_no_tasir",
            timed_out_primary=True,
        )
    except ContractTimeoutError as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="timeout_fallback",
            metadata=metadata,
            analysis_mode="fallback_no_tasir",
            timed_out_primary=True,
        )
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="fallback_no_tasir",
            metadata=metadata,
            analysis_mode="fallback_no_tasir",
            timed_out_primary=True,
        )


def analyze_contract(
    contract_dir: Path,
    per_contract_timeout_s: int = 0,
    timeout_fallback_no_tasir: bool = False,
    timeout_fallback_timeout_s: int = 0,
) -> AnalysisResult:
    """Analyze a single contract directory."""
    contract_id = contract_dir.name
    start_time = time.time()

    # Read metadata
    metadata_file = contract_dir / "metadata.json"
    metadata = None
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Find bytecode file
    bytecode_file = contract_dir / "bytecode.boc.b64"
    if not bytecode_file.exists():
        # Try other patterns
        for pattern in ["*.boc.b64", "*.boc"]:
            files = list(contract_dir.glob(pattern))
            if files:
                bytecode_file = files[0]
                break

    if not bytecode_file.exists():
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error="No bytecode file found",
            error_stage="file_read",
            metadata=metadata
        )

    fallback_timeout_s = (
        timeout_fallback_timeout_s
        if timeout_fallback_timeout_s > 0
        else per_contract_timeout_s
    )

    def _maybe_retry_timeout(primary_timeout: ContractTimeoutError) -> Optional[AnalysisResult]:
        if not timeout_fallback_no_tasir:
            return None
        fallback = _run_timeout_fallback_no_tasir(
            contract_id=contract_id,
            bytecode_file=bytecode_file,
            metadata=metadata,
            timeout_seconds=fallback_timeout_s,
        )
        if fallback.success:
            return fallback
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=(
                f"{primary_timeout}; fallback_no_tasir failed "
                f"({fallback.error_stage}: {fallback.error})"
            ),
            error_stage=fallback.error_stage or "timeout",
            metadata=metadata,
            analysis_mode="fallback_no_tasir",
            timed_out_primary=True,
        )

    # Read and decode bytecode (auto-detect hex or base64 for text files)
    try:
        boc_data = load_boc_file(bytecode_file)
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="decode",
            metadata=metadata
        )

    # Parse BOC
    try:
        cell = parse_boc(boc_data)
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="boc_parse",
            metadata=metadata
        )

    # Disassemble
    try:
        instructions = decompile_cell(cell)
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="disassemble",
            metadata=metadata
        )

    # Analyze
    try:
        analyzer = ProgramAnalyzer()
        with _contract_timeout(per_contract_timeout_s):
            facts = analyzer.analyze(instructions, cell)
    except ContractTimeoutError as e:
        fallback_result = _maybe_retry_timeout(e)
        if fallback_result is not None:
            return fallback_result
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="timeout",
            metadata=metadata
        )
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="analyze",
            metadata=metadata
        )

    # Run detectors
    try:
        detectors = default_detectors()
        vulnerabilities = []
        with _contract_timeout(per_contract_timeout_s):
            for detector in detectors:
                try:
                    results = detector.detect(facts)
                    for vuln in results:
                        vulnerabilities.append({
                            "detector": detector.name,
                            "severity": vuln.severity,
                            "message": vuln.message
                        })
                except Exception as e:
                    # Record detector failure but continue
                    vulnerabilities.append({
                        "detector": detector.name,
                        "severity": "error",
                        "message": f"Detector error: {str(e)}"
                    })
    except ContractTimeoutError as e:
        fallback_result = _maybe_retry_timeout(e)
        if fallback_result is not None:
            return fallback_result
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="timeout",
            metadata=metadata
        )
    except Exception as e:
        return AnalysisResult(
            contract_id=contract_id,
            success=False,
            error=str(e),
            error_stage="detect",
            metadata=metadata
        )

    elapsed_ms = (time.time() - start_time) * 1000

    return AnalysisResult(
        contract_id=contract_id,
        success=True,
        instruction_count=len(facts.instructions),
        block_count=len(facts.basic_blocks),
        vulnerability_count=len(vulnerabilities),
        vulnerabilities=vulnerabilities,
        analysis_time_ms=elapsed_ms,
        metadata=metadata
    )


def find_contract_dirs(dataset_path: Path) -> List[Path]:
    """Find all contract directories in the dataset."""
    contract_dirs = []

    # Directories to exclude (common dependency/source folders, not contracts)
    EXCLUDE_DIRS = {
        'node_modules', 'imports', 'contracts', 'sources', 'build',
        'dist', 'output', 'lib', 'libs', '__pycache__', '.git'
    }

    def is_contract_dir(item: Path) -> bool:
        """Check if directory looks like a contract (has bytecode file)."""
        if not item.is_dir():
            return False
        if item.name.startswith('.') or item.name in EXCLUDE_DIRS:
            return False
        # Contract dirs should have bytecode file
        return (item / "bytecode.boc.b64").exists() or \
               any(item.glob("*.boc.b64")) or \
               any(item.glob("*.boc"))

    # Check for mainnet/testnet subdirectories
    for network_dir in ["mainnet", "testnet"]:
        network_path = dataset_path / network_dir
        if network_path.exists():
            for item in network_path.iterdir():
                if is_contract_dir(item):
                    contract_dirs.append(item)

    # Also check root level
    if not contract_dirs:
        for item in dataset_path.iterdir():
            if is_contract_dir(item):
                contract_dirs.append(item)

    return contract_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Validate tasmscan against a dataset of TON contracts'
    )
    parser.add_argument('dataset', help='Path to dataset directory')
    parser.add_argument('--limit', '-n', type=int, help='Limit number of contracts to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    parser.add_argument(
        '--per-contract-timeout',
        type=int,
        default=0,
        help='Per-contract timeout in seconds (0 disables timeout).',
    )
    parser.add_argument(
        '--timeout-fallback-no-tasir',
        action='store_true',
        help=(
            'On primary timeout only, retry once with '
            'SecurityScanner(path_sensitive=False, use_tasir=False).'
        ),
    )
    parser.add_argument(
        '--timeout-fallback-timeout',
        type=int,
        default=0,
        help='Per-contract timeout for fallback retry (0 uses --per-contract-timeout).',
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Find all contract directories
    contract_dirs = find_contract_dirs(dataset_path)
    total = len(contract_dirs)
    print(f"Found {total} contracts in dataset")

    if args.limit:
        contract_dirs = contract_dirs[:args.limit]
        print(f"Limiting to {len(contract_dirs)} contracts")
    if args.timeout_fallback_no_tasir:
        fallback_timeout = (
            args.timeout_fallback_timeout
            if args.timeout_fallback_timeout > 0
            else args.per_contract_timeout
        )
        timeout_display = f"{fallback_timeout}s" if fallback_timeout > 0 else "disabled"
        print(f"Timeout fallback enabled: no-TASIR retry (timeout={timeout_display})")

    # Analyze contracts
    results: List[AnalysisResult] = []
    error_counts: Dict[str, int] = {}

    start_time = time.time()

    # Use parallel processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_contract,
                d,
                args.per_contract_timeout,
                args.timeout_fallback_no_tasir,
                args.timeout_fallback_timeout,
            ): d
            for d in contract_dirs
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)

                if not result.success:
                    stage = result.error_stage or "unknown"
                    error_counts[stage] = error_counts.get(stage, 0) + 1
                    if args.verbose:
                        print(f"[{i}/{len(contract_dirs)}] FAIL {result.contract_id[:20]}... ({stage}: {result.error[:50]})")
                else:
                    if args.verbose:
                        mode_suffix = (
                            f", mode={result.analysis_mode}"
                            if result.analysis_mode != "primary"
                            else ""
                        )
                        vuln_str = f", {result.vulnerability_count} vulns" if result.vulnerability_count > 0 else ""
                        print(
                            f"[{i}/{len(contract_dirs)}] OK   {result.contract_id[:20]}... "
                            f"({result.instruction_count} insts, {result.block_count} blocks"
                            f"{vuln_str}{mode_suffix})"
                        )

                # Progress indicator
                if not args.verbose and i % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    print(f"Progress: {i}/{len(contract_dirs)} ({i*100/len(contract_dirs):.1f}%) - {rate:.1f} contracts/sec")

            except Exception as e:
                print(f"Worker error: {e}", file=sys.stderr)

    elapsed_total = time.time() - start_time

    # Calculate statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_instructions = sum(r.instruction_count for r in successful)
    total_blocks = sum(r.block_count for r in successful)
    total_vulns = sum(r.vulnerability_count for r in successful)
    avg_time = sum(r.analysis_time_ms for r in successful) / len(successful) if successful else 0
    fallback_recovered = sum(1 for r in successful if r.analysis_mode == "fallback_no_tasir")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total contracts:     {len(results)}")
    print(f"Successful:          {len(successful)} ({len(successful)*100/len(results):.1f}%)")
    print(f"Failed:              {len(failed)} ({len(failed)*100/len(results):.1f}%)")
    print(f"Total time:          {elapsed_total:.1f}s ({len(results)/elapsed_total:.1f} contracts/sec)")
    print()
    print(f"Total instructions:  {total_instructions:,}")
    print(f"Total basic blocks:  {total_blocks:,}")
    print(f"Total findings:      {total_vulns:,}")
    print(f"Avg analysis time:   {avg_time:.1f}ms")
    if args.timeout_fallback_no_tasir:
        print(f"Fallback recovered:  {fallback_recovered}")

    if error_counts:
        print("\nError breakdown:")
        for stage, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {stage:<20s}: {count} ({count*100/len(failed):.1f}%)")

    # Vulnerability breakdown
    if total_vulns > 0:
        vuln_by_detector: Dict[str, int] = {}
        vuln_by_severity: Dict[str, int] = {}
        for r in successful:
            for v in r.vulnerabilities:
                detector = v.get("detector", "unknown")
                severity = v.get("severity", "unknown")
                vuln_by_detector[detector] = vuln_by_detector.get(detector, 0) + 1
                vuln_by_severity[severity] = vuln_by_severity.get(severity, 0) + 1

        print("\nFindings by detector:")
        for detector, count in sorted(vuln_by_detector.items(), key=lambda x: -x[1]):
            print(f"  {detector:<30s}: {count}")

        print("\nFindings by severity:")
        for severity, count in sorted(vuln_by_severity.items(), key=lambda x: -x[1]):
            print(f"  {severity:<10s}: {count}")

    # Output detailed results if requested
    if args.output:
        output_data = {
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0,
                "total_instructions": total_instructions,
                "total_blocks": total_blocks,
                "total_vulnerabilities": total_vulns,
                "avg_analysis_time_ms": avg_time,
                "total_time_seconds": elapsed_total,
                "fallback_recovered": fallback_recovered,
            },
            "error_breakdown": error_counts,
            "results": [
                {
                    "contract_id": r.contract_id,
                    "success": r.success,
                    "error": r.error,
                    "error_stage": r.error_stage,
                    "instruction_count": r.instruction_count,
                    "block_count": r.block_count,
                    "vulnerability_count": r.vulnerability_count,
                    "vulnerabilities": r.vulnerabilities,
                    "analysis_time_ms": r.analysis_time_ms,
                    "analysis_mode": r.analysis_mode,
                    "timed_out_primary": r.timed_out_primary,
                    "compiler": r.metadata.get("compiler") if r.metadata else None,
                    "network": r.metadata.get("network") if r.metadata else None
                }
                for r in results
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to: {args.output}")

    # Exit with error code if any failures
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
