#!/usr/bin/env python3
"""
Comprehensive benchmark test runner for tasmscan.
Tests against an externally supplied TON benchmark corpus.
Collects statistics on crashes, detections, performance, and edge cases.
"""

import json
import os
import sys
import time
import traceback
import base64
import signal
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasmscan.scanner import AnalysisIncompleteError, SecurityScanner
from tasmscan.config import AnalysisMode


BENCHMARK_DIR = Path(os.environ.get("TASMSCAN_BENCHMARK_DIR", "ton_benchmark/sources"))
RESULTS_DIR = Path(os.environ.get("TASMSCAN_RESULTS_DIR", "results"))


@dataclass
class ContractResult:
    """Result from scanning a single contract."""
    ipfs_hash: str
    network: str
    compiler: str
    status: str  # "success", "crash", "scan_failed", "timeout", "decode_error"
    duration_ms: float = 0.0
    vulnerabilities: List[Dict] = field(default_factory=list)
    error_message: str = ""
    error_type: str = ""
    bytecode_size: int = 0
    instruction_count: int = 0
    block_count: int = 0
    continuation_count: int = 0
    warnings: List[str] = field(default_factory=list)
    analysis_incomplete: bool = False
    analysis_incomplete_reasons: List[str] = field(default_factory=list)
    analysis_completeness: Dict[str, Any] = field(default_factory=dict)


class ContractTimeoutError(TimeoutError):
    """Raised when scanning a single contract exceeds its budget."""


@contextmanager
def _contract_timeout(seconds: int):
    """Apply a per-contract wall-clock timeout using SIGALRM."""
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


def find_all_contracts(benchmark_dir: Path) -> List[Dict]:
    """Find all contracts with bytecode in the benchmark."""
    contracts = []
    for network in ["mainnet", "testnet"]:
        network_dir = benchmark_dir / network
        if not network_dir.exists():
            continue
        for contract_dir in sorted(network_dir.iterdir()):
            if not contract_dir.is_dir():
                continue
            bytecode_file = contract_dir / "bytecode.boc.b64"
            metadata_file = contract_dir / "metadata.json"
            if not bytecode_file.exists():
                continue

            compiler = "unknown"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                    compiler = meta.get("compiler", "unknown")
                except Exception:
                    pass

            contracts.append({
                "ipfs_hash": contract_dir.name,
                "network": network,
                "bytecode_path": str(bytecode_file),
                "metadata_path": str(metadata_file) if metadata_file.exists() else None,
                "compiler": compiler,
            })
    return contracts


def scan_single_contract(
    contract_info: Dict,
    scanner: SecurityScanner,
    per_contract_timeout_s: int = 0,
) -> ContractResult:
    """Scan a single contract and return the result."""
    result = ContractResult(
        ipfs_hash=contract_info["ipfs_hash"],
        network=contract_info["network"],
        compiler=contract_info["compiler"],
        status="success",
    )

    try:
        # Use scan_file which handles base64/hex/binary formats
        bytecode_path = contract_info["bytecode_path"]

        # Get bytecode size for stats
        with open(bytecode_path, "r") as f:
            b64_data = f.read().strip()
        try:
            result.bytecode_size = len(base64.b64decode(b64_data + "=="))  # pad for safety
        except Exception:
            result.bytecode_size = len(b64_data)

        # Scan with timeout tracking
        start = time.time()
        with _contract_timeout(per_contract_timeout_s):
            vulnerabilities = scanner.scan_file(bytecode_path)
        elapsed = time.time() - start
        result.duration_ms = elapsed * 1000

        # Extract results - vulnerabilities is a List[Vulnerability]
        result.vulnerabilities = []
        for vuln in vulnerabilities:
            result.vulnerabilities.append({
                "detector": vuln.detector,
                "severity": vuln.severity,
                "confidence": vuln.confidence.value if hasattr(vuln.confidence, 'value') else str(vuln.confidence),
                "message": vuln.message,
                "remediation": vuln.remediation or "",
                "extra": vuln.extra or {},
            })

        analysis_info = scanner.get_last_analysis_info()
        metadata = analysis_info.get("metadata", {}) if isinstance(analysis_info, dict) else {}
        if isinstance(metadata, dict):
            result.instruction_count = int(metadata.get("instruction_count", 0) or 0)
            result.block_count = int(metadata.get("basic_block_count", 0) or 0)
            result.continuation_count = int(metadata.get("continuation_count", 0) or 0)
        completeness = analysis_info.get("completeness", {}) if isinstance(analysis_info, dict) else {}
        if isinstance(completeness, dict):
            result.analysis_completeness = dict(completeness)
            result.analysis_incomplete = bool(completeness.get("analysis_incomplete"))
            reasons = completeness.get("analysis_incomplete_reasons", [])
            if isinstance(reasons, list):
                result.analysis_incomplete_reasons = [r for r in reasons if isinstance(r, str)]
            if result.analysis_incomplete:
                summary = completeness.get("summary")
                if isinstance(summary, str) and summary:
                    result.warnings.append(f"analysis_incomplete: {summary}")

    except AnalysisIncompleteError as e:
        result.status = "scan_failed"
        result.error_type = type(e).__name__
        result.error_message = str(e)
        if isinstance(getattr(e, "completeness", None), dict):
            result.analysis_completeness = dict(e.completeness)
            result.analysis_incomplete = True
            reasons = e.completeness.get("analysis_incomplete_reasons", [])
            if isinstance(reasons, list):
                result.analysis_incomplete_reasons = [r for r in reasons if isinstance(r, str)]
        result.duration_ms = 0
    except ContractTimeoutError as e:
        result.status = "timeout"
        result.error_type = type(e).__name__
        result.error_message = str(e)
        result.duration_ms = 0
    except Exception as e:
        result.status = "crash"
        result.error_type = type(e).__name__
        result.error_message = str(e)
        result.duration_ms = 0

    return result


def evaluate_release_gate(
    results: List[ContractResult],
    *,
    max_incomplete_rate: float,
    max_dynamic_targets: int,
) -> Dict[str, Any]:
    """Compute a go/no-go release gate summary."""
    total = len(results)
    crashes = [r for r in results if r.status == "crash"]
    timeouts = [r for r in results if r.status == "timeout"]
    failures = [r for r in results if r.status == "scan_failed"]
    successes = [r for r in results if r.status == "success"]
    incomplete_success = [r for r in successes if r.analysis_incomplete]

    incomplete_rate = (len(incomplete_success) / len(successes)) if successes else 0.0
    reason_counts = Counter()
    for r in incomplete_success:
        reason_counts.update(r.analysis_incomplete_reasons)
    dynamic_targets = int(reason_counts.get("dynamic_continuation_targets", 0))

    checks = {
        "no_crashes": len(crashes) == 0,
        "no_timeouts": len(timeouts) == 0,
        "no_scan_failed": len(failures) == 0,
        "incomplete_rate_within_budget": incomplete_rate <= max_incomplete_rate,
        "dynamic_targets_within_budget": dynamic_targets <= max_dynamic_targets,
    }
    failed_checks = [name for name, ok in checks.items() if not ok]

    return {
        "go": not failed_checks,
        "failed_checks": failed_checks,
        "metrics": {
            "total_contracts": total,
            "success_contracts": len(successes),
            "incomplete_success_contracts": len(incomplete_success),
            "incomplete_rate": incomplete_rate,
            "dynamic_continuation_targets": dynamic_targets,
            "crash_count": len(crashes),
            "timeout_count": len(timeouts),
            "scan_failed_count": len(failures),
        },
        "budgets": {
            "max_incomplete_rate": max_incomplete_rate,
            "max_dynamic_targets": max_dynamic_targets,
        },
        "checks": checks,
    }


def run_benchmark(
    max_contracts: int = 0,
    network_filter: str = "",
    strict_completeness: bool = False,
    max_incomplete_rate: float = 1.0,
    max_dynamic_targets: int = 10**9,
    per_contract_timeout_s: int = 0,
):
    """Run the full benchmark."""
    print("=" * 80)
    print("TasmScan BENCHMARK TEST")
    print("=" * 80)

    # Find contracts
    contracts = find_all_contracts(BENCHMARK_DIR)
    print(f"\nFound {len(contracts)} contracts with bytecode")

    if network_filter:
        contracts = [c for c in contracts if c["network"] == network_filter]
        print(f"Filtered to {len(contracts)} {network_filter} contracts")

    if max_contracts > 0:
        contracts = contracts[:max_contracts]
        print(f"Limited to first {max_contracts} contracts")

    # Initialize scanner
    scanner = SecurityScanner(
        analysis_mode=AnalysisMode.SECURITY_AUDIT,
        use_tasir=True,
        strict_completeness=strict_completeness,
    )

    # Run scans
    results: List[ContractResult] = []
    crash_details: Dict[str, List[str]] = defaultdict(list)

    start_time = time.time()

    for i, contract in enumerate(contracts):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\n[{i+1}/{len(contracts)}] Scanning... ({rate:.1f} contracts/sec)")

        result = scan_single_contract(
            contract,
            scanner,
            per_contract_timeout_s=per_contract_timeout_s,
        )
        results.append(result)

        if result.status == "crash":
            crash_details[result.error_type].append(
                f"{result.network}/{result.ipfs_hash}: {result.error_message[:200]}"
            )

        # Print dots for progress
        if result.status == "success":
            print(".", end="", flush=True)
        elif result.status == "crash":
            print("X", end="", flush=True)
        elif result.status == "timeout":
            print("T", end="", flush=True)
        elif result.status == "scan_failed":
            print("F", end="", flush=True)
        else:
            print("?", end="", flush=True)

    total_time = time.time() - start_time

    # Generate report
    print("\n\n")
    gate = evaluate_release_gate(
        results,
        max_incomplete_rate=max_incomplete_rate,
        max_dynamic_targets=max_dynamic_targets,
    )
    generate_report(results, crash_details, total_time, gate)

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_detailed_results(results, crash_details, gate)

    return results


def generate_report(results: List[ContractResult], crash_details: Dict, total_time: float, gate: Dict[str, Any]):
    """Generate the benchmark report."""
    total = len(results)
    denominator = total if total > 0 else 1
    successes = [r for r in results if r.status == "success"]
    crashes = [r for r in results if r.status == "crash"]
    timeouts = [r for r in results if r.status == "timeout"]
    failures = [r for r in results if r.status == "scan_failed"]

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Overall stats
    print(f"\n## Overall Statistics")
    print(f"  Total contracts tested: {total}")
    print(f"  Successful scans:       {len(successes)} ({100*len(successes)/denominator:.1f}%)")
    print(f"  Crashes:                {len(crashes)} ({100*len(crashes)/denominator:.1f}%)")
    print(f"  Timeouts:               {len(timeouts)} ({100*len(timeouts)/denominator:.1f}%)")
    print(f"  Scan failures:          {len(failures)} ({100*len(failures)/denominator:.1f}%)")
    print(f"  Total time:             {total_time:.1f}s")
    print(f"  Average time/contract:  {(total_time/total*1000) if total else 0.0:.1f}ms")

    # By compiler
    print(f"\n## Results by Compiler")
    compiler_stats = defaultdict(
        lambda: {"total": 0, "success": 0, "crash": 0, "timeout": 0, "failed": 0}
    )
    for r in results:
        compiler_stats[r.compiler]["total"] += 1
        if r.status == "success":
            compiler_stats[r.compiler]["success"] += 1
        elif r.status == "crash":
            compiler_stats[r.compiler]["crash"] += 1
        elif r.status == "timeout":
            compiler_stats[r.compiler]["timeout"] += 1
        else:
            compiler_stats[r.compiler]["failed"] += 1

    for compiler, stats in sorted(compiler_stats.items()):
        pct = 100 * stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"  {compiler:12s}: {stats['total']:5d} total, {stats['success']:5d} OK ({pct:.1f}%), "
            f"{stats['crash']:4d} crash, {stats['timeout']:4d} timeout, {stats['failed']:4d} failed"
        )

    # By network
    print(f"\n## Results by Network")
    for network in ["mainnet", "testnet"]:
        net_results = [r for r in results if r.network == network]
        if net_results:
            net_success = len([r for r in net_results if r.status == "success"])
            print(f"  {network}: {len(net_results)} total, {net_success} success "
                  f"({100*net_success/len(net_results):.1f}%)")

    # Crash analysis
    if crashes:
        print(f"\n## Crash Analysis ({len(crashes)} crashes)")
        for error_type, examples in sorted(crash_details.items(), key=lambda x: -len(x[1])):
            print(f"\n  ### {error_type} ({len(examples)} occurrences)")
            for ex in examples[:5]:
                print(f"    - {ex}")
            if len(examples) > 5:
                print(f"    ... and {len(examples) - 5} more")

    # Vulnerability detection stats
    if successes:
        print(f"\n## Vulnerability Detection Statistics")
        vuln_counts = Counter()
        severity_counts = Counter()
        confidence_counts = Counter()
        contracts_with_vulns = 0

        for r in successes:
            if r.vulnerabilities:
                contracts_with_vulns += 1
            for v in r.vulnerabilities:
                vuln_counts[v.get("detector", "unknown")] += 1
                severity_counts[v.get("severity", "unknown")] += 1
                confidence_counts[v.get("confidence", "unknown")] += 1

        print(f"\n  Contracts with vulnerabilities: {contracts_with_vulns}/{len(successes)} "
              f"({100*contracts_with_vulns/len(successes):.1f}%)")
        print(f"  Total vulnerabilities found:    {sum(vuln_counts.values())}")

        print(f"\n  ### By Detector:")
        for detector, count in vuln_counts.most_common():
            print(f"    {detector:40s}: {count:5d}")

        print(f"\n  ### By Severity:")
        for sev, count in severity_counts.most_common():
            print(f"    {sev:12s}: {count:5d}")

        print(f"\n  ### By Confidence:")
        for conf, count in confidence_counts.most_common():
            print(f"    {conf:12s}: {count:5d}")

        # Detection rates by compiler
        print(f"\n  ### Detection Rate by Compiler:")
        for compiler in sorted(compiler_stats.keys()):
            comp_results = [r for r in successes if r.compiler == compiler]
            if comp_results:
                with_vulns = len([r for r in comp_results if r.vulnerabilities])
                total_vulns = sum(len(r.vulnerabilities) for r in comp_results)
                print(f"    {compiler:12s}: {with_vulns}/{len(comp_results)} contracts "
                      f"({100*with_vulns/len(comp_results):.1f}%), {total_vulns} total findings")

        print(f"\n## Completeness Statistics")
        incomplete_successes = [r for r in successes if r.analysis_incomplete]
        incomplete_rate = (100 * len(incomplete_successes) / len(successes)) if successes else 0.0
        print(
            f"  Incomplete analyses: {len(incomplete_successes)}/{len(successes)} "
            f"({incomplete_rate:.1f}%)"
        )
        reason_counts = Counter()
        for r in incomplete_successes:
            reason_counts.update(r.analysis_incomplete_reasons)
        if reason_counts:
            print("  ### Reasons:")
            for reason, count in reason_counts.most_common():
                print(f"    {reason:36s}: {count:5d}")
        else:
            print("  ### Reasons: none")

    # Performance stats
    if successes:
        print(f"\n## Performance Statistics")
        durations = [r.duration_ms for r in successes]
        durations.sort()
        print(f"  Min scan time:    {min(durations):.1f}ms")
        print(f"  Max scan time:    {max(durations):.1f}ms")
        print(f"  Median scan time: {durations[len(durations)//2]:.1f}ms")
        print(f"  P95 scan time:    {durations[int(len(durations)*0.95)]:.1f}ms")
        print(f"  P99 scan time:    {durations[int(len(durations)*0.99)]:.1f}ms")

        # Slow contracts
        slow = [r for r in successes if r.duration_ms > 5000]
        if slow:
            print(f"\n  ### Slow Contracts (>5s): {len(slow)}")
            for r in sorted(slow, key=lambda x: -x.duration_ms)[:10]:
                print(f"    {r.network}/{r.ipfs_hash}: {r.duration_ms:.0f}ms "
                      f"({r.instruction_count} instructions)")

    # Size statistics
    if successes:
        print(f"\n## Bytecode Size Statistics")
        sizes = [r.bytecode_size for r in successes if r.bytecode_size > 0]
        if sizes:
            sizes.sort()
            print(f"  Min size:    {min(sizes)} bytes")
            print(f"  Max size:    {max(sizes)} bytes")
            print(f"  Median size: {sizes[len(sizes)//2]} bytes")
            print(f"  Average:     {sum(sizes)/len(sizes):.0f} bytes")

    print(f"\n## Release Gate")
    print(f"  Status: {'GO' if gate.get('go') else 'NO-GO'}")
    budgets = gate.get("budgets", {})
    metrics = gate.get("metrics", {})
    print(
        f"  Budgets: incomplete_rate <= {budgets.get('max_incomplete_rate', 1.0):.3f}, "
        f"dynamic_targets <= {budgets.get('max_dynamic_targets', 0)}"
    )
    print(
        f"  Metrics: incomplete_rate={metrics.get('incomplete_rate', 0.0):.3f}, "
        f"dynamic_targets={metrics.get('dynamic_continuation_targets', 0)}, "
        f"crash={metrics.get('crash_count', 0)}, timeout={metrics.get('timeout_count', 0)}, "
        f"scan_failed={metrics.get('scan_failed_count', 0)}"
    )
    failed_checks = gate.get("failed_checks", [])
    if failed_checks:
        print("  Failed checks:")
        for check in failed_checks:
            print(f"    - {check}")


def save_detailed_results(results: List[ContractResult], crash_details: Dict, gate: Dict[str, Any]):
    """Save detailed results to JSON."""
    success_results = [r for r in results if r.status == "success"]
    incomplete_success = [r for r in success_results if r.analysis_incomplete]
    incomplete_reason_counts = Counter()
    for r in incomplete_success:
        incomplete_reason_counts.update(r.analysis_incomplete_reasons)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_contracts": len(results),
        "summary": {
            "success": len([r for r in results if r.status == "success"]),
            "crash": len([r for r in results if r.status == "crash"]),
            "timeout": len([r for r in results if r.status == "timeout"]),
            "scan_failed": len([r for r in results if r.status == "scan_failed"]),
            "analysis_incomplete_success": len(incomplete_success),
            "analysis_incomplete_reasons": dict(incomplete_reason_counts),
            "release_gate": gate,
        },
        "crash_types": {k: len(v) for k, v in crash_details.items()},
        "results": [],
    }

    for r in results:
        output["results"].append({
            "ipfs_hash": r.ipfs_hash,
            "network": r.network,
            "compiler": r.compiler,
            "status": r.status,
            "duration_ms": round(r.duration_ms, 2),
            "bytecode_size": r.bytecode_size,
            "instruction_count": r.instruction_count,
            "block_count": r.block_count,
            "continuation_count": r.continuation_count,
            "vulnerability_count": len(r.vulnerabilities),
            "vulnerabilities": r.vulnerabilities,
            "analysis_incomplete": r.analysis_incomplete,
            "analysis_incomplete_reasons": r.analysis_incomplete_reasons,
            "analysis_completeness": r.analysis_completeness,
            "error_type": r.error_type,
            "error_message": r.error_message[:500] if r.error_message else "",
            "warnings": r.warnings,
        })

    output_path = RESULTS_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")

    # Save crash details separately
    crash_path = RESULTS_DIR / "crash_details.json"
    with open(crash_path, "w") as f:
        json.dump(crash_details, f, indent=2, default=str)
    print(f"Crash details saved to: {crash_path}")

    # Save per-detector findings
    detector_findings = defaultdict(list)
    for r in results:
        for v in r.vulnerabilities:
            detector_findings[v.get("detector", "unknown")].append({
                "contract": f"{r.network}/{r.ipfs_hash}",
                "compiler": r.compiler,
                "severity": v.get("severity"),
                "confidence": v.get("confidence"),
                "message": v.get("message", ""),
                "location": v.get("location", ""),
            })

    findings_path = RESULTS_DIR / "detector_findings.json"
    with open(findings_path, "w") as f:
        json.dump(detector_findings, f, indent=2, default=str)
    print(f"Detector findings saved to: {findings_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tasmscan benchmark")
    parser.add_argument("--max", type=int, default=0, help="Max contracts to test (0=all)")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="External benchmark sources directory (or set TASMSCAN_BENCHMARK_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated reports (or set TASMSCAN_RESULTS_DIR).",
    )
    parser.add_argument("--network", type=str, default="", help="Filter by network (mainnet/testnet)")
    parser.add_argument(
        "--strict-completeness",
        action="store_true",
        default=False,
        help="Fail closed when analysis is incomplete",
    )
    parser.add_argument(
        "--max-incomplete-rate",
        type=float,
        default=1.0,
        help="Release gate budget for incomplete success rate (0.0-1.0).",
    )
    parser.add_argument(
        "--max-dynamic-targets",
        type=int,
        default=10**9,
        help="Release gate budget for dynamic_continuation_targets count.",
    )
    parser.add_argument(
        "--per-contract-timeout",
        type=int,
        default=0,
        help="Per-contract timeout in seconds (0 disables timeout).",
    )
    args = parser.parse_args()

    if args.dataset_dir is not None:
        BENCHMARK_DIR = args.dataset_dir
    if args.output_dir is not None:
        RESULTS_DIR = args.output_dir

    run_benchmark(
        max_contracts=args.max,
        network_filter=args.network,
        strict_completeness=args.strict_completeness,
        max_incomplete_rate=args.max_incomplete_rate,
        max_dynamic_targets=args.max_dynamic_targets,
        per_contract_timeout_s=args.per_contract_timeout,
    )
