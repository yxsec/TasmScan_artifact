"""
Main scanner interface for tasmscan.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pytoniq_core import Cell

logger = logging.getLogger("tasmscan.scanner")

from .disassembler import parse_boc, decompile_cell
from .boc_utils import detect_and_decode_boc, load_boc_file

from .analyzer.program_analyzer import ProgramAnalyzer
from .detectors.registry import DETECTOR_CLASSES, default_detectors
from .detectors.results import Confidence, Vulnerability
from .ir.dataflow import DataFlowAnalyzer
from .ir.ir_builder import IRBuilder
from .config import (
    AnalysisMode,
    STRICT_COMPLETENESS_DEFAULT,
    build_path_config,
)
from .completeness import build_completeness_report


class AnalysisIncompleteError(RuntimeError):
    """Raised when strict completeness mode rejects incomplete analysis."""

    def __init__(self, completeness: Dict[str, Any]):
        self.completeness = dict(completeness)
        summary = self.completeness.get("summary", "analysis incomplete")
        super().__init__(f"analysis_incomplete: {summary}")


def _filter_detectors(
    scanner: "SecurityScanner",
    detectors: Optional[List[str]],
) -> None:
    """
    Filter scanner detectors by name.

    Args:
        scanner: SecurityScanner instance to modify
        detectors: List of detector names to enable (None = keep all)

    Raises:
        ValueError: If detectors list is empty, contains unknown names,
                    or results in no detectors selected
    """
    if detectors is None:
        return

    requested_names = [name.strip() for name in detectors if name and name.strip()]
    if not requested_names:
        raise ValueError("Invalid detectors list: at least one detector name is required")

    available_names = set(DETECTOR_CLASSES.keys())
    unknown = sorted(set(requested_names) - available_names)
    if unknown:
        available_display = ", ".join(sorted(available_names))
        unknown_display = ", ".join(unknown)
        raise ValueError(
            f"Unknown detector(s): {unknown_display}. Available: {available_display}"
        )

    scanner.detectors = []
    seen = set()
    for name in requested_names:
        if name in seen:
            continue
        seen.add(name)
        scanner.detectors.append(DETECTOR_CLASSES[name]())

    if not scanner.detectors:
        raise ValueError("No detectors selected after filtering")


def _decode_explicit_hex(source: str) -> bytes:
    """
    Decode explicit hex string (with 0x prefix).

    Args:
        source: Hex string with 0x prefix

    Returns:
        Decoded bytes

    Raises:
        ValueError: If hex string is invalid
    """
    hex_str = "".join(source.split())
    if hex_str.lower().startswith("0x"):
        hex_str = hex_str[2:]

    if not hex_str:
        raise ValueError("Invalid hex string: empty input")
    if not all(c in "0123456789abcdefABCDEF" for c in hex_str):
        raise ValueError("Invalid hex string: contains non-hex characters")
    if len(hex_str) % 2 != 0:
        raise ValueError(f"Invalid hex string: odd length ({len(hex_str)})")

    return bytes.fromhex(hex_str)


def _resolve_source_to_bytes(source: str) -> bytes:
    """
    Resolve source string to BOC bytes.

    Handles:
    - Explicit hex (0x prefix)
    - File paths (binary or text-encoded BOC)
    - Auto-detected base64/hex strings

    Args:
        source: Input source string

    Returns:
        BOC data as bytes

    Raises:
        ValueError: If source cannot be resolved to valid BOC data
    """
    source_stripped = source.strip()

    # Handle explicit hex (0x prefix)
    if source_stripped.lower().startswith("0x"):
        try:
            return _decode_explicit_hex(source_stripped)
        except ValueError as exc:
            # Fallback: allow non-hex payloads that happen to start with 0x
            try:
                return detect_and_decode_boc(source_stripped)
            except ValueError:
                raise ValueError(
                    f"Invalid input: '{source_stripped[:50]}...' is not a valid hex string: {exc}"
                ) from exc

    # Check if source is a file path
    path = Path(source_stripped)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Invalid input: '{source_stripped}' is not a file")
        return load_boc_file(path)

    # Auto-detect base64/hex for non-file inputs
    try:
        return detect_and_decode_boc(source_stripped)
    except ValueError as exc:
        raise ValueError(
            f"Invalid input: '{source_stripped[:50]}...' is neither a valid file path "
            f"nor base64/hex data: {exc}"
        ) from exc




class SecurityScanner:
    """
    Main security scanner that coordinates analysis and detection.
    """

    def __init__(
        self,
        enable_all_detectors: bool = True,
        path_sensitive: bool = True,
        analysis_mode: Optional[Union[str, AnalysisMode]] = None,
        path_config: Optional[Dict[str, int]] = None,
        use_tasir: bool = True,
        use_savelist: bool = True,
        use_solver_ir: bool = False,
        strict_completeness: bool = STRICT_COMPLETENESS_DEFAULT,
    ):
        """
        Initialize scanner.

        Args:
            enable_all_detectors: If True, enable all built-in detectors
            path_sensitive: Enable path-sensitive dataflow analysis
            analysis_mode: Optional analysis mode preset for path-sensitive analysis
            path_config: Optional overrides for path-sensitive parameters
            use_tasir: Use TASIR for enhanced continuation tracking (default: True)
            use_solver_ir: Lower TASIR to Solver IR after TASIR build (default: False)
            strict_completeness: When True, incomplete analysis raises an error
        """
        self.analyzer = ProgramAnalyzer()
        self.detectors = default_detectors() if enable_all_detectors else []
        self.path_sensitive = path_sensitive
        self.path_config = build_path_config(analysis_mode, path_config)
        self.use_tasir = use_tasir
        self.use_savelist = use_savelist
        self.use_solver_ir = use_solver_ir
        self.strict_completeness = strict_completeness
        self.last_analysis_metadata: Dict[str, Any] = {}
        self.last_completeness: Dict[str, Any] = build_completeness_report({})
        self.last_facts: Any = None
        self._last_tasir_module: Any = None
        self._last_solver_module: Any = None

    def scan_cell(self, cell: Cell) -> List[Vulnerability]:
        """
        Scan a Cell for vulnerabilities.

        Args:
            cell: Root cell to analyze

        Returns:
            List of detected vulnerabilities

        Raises:
            ValueError: If the cell cannot be disassembled (invalid bytecode format)
            RuntimeError: If analysis encounters an unrecoverable error

        Note:
            Exceptions from disassembly or analysis are intentionally propagated
            to the caller. This fail-fast design ensures callers are aware of
            malformed input rather than silently returning empty results, which
            could be confused with a clean contract.
        """
        self.last_analysis_metadata = {}
        self.last_completeness = build_completeness_report({})
        self.last_facts = None
        self._last_tasir_module = None
        self._last_solver_module = None

        # Step 1: Disassemble using internal disassembler
        instructions = decompile_cell(cell)

        # Step 2: Analyze program
        facts = self.analyzer.analyze(instructions, cell)
        self.last_facts = facts

        # Step 3: Build TASIR if enabled
        tasir_module = None
        solver_ir_module = None
        if self.use_tasir:
            try:
                ir_builder = IRBuilder()
                tasir_module = ir_builder.build_tasir(facts, skip_savelists=not self.use_savelist)
                # Store in facts for detectors
                facts.metadata["tasir_module"] = tasir_module
                # Store facts reference in TVMModule for query delegation
                tasir_module._facts = facts
                if self.use_solver_ir:
                    try:
                        solver_ir_module = ir_builder.lower_tasir_to_solver_ir(tasir_module)
                        facts.metadata["solver_ir_module"] = solver_ir_module
                        facts.metadata["solver_ir"] = dict(solver_ir_module.metadata or {})
                    except Exception as e:
                        logger.warning(
                            "Solver IR lowering failed, continuing without solver IR: %s",
                            e,
                        )
                        facts.metadata["solver_ir_error"] = str(e)
            except Exception as e:
                logger.warning("TASIR build failed, falling back to legacy analysis: %s", e)
                tasir_module = None
                solver_ir_module = None
        self._last_tasir_module = tasir_module
        self._last_solver_module = solver_ir_module

        # Step 4: Run dataflow analysis and attach to facts
        dataflow_analyzer = DataFlowAnalyzer()
        if self.use_tasir and tasir_module is not None:
            facts.dataflow_graph = dataflow_analyzer.analyze_tasir(
                tasir_module,
                path_sensitive=self.path_sensitive,
                path_config=self.path_config or None,
            )
        else:
            facts.dataflow_graph = dataflow_analyzer.analyze(
                facts,
                path_sensitive=self.path_sensitive,
                path_config=self.path_config or None,
            )
        graph_metadata = dict(facts.dataflow_graph.analysis_metadata or {})
        program_metadata = dict(facts.metadata or {})
        structural_metadata = {
            "instruction_count": len(getattr(facts, "instructions", []) or []),
            "basic_block_count": len(getattr(facts, "basic_blocks", []) or []),
            "cfg_edge_count": len(getattr(facts, "cfg_edges", []) or []),
            "continuation_count": int(program_metadata.get("continuation_count", 0) or 0),
            "entry_point_count": len(getattr(facts, "entry_points", []) or []),
        }
        # Preserve richer program metadata while keeping dataflow completeness
        # fields authoritative when key names overlap.
        self.last_analysis_metadata = {
            **program_metadata,
            **structural_metadata,
            **graph_metadata,
        }
        self.last_completeness = build_completeness_report(self.last_analysis_metadata)

        if self.strict_completeness and self.last_completeness.get("analysis_incomplete"):
            raise AnalysisIncompleteError(self.last_completeness)

        # Step 4b: Store dataflow graph in TVMModule for integrated access
        if tasir_module is not None and facts.dataflow_graph is not None:
            tasir_module.dataflow_graph = facts.dataflow_graph

        # Step 4c: Build inter-procedural analysis view
        if tasir_module is not None:
            tasir_module.build_inter_procedural_view()

        # Step 5: Run detectors
        vulnerabilities = []
        failed_detectors = []
        for detector in self.detectors:
            try:
                findings = detector.detect(facts)
                vulnerabilities.extend(findings)
            except Exception as e:
                logger.error("Detector %s failed: %s", detector.name, e)
                failed_detectors.append({"name": detector.name, "error": str(e)})

        if failed_detectors:
            vulnerabilities.append(
                Vulnerability(
                    detector="analysis_incomplete",
                    severity="low",
                    message=(
                        "One or more detectors failed to execute; "
                        "analysis results may be incomplete."
                    ),
                    instruction=facts.instructions[0] if facts.instructions else None,
                    remediation="Review detector errors and rerun the analysis.",
                    extra={
                        "failed_detectors": failed_detectors,
                        "analysis_completeness": dict(self.last_completeness),
                    },
                    confidence=Confidence.HIGH,
                )
            )

        self._attach_completeness_to_findings(vulnerabilities)
        self._attach_evidence_to_findings(vulnerabilities, facts)
        self._enforce_evidence_policy(vulnerabilities)
        return vulnerabilities

    def get_last_analysis_info(self) -> Dict[str, Any]:
        """Return metadata from the most recent scan."""
        info: Dict[str, Any] = {
            "metadata": dict(self.last_analysis_metadata),
            "completeness": dict(self.last_completeness),
        }
        # Include analysis type from dataflow graph
        if (
            self.last_facts is not None
            and hasattr(self.last_facts, "dataflow_graph")
            and self.last_facts.dataflow_graph is not None
        ):
            info["type"] = self.last_facts.dataflow_graph.analysis_type
        # Include TASIR statistics
        if self._last_tasir_module is not None:
            info["tasir"] = {
                "functions": len(self._last_tasir_module.functions),
                "blocks": len(self._last_tasir_module.all_blocks()),
                "instructions": len(self._last_tasir_module.all_instructions()),
            }
        if self._last_solver_module is not None:
            info["solver_ir"] = {
                "blocks": len(self._last_solver_module.blocks),
                "instructions": len(self._last_solver_module.instructions),
                "symbols": len(self._last_solver_module.symbols),
                "constraints": len(self._last_solver_module.constraints),
                "obligations": len(self._last_solver_module.obligations),
                "metadata": dict(self._last_solver_module.metadata or {}),
            }
        return info

    def _attach_completeness_to_findings(self, vulnerabilities: List[Vulnerability]) -> None:
        """Attach normalized completeness metadata to each finding."""
        completeness = dict(self.last_completeness)
        for vuln in vulnerabilities:
            existing = vuln.extra if isinstance(vuln.extra, dict) else {}
            if "analysis_completeness" in existing:
                continue
            merged = dict(existing)
            merged["analysis_completeness"] = completeness
            vuln.extra = merged

    def _attach_evidence_to_findings(self, vulnerabilities: List[Vulnerability], facts) -> None:
        """Attach a normalized evidence schema to each finding."""
        graph = getattr(facts, "dataflow_graph", None)
        metadata = {}
        if graph is not None and isinstance(getattr(graph, "analysis_metadata", None), dict):
            metadata = graph.analysis_metadata

        solver_obligations = metadata.get("dynamic_target_solver_obligations", [])
        dynamic_target_count = self.last_completeness.get("dynamic_target_count", 0)
        solver_status = "unresolved_dynamic_targets" if dynamic_target_count > 0 else "resolved_or_not_applicable"

        for vuln in vulnerabilities:
            existing = vuln.extra if isinstance(vuln.extra, dict) else {}
            evidence = existing.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}

            # path_trace: minimal reproducible trace anchor
            if "path_trace" not in evidence:
                inst_idx = getattr(vuln.instruction, "index", None)
                evidence["path_trace"] = [inst_idx] if isinstance(inst_idx, int) else []

            # source_sink_chain: taint edges that end at finding instruction
            if "source_sink_chain" not in evidence:
                sink_idx = getattr(vuln.instruction, "index", None)
                chain = []
                if isinstance(sink_idx, int) and graph is not None:
                    for edge in getattr(graph, "tainted_propagation", []) or []:
                        if (
                            isinstance(edge, (list, tuple))
                            and len(edge) == 2
                            and isinstance(edge[0], int)
                            and isinstance(edge[1], int)
                            and edge[1] == sink_idx
                        ):
                            chain.append({"source": edge[0], "sink": edge[1]})
                evidence["source_sink_chain"] = chain

            # guard_state: whether analysis had completeness limitations
            if "guard_state" not in evidence:
                evidence["guard_state"] = {
                    "analysis_incomplete": bool(self.last_completeness.get("analysis_incomplete")),
                    "reasons": list(self.last_completeness.get("analysis_incomplete_reasons", [])),
                }

            # solver_status: dynamic continuation solving status + obligation count
            if "solver_status" not in evidence:
                evidence["solver_status"] = {
                    "status": solver_status,
                    "obligation_count": len(solver_obligations) if isinstance(solver_obligations, list) else 0,
                }

            merged = dict(existing)
            merged["evidence"] = evidence
            vuln.extra = merged

    def _enforce_evidence_policy(self, vulnerabilities: List[Vulnerability]) -> None:
        """Enforce evidence requirements for medium/high/critical findings."""
        required_fields = ("path_trace", "source_sink_chain", "guard_state", "solver_status")
        gated_severities = {"medium", "high", "critical"}

        for vuln in vulnerabilities:
            severity = str(getattr(vuln, "severity", "") or "").lower()
            if severity not in gated_severities:
                continue
            extra = vuln.extra if isinstance(vuln.extra, dict) else {}
            evidence = extra.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {}

            missing = [field for field in required_fields if field not in evidence]
            has_minimal_trace = bool(evidence.get("path_trace"))
            if not has_minimal_trace and "path_trace" not in missing:
                missing.append("path_trace")

            if not missing:
                policy = {
                    "required": True,
                    "status": "satisfied",
                    "action": "none",
                    "missing_fields": [],
                }
            else:
                # Fail-closed evidence policy: downgrade when required evidence is missing.
                vuln.severity = "low"
                if hasattr(vuln, "confidence") and vuln.confidence in {Confidence.HIGH, Confidence.MEDIUM}:
                    vuln.confidence = Confidence.LOW
                policy = {
                    "required": True,
                    "status": "missing",
                    "action": "downgraded_to_low",
                    "missing_fields": missing,
                }
                if "[evidence downgraded]" not in vuln.message:
                    vuln.message = f"{vuln.message} [evidence downgraded]"

            merged_extra = dict(extra)
            merged_extra["evidence_policy"] = policy
            merged_extra["evidence"] = evidence
            vuln.extra = merged_extra

    def scan_boc_bytes(self, boc_data: bytes) -> List[Vulnerability]:
        """
        Scan BOC data for vulnerabilities.

        Args:
            boc_data: BOC data as bytes

        Returns:
            List of detected vulnerabilities
        """
        cell = parse_boc(boc_data)
        return self.scan_cell(cell)

    def scan_file(self, file_path: str) -> List[Vulnerability]:
        """
        Scan a BOC file for vulnerabilities.

        Supports multiple formats:
        - Binary BOC files (.boc)
        - Base64-encoded BOC files (.b64, .base64)
        - Hex/text-encoded BOC files (.hex, .txt)

        Args:
            file_path: Path to BOC file

        Returns:
            List of detected vulnerabilities
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Invalid BOC file: {file_path}")
        if not path.is_file():
            raise ValueError(f"Invalid BOC file: {file_path} is not a file")

        boc_data = load_boc_file(path)
        return self.scan_boc_bytes(boc_data)


def scan_contract(
    source: str,
    detectors: Optional[List[str]] = None,
    path_sensitive: bool = True,
    analysis_mode: Optional[Union[str, AnalysisMode]] = None,
    path_config: Optional[Dict[str, int]] = None,
    use_tasir: bool = True,
    use_solver_ir: bool = False,
    strict_completeness: bool = STRICT_COMPLETENESS_DEFAULT,
) -> List[Vulnerability]:
    """
    Convenience function to scan a contract.

    Args:
        source: Path to BOC file (binary .boc or text-encoded .b64/.base64/.hex/.txt),
            or a base64/hex string
        detectors: List of detector names to enable (None = all)
        path_sensitive: Enable path-sensitive dataflow analysis
        analysis_mode: Optional analysis mode preset for path-sensitive analysis
        path_config: Optional path-sensitive config overrides
        use_tasir: Use TASIR for enhanced continuation tracking (default: True)
        use_solver_ir: Lower TASIR to Solver IR when TASIR is enabled (default: False)
        strict_completeness: When True, incomplete analysis raises an error

    Returns:
        List of detected vulnerabilities

    Raises:
        ValueError: If detectors include unknown names or none are selected

    Example:
        >>> vulnerabilities = scan_contract('contract.boc')
        >>> for vuln in vulnerabilities:
        ...     print(f"[{vuln.severity}] {vuln.message}")

    Note:
        Use a 0x prefix to force hex interpretation when a string could also
        be a valid file path.
    """
    if not source or not source.strip():
        raise ValueError("Invalid input: source is empty")

    scanner = SecurityScanner(
        path_sensitive=path_sensitive,
        analysis_mode=analysis_mode,
        path_config=path_config,
        use_tasir=use_tasir,
        use_solver_ir=use_solver_ir,
        strict_completeness=strict_completeness,
    )

    _filter_detectors(scanner, detectors)
    boc_data = _resolve_source_to_bytes(source)
    return scanner.scan_boc_bytes(boc_data)
