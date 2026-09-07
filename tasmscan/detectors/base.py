"""Base class for all detectors

Unified detection API
=====================

The recommended entry point for all detectors is the `detect(facts)` method.
This method provides access to both legacy analysis data and the new TASIR
module through `facts.metadata["tasir_module"]`.

For detectors that need TVMModule, use:
    module = self.get_tasir(facts)  # Returns TVMModule or None

CFG Analysis Guidelines for Detector Authors (CFG handling)
=====================================================

When implementing detectors that traverse the Control Flow Graph (CFG),
special care must be taken to handle blocks with unknown successors
(has_unknown_successor=True). These blocks indicate unresolved jump targets
such as computed continuations or indirect calls.

Conservative Handling Strategy:
------------------------------
1. **Detection Context**: If the analysis is looking for MISSING checks/guards,
   unknown successors should be treated as potentially UNGUARDED paths.
   Example: no_accept detector adds a low-severity finding when encountering
   unknown successors without ACCEPT seen.

2. **Guard Context**: If the analysis is looking for PRESENT checks/guards
   to validate safety, unknown successors should NOT be assumed to have
   the guard. Continue analysis conservatively.

3. **Sink Reachability**: When checking if sinks are reachable without guards,
   unknown successors represent paths that MIGHT reach sinks. Flag these
   with appropriate confidence levels (typically low/medium).

4. **Reporting**: When analysis encounters unknown successors in security-
   sensitive contexts, emit an informational finding to alert users that
   manual review may be needed.

Implementation Patterns:
-----------------------
- Check `block.has_unknown_successor` before processing successors
- Use `traverse_cfg_for_unguarded_sinks` with `flag_unknown_unguarded=True`
  for consistent handling across detectors
- Include `has_unknown_successors` in finding metadata for transparency

See `cfg_utils.py` for reusable CFG traversal utilities that implement
these patterns consistently.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..ir.dataflow.types import DataFlowGraph
    from ..ir.tasir_types import InstructionKind, TVMModule

from ..analyzer.facts import AnalysisFacts
from ..config import MIN_COMPLEXITY_THRESHOLD as DEFAULT_MIN_COMPLEXITY_THRESHOLD
from .results import Confidence, Vulnerability


class Detector(ABC):
    """
    Base interface for all detectors.

    CFG Analysis Note (CFG handling):
    When traversing the CFG, detectors must handle blocks with
    `has_unknown_successor=True` conservatively. See module docstring
    for detailed guidelines on handling unknown successors.

    Design Decision - Independent Detectors (independent detectors):
    ================================================
    Each detector operates independently without sharing intermediate results
    with other detectors. This design was chosen for several reasons:

    Benefits:
    - Simplicity: Detectors can be developed, tested, and maintained in isolation
    - Parallelization: Independent detectors can run concurrently without locks
    - Robustness: A bug in one detector cannot corrupt another detector's analysis
    - Modularity: Easy to add/remove detectors without impacting others

    Trade-offs:
    - Some redundant computation (e.g., multiple detectors may traverse CFG)
    - Cannot leverage findings from one detector to improve another's precision
    - Memory overhead from duplicated intermediate data structures

    If cross-detector information sharing becomes necessary, consider:
    1. Adding a shared AnalysisCache passed to all detectors
    2. Running detectors in phases where later phases can read earlier results
    3. Using AnalysisFacts as the shared state (current approach for basics)
    """

    name: str = "abstract"
    default_severity: str = "medium"
    description: str = ""
    experimental: bool = False
    MIN_COMPLEXITY_THRESHOLD = DEFAULT_MIN_COMPLEXITY_THRESHOLD  # Class-level default

    # Tags for categorization (e.g., ["critical", "gas", "reentrancy", "ton-specific"])
    tags: tuple = ()

    # Detector classification attributes
    category: str = "security"  # "security" or "code_quality"
    enabled_by_default: bool = True

    # Reserved field for future detector dependency support (future dependency support).
    # Currently unused - detectors run independently (see detector independence above).
    # Intended for future use cases like:
    # - Running detectors in topological order based on dependencies
    # - Passing results from prerequisite detectors to dependent ones
    # - Skipping detectors whose dependencies failed
    depends_on: tuple = ()

    def __init__(self, options: Optional[Dict[str, Any]] = None, min_complexity_threshold: Optional[int] = None):
        self.options = options or {}
        if min_complexity_threshold is not None:
            self.min_complexity_threshold = min_complexity_threshold
        else:
            self.min_complexity_threshold = self.MIN_COMPLEXITY_THRESHOLD

    def _is_trivial_contract(self, facts: AnalysisFacts) -> bool:
        """Check if contract is too simple."""
        return len(facts.instructions) < self.min_complexity_threshold

    @abstractmethod
    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """
        Run detector and return findings.

        This is the unified entry point for all detectors.

        For detectors that need access to the TVMModule, use:
            module = self.get_tasir(facts)

        The TVMModule (if available) is automatically stored in
        `facts.metadata["tasir_module"]` during analysis pipeline execution.

        Note: Detectors should prefer using ``self.get_tasir(facts)``
        for analysis data access. The TVMModule now provides delegating query
        methods that mirror AnalysisFacts: ``events_of()``, ``opcodes()``,
        ``entry_block_ids()``, ``basic_blocks``, ``cfg_edges``,
        ``stack_states``, ``instructions``, and ``summaries``.

        Args:
            facts: Analysis facts containing instructions, CFG, and metadata.
                   Access TVMModule via self.get_tasir(facts) if needed.

        Returns:
            List of detected vulnerabilities
        """
        pass

    def _build_vuln(
        self,
        message: str,
        instruction,
        severity: Optional[str] = None,
        remediation: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        confidence: Confidence = Confidence.MEDIUM,
    ) -> Vulnerability:
        return Vulnerability(
            detector=self.name,
            severity=severity or self.default_severity,
            message=message,
            instruction=instruction,
            remediation=remediation,
            extra=extra,
            confidence=confidence,
        )

    def get_tasir(self, facts: AnalysisFacts) -> Optional["TVMModule"]:
        """
        Get TASIR module from analysis facts if available.

        The TVMModule provides richer semantic information including:
        - InstructionKind classification for semantic matching
        - ContinuationDescriptor with SaveList for cross-continuation analysis
        - TVMBasicBlock with typed CFG edges

        Returns:
            TVMModule if TASIR was built, None otherwise
        """
        return facts.metadata.get("tasir_module")

    def get_dataflow(self, facts: AnalysisFacts) -> Optional["DataFlowGraph"]:
        """Get DataFlowGraph, preferring TVMModule's integrated copy."""
        module = self.get_tasir(facts)
        if module and module.dataflow_graph:
            return module.dataflow_graph
        return facts.dataflow_graph

    def get_instructions_by_kind(self, facts: AnalysisFacts, kind: "InstructionKind") -> List[Any]:
        """
        Get all TASIR instructions matching a specific InstructionKind.

        Args:
            facts: Analysis facts with optional tasir_module
            kind: The InstructionKind to filter by

        Returns:
            List of TVMInstruction matching the kind, empty if TASIR not available
        """
        module = self.get_tasir(facts)
        if module is None:
            return []
        return [inst for inst in module.all_instructions() if inst.kind == kind]
