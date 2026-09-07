"""TL-B structure violation detector

Detects potential TL-B structural violations in slice parsing:

1. **Excessive cell reads**: A slice is read (LDI/LDU/LDREF/LDSLICE etc.) after
   SBITS/SREFS returned 0 or after ENDS was called.
2. **Missing ENDS complement**: ENDS is called but prior reads haven't consumed
   all data (complement to LackEndParseDetector).
3. **Unbalanced ref reads**: More LDREF calls than SREFS indicates on a slice,
   suggesting reads may exceed the cell's reference count.

Detection tiers:
- **TASIR tier**: Walk each function's instructions. Track sequences of cell
  parse operations per slice source. Flag patterns like CTOS -> many loads
  without SBITS/SREFS checks, or LDREF count exceeding SREFS guards.
- **CFG tier**: Use basic block traversal to find paths where cell loads happen
  without prior size checks.
"""
from typing import Dict, List, Set

from ..analyzer.facts import AnalysisFacts, BasicBlock, InstructionFact
from ..config import SLICE_DATA_LOAD_OPCODES, SLICE_REF_LOAD_OPCODES
from .base import Detector
from .cfg_utils import (
    build_instruction_to_block_map,
    traverse_cfg_for_unguarded_sinks,
)
from .results import Confidence, Vulnerability


class TLBStructureViolationDetector(Detector):
    """Detects potential TL-B structural violations: reads from exhausted slices,
    missing size checks, and unbalanced reference reads."""

    name = "tlb_structure_violation"
    category = "security"
    enabled_by_default = True
    default_severity = "medium"
    description = (
        "Detects potential TL-B structural violations: reads from exhausted "
        "slices, missing size checks"
    )
    tags = ("tlb", "slice", "parsing", "cell", "security")

    # Slice creation opcodes
    SLICE_CREATE_OPCODES: Set[str] = {
        "CTOS",
        "LDSLICE",
        "LDSLICEX",
        "PLDSLICE",
        "PLDSLICEX",
    }

    # Data load opcodes - derived from shared config (single source of truth)
    DATA_LOAD_OPCODES: Set[str] = set(SLICE_DATA_LOAD_OPCODES)

    # Reference load opcodes - derived from shared config (single source of truth)
    REF_LOAD_OPCODES: Set[str] = set(SLICE_REF_LOAD_OPCODES)

    # All load opcodes combined
    ALL_LOAD_OPCODES: Set[str] = DATA_LOAD_OPCODES | REF_LOAD_OPCODES

    # Size-check opcodes that guard against reading from an exhausted slice
    SIZE_CHECK_OPCODES: Set[str] = {
        "SBITS",    # Remaining bits count
        "SREFS",    # Remaining refs count
        "SBITREFS", # Both bits and refs
        "SEMPTY",   # Slice empty check
        "SREMPTY",  # No refs remaining
        "SDEMPTY",  # No data remaining
        "SDEPTH",   # Remaining depth
        "SCHKBITS",   # Assert bits remaining
        "SCHKREFS",   # Assert refs remaining
        "SCHKBITREFS",  # Assert bits+refs
        "SCHKBITSQ",  # Quiet assert bits
        "SCHKREFSQ",  # Quiet assert refs
        "SCHKBITREFSQ",  # Quiet assert bits+refs
    }

    # Ref-specific size checks
    REF_CHECK_OPCODES: Set[str] = {
        "SREFS", "SBITREFS", "SREMPTY",
        "SCHKREFS", "SCHKBITREFS",
        "SCHKREFSQ", "SCHKBITREFSQ",
    }

    # ENDS opcode
    ENDS_OPCODES: Set[str] = {"ENDS"}

    # Threshold: number of consecutive loads without a size check before
    # flagging as "excessive reads without guard"
    EXCESSIVE_LOAD_THRESHOLD = 8

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings: List[Vulnerability] = []
        module = self.get_tasir(facts)

        if module is not None:
            # TASIR-based detection (preferred)
            for func in module.functions.values():
                findings.extend(self._check_function_tasir(func, facts))
            global_blocks_by_context: Dict[str, List] = {}
            for block in module.global_blocks:
                global_blocks_by_context.setdefault(block.context_id, []).extend(
                    block.instructions
                )
            for instructions in global_blocks_by_context.values():
                findings.extend(
                    self._check_instruction_sequence(
                        sorted(instructions, key=lambda i: i.index),
                        facts,
                    )
                )
        else:
            # CFG-based fallback
            findings.extend(self._detect_cfg_fallback(facts))

        return findings

    # ------------------------------------------------------------------
    # TASIR tier
    # ------------------------------------------------------------------

    def _check_function_tasir(self, func, facts: AnalysisFacts) -> List[Vulnerability]:
        """Analyze a single TVMFunction for TL-B structure violations."""
        all_insts = list(func.all_instructions())
        return self._check_instruction_sequence(all_insts, facts)

    def _check_instruction_sequence(self, all_insts: List, facts: AnalysisFacts) -> List[Vulnerability]:
        """Analyze one TASIR instruction sequence for TL-B structure violations."""
        findings: List[Vulnerability] = []
        if not all_insts:
            return findings

        findings.extend(self._check_excessive_loads(all_insts, facts))
        findings.extend(self._check_read_after_ends(all_insts, facts))
        findings.extend(self._check_unbalanced_refs(all_insts, facts))

        return findings

    def _check_excessive_loads(
        self, all_insts, facts: AnalysisFacts
    ) -> List[Vulnerability]:
        """Detect long sequences of loads without any size check.

        Walks instructions in order. After a CTOS (slice creation), counts
        consecutive load operations. If count exceeds the threshold without
        an intervening size check, the sequence is flagged.
        """
        findings: List[Vulnerability] = []

        tracking = False
        load_count = 0
        first_load_inst = None
        slice_create_inst = None

        for inst in all_insts:
            opcode = inst.opcode

            if opcode in self.SLICE_CREATE_OPCODES:
                # Starting a new slice parse sequence
                if tracking and load_count >= self.EXCESSIVE_LOAD_THRESHOLD:
                    findings.append(self._make_excessive_loads_finding(
                        slice_create_inst, first_load_inst, load_count, facts,
                    ))
                tracking = True
                load_count = 0
                first_load_inst = None
                slice_create_inst = inst

            elif tracking and opcode in self.SIZE_CHECK_OPCODES:
                # A size check resets the counter -- the developer is guarding
                load_count = 0
                first_load_inst = None

            elif tracking and opcode in self.ENDS_OPCODES:
                # ENDS terminates the parse, reset tracking
                tracking = False
                load_count = 0
                first_load_inst = None

            elif tracking and opcode in self.ALL_LOAD_OPCODES:
                if first_load_inst is None:
                    first_load_inst = inst
                load_count += 1

        # Flush final sequence
        if tracking and load_count >= self.EXCESSIVE_LOAD_THRESHOLD:
            findings.append(self._make_excessive_loads_finding(
                slice_create_inst, first_load_inst, load_count, facts,
            ))

        return findings

    def _check_read_after_ends(
        self, all_insts, facts: AnalysisFacts
    ) -> List[Vulnerability]:
        """Detect load operations that appear after ENDS on the same slice
        parse sequence, or after SBITS/SEMPTY indicated emptiness (heuristic:
        SBITS followed by a zero-branch-then-load pattern).

        This is a lightweight linear scan. It flags the *first* load after
        each ENDS within the same function (since we cannot perfectly track
        which stack slot corresponds to which slice).
        """
        findings: List[Vulnerability] = []

        saw_ends = False
        ends_inst = None

        for inst in all_insts:
            opcode = inst.opcode

            if opcode in self.SLICE_CREATE_OPCODES:
                # New slice creation resets state
                saw_ends = False
                ends_inst = None

            elif opcode in self.ENDS_OPCODES:
                saw_ends = True
                ends_inst = inst

            elif saw_ends and opcode in self.ALL_LOAD_OPCODES:
                # Load after ENDS on a potentially exhausted slice
                original = self._find_original_instruction(facts, inst.index)
                findings.append(self._build_vuln(
                    message=(
                        f"Potential read after ENDS: {opcode} at index "
                        f"{inst.index} follows ENDS at index {ends_inst.index}. "
                        f"Slice may already be consumed."
                    ),
                    instruction=original,
                    severity="high",
                    confidence=Confidence.MEDIUM,
                    remediation=(
                        "Do not read from a slice after calling end_parse() / ENDS. "
                        "If additional data is needed, parse it before ENDS or use "
                        "a separate slice."
                    ),
                    extra={
                        "sub_type": "read_after_ends",
                        "load_opcode": opcode,
                        "load_index": inst.index,
                        "ends_index": ends_inst.index,
                        "detection_method": "tasir",
                    },
                ))
                # Only flag once per ENDS to avoid noise
                saw_ends = False
                ends_inst = None

        return findings

    def _check_unbalanced_refs(
        self, all_insts, facts: AnalysisFacts
    ) -> List[Vulnerability]:
        """Detect sequences with more LDREF calls than SREFS guards.

        For each slice-creation point, count LDREF operations and SREFS
        checks that follow. If ref loads exceed ref checks and total ref
        loads >= 3, flag as a potential unbalanced read.
        """
        findings: List[Vulnerability] = []

        tracking = False
        ref_load_count = 0
        ref_check_count = 0
        slice_create_inst = None
        first_ref_load_inst = None

        for inst in all_insts:
            opcode = inst.opcode

            if opcode in self.SLICE_CREATE_OPCODES:
                # Flush previous sequence
                if tracking:
                    findings.extend(self._maybe_flag_unbalanced_refs(
                        slice_create_inst, first_ref_load_inst,
                        ref_load_count, ref_check_count, facts,
                    ))
                tracking = True
                ref_load_count = 0
                ref_check_count = 0
                slice_create_inst = inst
                first_ref_load_inst = None

            elif tracking and opcode in self.REF_LOAD_OPCODES:
                ref_load_count += 1
                if first_ref_load_inst is None:
                    first_ref_load_inst = inst

            elif tracking and opcode in self.REF_CHECK_OPCODES:
                ref_check_count += 1

            elif tracking and opcode in self.ENDS_OPCODES:
                findings.extend(self._maybe_flag_unbalanced_refs(
                    slice_create_inst, first_ref_load_inst,
                    ref_load_count, ref_check_count, facts,
                ))
                tracking = False

        # Flush final
        if tracking:
            findings.extend(self._maybe_flag_unbalanced_refs(
                slice_create_inst, first_ref_load_inst,
                ref_load_count, ref_check_count, facts,
            ))

        return findings

    # ------------------------------------------------------------------
    # CFG-based fallback
    # ------------------------------------------------------------------

    def _detect_cfg_fallback(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """CFG-based detection when TASIR is not available.

        Uses basic block traversal to find paths where cell loads happen
        without prior size checks.
        """
        findings: List[Vulnerability] = []

        if not facts.basic_blocks:
            return findings

        block_map: Dict[int, BasicBlock] = {b.id: b for b in facts.basic_blocks}
        entry_ids = facts.entry_block_ids()
        if not entry_ids:
            return findings

        instr_index: Dict[int, "InstructionFact"] = {
            inst.index: inst for inst in facts.instructions
        }

        # Collect indices
        slice_create_indices: List[int] = []
        load_indices: Set[int] = set()
        size_check_indices: Set[int] = set()

        for inst in facts.instructions:
            if inst.opcode in self.SLICE_CREATE_OPCODES:
                slice_create_indices.append(inst.index)
            if inst.opcode in self.ALL_LOAD_OPCODES:
                load_indices.add(inst.index)
            if inst.opcode in self.SIZE_CHECK_OPCODES:
                size_check_indices.add(inst.index)

        if not slice_create_indices or not load_indices:
            return findings

        # For each slice creation, check if loads are reachable without
        # passing through a size check
        instr_to_block = build_instruction_to_block_map(facts.basic_blocks)

        for create_idx in slice_create_indices:
            start_block = instr_to_block.get(create_idx)
            if start_block is None:
                continue

            result = traverse_cfg_for_unguarded_sinks(
                block_map=block_map,
                start_block_id=start_block,
                start_from_idx=create_idx,
                guard_indices=size_check_indices,
                sink_indices=load_indices,
            )

            # If many unguarded loads found, flag
            if len(result.flagged_indices) >= self.EXCESSIVE_LOAD_THRESHOLD:
                create_inst = instr_index.get(create_idx)
                if create_inst is None:
                    continue

                findings.append(self._build_vuln(
                    message=(
                        f"Potential TL-B structure violation: {len(result.flagged_indices)} "
                        f"cell load operations reachable from {create_inst.opcode} "
                        f"at index {create_idx} without size checks on any CFG path"
                    ),
                    instruction=create_inst,
                    severity="medium",
                    confidence=Confidence.LOW,
                    remediation=(
                        "Add SBITS/SREFS checks before reading from a slice to ensure "
                        "sufficient data remains. Use SCHKBITS/SCHKREFS for assertion-style "
                        "guards."
                    ),
                    extra={
                        "sub_type": "excessive_loads_no_guard",
                        "slice_create_opcode": create_inst.opcode,
                        "slice_create_index": create_idx,
                        "unguarded_load_count": len(result.flagged_indices),
                        "truncated": result.truncated,
                        "has_unknown_successors": result.encountered_unknown,
                        "detection_method": "cfg",
                    },
                ))

        return findings

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _make_excessive_loads_finding(
        self, slice_create_inst, first_load_inst, load_count: int,
        facts: AnalysisFacts,
    ) -> Vulnerability:
        """Build a finding for excessive loads without size checks."""
        report_inst = first_load_inst or slice_create_inst
        original = self._find_original_instruction(facts, report_inst.index)
        create_opcode = slice_create_inst.opcode if slice_create_inst else "unknown"
        create_index = slice_create_inst.index if slice_create_inst else -1

        return self._build_vuln(
            message=(
                f"Potential TL-B structure violation: {load_count} consecutive "
                f"cell load operations after {create_opcode} at index {create_index} "
                f"without SBITS/SREFS size checks"
            ),
            instruction=original,
            severity="medium",
            confidence=Confidence.MEDIUM,
            remediation=(
                "Insert SBITS or SREFS checks between load operations to verify "
                "sufficient data remains in the slice. Consider using SCHKBITS "
                "for an assertion-style guard."
            ),
            extra={
                "sub_type": "excessive_loads_no_guard",
                "slice_create_opcode": create_opcode,
                "slice_create_index": create_index,
                "load_count": load_count,
                "detection_method": "tasir",
            },
        )

    def _maybe_flag_unbalanced_refs(
        self, slice_create_inst, first_ref_load_inst,
        ref_load_count: int, ref_check_count: int,
        facts: AnalysisFacts,
    ) -> List[Vulnerability]:
        """Conditionally create a finding for unbalanced ref loads."""
        # Only flag if there are at least 3 ref loads and no ref checks
        if ref_load_count < 3 or ref_check_count >= ref_load_count:
            return []

        report_inst = first_ref_load_inst or slice_create_inst
        if report_inst is None:
            return []
        original = self._find_original_instruction(facts, report_inst.index)
        create_opcode = slice_create_inst.opcode if slice_create_inst else "unknown"
        create_index = slice_create_inst.index if slice_create_inst else -1

        return [self._build_vuln(
            message=(
                f"Potential unbalanced ref reads: {ref_load_count} LDREF operations "
                f"after {create_opcode} at index {create_index} but only "
                f"{ref_check_count} SREFS check(s). Cell refs may be exhausted."
            ),
            instruction=original,
            severity="medium",
            confidence=Confidence.LOW,
            remediation=(
                "Add SREFS or SCHKREFS checks before LDREF operations to ensure "
                "sufficient references remain in the slice. A TON cell can hold "
                "at most 4 references."
            ),
            extra={
                "sub_type": "unbalanced_ref_reads",
                "slice_create_opcode": create_opcode,
                "slice_create_index": create_index,
                "ref_load_count": ref_load_count,
                "ref_check_count": ref_check_count,
                "detection_method": "tasir",
            },
        )]

    def _find_original_instruction(self, facts: AnalysisFacts, index: int):
        """Find the original InstructionFact from facts by index."""
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
