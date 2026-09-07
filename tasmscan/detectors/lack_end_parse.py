"""Lack of end_parse() detector

Detects slices that are read but not validated with end_parse().

In TON, when parsing a slice (e.g., from a cell or message), developers should
call end_parse() (ENDS opcode) after reading to ensure no unexpected data remains.
Failing to do so can lead to:
- Accepting malformed messages with extra payload
- Missing data manipulation attacks
- Silent parsing errors
"""
from typing import Dict, List, Set

from ..analyzer.facts import AnalysisFacts
from ..config import SLICE_DATA_LOAD_OPCODES, SLICE_REF_LOAD_OPCODES
from .base import Detector
from .results import Confidence, Vulnerability


class LackEndParseDetector(Detector):
    """Detects potential missing end_parse() calls on slices."""

    name = "lack_end_parse"
    category = "security"
    enabled_by_default = True
    default_severity = "medium"
    description = "Detects slices that are read but not validated with end_parse()"
    tags = ["slice", "parsing", "validation"]

    # Slice creation opcodes - these produce slices that should be validated
    SLICE_CREATE_OPCODES: Set[str] = {
        "CTOS",       # Cell to slice
        "LDSLICE",    # Load slice from slice
        "LDSLICEX",   # Load slice (variable length)
        "PLDSLICE",   # Preload slice
        "PLDSLICEX",  # Preload slice (variable length)
    }

    # Slice read opcodes - derived from shared SLICE_DATA_LOAD_OPCODES + SLICE_REF_LOAD_OPCODES
    SLICE_READ_OPCODES: Set[str] = set(SLICE_DATA_LOAD_OPCODES) | set(SLICE_REF_LOAD_OPCODES)

    # Validation opcodes - these validate that slice is empty
    ENDS_OPCODES: Set[str] = {
        "ENDS",      # end_parse - throws if slice is not empty
        "SDEPTH",    # Get remaining depth - can be used for validation
    }

    # Opcodes that consume/terminate a slice
    SLICE_CONSUME_OPCODES: Set[str] = {
        "ENDS",      # Validates and consumes
        "SDEPTH",    # Gets depth (but doesn't consume)
        "SEMPTY",    # Checks if empty
        "SREMPTY",   # Checks if no refs
        "SDEMPTY",   # Checks if no data
    }

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        module = self.get_tasir(facts)

        if module is not None:
            # Use TASIR-based detection
            for func in module.functions.values():
                findings.extend(self._check_function(func, facts))
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
            # Fallback to simple opcode-based detection
            findings.extend(self._detect_simple(facts))

        return findings

    def _check_function(self, func, facts: AnalysisFacts) -> List[Vulnerability]:
        """Check a single function for missing end_parse calls using TASIR."""
        # Collect all instructions in this function
        all_insts = list(func.all_instructions())
        return self._check_instruction_sequence(all_insts, facts)

    def _check_instruction_sequence(self, all_insts: List, facts: AnalysisFacts) -> List[Vulnerability]:
        """Check one TASIR instruction sequence for missing end_parse calls."""
        findings = []
        if not all_insts:
            return findings

        # Find slice creation points
        slice_creates = [
            inst for inst in all_insts
            if inst.opcode in self.SLICE_CREATE_OPCODES
        ]

        # Find ENDS calls
        ends_calls = {
            inst.index for inst in all_insts
            if inst.opcode in self.ENDS_OPCODES
        }

        # Find slice read operations
        slice_reads = [
            inst for inst in all_insts
            if inst.opcode in self.SLICE_READ_OPCODES
        ]

        # Heuristic: report if slice creates exceed ENDS calls (some slices unchecked)
        # Original: only report when ENDS=0. Now also report when CTOS > ENDS.
        if slice_creates and slice_reads and (not ends_calls or len(slice_creates) > len(ends_calls)):
            # Report at the first slice creation point
            first_create = slice_creates[0]

            # Find the original instruction for location info
            instruction = self._find_original_instruction(facts, first_create.index)

            findings.append(self._build_vuln(
                message=(
                    f"Potential missing end_parse(): slice created by {first_create.opcode} "
                    f"at index {first_create.index}, {len(slice_reads)} read operation(s) found, "
                    f"but no ENDS call in function"
                ),
                instruction=instruction,
                remediation=(
                    "Call end_parse() (ENDS) after reading slice to validate no extra data. "
                    "Example: slice cs = in_msg.begin_parse(); int x = cs~load_uint(32); cs.end_parse();"
                ),
                severity="medium",
                confidence=Confidence.MEDIUM,
                extra={
                    "slice_create_opcode": first_create.opcode,
                    "slice_create_index": first_create.index,
                    "read_count": len(slice_reads),
                    "detection_method": "tasir",
                },
            ))

        return findings

    def _detect_simple(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """Simple opcode-based detection without TASIR (fallback)."""
        findings = []

        instructions = facts.instructions

        # Check if there are slice creation, read, and ENDS operations
        slice_create_insts = [
            inst for inst in instructions
            if inst.opcode in self.SLICE_CREATE_OPCODES
        ]

        slice_read_insts = [
            inst for inst in instructions
            if inst.opcode in self.SLICE_READ_OPCODES
        ]

        has_ends = any(
            inst.opcode in self.ENDS_OPCODES
            for inst in instructions
        )

        # If slice creates exceed ENDS (some slices not validated)
        ends_count = sum(1 for inst in instructions if inst.opcode in self.ENDS_OPCODES)
        if slice_create_insts and slice_read_insts and (not has_ends or len(slice_create_insts) > ends_count):
            # Report at the first slice creation
            first_create = slice_create_insts[0]

            findings.append(self._build_vuln(
                message=(
                    f"Potential missing end_parse(): slice created by {first_create.opcode}, "
                    f"{len(slice_read_insts)} read operation(s) found, but no ENDS call"
                ),
                instruction=first_create,
                remediation=(
                    "Call end_parse() (ENDS) after reading slice to validate no extra data. "
                    "Example: slice cs = in_msg.begin_parse(); int x = cs~load_uint(32); cs.end_parse();"
                ),
                severity="medium",
                confidence=Confidence.MEDIUM,
                extra={
                    "slice_create_opcode": first_create.opcode,
                    "slice_create_index": first_create.index,
                    "read_count": len(slice_read_insts),
                    "detection_method": "simple",
                },
            ))

        return findings

    def _find_original_instruction(self, facts: AnalysisFacts, index: int):
        """Find the original instruction from facts by index."""
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        # Fallback: return first instruction if not found
        return facts.instructions[0] if facts.instructions else None
