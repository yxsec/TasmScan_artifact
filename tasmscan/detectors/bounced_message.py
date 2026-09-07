"""Bounced message detectors (Critical)"""
from typing import Dict, List, Optional

from ..analyzer.facts import AnalysisFacts
from ..config import BOUNCED_SEARCH_RANGE, MAX_SEARCH_ITERATIONS
from ..ir.tasir_types import SemanticLabel
from .base import Detector
from .cfg_utils import GUARD_OPCODES, build_instruction_to_block_map, traverse_cfg_for_specific_guard
from .results import Confidence, Vulnerability


class BouncedMessageDetector(Detector):
    """Critical: Detects missing bounced flag check."""

    name = "bounced_message"
    category = "security"
    default_severity = "critical"
    description = "Checks if bounced flag is validated in recv_internal"

    # Default search range for message header parsing, configurable via options
    DEFAULT_SEARCH_RANGE = BOUNCED_SEARCH_RANGE

    # Lookahead window for bit extraction pattern matching
    # Value of 8 covers typical FunC/Tact patterns:
    # LDU 4 -> DROP/NIP -> PUSHINT -> AND -> ... (up to 6-7 instructions)
    # Plus safety margin for stack shuffles between operations
    BIT_EXTRACTION_LOOKAHEAD = 8

    # Lookahead window for guard instruction search after bit extraction.
    # In practice many compiler-generated recv_internal paths insert
    # message-address parsing and validation between `flags & 1` and the
    # eventual THROWIFNOT/IF guard, so the window needs to cover those steps.
    GUARD_SEARCH_LOOKAHEAD = 10

    # In TON message format, the bounced flag is bit 31 in the 32-bit flags field
    # When loading 32 bits at once, AND with this mask extracts the bounced flag
    BOUNCED_FLAG_MASK_32BIT = 0x80000000

    # Common boundary values used in GTINT/LESSINT/LEQINT/GEQINT checks:
    # 0: for GTINT 0 (value > 0, i.e., non-negative check)
    # 1: for LESSINT 1 (value < 1, i.e., zero check)
    # -1: for edge cases with signed comparisons
    GUARD_BOUNDARY_VALUES = frozenset({0, 1, -1})

    # ==========================================================================
    # ==========================================================================
    # Each pattern defines opcodes and conditions for detecting bounced flag extraction.
    # Users can extend this via options["additional_patterns"] in __init__.
    #
    # Pattern structure:
    #   "pattern_name": {
    #       "description": str,           # Human-readable explanation
    #       "opcodes": List[str],         # Opcodes that match this pattern
    #       "requires_prev_push": int,    # Optional: requires PUSHINT with this value before
    #       "arg_value": int,             # Optional: instruction argument must equal this
    #       "arg_in_set": Set[int],       # Optional: instruction argument must be in this set
    #       "chain_next": List[str],      # Optional: next instruction must be one of these
    #   }
    #
    # To extend patterns, pass options={"additional_patterns": {...}} to __init__.
    # Example:
    #   detector = BouncedMessageDetector(options={
    #       "additional_patterns": {
    #           "custom_pattern": {
    #               "description": "Custom extraction pattern",
    #               "opcodes": ["CUSTOM_OP"],
    #               "arg_value": 42,
    #           }
    #       }
    #   })
    DEFAULT_BOUNCED_PATTERNS = {
        "push_one_and": {
            "description": "PUSHINT 1 + AND pattern (mask with 1 to extract bit 0)",
            "opcodes": ["AND"],
            "requires_prev_push": 1,
        },
        "modpow2_1": {
            "description": "MODPOW2 1 pattern (mod 2 extracts bit 0)",
            "opcodes": ["MODPOW2"],
            "arg_value": 1,
        },
        "rshift_3": {
            "description": "RSHIFT 3 pattern (extract bit 3 from 4-bit flags = bounced)",
            "opcodes": ["RSHIFT"],
            "arg_value": 3,
        },
        "push_three_rshift": {
            "description": "PUSHINT 3 + RSHIFT pattern (non-immediate RSHIFT)",
            "opcodes": ["RSHIFT"],
            "requires_prev_push": 3,
        },
        "pldu_1": {
            "description": "PLDU 1 pattern (extract single bit directly)",
            "opcodes": ["PLDU"],
            "arg_value": 1,
        },
        "booleval": {
            "description": "BOOLEVAL pattern (convert to boolean 0/-1)",
            "opcodes": ["BOOLEVAL"],
        },
        "zero_comparison": {
            "description": "Zero/non-zero comparison patterns",
            "opcodes": ["ISZERO", "EQINT", "NEQINT"],
        },
        "boundary_comparison": {
            "description": "GTINT/LESSINT/LEQINT/GEQINT with boundary values",
            "opcodes": ["GTINT", "LESSINT", "LEQINT", "GEQINT"],
            "arg_in_set": GUARD_BOUNDARY_VALUES,
        },
        "chained_shift": {
            "description": "LSHIFT/RSHIFT chained bit operations",
            "opcodes": ["LSHIFT", "RSHIFT"],
            "chain_next": ["LSHIFT", "RSHIFT"],
        },
    }

    # Similar structure to DEFAULT_BOUNCED_PATTERNS but for 32-bit flag loads.
    # The "requires_prev_push_mask" flag indicates checking for BOUNCED_FLAG_MASK_32BIT.
    DEFAULT_BOUNCED_PATTERNS_32BIT = {
        "push_mask_and": {
            "description": "PUSHINT 0x80000000 + AND pattern (extract bit 31)",
            "opcodes": ["AND"],
            "requires_prev_push_mask": True,  # Special: checks for BOUNCED_FLAG_MASK_32BIT
        },
        "rshift_31": {
            "description": "RSHIFT 31 pattern (extract highest bit from 32-bit value)",
            "opcodes": ["RSHIFT"],
            "arg_value": 31,
        },
    }

    # Opcodes that reset PUSHINT tracking (not stack-preserving)
    PUSHINT_RESET_EXCLUDE = frozenset({
        "DUP", "OVER", "SWAP", "ROT", "DUP2", "OVER2"
    })

    # Opcodes that signal unrelated operations (stop pattern search)
    UNRELATED_OPCODES = frozenset({
        "ADD", "MUL", "SUB", "NEWC", "ENDC", "STREF", "LDREF", "SENDRAWMSG"
    })

    def __init__(self, options=None, **kwargs):
        super().__init__(options, **kwargs)
        # Allow override via options
        self.search_range = self.options.get("search_range", self.DEFAULT_SEARCH_RANGE)

        self.bounced_patterns = dict(self.DEFAULT_BOUNCED_PATTERNS)
        if self.options.get("additional_patterns"):
            self.bounced_patterns.update(self.options["additional_patterns"])

        self.bounced_patterns_32bit = dict(self.DEFAULT_BOUNCED_PATTERNS_32BIT)
        if self.options.get("additional_patterns_32bit"):
            self.bounced_patterns_32bit.update(self.options["additional_patterns_32bit"])

    @staticmethod
    def _is_pushint_like(opcode: str) -> bool:
        """Recognize PUSHINT immediate specializations emitted by the disassembler."""
        return opcode == "PUSHPOW2" or opcode.startswith("PUSHINT")

    @classmethod
    def _preserves_prev_push(cls, opcode: str) -> bool:
        """Opcodes that do not invalidate the previous pushed immediate."""
        return cls._is_pushint_like(opcode) or opcode in cls.PUSHINT_RESET_EXCLUDE

    @staticmethod
    def _instruction_scope_indices(
        module,
        facts: Optional[AnalysisFacts] = None,
    ) -> Optional[set[int]]:
        """Prefer recv_internal scope, extended with CFG-reachable continuation code."""
        if module is None:
            return None
        recv_internal = module.get_recv_internal()
        if recv_internal is None:
            return None
        scoped = {inst.index for inst in recv_internal.all_instructions()}
        if not scoped:
            return None
        if facts is None or not facts.cfg_edges:
            return scoped

        adjacency: Dict[int, List[int]] = {}
        for edge in facts.cfg_edges:
            if edge.target is None:
                continue
            adjacency.setdefault(edge.source, []).append(edge.target)

        reachable: set[int] = set()
        stack = list(scoped)
        while stack:
            idx = stack.pop()
            if idx in reachable:
                continue
            reachable.add(idx)
            for nxt in adjacency.get(idx, []):
                if nxt not in reachable:
                    stack.append(nxt)

        return reachable or scoped

    @staticmethod
    def _scoped_fact_instructions(
        facts: AnalysisFacts,
        scope_indices: Optional[set[int]],
    ) -> List:
        """Return facts instructions limited to scope_indices when provided."""
        if scope_indices is None:
            return list(facts.instructions)
        return [inst for inst in facts.instructions if inst.index in scope_indices]

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        module = self.get_tasir(facts)
        scope_indices = self._instruction_scope_indices(module, facts)
        instructions = self._scoped_fact_instructions(facts, scope_indices)

        # Fallback to all instructions when TASIR scoping is too restrictive
        # (e.g. recv_internal has very few own instructions, delegates to continuations)
        if len(instructions) < 2 and scope_indices is not None:
            scope_indices = None
            instructions = self._scoped_fact_instructions(facts, scope_indices)

        # Boundary check: skip detection for contracts that are too short
        if len(instructions) < 2:
            return []
        check_range = min(self.search_range, len(instructions))

        # Try TASIR-based detection first
        tasir_result = self._detect_with_tasir(facts, instructions, scope_indices)
        if tasir_result is not None:
            if tasir_result:
                return tasir_result
            sends = [inst for inst in facts.instructions
                     if inst.opcode in ("SENDRAWMSG", "SENDMSG")]
            if sends:
                return [
                    self._build_vuln(
                        message=f"Contract sends {len(sends)} message(s) but "
                                f"bounced handler does not check specific "
                                f"opcodes — may miss bounced responses",
                        instruction=instructions[0] if instructions else None,
                        severity="medium",
                        confidence=Confidence.LOW,
                        remediation="Ensure bounced messages are properly "
                                    "handled for all sent message types."
                    )
                ]
            return tasir_result

        # Quick check via SemanticLabel
        if module and module.has_pattern(SemanticLabel.BOUNCED_CHECK):
            # Contract has bounced check pattern via semantic label, likely OK
            # But still run detailed analysis to verify completeness
            pass

        # Fallback to traditional opcode-based detection
        has_bounced_check = False

        # Check first N instructions (message header parsing is typically at the start)
        for i in range(check_range):
            inst1 = instructions[i]

            # Pattern 1: LDU 4 / PLDU 4 followed by bit extraction
            if inst1.opcode in {"LDU", "PLDU"}:
                if inst1.arguments and len(inst1.arguments) > 0:
                    arg = inst1.arguments[0]
                    if hasattr(arg, 'value') and arg.value == 4:
                        if self._check_bounced_extraction(instructions, i):
                            has_bounced_check = True
                            break

            # Pattern 2: LDU 32 + AND BOUNCED_FLAG_MASK_32BIT (flags + more bits loaded together)
            if inst1.opcode in {"LDU", "PLDU"}:
                if inst1.arguments and len(inst1.arguments) > 0:
                    arg = inst1.arguments[0]
                    if hasattr(arg, 'value') and arg.value == 32:
                        if self._check_bounced_extraction_32bit(instructions, i):
                            has_bounced_check = True
                            break

        if not has_bounced_check:
            return [
                self._build_vuln(
                    message="Contract does not check bounced flag in message. "
                            "This can lead to incorrect handling of failed transaction bounces.",
                    instruction=instructions[0] if instructions else None,
                    severity="critical",
                    remediation="Add bounced flag check at the start of recv_internal: "
                                "slice cs = in_msg_full.begin_parse(); "
                                "int flags = cs~load_uint(4); "
                                "if (flags & 1) { return(); }"
                )
            ]

        # Contract has bounced check — verify the handler is not trivially incomplete.
        # If the contract sends messages but the bounced handler lacks op-specific
        # dispatch or a proper return/throw, flag as incomplete.
        sends = [inst for inst in facts.instructions
                 if inst.opcode in ("SENDRAWMSG", "SENDMSG")]
        if sends:
            incomplete = self._check_incomplete_bounced_handler(
                facts.instructions, check_range)
            if incomplete:
                return [
                    self._build_vuln(
                        message="Bounced flag is loaded but not checked with a guard. "
                                "Send operations may execute on bounced message path.",
                        instruction=instructions[0] if instructions else None,
                        severity="medium",
                        confidence=Confidence.MEDIUM,
                        remediation="Ensure bounced messages are properly handled "
                                    "before executing send operations."
                    )
                ]

        return []

    def _check_incomplete_bounced_handler(self, instructions, search_range):
        """Check if the bounced handler after the flag check is incomplete.

        Returns True if:
        - There is a bounced flag extraction (AND/MODPOW2) followed by a guard
        - But the guard branch does NOT contain: return, throw, or op-comparison
        """
        GUARD_OPS = {"IF", "IFNOT", "IFJMP", "IFNOTJMP", "THROWIF",
                     "THROWIFNOT", "IFELSE"}
        RETURN_OPS = {"RET", "RETALT", "RETFALSE", "RETBOOL"}
        THROW_OPS = {"THROW", "THROWARG"}
        OP_CHECK_OPS = {"EQUAL", "THROWIFNOT", "THROWIF"}

        # Find bounced check position
        bounced_pos = -1
        for i, inst in enumerate(instructions[:search_range + 20]):
            if inst.opcode in ("AND", "MODPOW2"):
                bounced_pos = i
                break

        if bounced_pos < 0:
            return False

        # Find guard after bounced check
        guard_pos = -1
        for i in range(bounced_pos + 1,
                       min(bounced_pos + 10, len(instructions))):
            if instructions[i].opcode in GUARD_OPS:
                guard_pos = i
                break

        if guard_pos < 0:
            # Bounced flag extracted but no guard at all — incomplete
            return True

        # Check what follows the guard (the bounced branch, ~20 instructions)
        handler_section = instructions[guard_pos + 1:guard_pos + 20]
        has_return = any(inst.opcode in RETURN_OPS for inst in handler_section)
        has_throw = any(inst.opcode in THROW_OPS for inst in handler_section)
        has_op_check = any(inst.opcode in OP_CHECK_OPS for inst in handler_section)

        # If handler has return/throw/op-check, it's at least minimally complete
        if has_return or has_throw or has_op_check:
            return False

        # No return, no throw, no op check — handler is incomplete
        return True

    def _detect_with_tasir(
        self,
        facts: AnalysisFacts,
        instructions: Optional[List] = None,
        scope_indices: Optional[set[int]] = None,
    ) -> Optional[List[Vulnerability]]:
        """
        Attempt TASIR-based detection of bounced flag check.

        Uses InstructionKind semantic classification:
        - CELL_LOAD: Identifies load operations (LDU, PLDU, etc.)
        - BRANCH_CONDITIONAL: Identifies guard instructions (IF, IFNOT, etc.)

        Returns:
            List of findings if TASIR detection is conclusive, None to signal fallback.
            Empty list means bounced check was found via TASIR.

        Note: Bounced flag detection is pattern-based, so TASIR mainly provides
        semantic classification benefit. The detailed bit extraction patterns
        still need the existing opcode-level analysis for full coverage.
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        from ..ir.tasir_types import InstructionKind

        if instructions is None:
            instructions = self._scoped_fact_instructions(facts, scope_indices)
        if not instructions:
            return None

        check_range = min(self.search_range, len(instructions))
        index_to_pos = {inst.index: pos for pos, inst in enumerate(instructions)}

        # Get all TASIR instructions
        tasir_instructions = module.all_instructions()
        if scope_indices is not None:
            tasir_instructions = [
                inst for inst in tasir_instructions if inst.index in scope_indices
            ]
        tasir_by_pos = {
            index_to_pos[inst.index]: inst
            for inst in tasir_instructions
            if inst.index in index_to_pos
        }

        # Find CELL_LOAD instructions in the search range
        cell_loads_in_range = [
            (pos, inst)
            for pos, inst in sorted(tasir_by_pos.items())
            if inst.kind == InstructionKind.CELL_LOAD and pos < check_range
        ]

        # Check for LDU 4 / LDU 32 patterns among CELL_LOAD instructions
        for pos, tasir_inst in cell_loads_in_range:
            if tasir_inst.opcode not in {"LDU", "PLDU"}:
                continue

            # Get the original instruction to check arguments
            orig_inst = instructions[pos]
            if not orig_inst.arguments or len(orig_inst.arguments) == 0:
                continue
            arg = orig_inst.arguments[0]
            if not hasattr(arg, 'value'):
                continue

            load_bits = arg.value

            # Check for bounced flag load (4-bit or 32-bit)
            if load_bits == 4:
                if self._check_bounced_extraction_tasir(
                    instructions, tasir_by_pos, pos
                ):
                    return []  # Bounced check found
            elif load_bits == 32:
                if self._check_bounced_extraction_32bit_tasir(
                    instructions, tasir_by_pos, pos
                ):
                    return []  # Bounced check found

        # TASIR analysis found load operations but no bounced check pattern
        # Return None to let the fallback handle edge cases not covered by TASIR
        return None

    def _check_bounced_extraction_tasir(
        self,
        instructions: List,
        tasir_by_pos: Dict[int, object],
        start_pos: int,
    ) -> bool:
        """
        Check for bounced flag extraction using TASIR semantic classification.

        Looks for bit extraction followed by BRANCH_CONDITIONAL guard.
        Falls back to traditional opcode check for detailed pattern matching.
        """
        from ..ir.tasir_types import InstructionKind

        max_lookahead = min(start_pos + self.BIT_EXTRACTION_LOOKAHEAD, len(instructions))

        # Look for bit extraction followed by conditional branch
        found_extraction = False

        for j in range(start_pos + 1, max_lookahead):
            tasir_inst = tasir_by_pos.get(j)
            if tasir_inst is None:
                # Fall back to opcode check for this instruction
                inst = instructions[j]
                if inst.opcode in {"AND", "MODPOW2", "RSHIFT"}:
                    found_extraction = True
                continue

            # Check for bitwise operations (bit extraction)
            if tasir_inst.kind == InstructionKind.BITWISE:
                found_extraction = True
                continue

            # Check for BRANCH_CONDITIONAL (guard) after extraction
            if tasir_inst.kind == InstructionKind.BRANCH_CONDITIONAL and found_extraction:
                return True

            # Also check THROW as guard
            if tasir_inst.kind == InstructionKind.THROW and found_extraction:
                return True

        # Fallback: use traditional method if TASIR didn't conclusively find pattern
        return self._check_bounced_extraction(instructions, start_pos)

    def _check_bounced_extraction_32bit_tasir(
        self,
        instructions: List,
        tasir_by_pos: Dict[int, object],
        start_pos: int,
    ) -> bool:
        """
        Check for bounced flag extraction from 32-bit load using TASIR.

        Similar to _check_bounced_extraction_tasir but for 32-bit pattern.
        """
        from ..ir.tasir_types import InstructionKind

        max_lookahead = min(start_pos + self.BIT_EXTRACTION_LOOKAHEAD, len(instructions))

        found_extraction = False

        for j in range(start_pos + 1, max_lookahead):
            tasir_inst = tasir_by_pos.get(j)
            if tasir_inst is None:
                inst = instructions[j]
                if inst.opcode in {"AND", "RSHIFT"}:
                    found_extraction = True
                continue

            # Check for bitwise operations
            if tasir_inst.kind == InstructionKind.BITWISE:
                found_extraction = True
                continue

            # Check for guard
            if tasir_inst.kind == InstructionKind.BRANCH_CONDITIONAL and found_extraction:
                return True
            if tasir_inst.kind == InstructionKind.THROW and found_extraction:
                return True

        # Fallback to traditional method
        return self._check_bounced_extraction_32bit(instructions, start_pos)

    def _check_bounced_extraction(self, instructions, start_idx: int) -> bool:
        """
        Check for bounced flag extraction patterns after LDU 4.

        Improved pattern handling: Now uses self.bounced_patterns for extensible pattern matching.
        Patterns are defined in DEFAULT_BOUNCED_PATTERNS and can be extended
        via options["additional_patterns"].

        Supported patterns (from DEFAULT_BOUNCED_PATTERNS):
        1. push_one_and: PUSHINT 1 + AND (mask with 1)
        2. modpow2_1: MODPOW2 1 (modulo 2)
        3. rshift_3: RSHIFT 3 (extract highest bit of 4-bit field = bounced)
        4. push_three_rshift: PUSHINT 3 + RSHIFT (non-immediate RSHIFT)
        5. pldu_1: PLDU 1 (extract single bit directly)
        6. booleval: BOOLEVAL (convert to boolean 0/-1)
        7. zero_comparison: ISZERO / EQINT / NEQINT (zero/non-zero comparison)
        8. boundary_comparison: GTINT/LESSINT/LEQINT/GEQINT with boundary values
        9. chained_shift: LSHIFT/RSHIFT chained bit operations

        All patterns must be followed by a guard (IF/IFNOT/THROW).

        Known Limitations:
        This pattern matching approach may miss equivalent extraction patterns,
        including but not limited to:
        - Indirect flag checks via computed jumps or table lookups

        To support additional patterns, pass options={"additional_patterns": {...}}.
        """
        max_lookahead = min(start_idx + self.BIT_EXTRACTION_LOOKAHEAD, len(instructions))
        prev_push_value: Optional[int] = None

        for j in range(start_idx + 1, max_lookahead):
            inst = instructions[j]
            opcode = inst.opcode

            # Track PUSHINT values for patterns that require them
            if self._is_pushint_like(opcode):
                val = self._get_arg_value(inst)
                # PUSHPOW2 n = 2^n, so PUSHPOW2 0 = 1
                if opcode == "PUSHPOW2" and val is not None:
                    prev_push_value = 1 << val
                else:
                    prev_push_value = val
                continue

            # Check each pattern in self.bounced_patterns
            if self._match_pattern(inst, opcode, j, instructions, prev_push_value):
                return True

            # Reset PUSHINT tracking if non-stack-preserving opcode encountered
            if not self._preserves_prev_push(opcode):
                prev_push_value = None

            # Stop if we see unrelated operations
            if opcode in self.UNRELATED_OPCODES:
                break

        return False

    def _match_pattern(
        self,
        inst,
        opcode: str,
        idx: int,
        instructions,
        prev_push_value: Optional[int]
    ) -> bool:
        """
        Improved pattern handling: Match instruction against configured patterns.

        Args:
            inst: Current instruction
            opcode: Current instruction opcode
            idx: Current instruction index
            instructions: Full instruction list
            prev_push_value: Value from previous PUSHINT (if any)

        Returns:
            True if a pattern matches AND has a guard after, False otherwise.
        """
        for pattern_name, pattern in self.bounced_patterns.items():
            pattern_opcodes = pattern.get("opcodes", [])

            # Check if opcode matches this pattern
            if opcode not in pattern_opcodes:
                continue

            # Check requires_prev_push condition
            if "requires_prev_push" in pattern:
                required_value = pattern["requires_prev_push"]
                if prev_push_value != required_value:
                    continue

            # Check arg_value condition
            if "arg_value" in pattern:
                arg_val = self._get_arg_value(inst)
                if arg_val is None or arg_val != pattern["arg_value"]:
                    continue

            # Check arg_in_set condition
            if "arg_in_set" in pattern:
                arg_val = self._get_arg_value(inst)
                if arg_val is None or arg_val not in pattern["arg_in_set"]:
                    continue

            # Check chain_next condition (for chained operations like LSHIFT+RSHIFT)
            if "chain_next" in pattern:
                chain_opcodes = pattern["chain_next"]
                if idx + 1 < len(instructions):
                    next_inst = instructions[idx + 1]
                    if next_inst.opcode in chain_opcodes:
                        if self._has_guard_after(instructions, idx + 1):
                            return True
                # chain_next pattern requires next instruction match, so skip guard check
                continue

            # Pattern matched, check for guard
            if self._has_guard_after(instructions, idx):
                return True

        return False

    def _check_bounced_extraction_32bit(self, instructions, start_idx: int) -> bool:
        """
        Check for bounced flag extraction when 32 bits are loaded at once.

        Improved pattern handling: Now uses self.bounced_patterns_32bit for extensible pattern matching.
        Patterns are defined in DEFAULT_BOUNCED_PATTERNS_32BIT and can be extended
        via options["additional_patterns_32bit"].

        Supported patterns (from DEFAULT_BOUNCED_PATTERNS_32BIT):
        1. push_mask_and: PUSHINT 0x80000000 + AND (extract bit 31)
        2. rshift_31: RSHIFT 31 (extract highest bit from 32-bit value)

        The bounced bit is at position 31 (bit 3 of original 4-bit flags, shifted left by 28).
        """
        max_lookahead = min(start_idx + self.BIT_EXTRACTION_LOOKAHEAD, len(instructions))
        prev_push_value: Optional[int] = None

        for j in range(start_idx + 1, max_lookahead):
            inst = instructions[j]
            opcode = inst.opcode

            # Track PUSHINT values for AND mask
            if self._is_pushint_like(opcode):
                prev_push_value = self._get_arg_value(inst)
                continue

            # Check each pattern in self.bounced_patterns_32bit
            if self._match_pattern_32bit(inst, opcode, j, instructions, prev_push_value):
                return True

            # Reset tracking
            if not self._preserves_prev_push(opcode):
                prev_push_value = None

            # Stop if we see unrelated operations
            if opcode in self.UNRELATED_OPCODES:
                break

        return False

    def _match_pattern_32bit(
        self,
        inst,
        opcode: str,
        idx: int,
        instructions,
        prev_push_value: Optional[int]
    ) -> bool:
        """
        Improved pattern handling: Match instruction against configured 32-bit patterns.

        Args:
            inst: Current instruction
            opcode: Current instruction opcode
            idx: Current instruction index
            instructions: Full instruction list
            prev_push_value: Value from previous PUSHINT (if any)

        Returns:
            True if a pattern matches AND has a guard after, False otherwise.
        """
        for pattern_name, pattern in self.bounced_patterns_32bit.items():
            pattern_opcodes = pattern.get("opcodes", [])

            # Check if opcode matches this pattern
            if opcode not in pattern_opcodes:
                continue

            # Check requires_prev_push_mask condition (special case for 32-bit mask)
            if pattern.get("requires_prev_push_mask"):
                if prev_push_value is None:
                    continue
                target_mask = self.BOUNCED_FLAG_MASK_32BIT  # 0x80000000 = 2147483648
                # TVM uses signed 257-bit integers, so the 32-bit bounced flag mask
                # 0x80000000 may be pushed as either positive (2147483648) or
                # negative (-2147483648) depending on compiler/source representation
                if prev_push_value != target_mask and prev_push_value != -target_mask:
                    continue

            # Check arg_value condition
            if "arg_value" in pattern:
                arg_val = self._get_arg_value(inst)
                if arg_val is None or arg_val != pattern["arg_value"]:
                    continue

            # Pattern matched, check for guard
            if self._has_guard_after(instructions, idx):
                return True

        return False

    def _get_arg_value(self, inst) -> Optional[int]:
        """Extract integer value from instruction's first argument.

        Improved handling: Returns None instead of -1 when not found, since -1 is a
        valid TVM integer value. Callers must check for None explicitly.
        """
        if inst.arguments and len(inst.arguments) > 0:
            arg = inst.arguments[0]
            if hasattr(arg, 'value') and isinstance(arg.value, int):
                return arg.value
        return None

    def _has_guard_after(self, instructions, idx: int) -> bool:
        """
        Check if there's a guard (IF/IFNOT/THROW) within next few instructions.

        This validates that the bounced check result is actually used for control flow.
        """
        max_lookahead = min(idx + self.GUARD_SEARCH_LOOKAHEAD, len(instructions))

        for k in range(idx + 1, max_lookahead):
            inst = instructions[k]
            if inst.opcode in GUARD_OPCODES:
                return True
            # Stack manipulation is OK
            if inst.opcode in {"DUP", "OVER", "SWAP", "ROT"}:
                continue
            # If we see consuming operations without guard, stop
            if inst.opcode in {"ADD", "SUB", "MUL", "DIV", "DROP", "POP",
                              "NEWC", "STREF", "SENDRAWMSG"}:
                return False

        return False


class BouncedHandlerPatternDetector(Detector):
    """
    [EXPERIMENTAL] Checks bounced handler correctness using CFG-based path analysis.

    This detector attempts to verify that bounced messages are properly handled by
    checking if send operations can be reached without passing through a bounced
    flag guard. Due to the complexity of control flow analysis, this detector may
    produce false positives in contracts with complex branching structures.

    Detection approach:
    1. Find the bounced flag load instruction (LDU 4 or LDU 32)
    2. Find subsequent guard instructions that check the bounced flag
    3. Use CFG path analysis to check if any send operation is reachable from
       the bounced check WITHOUT going through a proper guard

    Limitations:
    - May not detect all bounced handling patterns (e.g., computed jumps)
    - May produce false positives when guards exist but are not recognized
    - Requires CFG information for accurate analysis; falls back to heuristics otherwise
    """

    name = "bounced_handler_pattern"
    category = "security"
    default_severity = "medium"  # Lowered from "high" due to heuristic nature
    description = "Ensures bounced messages are handled correctly (experimental)"
    experimental = True  # Mark as experimental for tooling awareness
    enabled_by_default = False  # Experimental detector; opt-in via config

    # Default search range for message header parsing, configurable via options
    DEFAULT_SEARCH_RANGE = BOUNCED_SEARCH_RANGE

    # Extended lookahead for guard search in bounced handler pattern detection.
    # Compiler-generated recv paths often insert address parsing and validation
    # before the actual THROWIFNOT/IF guard becomes visible.
    GUARD_SEARCH_EXTENDED_LOOKAHEAD = 14

    def __init__(self, options=None, **kwargs):
        super().__init__(options, **kwargs)
        # Allow override via options
        self.search_range = self.options.get("search_range", self.DEFAULT_SEARCH_RANGE)

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        module = self.get_tasir(facts)
        scope_indices = BouncedMessageDetector._instruction_scope_indices(module, facts)
        instructions = BouncedMessageDetector._scoped_fact_instructions(facts, scope_indices)

        # Fallback to all instructions when TASIR scoping is too restrictive
        if len(instructions) < 2 and scope_indices is not None:
            scope_indices = None
            instructions = BouncedMessageDetector._scoped_fact_instructions(facts, scope_indices)

        # Skip contracts that are too short
        if len(instructions) < 2:
            return findings

        # Try TASIR-based detection first
        tasir_result = self._detect_with_tasir(
            facts,
            instructions=instructions,
            scope_indices=scope_indices,
        )
        if tasir_result is not None:
            return tasir_result

        # Quick check via SemanticLabel + has_pattern
        if module and module.has_pattern(SemanticLabel.BOUNCED_CHECK):
            # Contract has bounced check pattern, still run detailed analysis
            pass

        # Fallback to traditional detection
        # Find bounced check position (LDU 4 or LDU 32 in message header parsing)
        bounced_check_pos = self._find_bounced_load(instructions)

        if bounced_check_pos is None:
            # No bounced check - handled by BouncedMessageDetector
            return findings

        # Find the guard instruction that actually checks the bounced flag
        bounced_guard_pos = self._find_bounced_guard(instructions, bounced_check_pos)

        if bounced_guard_pos is None:
            # Has flag load but no guard - high confidence issue
            send_events = [
                e for e in facts.events_of("send")
                if scope_indices is None or e.instruction.index in scope_indices
            ]
            if send_events:
                findings.append(
                    self._build_vuln(
                        message="Bounced flag is loaded but not checked with a guard. "
                        "Send operations may execute on bounced message path.",
                        instruction=instructions[bounced_check_pos],
                        severity="high",  # High confidence for this case
                        remediation="Add 'if (flags & 1) { return(); }' check after loading flags",
                        extra={"confidence": "high", "signal": "no_guard_after_flag_load"},
                    )
                )
            return findings

        # Use CFG-based analysis if available for more accurate path analysis
        if facts.basic_blocks and len(facts.basic_blocks) > 1:
            return self._detect_with_cfg(
                facts,
                bounced_check_idx=instructions[bounced_check_pos].index,
                bounced_guard_idx=instructions[bounced_guard_pos].index,
                scope_indices=scope_indices,
            )

        # Fallback: simple heuristic analysis when CFG is not available
        return self._detect_without_cfg(
            facts,
            bounced_check_pos=bounced_check_pos,
            bounced_guard_pos=bounced_guard_pos,
            instructions=instructions,
            scope_indices=scope_indices,
        )

    def _detect_with_tasir(
        self,
        facts: AnalysisFacts,
        instructions: Optional[List] = None,
        scope_indices: Optional[set[int]] = None,
    ) -> Optional[List[Vulnerability]]:
        """
        Attempt TASIR-based detection of bounced handler patterns.

        Uses InstructionKind semantic classification:
        - CELL_LOAD: Identifies load operations for flag parsing
        - BRANCH_CONDITIONAL: Identifies guard instructions
        - SEND_MESSAGE: Identifies send operations that need protection

        Returns:
            List of findings if TASIR detection is conclusive, None to signal fallback.

        Note: This provides semantic classification benefit but still relies on
        traditional pattern matching for detailed bit extraction analysis.
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        from ..ir.tasir_types import InstructionKind

        if instructions is None:
            instructions = BouncedMessageDetector._scoped_fact_instructions(facts, scope_indices)
        if not instructions:
            return None  # Inconclusive — let fallback handle

        index_to_pos = {inst.index: pos for pos, inst in enumerate(instructions)}
        tasir_instructions = module.all_instructions()
        if scope_indices is not None:
            tasir_instructions = [
                inst for inst in tasir_instructions if inst.index in scope_indices
            ]
        tasir_by_pos = {
            index_to_pos[inst.index]: inst
            for inst in tasir_instructions
            if inst.index in index_to_pos
        }

        # Find bounced load using TASIR semantic classification
        bounced_check_pos = self._find_bounced_load_tasir(
            instructions, tasir_by_pos
        )

        if bounced_check_pos is None:
            # TASIR didn't find bounced load pattern — fallback may use
            # different heuristics that cover additional patterns.
            return None

        # Find guard using TASIR
        bounced_guard_pos = self._find_bounced_guard_tasir(
            instructions, tasir_by_pos, bounced_check_pos
        )

        if bounced_guard_pos is None:
            # Has flag load but no guard - check for send operations
            send_instructions = [
                inst for inst in tasir_by_pos.values()
                if inst.kind == InstructionKind.SEND_MESSAGE
            ]
            send_events = [
                event for event in facts.events_of("send")
                if scope_indices is None or event.instruction.index in scope_indices
            ]
            if send_instructions or send_events:
                return [
                    self._build_vuln(
                        message="Bounced flag is loaded but not checked with a guard. "
                        "Send operations may execute on bounced message path.",
                        instruction=instructions[bounced_check_pos],
                        severity="high",
                        remediation="Add 'if (flags & 1) { return(); }' check after loading flags",
                        extra={
                            "confidence": "high",
                            "signal": "no_guard_after_flag_load",
                            "detection_method": "tasir",
                        },
                    )
                ]
            return []

        # Use CFG-based analysis if available
        if facts.basic_blocks and len(facts.basic_blocks) > 1:
            return self._detect_with_cfg(
                facts,
                bounced_check_idx=instructions[bounced_check_pos].index,
                bounced_guard_idx=instructions[bounced_guard_pos].index,
                scope_indices=scope_indices,
            )

        # Fallback to non-CFG analysis
        return self._detect_without_cfg(
            facts,
            bounced_check_pos=bounced_check_pos,
            bounced_guard_pos=bounced_guard_pos,
            instructions=instructions,
            scope_indices=scope_indices,
        )

    def _find_bounced_load_tasir(
        self,
        instructions: List,
        tasir_by_pos: Dict[int, object],
    ) -> Optional[int]:
        """Find bounced flag load instruction using TASIR classification."""
        from ..ir.tasir_types import InstructionKind

        # Find CELL_LOAD instructions in local search range
        for pos in range(min(self.search_range, len(instructions))):
            tasir_inst = tasir_by_pos.get(pos)
            if tasir_inst is None:
                continue
            if tasir_inst.kind != InstructionKind.CELL_LOAD:
                continue
            if tasir_inst.opcode not in {"LDU", "PLDU"}:
                continue

            # Check argument for 4 or 32 bit load
            orig_inst = instructions[pos]
            if orig_inst.arguments and hasattr(orig_inst.arguments[0], "value"):
                val = orig_inst.arguments[0].value
                if val in {4, 32}:
                    return pos

        # Fallback to traditional method
        return self._find_bounced_load(instructions)

    def _find_bounced_guard_tasir(
        self,
        instructions: List,
        tasir_by_pos: Dict[int, object],
        flag_load_pos: int,
    ) -> Optional[int]:
        """Find bounced guard instruction using TASIR classification."""
        from ..ir.tasir_types import InstructionKind

        max_lookahead = min(
            flag_load_pos + self.GUARD_SEARCH_EXTENDED_LOOKAHEAD, len(instructions)
        )

        found_bit_extraction = False

        for i in range(flag_load_pos + 1, max_lookahead):
            tasir_inst = tasir_by_pos.get(i)

            if tasir_inst is not None:
                # Use TASIR semantic classification
                if tasir_inst.kind == InstructionKind.BITWISE:
                    found_bit_extraction = True
                    continue

                # Check for guard (BRANCH_CONDITIONAL or THROW)
                if tasir_inst.kind == InstructionKind.BRANCH_CONDITIONAL and found_bit_extraction:
                    return i
                if tasir_inst.kind == InstructionKind.THROW and found_bit_extraction:
                    return i

                # Accept guard immediately after flag load
                if (tasir_inst.kind == InstructionKind.BRANCH_CONDITIONAL and
                        i == flag_load_pos + 1):
                    return i
                if tasir_inst.kind == InstructionKind.THROW and i == flag_load_pos + 1:
                    return i

                # Stop at unrelated operations
                if tasir_inst.kind in {
                    InstructionKind.CELL_STORE,
                    InstructionKind.SEND_MESSAGE,
                    InstructionKind.RESERVE,
                }:
                    break
            else:
                # Fallback to opcode check
                if i < len(instructions):
                    inst = instructions[i]
                    if inst.opcode in {"AND", "MODPOW2", "RSHIFT"}:
                        found_bit_extraction = True
                    elif inst.opcode in GUARD_OPCODES and found_bit_extraction:
                        return i
                    elif inst.opcode in GUARD_OPCODES and i == flag_load_pos + 1:
                        return i
                    elif inst.opcode in {"NEWC", "ENDC", "STREF", "LDREF", "SENDRAWMSG", "RAWRESERVE"}:
                        break

        # Fallback to traditional method if TASIR didn't find it
        return self._find_bounced_guard(instructions, flag_load_pos)

    def _find_bounced_load(self, instructions) -> Optional[int]:
        """Find the instruction that loads message flags (containing bounced bit)."""
        for i in range(min(self.search_range, len(instructions))):
            inst = instructions[i]
            if inst.opcode in {"LDU", "PLDU"}:
                if inst.arguments and hasattr(inst.arguments[0], "value"):
                    val = inst.arguments[0].value
                    if val in {4, 32}:  # Support both 4-bit and 32-bit flag loading
                        return i
        return None

    def _find_bounced_guard(self, instructions, flag_load_idx: int) -> Optional[int]:
        """
        Find the guard instruction that checks the bounced flag after flag load.

        Returns the index of the guard instruction, or None if no guard found.
        A valid bounced guard pattern requires:
        1. Bit extraction (AND, MODPOW2, RSHIFT) after flag load
        2. Guard instruction (IF/IFNOT/THROW) within a few instructions
        """
        # Look for bit extraction followed by guard within extended lookahead window
        max_lookahead = min(flag_load_idx + self.GUARD_SEARCH_EXTENDED_LOOKAHEAD, len(instructions))
        found_bit_extraction = False

        for i in range(flag_load_idx + 1, max_lookahead):
            inst = instructions[i]
            opcode = inst.opcode

            # Check for bounced bit extraction patterns
            # Note: FIRST is for tuples, not for bit extraction - excluded for consistency
            # with _has_bounced_check (see Pattern 4 comment)
            if opcode in {"AND", "MODPOW2", "RSHIFT"}:
                found_bit_extraction = True
                continue

            # Check for guard after bit extraction
            if opcode in GUARD_OPCODES and found_bit_extraction:
                return i

            # Also accept guard immediately after flag load (some compilers)
            if opcode in GUARD_OPCODES and i == flag_load_idx + 1:
                return i

            # Stop if we see unrelated operations that would consume the flag value
            if opcode in {"NEWC", "ENDC", "STREF", "LDREF", "SENDRAWMSG", "RAWRESERVE"}:
                break

        return None

    def _detect_with_cfg(
        self,
        facts: AnalysisFacts,
        bounced_check_idx: int,
        bounced_guard_idx: int,
        scope_indices: Optional[set[int]] = None,
    ) -> List[Vulnerability]:
        """
        Use CFG to check if send operations are reachable without going through
        the bounced guard.

        This is more accurate than simple instruction counting because it considers
        actual control flow paths. Uses common CFG traversal utility.
        """
        findings = []

        # Build block map and instruction-to-block mapping
        block_map = {block.id: block for block in facts.basic_blocks}
        instr_to_block = build_instruction_to_block_map(facts.basic_blocks)

        # Check for unknown successors which indicate incomplete CFG analysis
        has_unknown_successors = any(
            getattr(block, 'has_unknown_successor', False)
            for block in facts.basic_blocks
        )

        # Get send instruction indices
        send_indices = {
            event.instruction.index
            for event in facts.events_of("send")
            if scope_indices is None or event.instruction.index in scope_indices
        }
        if not send_indices:
            return findings  # Early return: no send operations to check

        # Find the start block
        start_block_id = instr_to_block.get(bounced_check_idx)
        if start_block_id is None:
            return findings

        # Use common CFG traversal utility to find unguarded sends
        result = traverse_cfg_for_specific_guard(
            block_map=block_map,
            start_block_id=start_block_id,
            start_from_idx=bounced_check_idx,
            specific_guard_idx=bounced_guard_idx,
            sink_indices=send_indices,
            max_iterations=MAX_SEARCH_ITERATIONS,
        )
        flagged_sends = result.flagged_indices
        truncated = result.truncated
        instruction_by_index = {inst.index: inst for inst in facts.instructions}

        # Report findings for sends reachable without bounced guard
        for send_idx in sorted(flagged_sends):
            instruction = instruction_by_index.get(send_idx)
            if instruction is None:
                continue

            severity = "medium"
            message = (
                "Send operation may be reachable on bounced message path "
                "without proper guard (CFG path analysis). "
                "Ensure bounced messages return early without sending new messages."
            )
            if has_unknown_successors:
                severity = "low"  # Downgrade due to incomplete CFG
                message += " (CFG incomplete - manual review recommended)"

            findings.append(
                self._build_vuln(
                    message=message,
                    instruction=instruction,
                    severity=severity,
                    remediation="Add 'if (flags & 1) { return(); }' check before any send operations",
                    extra={
                        "confidence": "medium" if not has_unknown_successors else "low",
                        "signal": "cfg_path_analysis",
                        "bounced_guard_idx": bounced_guard_idx,
                        "analysis_truncated": truncated,
                        "has_unknown_successors": has_unknown_successors,
                    },
                )
            )

        return findings

    def _detect_without_cfg(
        self,
        facts: AnalysisFacts,
        bounced_check_pos: int,
        bounced_guard_pos: int,
        instructions: List,
        scope_indices: Optional[set[int]] = None,
    ) -> List[Vulnerability]:
        """
        Fallback detection when CFG is not available.

        Uses a simple heuristic: if there's a send operation BEFORE the bounced guard,
        it's definitely a problem. If it's after, we assume the guard protects it.
        """
        findings = []
        send_events = sorted(
            (
                event for event in facts.events_of("send")
                if scope_indices is None or event.instruction.index in scope_indices
            ),
            key=lambda e: e.instruction.index,
        )
        index_to_pos = {inst.index: pos for pos, inst in enumerate(instructions)}

        for send_event in send_events:
            send_pos = index_to_pos.get(send_event.instruction.index)
            if send_pos is None:
                continue

            # Only check sends after the bounced flag load
            if send_pos <= bounced_check_pos:
                continue

            # If send is before the guard, it's definitely not protected
            if send_pos < bounced_guard_pos:
                findings.append(
                    self._build_vuln(
                        message="Send operation occurs before bounced flag guard. "
                        "This send will execute even on bounced messages.",
                        instruction=send_event.instruction,
                        severity="high",  # High confidence for this case
                        remediation="Move bounced check before any send operations",
                        extra={
                            "confidence": "high",
                            "signal": "send_before_guard",
                            "bounced_guard_idx": instructions[bounced_guard_pos].index,
                        },
                    )
                )
                break  # Report only the first issue

        return findings
