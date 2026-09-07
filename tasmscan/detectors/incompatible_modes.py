"""
Incompatible Message Modes Detector

Detects two issues with SENDRAWMSG mode flags:
1. SendRemainingValue (0x40) + SendRemainingBalance (0x80) used in same transaction
   -> TVM exit code 34
2. SendRemainingValue (0x40) used more than once in same transaction
   -> Silently loses funds (only first message gets remaining value)

Reference: TSA detects these as 'incompatible-message-modes' (exit 34) and
'double-send-remaining-value'.
"""
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..analyzer.facts import AnalysisFacts, Event
from ..analyzer.utils import extract_int_arg
from .base import Detector
from .results import Confidence, Vulnerability

if TYPE_CHECKING:
    from ..ir.tasir_types import TVMInstruction, TVMModule

logger = logging.getLogger(__name__)

# Mode flag constants
SEND_REMAINING_VALUE = 0x40   # flag 64: attach all remaining value to message
SEND_REMAINING_BALANCE = 0x80  # flag 128: send entire remaining balance


class IncompatibleMessageModesDetector(Detector):
    """Detects incompatible SENDRAWMSG mode flag combinations.

    Detection Strategy
    ------------------
    This detector uses a tiered detection approach:

    1. **TASIR-based detection** (preferred): Uses InstructionKind.SEND_MESSAGE
       to find all send instructions, then resolves mode constants from
       preceding PUSHINT instructions or TVMInstruction immediates.

    2. **CFG-based fallback**: Uses AnalysisFacts events and instruction
       sequence to find SENDRAWMSG opcodes, then walks backward to find
       the mode constant (PUSHINT immediately before the send).

    Two vulnerability classes:

    - **exit_code_34**: Mode has both 0x40 and 0x80 set simultaneously.
      TVM raises exit code 34 at runtime because these flags are mutually
      exclusive (remaining value vs remaining balance).

    - **double_remaining_value**: Multiple SENDRAWMSG calls in the same
      function/transaction use flag 0x40. Only the first message actually
      receives the remaining value; subsequent ones silently get zero,
      causing fund loss.
    """

    name = "incompatible_message_modes"
    category = "security"
    default_severity = "high"
    description = (
        "Detects incompatible SENDRAWMSG mode flags: "
        "0x40+0x80 (exit code 34) and multiple 0x40 sends (fund loss)."
    )

    @staticmethod
    def _decode_push_constant(opcode: str, raw_value: int) -> int:
        """Decode immediate push variants into the actual pushed integer value."""
        upper = (opcode or "").upper()
        if upper == "PUSHPOW2":
            return 1 << raw_value
        if upper == "PUSHNEGPOW2":
            return -(1 << raw_value)
        if upper == "PUSHPOW2DEC":
            return (1 << raw_value) - 1
        return raw_value

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        # Try TASIR-based detection first
        tasir_findings = self._detect_with_tasir(facts)
        if tasir_findings is not None:
            return tasir_findings

        # Fall back to CFG / linear analysis
        return self._detect_with_cfg(facts)

    # ------------------------------------------------------------------
    # Tier 1: TASIR-based detection
    # ------------------------------------------------------------------

    def _detect_with_tasir(self, facts: AnalysisFacts) -> Optional[List[Vulnerability]]:
        """TASIR-based detection using InstructionKind for semantic matching.

        Returns list of findings if TASIR is available, None to signal fallback.
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        from ..ir.tasir_types import InstructionKind

        send_instructions = self.get_instructions_by_kind(facts, InstructionKind.SEND_MESSAGE)
        if not send_instructions:
            return []

        all_instructions = module.all_instructions()
        instr_by_index: Dict[int, "TVMInstruction"] = {i.index: i for i in all_instructions}

        # Group sends by function (method_id) to detect per-transaction issues
        sends_by_function: Dict[Optional[int], List[Tuple["TVMInstruction", Optional[int]]]] = defaultdict(list)

        for send_instr in send_instructions:
            mode = self._resolve_mode_tasir(send_instr, instr_by_index, module)
            func_id = self._get_function_id(send_instr, module)
            sends_by_function[func_id].append((send_instr, mode))

        findings: List[Vulnerability] = []

        for func_id, sends in sends_by_function.items():
            findings.extend(self._check_mode_conflicts(sends, facts))

        return findings

    def _resolve_mode_tasir(
        self,
        send_instr: "TVMInstruction",
        instr_by_index: Dict[int, "TVMInstruction"],
        module: "TVMModule",
    ) -> Optional[int]:
        """Try to resolve the integer mode argument for a SENDRAWMSG instruction.

        SENDRAWMSG pops two values: the message cell and the mode integer.
        The mode is typically the top-of-stack value, pushed by a PUSHINT
        immediately before (or near) the SENDRAWMSG.

        Strategy:
        1. Check the send instruction's own immediates (some IR builders inline it).
        2. Walk backward from the send instruction looking for a PUSHINT/PUSHINT_4.
        """
        # Strategy 1: check immediates on the send instruction itself
        for imm in send_instr.immediates:
            if isinstance(imm, int):
                return imm

        # Strategy 2: walk backward up to 4 instructions looking for PUSHINT
        from ..ir.tasir_types import InstructionKind

        for offset in range(1, 5):
            prev_idx = send_instr.index - offset
            prev = instr_by_index.get(prev_idx)
            if prev is None:
                continue
            if prev.kind == InstructionKind.STACK_PUSH and prev.opcode.startswith("PUSH"):
                # Extract and decode pushed constants (supports PUSHPOW2 family).
                raw_value = None
                for imm in prev.immediates:
                    if isinstance(imm, int):
                        raw_value = imm
                        break
                if raw_value is None:
                    raw_value = extract_int_arg(prev.original_args, 0)
                if raw_value is not None:
                    return self._decode_push_constant(prev.opcode, raw_value)
            # If we hit a non-stack-manipulation instruction, stop
            if prev.kind not in (
                InstructionKind.STACK_PUSH,
                InstructionKind.STACK_POP,
                InstructionKind.STACK_SHUFFLE,
                InstructionKind.NOP,
            ):
                break

        return None

    def _get_function_id(
        self, instr: "TVMInstruction", module: "TVMModule"
    ) -> Optional[int]:
        """Determine which function (method_id) an instruction belongs to."""
        for method_id, func in module.functions.items():
            for block in func.blocks.values():
                for bi in block.instructions:
                    if bi.index == instr.index:
                        return method_id
        return None

    # ------------------------------------------------------------------
    # Tier 2: CFG / linear fallback
    # ------------------------------------------------------------------

    def _detect_with_cfg(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """Fallback detection using AnalysisFacts events and raw instructions."""
        send_events = facts.events_of("send")
        if not send_events:
            return []

        # Build instruction lookup
        instr_by_index = {i.index: i for i in facts.instructions}

        # Group sends by continuation context (proxy for per-function grouping)
        sends_by_context: Dict[Optional[str], List[Tuple[Event, Optional[int]]]] = defaultdict(list)

        for event in send_events:
            mode = self._resolve_mode_cfg(event.instruction, instr_by_index)
            ctx = event.instruction.continuation_id
            sends_by_context[ctx].append((event, mode))

        findings: List[Vulnerability] = []

        for ctx, sends in sends_by_context.items():
            send_tuples = [
                (ev.instruction, mode) for ev, mode in sends
            ]
            findings.extend(self._check_mode_conflicts_cfg(send_tuples, facts))

        return findings

    def _resolve_mode_cfg(
        self, send_instr, instr_by_index
    ) -> Optional[int]:
        """Resolve mode constant from instructions preceding SENDRAWMSG.

        Walks backward from the send instruction looking for a PUSHINT
        that supplies the mode argument.
        """
        for offset in range(1, 5):
            prev_idx = send_instr.index - offset
            prev = instr_by_index.get(prev_idx)
            if prev is None:
                continue
            opcode = prev.opcode
            if opcode.startswith("PUSH"):
                # Extract and decode pushed constants (supports MockArg/Arg + PUSHPOW2 family).
                raw_value = extract_int_arg(prev.arguments, 0)
                if raw_value is not None:
                    return self._decode_push_constant(opcode, raw_value)
                break
            # Stop on non-trivial instructions that would change stack layout
            if opcode not in ("NOP", "SWAP", "XCHG", "PUSH", "POP", "DROP"):
                break

        return None

    # ------------------------------------------------------------------
    # Shared analysis logic
    # ------------------------------------------------------------------

    def _check_mode_conflicts(
        self,
        sends: List[Tuple["TVMInstruction", Optional[int]]],
        facts: AnalysisFacts,
    ) -> List[Vulnerability]:
        """Check a set of sends (within one function) for mode conflicts."""
        findings: List[Vulnerability] = []
        remaining_value_sends: List["TVMInstruction"] = []

        for send_instr, mode in sends:
            if mode is None:
                continue

            # Check 1: 0x40 + 0x80 simultaneously
            if (mode & SEND_REMAINING_VALUE) and (mode & SEND_REMAINING_BALANCE):
                # Map back to InstructionFact for consistent reporting
                original = self._find_original_instruction(send_instr.index, facts)
                findings.append(
                    self._build_vuln(
                        message=(
                            f"SENDRAWMSG mode {mode} (0x{mode:02x}) combines "
                            f"SendRemainingValue (0x40) and SendRemainingBalance (0x80). "
                            f"This causes TVM exit code 34 at runtime."
                        ),
                        instruction=original,
                        severity="critical",
                        confidence=Confidence.HIGH,
                        remediation=(
                            "Use only one of mode flag 64 (SendRemainingValue) or "
                            "128 (SendRemainingBalance), not both."
                        ),
                        extra={
                            "mode_value": mode,
                            "issue": "exit_code_34",
                            "flags": {"remaining_value": True, "remaining_balance": True},
                        },
                    )
                )

            # Track sends with 0x40 for duplicate check
            if mode & SEND_REMAINING_VALUE:
                remaining_value_sends.append(send_instr)

        # Check 2: multiple sends with 0x40 in same function
        if len(remaining_value_sends) > 1:
            # Flag all but the first (the first one works correctly)
            for send_instr in remaining_value_sends[1:]:
                original = self._find_original_instruction(send_instr.index, facts)
                findings.append(
                    self._build_vuln(
                        message=(
                            "Multiple SENDRAWMSG instructions use SendRemainingValue "
                            "(mode flag 0x40) in the same transaction. Only the first "
                            "message receives the remaining value; this message silently "
                            "gets zero value, causing fund loss."
                        ),
                        instruction=original,
                        severity="high",
                        confidence=Confidence.HIGH,
                        remediation=(
                            "Use SendRemainingValue (mode 64) on at most one message "
                            "per transaction. For additional messages, specify an explicit "
                            "value or use RAWRESERVE to control fund distribution."
                        ),
                        extra={
                            "issue": "double_send_remaining_value",
                            "remaining_value_send_count": len(remaining_value_sends),
                        },
                    )
                )

        return findings

    def _check_mode_conflicts_cfg(
        self,
        sends: List[Tuple],
        facts: AnalysisFacts,
    ) -> List[Vulnerability]:
        """Check mode conflicts for CFG-tier (InstructionFact, mode) tuples."""
        findings: List[Vulnerability] = []
        remaining_value_sends = []

        for send_instr, mode in sends:
            if mode is None:
                continue

            # Check 1: 0x40 + 0x80 simultaneously
            if (mode & SEND_REMAINING_VALUE) and (mode & SEND_REMAINING_BALANCE):
                findings.append(
                    self._build_vuln(
                        message=(
                            f"SENDRAWMSG mode {mode} (0x{mode:02x}) combines "
                            f"SendRemainingValue (0x40) and SendRemainingBalance (0x80). "
                            f"This causes TVM exit code 34 at runtime."
                        ),
                        instruction=send_instr,
                        severity="critical",
                        confidence=Confidence.HIGH,
                        remediation=(
                            "Use only one of mode flag 64 (SendRemainingValue) or "
                            "128 (SendRemainingBalance), not both."
                        ),
                        extra={
                            "mode_value": mode,
                            "issue": "exit_code_34",
                            "flags": {"remaining_value": True, "remaining_balance": True},
                        },
                    )
                )

            if mode & SEND_REMAINING_VALUE:
                remaining_value_sends.append(send_instr)

        # Check 2: multiple sends with 0x40
        if len(remaining_value_sends) > 1:
            for send_instr in remaining_value_sends[1:]:
                findings.append(
                    self._build_vuln(
                        message=(
                            "Multiple SENDRAWMSG instructions use SendRemainingValue "
                            "(mode flag 0x40) in the same transaction. Only the first "
                            "message receives the remaining value; this message silently "
                            "gets zero value, causing fund loss."
                        ),
                        instruction=send_instr,
                        severity="high",
                        confidence=Confidence.HIGH,
                        remediation=(
                            "Use SendRemainingValue (mode 64) on at most one message "
                            "per transaction. For additional messages, specify an explicit "
                            "value or use RAWRESERVE to control fund distribution."
                        ),
                        extra={
                            "issue": "double_send_remaining_value",
                            "remaining_value_send_count": len(remaining_value_sends),
                        },
                    )
                )

        return findings

    def _find_original_instruction(self, index: int, facts: AnalysisFacts):
        """Map a TVMInstruction index back to the original InstructionFact."""
        for instr in facts.instructions:
            if instr.index == index:
                return instr
        # Fallback: return first instruction if available
        return facts.instructions[0] if facts.instructions else None
