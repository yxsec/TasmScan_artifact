"""Unhandled bounced message op detector

Detects contracts that send messages with specific opcodes but fail to handle
the bounced response for those opcodes. This complements bounced_message
(which checks if bounced flag is checked at all) by verifying that the
bounced handler covers all sent opcodes.

Pattern: contract sends SENDRAWMSG with op X, but the bounced handler
does not check for op X.

Corresponds to TONScanner's "UnHandleBouncedMessage" vulnerability class
(the specific "UnHandle bounced message with op: 0x..." pattern).
"""
from typing import List, Set

from ..analyzer.facts import AnalysisFacts
from .base import Detector
from .results import Confidence, Vulnerability


class UnhandledBouncedOpDetector(Detector):
    """Detects unhandled bounced message opcodes."""

    name = "unhandled_bounced_op"
    category = "security"
    enabled_by_default = True
    default_severity = "medium"
    description = "Detects contracts that send messages but may not fully handle bounced responses"
    tags = ("bounced", "message-handling", "tonscanner-compat")

    SEND_OPCODES: Set[str] = {"SENDRAWMSG", "SENDMSG"}

    # Opcodes that build message body (contain op code)
    STORE_UINT_OPCODES: Set[str] = {"STU", "PUSHINT", "PUSHINT_4", "PUSHINT_8",
                                     "PUSHINT_16", "PUSHINT_LONG"}

    # Bounced flag check opcodes
    BOUNCED_CHECK_OPCODES: Set[str] = {"AND", "MODPOW2"}

    # Guard opcodes for bounced handling
    GUARD_OPCODES: Set[str] = {"IF", "IFNOT", "IFJMP", "IFNOTJMP",
                                "THROWIF", "THROWIFNOT", "IFELSE"}

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        insts = facts.instructions

        # Step 1: Does contract send messages?
        sends = [i for i in insts if i.opcode in self.SEND_OPCODES]
        if not sends:
            return findings

        # Step 2: Does contract have a bounced handler?
        # Look for bounced flag extraction pattern in first ~30 instructions
        has_bounced_check = False
        bounced_handler_start = -1
        for i, inst in enumerate(insts[:40]):
            if inst.opcode in self.BOUNCED_CHECK_OPCODES:
                # Check if followed by guard
                for j in range(i + 1, min(i + 6, len(insts))):
                    if insts[j].opcode in self.GUARD_OPCODES:
                        has_bounced_check = True
                        bounced_handler_start = j
                        break
            if has_bounced_check:
                break

        if not has_bounced_check:
            # No bounced check at all — already caught by bounced_message detector
            return findings

        # Step 3: Check if bounced handler is a simple return/throw (valid pattern)
        # A simple handler that just returns or throws is acceptable
        RETURN_OPCODES = {"RET", "RETALT", "RETFALSE", "RETBOOL"}
        THROW_OPCODES = {"THROW", "THROWARG", "THROWIF", "THROWIFNOT",
                         "THROWARGIF", "THROWARGIFNOT"}
        if bounced_handler_start > 0:
            bounce_section = insts[bounced_handler_start:bounced_handler_start + 10]
            # If the handler immediately returns or throws, it's a valid
            # "ignore bounced" or "fail on bounced" pattern
            for inst in bounce_section:
                if inst.opcode in RETURN_OPCODES | THROW_OPCODES:
                    return findings  # Valid simple handler, no finding

        # Step 4: Extract sent op codes (look for STU 32 pattern near SENDRAWMSG)
        sent_ops = set()
        for send in sends:
            send_idx = send.index if hasattr(send, 'index') else 0
            # Look backwards for STU 32 preceded by PUSHINT (standard op-store pattern)
            for j in range(max(0, send_idx - 15), send_idx):
                if j < len(insts) and insts[j].opcode in ("STU",):
                    # Check if this stores 32 bits (op code width)
                    args = getattr(getattr(insts[j], 'instruction', None), 'args', [])
                    arg_val = args[0] if args else None
                    if hasattr(arg_val, 'value'):
                        arg_val = arg_val.value
                    if arg_val == 32:
                        # Look further back for the PUSHINT (the actual op code)
                        for k in range(max(0, j - 3), j):
                            if k < len(insts) and insts[k].opcode in (
                                "PUSHINT", "PUSHINT_LONG", "PUSHINT_16"):
                                op_args = getattr(getattr(insts[k], 'instruction', None), 'args', [])
                                if op_args:
                                    op_val = op_args[0]
                                    if hasattr(op_val, 'value'):
                                        op_val = op_val.value
                                    if isinstance(op_val, int) and op_val > 0:
                                        sent_ops.add(op_val)

        # Only flag if contract sends 2+ distinct opcodes (single-op contracts
        # don't need per-op bounced handling)
        if len(sent_ops) < 2:
            return findings

        # Step 5: Count op code checks in bounced handler
        # Recognize EQUAL, THROWIFNOT, IFJMP as valid op-check patterns
        OP_CHECK_OPCODES = {"EQUAL", "THROWIFNOT", "THROWIF", "IFJMP", "IFNOTJMP"}
        handled_ops = set()
        if bounced_handler_start > 0:
            bounce_section = insts[bounced_handler_start:bounced_handler_start + 80]
            for i, inst in enumerate(bounce_section):
                if inst.opcode in OP_CHECK_OPCODES:
                    for j in range(max(0, i - 5), i):
                        prev = bounce_section[j]
                        if prev.opcode in ("PUSHINT", "PUSHINT_LONG", "PUSHINT_16"):
                            args = getattr(getattr(prev, 'instruction', None), 'args', [])
                            if args:
                                val = args[0]
                                if hasattr(val, 'value'):
                                    val = val.value
                                if isinstance(val, int):
                                    handled_ops.add(val)

        # Step 6: Flag only if there are unhandled ops
        if sent_ops and len(handled_ops) < len(sent_ops):
            unhandled = sent_ops - handled_ops
            if unhandled:
                findings.append(self._build_vuln(
                    message=f"Contract sends {len(sent_ops)} distinct op(s) but bounced "
                            f"handler only covers {len(handled_ops)} — "
                            f"{len(unhandled)} op(s) may be unhandled",
                    instruction=sends[0],
                    remediation="Ensure bounced handler covers all sent message opcodes",
                    severity="medium",
                    confidence=Confidence.MEDIUM,
                    extra={"sent_ops": len(sent_ops), "handled_ops": len(handled_ops),
                           "unhandled_count": len(unhandled),
                           "detection_method": "bounced_op_diff"},
                ))

        return findings
