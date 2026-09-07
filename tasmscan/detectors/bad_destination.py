"""Detector for unvalidated destination addresses and message cell overflow risks.

Checks two properties of SENDRAWMSG message construction:

1. **Unvalidated destination address**: The message cell passed to SENDRAWMSG
   contains a destination address that flows from untrusted input (incoming
   message) without validation.  Detection traces backward from SENDRAWMSG
   through the builder chain (NEWC -> store ops -> ENDC) to find address
   store operations (STSLICER / STSLICE after LDMSGADDR).  If the address
   originates from a taint source and no guard intervenes, a finding is
   emitted.

2. **Message cell overflow risk**: The builder chain for SENDRAWMSG
   accumulates more than 1023 bits or 4 refs (TVM Cell limits).  Detection
   sums known constant bit widths (STI n -> n bits, STU n -> n bits,
   STREF -> 1 ref) along the builder chain.  If the sum provably exceeds
   the Cell capacity, a finding is emitted.
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..analyzer.facts import AnalysisFacts, BasicBlock
from ..analyzer.utils import extract_int_arg
from ..config import MAX_SEARCH_ITERATIONS
from .base import Detector
from .cfg_utils import (
    build_instruction_to_block_map,
    traverse_cfg_for_unguarded_sinks,
)
from .results import Confidence, Vulnerability

if TYPE_CHECKING:
    from ..ir.tasir_types import TVMInstruction, TVMModule


# ---- TVM Cell hard limits ------------------------------------------------
_MAX_CELL_BITS = 1023
_MAX_CELL_REFS = 4

# Opcodes that load a message address from a slice (taint-relevant)
_ADDR_LOAD_OPCODES = frozenset({
    "LDMSGADDR", "LDMSGADDRQ",
})

# Opcodes that store a slice into a builder (potential address stores)
_ADDR_STORE_OPCODES = frozenset({
    "STSLICER", "STSLICE", "STSLICECONST",
})

# Opcodes that contribute a known number of bits to a builder
# Mapping: opcode -> (bits_from_immediate, refs)
# For STI/STU the immediate argument gives the bit width.
# For fixed-width stores the constant is used directly.
_STORE_BIT_COSTS: Dict[str, Tuple[Optional[int], int]] = {
    # Variable-width integer stores: bits = first immediate arg
    "STI":   (None, 0),   # bits from immediate
    "STU":   (None, 0),
    "STIR":  (None, 0),
    "STUR":  (None, 0),
    # Fixed stores
    "STZERO":  (1, 0),     # store one 0-bit
    "STONE":   (1, 0),     # store one 1-bit
    "STZEROES": (None, 0), # variable count of 0-bits
    "STONES":  (None, 0),  # variable count of 1-bits
    # Reference stores
    "STREF":   (0, 1),
    "STREFR":  (0, 1),
    "STBREFR": (0, 1),
    "STREF2CONST": (0, 2),
    # Slice stores (unknown width unless we track the slice)
    "STSLICE":  (None, 0),
    "STSLICER": (None, 0),
    "STSLICECONST": (None, 0),
    # Grams / variable-length
    "STGRAMS":   (None, 0),
    "STVARINT16": (None, 0),
    "STVARUINT16": (None, 0),
    # Builder-to-builder
    "STBR":    (None, 0),
    "STB":     (None, 0),
}

# Opcodes that begin a new builder (reset bit/ref counters)
_NEWC_OPCODES = frozenset({"NEWC"})

# Opcodes that finalize a builder into a cell
_ENDC_OPCODES = frozenset({"ENDC", "ENDCST"})


class BadDestinationAddressDetector(Detector):
    """Detects unvalidated destination addresses and message cell overflow risks."""

    name = "bad_destination_address"
    default_severity = "high"
    category = "security"
    description = (
        "Detects unvalidated destination addresses and message cell overflow risks"
    )
    tags = ("critical", "ton-specific")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        if self._is_trivial_contract(facts):
            return findings

        # -- TASIR-tier detection (preferred) --
        module = self.get_tasir(facts)
        if module is not None:
            findings.extend(self._detect_tasir(facts, module))
            return findings

        # -- Legacy opcode-level detection --
        findings.extend(self._detect_legacy(facts))
        return findings

    # ==================================================================
    # TASIR-tier detection
    # ==================================================================

    def _detect_tasir(
        self, facts: AnalysisFacts, module: "TVMModule"
    ) -> List[Vulnerability]:
        from ..ir.tasir_types import InstructionKind

        findings: List[Vulnerability] = []

        send_instructions = [
            inst for inst in module.all_instructions()
            if inst.kind == InstructionKind.SEND_MESSAGE
        ]
        if not send_instructions:
            return findings

        all_instructions = module.all_instructions()
        idx_to_inst: Dict[int, "TVMInstruction"] = {
            inst.index: inst for inst in all_instructions
        }
        idx_to_context: Dict[int, str] = {}
        for block in module.all_blocks():
            for inst in block.instructions:
                idx_to_context[inst.index] = block.context_id

        for send_inst in send_instructions:
            # ---- Check 1: unvalidated destination address ----
            builder_chain = self._trace_builder_chain_tasir(
                send_inst, all_instructions, idx_to_context
            )
            addr_findings = self._check_unvalidated_addr_tasir(
                facts, module, send_inst, builder_chain, idx_to_inst
            )
            findings.extend(addr_findings)

            # ---- Check 2: message cell overflow ----
            overflow_findings = self._check_cell_overflow_tasir(
                facts, send_inst, builder_chain, idx_to_inst
            )
            findings.extend(overflow_findings)

        return findings

    # ---- builder chain tracing (TASIR) --------------------------------

    def _trace_builder_chain_tasir(
        self,
        send_inst: "TVMInstruction",
        all_instructions: List["TVMInstruction"],
        idx_to_context: Dict[int, str],
    ) -> List["TVMInstruction"]:
        """Walk backward from *send_inst* to collect the builder chain.

        Returns instructions in program order (NEWC first, ENDC last).
        """
        from ..ir.tasir_types import InstructionKind

        chain: List["TVMInstruction"] = []
        # Walk backward from the send instruction
        target_idx = send_inst.index
        target_context = idx_to_context.get(target_idx)
        for inst in reversed(all_instructions):
            if inst.index >= target_idx:
                continue
            if target_context is not None and idx_to_context.get(inst.index) != target_context:
                continue
            opcode = inst.opcode.upper()
            # Once we hit NEWC we have the start of the builder
            if opcode in _NEWC_OPCODES:
                chain.insert(0, inst)
                break
            # Collect store / ENDC operations
            if (opcode in _STORE_BIT_COSTS
                    or opcode in _ENDC_OPCODES
                    or inst.kind == InstructionKind.CELL_STORE):
                chain.insert(0, inst)

        return chain

    # ---- Check 1 helpers (TASIR) --------------------------------------

    def _check_unvalidated_addr_tasir(
        self,
        facts: AnalysisFacts,
        module: "TVMModule",
        send_inst: "TVMInstruction",
        builder_chain: List["TVMInstruction"],
        idx_to_inst: Dict[int, "TVMInstruction"],
    ) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        # Find address-store instructions in the builder chain
        addr_store_indices: List[int] = []
        for inst in builder_chain:
            opcode = inst.opcode.upper()
            if opcode in _ADDR_STORE_OPCODES:
                addr_store_indices.append(inst.index)

        if not addr_store_indices:
            return findings

        # For each address store, check whether it came from a tainted
        # source (LDMSGADDR family) without an intervening guard.
        all_instructions = module.all_instructions()
        for addr_idx in addr_store_indices:
            # Walk backward from addr_store to find the address source
            taint_source_idx = self._find_addr_taint_source(
                addr_idx, all_instructions
            )
            if taint_source_idx is None:
                continue

            # Check taint flow via dataflow graph if available
            if module.dataflow_graph is not None:
                if not module.get_taint_flow(taint_source_idx, addr_idx):
                    continue  # no confirmed taint propagation
                if module.get_guarded_taint_at(addr_idx):
                    continue  # taint was guarded

            # CFG-based guard check as fallback / reinforcement
            guarded = self._is_guarded_between(
                facts, taint_source_idx, send_inst.index
            )
            if guarded:
                continue

            # Map back to InstructionFact for the finding
            target_fact = self._resolve_instruction_fact(
                facts, send_inst.index
            )
            if target_fact is None:
                continue

            findings.append(
                self._build_vuln(
                    message=(
                        "Destination address in SENDRAWMSG message cell "
                        "flows from untrusted input (LDMSGADDR) without "
                        "validation."
                    ),
                    instruction=target_fact,
                    remediation=(
                        "Validate the destination address against a whitelist "
                        "or known contract address before sending."
                    ),
                    confidence=Confidence.MEDIUM,
                    extra={
                        "signal": "unvalidated_destination",
                        "taint_source_index": taint_source_idx,
                        "addr_store_index": addr_idx,
                        "send_index": send_inst.index,
                    },
                )
            )

        return findings

    def _find_addr_taint_source(
        self,
        addr_store_idx: int,
        all_instructions: List["TVMInstruction"],
    ) -> Optional[int]:
        """Walk backward from an address store to find an LDMSGADDR source."""
        # Simple backward scan within a limited window
        window = 30
        for inst in reversed(all_instructions):
            if inst.index >= addr_store_idx:
                continue
            if inst.index < addr_store_idx - window:
                break
            opcode = inst.opcode.upper()
            if opcode in _ADDR_LOAD_OPCODES:
                return inst.index
        return None

    # ---- Check 2 helpers (TASIR) --------------------------------------

    def _check_cell_overflow_tasir(
        self,
        facts: AnalysisFacts,
        send_inst: "TVMInstruction",
        builder_chain: List["TVMInstruction"],
        idx_to_inst: Dict[int, "TVMInstruction"],
    ) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        total_bits = 0
        total_refs = 0
        saw_endc = False

        for inst in builder_chain:
            opcode = inst.opcode.upper()
            if opcode in _NEWC_OPCODES:
                total_bits = 0
                total_refs = 0
                continue
            if opcode in _ENDC_OPCODES:
                saw_endc = True
                # Check limits at finalization
                if total_bits > _MAX_CELL_BITS or total_refs > _MAX_CELL_REFS:
                    target_fact = self._resolve_instruction_fact(
                        facts, inst.index
                    )
                    if target_fact is not None:
                        findings.append(self._build_overflow_finding(
                            target_fact, total_bits, total_refs,
                            send_inst.index,
                        ))
                continue

            cost = _STORE_BIT_COSTS.get(opcode)
            if cost is not None:
                bit_cost, ref_cost = cost
                if bit_cost is not None:
                    total_bits += bit_cost
                elif opcode in ("STI", "STU", "STIR", "STUR"):
                    # Extract bit width from immediate argument
                    width = self._get_immediate_int(inst)
                    if width is not None:
                        total_bits += width
                total_refs += ref_cost

        # Final check only when no ENDC was found in chain.
        if (not saw_endc) and (total_bits > _MAX_CELL_BITS or total_refs > _MAX_CELL_REFS):
            target_fact = self._resolve_instruction_fact(
                facts, send_inst.index
            )
            if target_fact is not None:
                findings.append(self._build_overflow_finding(
                    target_fact, total_bits, total_refs, send_inst.index,
                ))

        return findings

    @staticmethod
    def _get_immediate_int(inst: "TVMInstruction") -> Optional[int]:
        """Extract the first integer immediate from a TASIR instruction."""
        for imm in inst.immediates:
            if isinstance(imm, int):
                return imm
        # Fallback: check original_args
        for arg in inst.original_args:
            if isinstance(arg, int):
                return arg
        return None

    def _build_overflow_finding(
        self,
        instruction,
        total_bits: int,
        total_refs: int,
        send_index: int,
    ) -> Vulnerability:
        parts = []
        if total_bits > _MAX_CELL_BITS:
            parts.append(f"{total_bits} bits (limit {_MAX_CELL_BITS})")
        if total_refs > _MAX_CELL_REFS:
            parts.append(f"{total_refs} refs (limit {_MAX_CELL_REFS})")
        detail = " and ".join(parts)
        return self._build_vuln(
            message=(
                f"Message cell builder may overflow: {detail}. "
                f"This will cause a runtime exception in SENDRAWMSG."
            ),
            instruction=instruction,
            severity="medium",
            remediation=(
                "Split the message payload across multiple cells using "
                "STREF, or reduce the data stored in a single cell."
            ),
            confidence=Confidence.HIGH,
            extra={
                "signal": "cell_overflow",
                "total_bits": total_bits,
                "total_refs": total_refs,
                "send_index": send_index,
            },
        )

    # ==================================================================
    # Legacy opcode-level detection
    # ==================================================================

    def _detect_legacy(self, facts: AnalysisFacts) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        # Find SENDRAWMSG instructions
        send_facts: List[object] = []
        for inst_fact in facts.instructions:
            if inst_fact.opcode.upper() in ("SENDRAWMSG", "SENDMSG"):
                send_facts.append(inst_fact)

        if not send_facts:
            return findings

        instructions = facts.instructions
        idx_to_fact: Dict[int, object] = {
            inst.index: inst for inst in instructions
        }

        for send_fact in send_facts:
            send_idx = send_fact.index
            # ---- Check 1: unvalidated address ----
            chain = self._trace_builder_chain_legacy(
                send_idx,
                instructions,
                context_id=send_fact.continuation_id,
            )
            addr_findings = self._check_unvalidated_addr_legacy(
                facts, send_idx, chain, idx_to_fact
            )
            findings.extend(addr_findings)

            # ---- Check 2: cell overflow ----
            overflow_findings = self._check_cell_overflow_legacy(
                facts, send_idx, chain, idx_to_fact
            )
            findings.extend(overflow_findings)

        return findings

    def _trace_builder_chain_legacy(
        self,
        send_idx: int,
        instructions: list,
        *,
        context_id: Optional[str],
    ) -> list:
        """Backward trace from send to NEWC through store instructions."""
        chain = []
        for inst_fact in reversed(instructions):
            if inst_fact.index >= send_idx:
                continue
            if inst_fact.continuation_id != context_id:
                continue
            opcode = inst_fact.opcode.upper()
            if opcode in _NEWC_OPCODES:
                chain.insert(0, inst_fact)
                break
            if opcode in _STORE_BIT_COSTS or opcode in _ENDC_OPCODES:
                chain.insert(0, inst_fact)
        return chain

    def _check_unvalidated_addr_legacy(
        self,
        facts: AnalysisFacts,
        send_idx: int,
        chain: list,
        idx_to_fact: Dict[int, object],
    ) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        addr_store_indices: List[int] = []
        for inst_fact in chain:
            if inst_fact.opcode.upper() in _ADDR_STORE_OPCODES:
                addr_store_indices.append(inst_fact.index)

        if not addr_store_indices:
            return findings

        for addr_idx in addr_store_indices:
            # Backward scan for LDMSGADDR source
            taint_source_idx = None
            window = 30
            for inst_fact in reversed(facts.instructions):
                if inst_fact.index >= addr_idx:
                    continue
                if inst_fact.index < addr_idx - window:
                    break
                if inst_fact.opcode.upper() in _ADDR_LOAD_OPCODES:
                    taint_source_idx = inst_fact.index
                    break

            if taint_source_idx is None:
                continue

            guarded = self._is_guarded_between(
                facts, taint_source_idx, send_idx
            )
            if guarded:
                continue

            send_fact = idx_to_fact.get(send_idx)
            if send_fact is None:
                continue

            findings.append(
                self._build_vuln(
                    message=(
                        "Destination address in SENDRAWMSG message cell "
                        "flows from untrusted input (LDMSGADDR) without "
                        "validation."
                    ),
                    instruction=send_fact,
                    remediation=(
                        "Validate the destination address against a whitelist "
                        "or known contract address before sending."
                    ),
                    confidence=Confidence.LOW,
                    extra={
                        "signal": "unvalidated_destination",
                        "taint_source_index": taint_source_idx,
                        "addr_store_index": addr_idx,
                        "send_index": send_idx,
                        "analysis_tier": "legacy",
                    },
                )
            )

        return findings

    def _check_cell_overflow_legacy(
        self,
        facts: AnalysisFacts,
        send_idx: int,
        chain: list,
        idx_to_fact: Dict[int, object],
    ) -> List[Vulnerability]:
        findings: List[Vulnerability] = []

        total_bits = 0
        total_refs = 0
        saw_endc = False

        for inst_fact in chain:
            opcode = inst_fact.opcode.upper()
            if opcode in _NEWC_OPCODES:
                total_bits = 0
                total_refs = 0
                continue
            if opcode in _ENDC_OPCODES:
                saw_endc = True
                if total_bits > _MAX_CELL_BITS or total_refs > _MAX_CELL_REFS:
                    findings.append(self._build_overflow_finding(
                        inst_fact, total_bits, total_refs, send_idx,
                    ))
                continue

            cost = _STORE_BIT_COSTS.get(opcode)
            if cost is not None:
                bit_cost, ref_cost = cost
                if bit_cost is not None:
                    total_bits += bit_cost
                elif opcode in ("STI", "STU", "STIR", "STUR"):
                    width = self._get_immediate_int_legacy(inst_fact)
                    if width is not None:
                        total_bits += width
                total_refs += ref_cost

        # Check after chain only if no ENDC found
        if (not saw_endc) and (total_bits > _MAX_CELL_BITS or total_refs > _MAX_CELL_REFS):
            send_fact = idx_to_fact.get(send_idx)
            if send_fact is not None:
                findings.append(self._build_overflow_finding(
                    send_fact, total_bits, total_refs, send_idx,
                ))

        return findings

    @staticmethod
    def _get_immediate_int_legacy(inst_fact) -> Optional[int]:
        """Extract the first integer argument from a legacy InstructionFact."""
        args = getattr(inst_fact, "arguments", None)
        return extract_int_arg(args, 0)

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _is_guarded_between(
        self, facts: AnalysisFacts, source_idx: int, sink_idx: int
    ) -> bool:
        """Check if there is a guard (IF*/THROW*) between source and sink.

        Uses CFG traversal when basic blocks are available, otherwise falls
        back to a linear scan of guard events.
        """
        guard_events = facts.events_of("guard")
        guard_indices = {evt.instruction.index for evt in guard_events}

        # Prefer CFG-based validation when block structure is available.
        if facts.basic_blocks:
            block_map: Dict[int, BasicBlock] = {
                b.id: b for b in facts.basic_blocks
            }
            instr_to_block = build_instruction_to_block_map(
                facts.basic_blocks
            )
            start_block_id = instr_to_block.get(source_idx)
            if start_block_id is not None:
                result = traverse_cfg_for_unguarded_sinks(
                    block_map=block_map,
                    start_block_id=start_block_id,
                    start_from_idx=source_idx,
                    guard_indices=guard_indices,
                    sink_indices={sink_idx},
                    max_iterations=MAX_SEARCH_ITERATIONS,
                )
                # When CFG traversal is available, it is authoritative.
                return sink_idx not in result.flagged_indices

        # Fallback: linear check when CFG is unavailable/incomplete.
        for g_idx in guard_indices:
            if source_idx < g_idx < sink_idx:
                return True

        return False

    @staticmethod
    def _resolve_instruction_fact(
        facts: AnalysisFacts, index: int
    ) -> Optional[object]:
        """Map a TASIR instruction index back to an InstructionFact."""
        for inst_fact in facts.instructions:
            if inst_fact.index == index:
                return inst_fact
        return None
