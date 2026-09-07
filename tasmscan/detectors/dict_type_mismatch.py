"""Detector for dictionary/data cell type mismatches.

Detects two classes of issues:

1. **Dict operation on data cell**: A cell loaded via CTOS (not LDDICT/PLDDICT)
   is used as input to DICTGET/DICTIGET/DICTUGET etc.  The cell likely holds
   raw data, not a dictionary tree, and the dict operation will fail at runtime
   or produce garbage.

2. **Data operation on dict cell**: A cell loaded via LDDICT/PLDDICT is used as
   input to LDI/LDU/LDSLICE (treating dict internal structure as raw data).
   Reading raw integers from dictionary internals is almost always a bug.

Detection operates at the TASIR tier with a conservative, best-effort heuristic:
instruction-level origin tracking (not full SSA).  False negatives are possible
when values flow through stack shuffles or continuations; false positives are
minimised by only flagging clear origin->use chains within the same function.
"""

import logging
from typing import Dict, List, Optional, Set

from ..analyzer.facts import AnalysisFacts
from ..ir.tasir_types import InstructionKind, TVMFunction, TVMInstruction
from .base import Detector
from .results import Confidence, Vulnerability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opcode sets for origin classification
# ---------------------------------------------------------------------------

# Opcodes that produce a *dictionary* value (the output is a dict cell/slice)
_DICT_LOAD_OPCODES: Set[str] = frozenset({
    "LDDICT", "LDDICTS", "LDDICTQ",
    "PLDDICT", "PLDDICTS", "PLDDICTQ",
})

# Opcodes that convert a cell to a slice (general data path)
_CTOS_OPCODES: Set[str] = frozenset({
    "CTOS",
})

# Opcodes that load a child cell reference (general data path)
_LDREF_OPCODES: Set[str] = frozenset({
    "LDREF", "LDREFRTOS", "LDREFIDX",
    "PLDREFVAR", "PLDREFIDX",
})

# InstructionKinds that represent dictionary *use* (get/set/delete)
_DICT_USE_KINDS: Set[InstructionKind] = frozenset({
    InstructionKind.DICT_GET,
    InstructionKind.DICT_SET,
    InstructionKind.DICT_DELETE,
})

# InstructionKinds that represent raw data reading from a slice
_DATA_READ_KINDS: Set[InstructionKind] = frozenset({
    InstructionKind.CELL_LOAD,   # LDI, LDU, LDSLICE, LDREF, etc.
})

# Finer-grained: opcodes that read raw integers/bits (subset of CELL_LOAD)
_DATA_READ_OPCODES: Set[str] = frozenset({
    # Signed integer loads
    "LDI", "LDIX", "LDIQ", "LDIXQ",
    # Unsigned integer loads
    "LDU", "LDUX", "LDUQ", "LDUXQ", "PLDUZ",
    # Bit-slice loads
    "LDSLICE", "LDSLICEX", "LDSLICEXQ",
    # Preloads (also read raw data)
    "PLDI", "PLDIX", "PLDIQ", "PLDIXQ",
    "PLDU", "PLDUX", "PLDUQ", "PLDUXQ",
    "PLDSLICE", "PLDSLICEX", "PLDSLICEXQ",
})

# Opcodes that push control register c4 (persistent storage root)
_PUSHCTR_C4_OPCODES: Set[str] = frozenset({
    "PUSHCTR",  # with arg c4
    "PUSH",     # alias sometimes used
})


class DictTypeMismatchDetector(Detector):
    """Detects dictionary operations on data cells and data operations on dictionary cells.

    This is a conservative heuristic operating at the TASIR instruction level.
    It tracks the "origin" of values (dict-load vs data-load) per instruction
    index within each function, then flags suspicious cross-use patterns.
    """

    name = "dict_type_mismatch"
    default_severity = "medium"
    category = "security"
    description = (
        "Detects dictionary operations on data cells and "
        "data operations on dictionary cells"
    )

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        module = self.get_tasir(facts)
        if module is None:
            return []

        findings: List[Vulnerability] = []
        for func in module.functions.values():
            findings.extend(self._analyse_function(func, facts))
        global_blocks_by_context: Dict[str, List[TVMInstruction]] = {}
        for block in module.global_blocks:
            global_blocks_by_context.setdefault(block.context_id, []).extend(
                block.instructions
            )
        for context_id, instructions in global_blocks_by_context.items():
            findings.extend(
                self._analyse_instruction_sequence(
                    sorted(instructions, key=lambda i: i.index),
                    facts,
                    method_id=None,
                    context_id=context_id,
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Per-function analysis
    # ------------------------------------------------------------------

    def _analyse_function(
        self, func: TVMFunction, facts: AnalysisFacts
    ) -> List[Vulnerability]:
        """Walk a single function's instructions and flag mismatches."""

        # Collect instructions in program order across all blocks
        all_instrs: List[TVMInstruction] = []
        for block in func.blocks.values():
            all_instrs.extend(block.instructions)
        all_instrs.sort(key=lambda i: i.index)

        return self._analyse_instruction_sequence(
            all_instrs,
            facts,
            method_id=func.method_id,
            context_id=None,
        )

    def _analyse_instruction_sequence(
        self,
        all_instrs: List[TVMInstruction],
        facts: AnalysisFacts,
        *,
        method_id: Optional[int],
        context_id: Optional[str],
    ) -> List[Vulnerability]:
        """Walk a TASIR instruction sequence and flag mismatches."""

        if not all_instrs:
            return []

        # Origin maps: instruction index -> origin tag
        # "dict"  = value came from LDDICT / PLDDICT family
        # "data"  = value came from CTOS / LDREF (general data cell)
        origins: Dict[int, str] = {}

        # Pass 1: classify origins
        for inst in all_instrs:
            opcode_upper = inst.opcode.upper()

            # Dict-load opcodes produce dict-origin values
            if opcode_upper in _DICT_LOAD_OPCODES:
                origins[inst.index] = "dict"
                continue

            # CTOS / LDREF produce data-origin values
            if opcode_upper in _CTOS_OPCODES or opcode_upper in _LDREF_OPCODES:
                origins[inst.index] = "data"
                continue

            # Track PUSHCTR c4 -> CTOS pattern
            # PUSHCTR c4 is a common pattern: the c4 register holds the
            # persistent storage root which typically contains *both* raw
            # data fields and dictionary sub-trees.  We mark CTOS after
            # PUSHCTR c4 as "data" (the default) since the developer must
            # then use LDDICT to extract dict portions.  This is already
            # handled by the CTOS branch above, but we track the pattern
            # for potential future refinement.
            if opcode_upper == "PUSHCTR":
                # Check if this pushes c4 specifically
                # original_args typically contains the register index
                args = inst.original_args
                if args and _is_c4_arg(args):
                    continue

        # Pass 2: detect mismatches
        findings: List[Vulnerability] = []

        # Build a quick lookup of instruction index -> instruction for facts
        facts_instr_map: Dict[int, object] = {
            i.index: i for i in facts.instructions
        }

        # We use a simple "nearest preceding origin" heuristic:
        # scan instructions in order and maintain the most recent origin.
        # When we encounter a dict-use or data-read instruction, check
        # if the most recent relevant origin is mismatched.
        #
        # This is deliberately conservative: stack shuffles, DUPs, and
        # cross-block flows can invalidate the heuristic, so we only
        # flag when the origin instruction is "close" (within a window).

        recent_dict_origin_idx: Optional[int] = None
        recent_data_origin_idx: Optional[int] = None

        # Maximum instruction distance for origin tracking
        MAX_ORIGIN_DISTANCE = 12

        for inst in all_instrs:
            idx = inst.index
            opcode_upper = inst.opcode.upper()

            # Update recent origins
            if idx in origins:
                tag = origins[idx]
                if tag == "dict":
                    recent_dict_origin_idx = idx
                elif tag == "data":
                    recent_data_origin_idx = idx

            # Check 1: Dict operation on data-origin value
            # If we see a DICT_GET/DICT_SET/DICT_DELETE and the most recent
            # origin is "data" (not "dict"), flag it.
            if inst.kind in _DICT_USE_KINDS:
                if (
                    recent_data_origin_idx is not None
                    and (recent_dict_origin_idx is None
                         or recent_data_origin_idx > recent_dict_origin_idx)
                    and (idx - recent_data_origin_idx) <= MAX_ORIGIN_DISTANCE
                ):
                    # The dict operand likely came from a data cell
                    target_instr = facts_instr_map.get(idx)
                    if target_instr is not None:
                        findings.append(
                            self._build_vuln(
                                message=(
                                    f"Dictionary operation {inst.opcode} appears to use "
                                    f"a data cell/slice (loaded via CTOS/LDREF at "
                                    f"instruction {recent_data_origin_idx}) as dictionary input. "
                                    f"The value may not be a valid dictionary."
                                ),
                                instruction=target_instr,
                                confidence=Confidence.LOW,
                                remediation=(
                                    "Ensure the dictionary operand comes from "
                                    "LDDICT/PLDDICT rather than raw CTOS/LDREF. "
                                    "If intentional, load the dict with the appropriate "
                                    "dictionary load instruction."
                                ),
                                extra={
                                    "issue_type": "dict_op_on_data_cell",
                                    "dict_opcode": inst.opcode,
                                    "origin_instruction": recent_data_origin_idx,
                                    "method_id": method_id,
                                    **({"context_id": context_id} if context_id is not None else {}),
                                },
                            )
                        )

            # Check 2: Data-read operation on dict-origin value
            # If we see LDI/LDU/LDSLICE and the most recent origin is
            # "dict" (not "data"), flag it.
            if opcode_upper in _DATA_READ_OPCODES:
                if (
                    recent_dict_origin_idx is not None
                    and (recent_data_origin_idx is None
                         or recent_dict_origin_idx > recent_data_origin_idx)
                    and (idx - recent_dict_origin_idx) <= MAX_ORIGIN_DISTANCE
                ):
                    # The slice being read likely came from a dict load
                    target_instr = facts_instr_map.get(idx)
                    if target_instr is not None:
                        findings.append(
                            self._build_vuln(
                                message=(
                                    f"Data read operation {inst.opcode} appears to read "
                                    f"from a dictionary cell/slice (loaded via LDDICT/PLDDICT at "
                                    f"instruction {recent_dict_origin_idx}). "
                                    f"Reading raw integers from dictionary internals is likely a bug."
                                ),
                                instruction=target_instr,
                                confidence=Confidence.LOW,
                                remediation=(
                                    "Use DICTGET/DICTIGET/DICTUGET to access dictionary "
                                    "entries instead of reading raw data from the dict cell. "
                                    "If this is intentional (e.g., inspecting dict metadata), "
                                    "consider suppressing this finding."
                                ),
                                extra={
                                    "issue_type": "data_op_on_dict_cell",
                                    "data_opcode": inst.opcode,
                                    "origin_instruction": recent_dict_origin_idx,
                                    "method_id": method_id,
                                    **({"context_id": context_id} if context_id is not None else {}),
                                },
                            )
                        )

        return findings


def _is_c4_arg(args: list) -> bool:
    """Check if instruction arguments reference control register c4."""
    for arg in args:
        if isinstance(arg, int) and arg == 4:
            return True
        if isinstance(arg, str):
            arg_lower = arg.lower()
            if arg_lower in ("c4", "4"):
                return True
    return False
