"""Stack underflow detector"""
from typing import Dict, List, Optional

from ..analyzer.facts import AnalysisFacts
from ..config import (
    CORE_SENSITIVE_OPCODES,
    CUMULATIVE_STACK_CONSUMPTION_THRESHOLD,
    STACK_UNCERTAINTY_THRESHOLD,
    STACK_SHALLOW_UNDERFLOW_THRESHOLD,
)
from ..ir.stack_effects import StackEffect, get_stack_effect
from .base import Detector
from .results import Vulnerability


class StackUnderflowDetector(Detector):
    """Detects potential stack underflows and uncertain stack states (code quality check)."""

    name = "stack_underflow"
    category = "code_quality"  # Distinguishes from security vulnerabilities
    enabled_by_default = True  # Enabled when code_quality is included
    default_severity = "high"
    description = "Detects potential stack underflows (code quality check)"

    # Sensitive operations that require known stack state
    # Note: Only include opcodes that actually exist in TVM spec (cp0.json)
    # SENDRAWMSGIMM, ACCEPTQ, SETGASLIMITVAR do not exist
    # CALL* opcodes intentionally excluded: they are very common in normal code,
    # and flagging them under uncertain stack state would cause too many false positives.
    # They will still be detected if height_min definitively goes negative.
    # Derived from CORE_SENSITIVE_OPCODES (config.py) - single source of truth
    SENSITIVE_OPCODES = set(CORE_SENSITIVE_OPCODES)

    # THROW instructions that legitimately terminate with negative stack
    # These pop values and then terminate execution, so negative stack is acceptable
    # Note: Include all THROW variants from TVM spec (cp0.json)
    THROW_INSTRUCTIONS = {
        # Standard throw (with immediate exception code)
        "THROW", "THROWIF", "THROWIFNOT",
        # Short forms (compact encoding)
        "THROWIF_SHORT", "THROWIFNOT_SHORT",
        # With argument (exception value on stack)
        "THROWARG", "THROWARGIF", "THROWARGIFNOT",
        # Any exception code (code from stack)
        "THROWANY", "THROWANYIF", "THROWANYIFNOT",
        # Any with argument (both code and value from stack)
        "THROWARGANY", "THROWARGANYIF", "THROWARGANYIFNOT",
    }

    def _build_tasir_stack_effect_index(self, facts: AnalysisFacts) -> Optional[Dict[int, "StackEffect"]]:
        """
        Build an index mapping instruction index to stack effect from TASIR.

        Returns:
            Dict mapping instruction index to StackEffect, or None if TASIR unavailable
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        effect_index = {}
        for inst in module.all_instructions():
            if inst.stack_effect is not None:
                effect_index[inst.index] = inst.stack_effect
        return effect_index if effect_index else None

    def _get_stack_effect_from_tasir(
        self,
        tasir_index: Optional[Dict[int, "StackEffect"]],
        instruction_index: int
    ) -> Optional["StackEffect"]:
        """
        Get stack effect from TASIR index if available.

        Args:
            tasir_index: Pre-built index from _build_tasir_stack_effect_index
            instruction_index: Index of the instruction

        Returns:
            StackEffect if found in TASIR, None to use fallback
        """
        if tasir_index is None:
            return None
        return tasir_index.get(instruction_index)

    def _min_required_inputs_tasir(
        self,
        tasir_index: Optional[Dict[int, "StackEffect"]],
        instruction_index: int
    ) -> Optional[int]:
        """
        Get minimum required inputs from TASIR if available.

        Args:
            tasir_index: Pre-built index from _build_tasir_stack_effect_index
            instruction_index: Index of the instruction

        Returns:
            min_inputs if found in TASIR, None to use fallback
        """
        effect = self._get_stack_effect_from_tasir(tasir_index, instruction_index)
        if effect is not None and effect.min_inputs is not None:
            return effect.min_inputs
        return None

    @staticmethod
    def _min_required_inputs(opcode: str) -> Optional[int]:
        try:
            effect = get_stack_effect(opcode)
        except (AttributeError, KeyError):
            return None
        if effect and effect.min_inputs is not None:
            return effect.min_inputs
        return None

    @staticmethod
    def _estimate_height_before(state, use_max: bool = False) -> Optional[int]:
        """Estimate stack height before an instruction from its effect.

        Args:
            state: StackState for the instruction
            use_max: If True, use height_max (upper bound); if False, use height_min

        Returns:
            Estimated height before the instruction, or None if unbounded/unknown.

        Note:
            When use_max=True and height_max is None, this means the upper bound
            is unbounded (infinity), so we return None rather than falling back
            to height_before, which would incorrectly treat unbounded as bounded.
        """
        value = state.height_max if use_max else state.height_min
        if value is None:
            # For use_max=True: None means unbounded (no upper limit)
            # For use_max=False: None means we don't know the minimum
            # In both cases, don't fall back to height_before as it would
            # lose the semantic meaning of "unbounded" vs "bounded"
            if use_max:
                return None  # Unbounded upper limit, cannot estimate
            # For min, fall back to height_before as a conservative estimate
            value = state.height_before
        if value is None:
            return None
        if state.delta is None:
            return value
        return value - state.delta

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        reported_indices = set()
        instruction_by_index = {inst.index: inst for inst in facts.instructions}

        # Build taint context for annotating findings
        module = self.get_tasir(facts)
        _tainted_set: set = set()
        if module:
            _tainted_set = set(module.get_tainted_instructions())

        # Build TASIR stack effect index once for efficient lookups
        # This avoids repeated TASIR traversal during the analysis passes
        tasir_index = self._build_tasir_stack_effect_index(facts)

        # Pass 1: Original single-instruction underflow checks
        for state in facts.stack_states:
            instruction = instruction_by_index.get(state.instruction_index)
            if instruction is None:
                continue

            # Use height_min for underflow detection (worst-case analysis)
            # If height_min < 0, stack MIGHT underflow even in the best interpretation
            underflow_detected = False

            # Case 1: Check height_min (range-based detection)
            # This catches both definite and potential underflows
            if state.height_min is not None and state.height_min < 0:
                # Skip THROW instructions - they terminate execution legitimately
                if instruction.opcode not in self.THROW_INSTRUCTIONS:
                    if state.instruction_index not in reported_indices:
                        if state.unknown:
                            # Uncertain underflow: filter out low-confidence reports
                            # When uncertainty range (height_max - height_min) is too large,
                            # the negative height_min is likely due to conservative estimates
                            # from dynamic instructions (DROPX, BLKDROP, etc.) accumulating
                            if state.height_max is None:
                                uncertainty_range = STACK_UNCERTAINTY_THRESHOLD + 1  # Unbounded = high uncertainty
                            else:
                                uncertainty_range = state.height_max - state.height_min

                            # Guard clause: Skip high-uncertainty reports
                            # See config.py for threshold rationale
                            should_skip = (
                                uncertainty_range > STACK_UNCERTAINTY_THRESHOLD  # High uncertainty
                                or state.height_min > STACK_SHALLOW_UNDERFLOW_THRESHOLD  # Shallow underflow
                            )
                            if not should_skip:
                                # Significant potential underflow even with uncertainty
                                findings.append(
                                    self._build_vuln(
                                        message=f"Potential underflow: stack might go negative (min_height={state.height_min}, delta~{state.delta})",
                                        instruction=instruction,
                                        remediation="Verify this operation has sufficient operands. Consider the worst-case stack depth.",
                                        severity="medium",
                                    )
                                )
                                reported_indices.add(state.instruction_index)
                        else:
                            # Definite underflow: we're certain this will underflow
                            findings.append(
                                self._build_vuln(
                                    message=f"Stack underflow detected: height becomes {state.height_min}",
                                    instruction=instruction,
                                    remediation="Ensure sufficient values on stack before this instruction.",
                                )
                            )
                            reported_indices.add(state.instruction_index)
                    underflow_detected = True

            # Fallback: Use height_after if height_min not available (backward compatibility)
            elif state.height_after is not None and state.height_after < 0:
                if instruction.opcode not in self.THROW_INSTRUCTIONS:
                    if state.instruction_index not in reported_indices:
                        findings.append(
                            self._build_vuln(
                                message=f"Stack underflow detected: height becomes {state.height_after}",
                                instruction=instruction,
                                remediation="Ensure sufficient values on stack before this instruction.",
                            )
                        )
                        reported_indices.add(state.instruction_index)
                    underflow_detected = True

            # Case 2: Unknown stack state before sensitive operation
            # We lost track of stack height, but a sensitive operation needs certain stack layout
            if not underflow_detected and state.unknown and instruction.opcode in self.SENSITIVE_OPCODES:
                if state.instruction_index not in reported_indices:
                    findings.append(
                        self._build_vuln(
                            message=f"Sensitive operation '{instruction.opcode}' with uncertain stack state",
                            instruction=instruction,
                            remediation="Analysis lost track of stack height before this critical operation. Verify stack manually.",
                            severity="medium",
                        )
                    )
                    reported_indices.add(state.instruction_index)

            # Case 3: Instruction requires minimum stack inputs (even if net delta is 0)
            if not underflow_detected and instruction.opcode not in self.THROW_INSTRUCTIONS:
                # Try TASIR first for stack effect (avoids repeated lookups)
                min_inputs = self._min_required_inputs_tasir(tasir_index, state.instruction_index)
                # Fallback to opcode-based lookup if TASIR unavailable
                if min_inputs is None:
                    min_inputs = self._min_required_inputs(instruction.opcode)
                if min_inputs is not None and min_inputs > 0:
                    min_before = self._estimate_height_before(state, use_max=False)
                    if min_before is None:
                        continue
                    max_before = self._estimate_height_before(state, use_max=True)

                    if min_before < min_inputs and state.instruction_index not in reported_indices:
                        # Definite underflow if even the max possible height is too small
                        if max_before is not None and max_before < min_inputs:
                            findings.append(
                                self._build_vuln(
                                    message=(
                                        f"Stack underflow detected: '{instruction.opcode}' "
                                        f"requires >= {min_inputs} stack items, but min is {min_before}"
                                    ),
                                    instruction=instruction,
                                    remediation="Ensure sufficient values on stack before this instruction.",
                                )
                            )
                            reported_indices.add(state.instruction_index)
                        else:
                            # Uncertain underflow: filter out low-confidence reports
                            if max_before is None:
                                uncertainty_range = STACK_UNCERTAINTY_THRESHOLD + 1  # Unbounded = high uncertainty
                            else:
                                uncertainty_range = max_before - min_before
                            if state.unknown and uncertainty_range > STACK_UNCERTAINTY_THRESHOLD:
                                continue
                            findings.append(
                                self._build_vuln(
                                    message=(
                                        f"Potential underflow: '{instruction.opcode}' "
                                        f"requires >= {min_inputs} stack items, but min is {min_before}"
                                    ),
                                    instruction=instruction,
                                    remediation="Verify this instruction has sufficient operands on all paths.",
                                    severity="medium",
                                )
                            )
                            reported_indices.add(state.instruction_index)

            # Case 4: Dynamic stack ops with unbounded inputs
            # If the stack height is at or below the minimum inputs, the dynamic
            # count could still cause an underflow (e.g., DROPX with only 1 item).
            if not underflow_detected and instruction.opcode not in self.THROW_INSTRUCTIONS:
                # Try TASIR first for stack effect
                effect = self._get_stack_effect_from_tasir(tasir_index, state.instruction_index)
                # Fallback to opcode-based lookup if TASIR unavailable
                if effect is None:
                    try:
                        from ..ir.stack_effects import get_stack_effect
                        effect = get_stack_effect(instruction.opcode)
                    except (ImportError, AttributeError, KeyError):
                        effect = None
                if effect and effect.is_dynamic and effect.max_inputs is None and effect.min_inputs:
                    min_before = self._estimate_height_before(state, use_max=False)
                    if min_before is not None and min_before <= effect.min_inputs:
                        if state.instruction_index not in reported_indices:
                            findings.append(
                                self._build_vuln(
                                    message=(
                                        f"Potential underflow: dynamic '{instruction.opcode}' "
                                        f"may consume more than available stack items (min={min_before})"
                                    ),
                                    instruction=instruction,
                                    remediation=(
                                        "Dynamic stack count may exceed current depth. "
                                        "Verify runtime bounds or add explicit checks."
                                    ),
                                    severity="medium",
                                )
                            )
                            reported_indices.add(state.instruction_index)

        # Pass 2: Detect cumulative stack consumption sequences
        # E.g., [-2, +1, -2, +1, -2] has net consumption of -4, which is flagged.
        # This is intentional: even if pushes occur mid-sequence, the overall deficit
        # represents potential underflow risk that static analysis cannot disprove.
        # Positive deltas reduce the deficit but don't reset the sequence start unless
        # cumulative becomes >= 0, because the sequence started at a consumption point.
        cumulative_delta = 0
        sequence_start_idx = None
        sequence_start_state = None

        for i, state in enumerate(facts.stack_states):
            instruction = instruction_by_index.get(state.instruction_index)
            if instruction is None:
                # Reset sequence on missing instruction
                cumulative_delta = 0
                sequence_start_idx = None
                sequence_start_state = None
                continue

            # Skip THROW instructions in sequence detection
            if instruction.opcode in self.THROW_INSTRUCTIONS:
                cumulative_delta = 0
                sequence_start_idx = None
                sequence_start_state = None
                continue

            # Track sequences of stack-consuming instructions
            if state.delta is not None and state.delta < 0:
                # Negative delta - consuming stack elements
                if sequence_start_idx is None:
                    sequence_start_idx = i
                    sequence_start_state = state
                cumulative_delta += state.delta

                # If cumulative consumption exceeds threshold, report the sequence start
                if cumulative_delta < -CUMULATIVE_STACK_CONSUMPTION_THRESHOLD:
                    start_instruction = instruction_by_index.get(sequence_start_state.instruction_index)
                    if start_instruction and sequence_start_state.instruction_index not in reported_indices:
                        findings.append(
                            self._build_vuln(
                                message=f"Potential stack underflow: net stack consumption of "
                                        f"{-cumulative_delta} elements (may include intermediate pushes)",
                                instruction=start_instruction,
                                remediation="Verify sufficient stack depth before this instruction sequence.",
                                severity="high",
                            )
                        )
                        reported_indices.add(sequence_start_state.instruction_index)
                        # Reset to avoid duplicate reports for overlapping sequences
                        cumulative_delta = 0
                        sequence_start_idx = None
                        sequence_start_state = None
            elif state.delta is not None and state.delta > 0:
                # Positive delta - add to cumulative instead of resetting
                cumulative_delta += state.delta
                # Only reset sequence start when cumulative becomes non-negative
                if cumulative_delta >= 0:
                    sequence_start_idx = None
                    sequence_start_state = None
            elif state.delta is None:
                # Unknown effect - reset sequence since we can't determine cumulative impact
                cumulative_delta = 0
                sequence_start_idx = None
                sequence_start_state = None
            # delta == 0: keep sequence going (e.g., SWAP, XCHG stack reordering ops)

        # Annotate findings on tainted paths
        if _tainted_set:
            for finding in findings:
                inst = finding.instruction
                if inst is not None and getattr(inst, 'index', None) in _tainted_set:
                    if finding.extra is None:
                        finding.extra = {}
                    finding.extra["on_tainted_path"] = True

        return findings
