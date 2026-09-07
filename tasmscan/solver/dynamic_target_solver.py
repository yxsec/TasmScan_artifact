"""Dynamic continuation target solver (v1)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ir.tasir_types import RegisterLocation, TVMInstruction, TVMModule
from .types import DynamicTargetSolveResult


# c3-register dispatch: push method_id then call/jump via c3.
# Names must match decoder output (instruction_table.json).
DICT_DISPATCH_OPCODES = {
    "CALLDICT",
    "CALLDICT_LONG",
    "JMPDICT",
    "JMPDICT_LONG",
}

# Cell/reference dispatch: target continuation is carried by a cell/reference value.
REF_DISPATCH_OPCODES = {
    "CALLREF",
    "JMPREF",
    "JMPREFDATA",
}

# Stack dispatch: pop continuation from stack and call/jump.
STACK_DISPATCH_OPCODES = {
    "EXECUTE",
    "JMPX",
    "JMPXARGS",
    "JMPXVARARGS",
    "JMPXDATA",
    "CALLXARGS",
    "CALLXVARARGS",
    "CALLCC",
    "CALLCCARGS",
    "CALLCCVARARGS",
    "BOOLEVAL",
}

# Suffix prefixes for parameterized call variants (e.g. CALLXARGS_1).
_STACK_DISPATCH_SUFFIX_PREFIXES = ("CALLXARGS_", "CALLCCARGS_")

NormalizedReason = Tuple[str, int, str, str]


class DynamicTargetSolver:
    """Lightweight symbolic/summary solver for dynamic continuation targets."""

    CTRL_REG_C3 = 3
    # Candidate cap for conservative dynamic dispatch expansion.
    MAX_C3_DISPATCH_CANDIDATES = 64
    CONT_CREATE_OPCODES = {
        "PUSHCONT",
        "PUSHREFCONT",
        "PUSHREF",
        "BLESS",
        "BLESSARGS",
        "BLESSVARARGS",
    }
    _STACK_ONE_HOP_PASSTHROUGH_OPCODES = {
        "NOP",
        "DUP",
        "PUSH",
        "OVER",
        "XCHG",
        "SWAP",
    }

    def __init__(self) -> None:
        self._cache: Dict[Tuple, DynamicTargetSolveResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.solve_count = 0
        self.solved_count = 0
        self.unsolved_count = 0

    def solve(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
        prev_inst: Optional[TVMInstruction] = None,
        pre_prev_inst: Optional[TVMInstruction] = None,
        uncertain_reasons: Optional[List[Any]] = None,
    ) -> DynamicTargetSolveResult:
        """Solve candidate continuation targets for one instruction."""
        self.solve_count += 1
        upper = inst.opcode.upper()
        method_id = self._extract_method_id(inst)
        prev_sig = self._prev_signature(prev_inst)
        pre_prev_sig = self._prev_method_hint_signature(pre_prev_inst)
        inst_cont_refs_sig = tuple(sorted(
            ref for ref in inst.continuation_refs if isinstance(ref, str)
        ))
        normalized_reasons = self._normalize_uncertain_reasons(uncertain_reasons)
        reasons_sig = tuple(normalized_reasons)
        cache_key = (
            upper,
            context_id,
            method_id,
            inst_cont_refs_sig,
            prev_sig,
            pre_prev_sig,
            tuple(self._method_context_candidates(module)),
            reasons_sig,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            result = replace(cached, from_cache=True)
            self._update_outcome_stats(result)
            return result

        self.cache_misses += 1

        if normalized_reasons:
            reason_result, should_fallback = self._solve_by_reason(
                inst=inst,
                module=module,
                context_id=context_id,
                reasons=normalized_reasons,
            )
            if not should_fallback:
                self._cache[cache_key] = replace(reason_result, from_cache=False)
                self._update_outcome_stats(reason_result)
                return reason_result

        is_stack_dispatch = (
            upper in STACK_DISPATCH_OPCODES
            or any(upper.startswith(p) for p in _STACK_DISPATCH_SUFFIX_PREFIXES)
        )

        if upper in DICT_DISPATCH_OPCODES:
            result = self._solve_dict_dispatch(inst, module, context_id, method_id)
        elif upper in REF_DISPATCH_OPCODES:
            result = self._solve_ref_dispatch(inst, module, context_id, prev_inst, pre_prev_inst)
        elif is_stack_dispatch:
            result = self._solve_stack_dispatch(
                inst, module, context_id, prev_inst, pre_prev_inst
            )
        else:
            result = DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="unsupported_opcode_family",
                reason="opcode_not_modeled",
            )

        self._cache[cache_key] = replace(result, from_cache=False)
        self._update_outcome_stats(result)
        return result

    def stats(self) -> Dict[str, int]:
        """Return aggregate solver statistics."""
        return {
            "solve_count": self.solve_count,
            "solved_count": self.solved_count,
            "unsolved_count": self.unsolved_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self._cache),
        }

    def _solve_dict_dispatch(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
        method_id: Optional[int],
    ) -> DynamicTargetSolveResult:
        if method_id is None:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="dict_dispatch_immediate",
                reason="missing_method_id_immediate",
            )

        method_context_map = self._method_context_map(module)
        target_ctx = method_context_map.get(method_id)
        if target_ctx is None:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unsat",
                strategy="dict_dispatch_immediate",
                reason="method_id_not_in_module",
                model={"method_id": method_id},
            )

        return DynamicTargetSolveResult(
            instruction_index=inst.index,
            opcode=inst.opcode,
            status="sat",
            strategy="dict_dispatch_immediate",
            candidate_continuations=[target_ctx],
            model={"method_id": method_id, "target_context": target_ctx},
        )

    def _solve_stack_dispatch(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
        prev_inst: Optional[TVMInstruction],
        pre_prev_inst: Optional[TVMInstruction],
    ) -> DynamicTargetSolveResult:
        direct_stack_candidates = self._resolve_stack_neighbor_candidates(
            module=module,
            prev_inst=prev_inst,
            pre_prev_inst=pre_prev_inst,
        )
        if direct_stack_candidates:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="sat",
                strategy="stack_dispatch_prev_cont_create",
                candidate_continuations=direct_stack_candidates,
                model={
                    "candidate_count": len(direct_stack_candidates),
                    "source": "neighbor_cont_create",
                },
            )

        # Common dynamic-code idiom:
        #   ... <cell_or_slice>; CTOS; BLESS; EXECUTE
        # We cannot recover a single exact target, so conservatively fall back
        # to known method contexts in this module.
        if self._is_bless_execute_pattern(prev_inst, pre_prev_inst):
            return self._solve_bless_execute_dispatch(inst, module, context_id)

        if not self._is_pushctr_c3(prev_inst):
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="stack_dispatch_pattern",
                reason="unsupported_stack_target_pattern",
            )

        # Common TVM dispatch idiom:
        #   PUSHINT <method_id>; PUSHCTR c3; EXECUTE
        # Prefer this precise hint over broad c3 dictionary expansion.
        hinted_method_id = self._extract_method_hint(pre_prev_inst)
        if hinted_method_id is not None:
            method_context_map = self._method_context_map(module)
            target_ctx = method_context_map.get(hinted_method_id)
            if target_ctx is not None:
                return DynamicTargetSolveResult(
                    instruction_index=inst.index,
                    opcode=inst.opcode,
                    status="sat",
                    strategy="pushctr_c3_method_id_hint",
                    candidate_continuations=[target_ctx],
                    model={
                        "source_register": "c3",
                        "hinted_method_id": hinted_method_id,
                        "target_context": target_ctx,
                    },
                )
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unsat",
                strategy="pushctr_c3_method_id_hint",
                reason="method_id_not_in_module",
                model={"hinted_method_id": hinted_method_id},
            )

        candidates = self._method_context_candidates(module)
        if not candidates:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="pushctr_c3_method_dict",
                reason="no_method_context_candidates",
            )

        # Prefer inter-context dispatch targets to avoid adding trivial self loops.
        filtered_candidates = tuple(c for c in candidates if c != context_id)
        if filtered_candidates:
            candidates = filtered_candidates

        if len(candidates) > self.MAX_C3_DISPATCH_CANDIDATES:
            if inst.opcode.upper() == "EXECUTE":
                bounded = list(candidates[: self.MAX_C3_DISPATCH_CANDIDATES])
                return DynamicTargetSolveResult(
                    instruction_index=inst.index,
                    opcode=inst.opcode,
                    status="unknown",
                    strategy="pushctr_c3_method_dict_truncated",
                    reason="candidate_set_truncated",
                    candidate_continuations=bounded,
                    model={
                        "source_register": "c3",
                        "context_id": context_id,
                        "candidate_count": len(candidates),
                        "truncated_to": len(bounded),
                    },
                )
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="pushctr_c3_method_dict",
                reason="candidate_set_too_large",
                model={"candidate_count": len(candidates)},
            )

        return DynamicTargetSolveResult(
            instruction_index=inst.index,
            opcode=inst.opcode,
            status="sat",
            strategy="pushctr_c3_method_dict",
            candidate_continuations=list(candidates),
            model={
                "source_register": "c3",
                "context_id": context_id,
                "candidate_count": len(candidates),
            },
        )

    def _solve_bless_execute_dispatch(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
    ) -> DynamicTargetSolveResult:
        candidates = self._method_context_candidates(module)
        if not candidates:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="bless_execute_dynamic_cell",
                reason="no_method_context_candidates",
            )

        filtered_candidates = tuple(c for c in candidates if c != context_id)
        if filtered_candidates:
            candidates = filtered_candidates

        if len(candidates) > self.MAX_C3_DISPATCH_CANDIDATES:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="bless_execute_dynamic_cell",
                reason="candidate_set_too_large",
                model={"candidate_count": len(candidates)},
            )

        return DynamicTargetSolveResult(
            instruction_index=inst.index,
            opcode=inst.opcode,
            status="sat",
            strategy="bless_execute_dynamic_cell",
            candidate_continuations=list(candidates),
            model={
                "source": "bless_cell",
                "context_id": context_id,
                "candidate_count": len(candidates),
            },
        )

    def _solve_ref_dispatch(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
        prev_inst: Optional[TVMInstruction],
        pre_prev_inst: Optional[TVMInstruction],
    ) -> DynamicTargetSolveResult:
        candidates: List[str] = []
        direct_known = self._extract_known_continuation_refs(inst, module)
        if direct_known:
            candidates.extend(direct_known)

        neighbor_candidates = self._resolve_stack_neighbor_candidates(
            module=module,
            prev_inst=prev_inst,
            pre_prev_inst=pre_prev_inst,
        )
        if neighbor_candidates:
            candidates.extend(neighbor_candidates)

        candidates = list(dict.fromkeys(candidates))
        if context_id in candidates and len(candidates) > 1:
            candidates = [c for c in candidates if c != context_id]
        if not candidates:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="ref_dispatch_neighbor_pattern",
                reason="unsupported_ref_target_pattern",
            )
        if len(candidates) > self.MAX_C3_DISPATCH_CANDIDATES:
            return DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="ref_dispatch_neighbor_pattern",
                reason="candidate_set_too_large",
                model={"candidate_count": len(candidates)},
            )
        return DynamicTargetSolveResult(
            instruction_index=inst.index,
            opcode=inst.opcode,
            status="sat",
            strategy="ref_dispatch_neighbor_pattern",
            candidate_continuations=candidates,
            model={"candidate_count": len(candidates)},
        )

    def _resolve_stack_neighbor_candidates(
        self,
        *,
        module: TVMModule,
        prev_inst: Optional[TVMInstruction],
        pre_prev_inst: Optional[TVMInstruction],
    ) -> List[str]:
        """Resolve continuation targets from immediate stack neighborhood."""
        direct_prev = self._extract_known_continuation_refs(prev_inst, module)
        if direct_prev and self._is_cont_create(prev_inst):
            return direct_prev

        # One-hop passthrough (e.g., PUSHCONT ; NOP ; EXECUTE).
        if prev_inst is not None and pre_prev_inst is not None:
            if (
                prev_inst.opcode.upper() in self._STACK_ONE_HOP_PASSTHROUGH_OPCODES
                and self._is_cont_create(pre_prev_inst)
            ):
                one_hop = self._extract_known_continuation_refs(pre_prev_inst, module)
                if one_hop:
                    return one_hop
        return []

    def _extract_known_continuation_refs(
        self,
        inst: Optional[TVMInstruction],
        module: TVMModule,
    ) -> List[str]:
        if inst is None:
            return []
        known: List[str] = []
        for ref in inst.continuation_refs:
            if not isinstance(ref, str) or not ref or ref.startswith("__"):
                continue
            if self._is_known_continuation(ref, module) and ref not in known:
                known.append(ref)
        return known

    def _is_known_continuation(self, cont_ref: str, module: TVMModule) -> bool:
        if cont_ref in module.cross_function_continuations:
            return True
        for func in module.functions.values():
            if cont_ref in func.continuations:
                return True
        return False

    def _is_cont_create(self, inst: Optional[TVMInstruction]) -> bool:
        if inst is None:
            return False
        return inst.opcode.upper() in self.CONT_CREATE_OPCODES

    def _solve_by_reason(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        context_id: str,
        reasons: Sequence[NormalizedReason],
    ) -> Tuple[DynamicTargetSolveResult, bool]:
        """Solve using uncertain-branch reason metadata before legacy pattern matching."""
        all_candidates: List[str] = []
        strategies: List[str] = []
        should_fallback = False
        handled_reason = False
        method_candidates = self._method_context_candidates(module)

        for reason, source_index, source_opcode, detail in reasons:
            lower_reason = reason.lower()
            if lower_reason == "bless":
                handled_reason = True
                filtered = tuple(c for c in method_candidates if c != context_id)
                if filtered:
                    all_candidates.extend(filtered)
                else:
                    all_candidates.extend(method_candidates)
                strategies.append(f"bless_from_{source_index}")
                continue

            if lower_reason == "ctrl_reg_unknown":
                handled_reason = True
                all_candidates.extend(method_candidates)
                suffix = detail or "reg"
                strategies.append(f"ctrl_reg_{suffix}_from_{source_index}")
                continue

            if lower_reason == "dynamic_effect":
                handled_reason = True
                all_candidates.extend(method_candidates)
                strategies.append(f"dynamic_effect_from_{source_opcode}@{source_index}")
                continue

            if lower_reason in {"stack_invalidated", "stack_underflow", "terminator"}:
                handled_reason = True
                strategies.append(f"{lower_reason}_by_{source_opcode}@{source_index}")
                continue

            if lower_reason == "unresolved":
                strategies.append(f"unresolved@{source_index}")
                should_fallback = True
                continue

            strategies.append(f"unknown_reason_{reason}@{source_index}")
            should_fallback = True

        if all_candidates:
            unique_candidates = list(dict.fromkeys(all_candidates))
            if len(unique_candidates) > self.MAX_C3_DISPATCH_CANDIDATES:
                return (
                    DynamicTargetSolveResult(
                        instruction_index=inst.index,
                        opcode=inst.opcode,
                        status="unknown",
                        strategy="unified_reason_solver",
                        reason="too_many_candidates",
                        model={
                            "candidate_count": len(unique_candidates),
                            "strategies": strategies,
                            "reason_count": len(reasons),
                        },
                    ),
                    False,
                )
            return (
                DynamicTargetSolveResult(
                    instruction_index=inst.index,
                    opcode=inst.opcode,
                    status="sat",
                    strategy="unified_reason_solver",
                    candidate_continuations=unique_candidates,
                    model={
                        "candidate_count": len(unique_candidates),
                        "strategies": strategies,
                        "reason_count": len(reasons),
                    },
                ),
                False,
            )

        if should_fallback and not handled_reason:
            return (
                DynamicTargetSolveResult(
                    instruction_index=inst.index,
                    opcode=inst.opcode,
                    status="unknown",
                    strategy="unified_reason_solver",
                    reason="fallback_to_legacy",
                    model={"strategies": strategies, "reason_count": len(reasons)},
                ),
                True,
            )

        return (
            DynamicTargetSolveResult(
                instruction_index=inst.index,
                opcode=inst.opcode,
                status="unknown",
                strategy="unified_reason_solver",
                reason="no_candidates_from_reasons",
                model={"strategies": strategies, "reason_count": len(reasons)},
            ),
            False,
        )

    def _normalize_uncertain_reasons(
        self,
        uncertain_reasons: Optional[Sequence[Any]],
    ) -> List[NormalizedReason]:
        """Normalize reason payload from analyzer metadata into stable tuples."""
        normalized: List[NormalizedReason] = []
        for item in uncertain_reasons or []:
            if isinstance(item, dict):
                reason = str(item.get("reason", "") or "")
                source_index = self._as_int(item.get("source_index"), default=-1)
                source_opcode = str(item.get("source_opcode", "") or "")
                detail = str(item.get("detail", "") or "")
            else:
                reason = str(getattr(item, "reason", "") or "")
                source_index = self._as_int(getattr(item, "source_index", -1), default=-1)
                source_opcode = str(getattr(item, "source_opcode", "") or "")
                detail = str(getattr(item, "detail", "") or "")

            if not reason:
                continue
            normalized.append((reason, source_index, source_opcode, detail))
        return normalized

    @staticmethod
    def _as_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _method_context_candidates(self, module: TVMModule) -> Tuple[str, ...]:
        candidates = []
        for _, func in sorted(module.functions.items()):
            ctx = func.metadata.get("context")
            if isinstance(ctx, str) and ctx and ctx in module.cross_function_continuations:
                if ctx not in candidates:
                    candidates.append(ctx)
        return tuple(candidates)

    def _method_context_map(self, module: TVMModule) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for method_id, func in module.functions.items():
            ctx = func.metadata.get("context")
            if isinstance(ctx, str) and ctx and ctx in module.cross_function_continuations:
                mapping[method_id] = ctx
        return mapping

    def _extract_method_id(self, inst: TVMInstruction) -> Optional[int]:
        for imm in inst.immediates:
            if isinstance(imm, int):
                return int(imm)
        return None

    def _prev_signature(
        self, prev_inst: Optional[TVMInstruction]
    ) -> Tuple[Optional[str], Optional[int], Tuple[str, ...]]:
        if prev_inst is None:
            return (None, None, tuple())
        return (
            prev_inst.opcode.upper(),
            self._extract_ctrl_reg_index(prev_inst),
            tuple(sorted(prev_inst.continuation_refs)),
        )

    def _prev_method_hint_signature(
        self, pre_prev_inst: Optional[TVMInstruction]
    ) -> Tuple[Optional[str], Optional[int], Tuple[str, ...]]:
        if pre_prev_inst is None:
            return (None, None, tuple())
        return (
            pre_prev_inst.opcode.upper(),
            self._extract_method_hint(pre_prev_inst),
            tuple(sorted(pre_prev_inst.continuation_refs)),
        )

    def _is_pushctr_c3(self, inst: Optional[TVMInstruction]) -> bool:
        if inst is None:
            return False
        if inst.opcode.upper() != "PUSHCTR":
            return False
        return self._extract_ctrl_reg_index(inst) == self.CTRL_REG_C3

    def _is_bless_execute_pattern(
        self,
        prev_inst: Optional[TVMInstruction],
        pre_prev_inst: Optional[TVMInstruction],
    ) -> bool:
        if prev_inst is None or pre_prev_inst is None:
            return False
        if prev_inst.opcode.upper() not in {"BLESS", "BLESSARGS", "BLESSVARARGS"}:
            return False
        return pre_prev_inst.opcode.upper() == "CTOS"

    def _extract_method_hint(self, inst: Optional[TVMInstruction]) -> Optional[int]:
        if inst is None:
            return None
        upper = inst.opcode.upper()
        if not upper.startswith("PUSHINT"):
            return None
        return self._extract_method_id(inst)

    def _extract_ctrl_reg_index(self, inst: TVMInstruction) -> Optional[int]:
        for loc in self._iter_locations(inst):
            if isinstance(loc, RegisterLocation):
                return int(loc.index)
        for imm in inst.immediates:
            if isinstance(imm, int):
                return int(imm)
        return None

    def _iter_locations(self, inst: TVMInstruction) -> Iterable[object]:
        for loc in inst.inputs:
            yield loc
        for loc in inst.outputs:
            yield loc

    def _update_outcome_stats(self, result: DynamicTargetSolveResult) -> None:
        if result.status == "sat" and result.candidate_continuations:
            self.solved_count += 1
            return
        self.unsolved_count += 1
