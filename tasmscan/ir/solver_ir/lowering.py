"""
Lowering from TASIR (TVMModule) to Solver IR.

Current scope intentionally focuses on dynamic continuation dispatch core:
- control register c3 state threading
- PUSHCTR/POPCTR c3 modeling
- CALLDICT/JMPDICT family constraint extraction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..tasir_types import RegisterLocation, TVMBasicBlock, TVMInstruction, TVMModule
from .types import (
    ContinuationState,
    DynamicTargetObligation,
    SolverBlock,
    SolverConstraint,
    SolverInstruction,
    SolverModule,
    SolverSymbol,
)


DICT_DISPATCH_OPCODES: Set[str] = {
    "CALLDICT",
    "CALLDICT_LONG",
    "JMPDICT",
    "JMPDICT_LONG",
}


@dataclass
class _LoweringState:
    instruction_map: Dict[int, TVMInstruction] = field(default_factory=dict)
    block_by_id: Dict[int, TVMBasicBlock] = field(default_factory=dict)
    pred_map: Dict[int, List[int]] = field(default_factory=dict)
    succ_map: Dict[int, List[int]] = field(default_factory=dict)
    out_c3_by_block: Dict[int, str] = field(default_factory=dict)
    fallback_exit_symbol_by_block: Dict[int, str] = field(default_factory=dict)
    known_method_ids: Set[int] = field(default_factory=set)


class SolverIRLowering:
    """Convert TVMModule into SolverModule with symbolic constraints."""

    _CTRL_REG_C3 = 3

    def lower(self, module: TVMModule) -> SolverModule:
        state = _LoweringState()
        solver_module = SolverModule(
            metadata={
                "source": "solver_ir_lowering",
                "tasir_instruction_count": len(module.all_instructions()),
                "tasir_block_count": len(module.all_blocks()),
            }
        )

        self._index_module(module, state)
        self._init_solver_blocks(solver_module, state)
        self._thread_c3_state_and_lower_instructions(solver_module, state)
        self._set_summary_metadata(solver_module)
        return solver_module

    def _index_module(self, module: TVMModule, state: _LoweringState) -> None:
        state.known_method_ids = set(module.functions.keys())
        for block in module.all_blocks():
            state.block_by_id[block.id] = block
            state.pred_map.setdefault(block.id, [])
            state.succ_map.setdefault(block.id, [])
            for inst in block.instructions:
                state.instruction_map[inst.index] = inst

        for block in module.all_blocks():
            for edge in block.successors:
                src = edge.source_block
                dst = edge.target_block
                if src not in state.succ_map:
                    state.succ_map[src] = []
                if dst not in state.pred_map:
                    state.pred_map[dst] = []
                if dst not in state.succ_map[src]:
                    state.succ_map[src].append(dst)
                if src not in state.pred_map[dst]:
                    state.pred_map[dst].append(src)

    def _init_solver_blocks(self, solver_module: SolverModule, state: _LoweringState) -> None:
        for block_id in sorted(state.block_by_id):
            block = state.block_by_id[block_id]
            solver_block = SolverBlock(
                id=block_id,
                context_id=block.context_id,
                instruction_indices=[inst.index for inst in sorted(block.instructions, key=lambda i: i.index)],
                predecessors=sorted(state.pred_map.get(block_id, [])),
                successors=sorted(state.succ_map.get(block_id, [])),
            )
            solver_module.blocks[block_id] = solver_block

    def _thread_c3_state_and_lower_instructions(
        self, solver_module: SolverModule, state: _LoweringState
    ) -> None:
        for block_id in sorted(solver_module.blocks):
            block = solver_module.blocks[block_id]
            entry_c3 = self._resolve_entry_c3_symbol(solver_module, state, block_id)
            block.entry_state.ctrl_regs[self._CTRL_REG_C3] = entry_c3

            current_c3 = entry_c3
            for inst_idx in block.instruction_indices:
                inst = state.instruction_map[inst_idx]
                lowered = self._lower_instruction(
                    solver_module=solver_module,
                    state=state,
                    block=block,
                    inst=inst,
                    current_c3=current_c3,
                )
                solver_module.instructions[inst_idx] = lowered
                if lowered.writes:
                    current_c3 = lowered.writes[-1]

            block.exit_state.ctrl_regs[self._CTRL_REG_C3] = current_c3
            state.out_c3_by_block[block_id] = current_c3
            self._bind_fallback_exit_symbol(
                solver_module=solver_module,
                state=state,
                block_id=block_id,
                actual_exit_symbol=current_c3,
            )

    def _resolve_entry_c3_symbol(
        self,
        solver_module: SolverModule,
        state: _LoweringState,
        block_id: int,
    ) -> str:
        preds = state.pred_map.get(block_id, [])
        if not preds:
            name = self._fresh_symbol_name("c3_entry")
            self._declare_symbol(solver_module, name, "ContRef")
            return name

        if len(preds) == 1:
            pred = preds[0]
            out = state.out_c3_by_block.get(pred)
            if out:
                return out
            name = self._fresh_symbol_name(f"c3_b{block_id}_in")
            self._declare_symbol(solver_module, name, "ContRef")
            self._append_constraint(
                solver_module,
                SolverConstraint(
                    expression=f"(= {name} {self._fallback_block_exit_symbol(solver_module, state, pred)})",
                    metadata={"kind": "edge_c3_flow", "from_block": pred, "to_block": block_id},
                ),
            )
            return name

        # Multi-predecessor merge: explicit path choice guards.
        phi = self._fresh_symbol_name(f"c3_b{block_id}_phi")
        self._declare_symbol(solver_module, phi, "ContRef")

        edge_guards: List[str] = []
        for pred in sorted(preds):
            guard = f"edge_b{pred}_to_b{block_id}"
            self._declare_symbol(solver_module, guard, "Bool")
            edge_guards.append(guard)
            pred_out = state.out_c3_by_block.get(pred) or self._fallback_block_exit_symbol(
                solver_module,
                state,
                pred,
            )
            self._append_constraint(
                solver_module,
                SolverConstraint(
                    expression=f"(=> {guard} (= {phi} {pred_out}))",
                    metadata={
                        "kind": "edge_guarded_c3_flow",
                        "from_block": pred,
                        "to_block": block_id,
                    },
                ),
            )

        if edge_guards:
            self._append_constraint(
                solver_module,
                SolverConstraint(
                    expression=f"(or {' '.join(edge_guards)})",
                    metadata={"kind": "merge_requires_predecessor_path", "block_id": block_id},
                ),
            )
        return phi

    def _fallback_block_exit_symbol(
        self,
        solver_module: SolverModule,
        state: _LoweringState,
        block_id: int,
    ) -> str:
        existing = state.fallback_exit_symbol_by_block.get(block_id)
        if existing:
            return existing
        name = self._fresh_symbol_name(f"c3_b{block_id}_exit")
        self._declare_symbol(solver_module, name, "ContRef")
        state.fallback_exit_symbol_by_block[block_id] = name
        return name

    def _bind_fallback_exit_symbol(
        self,
        solver_module: SolverModule,
        state: _LoweringState,
        block_id: int,
        actual_exit_symbol: str,
    ) -> None:
        """Bind deferred fallback block-exit symbol to the final block exit symbol."""
        fallback = state.fallback_exit_symbol_by_block.get(block_id)
        if not fallback or fallback == actual_exit_symbol:
            return
        self._append_constraint(
            solver_module,
            SolverConstraint(
                expression=f"(= {fallback} {actual_exit_symbol})",
                metadata={
                    "kind": "late_block_exit_bind",
                    "block_id": block_id,
                    "fallback_symbol": fallback,
                    "actual_exit_symbol": actual_exit_symbol,
                },
            ),
        )

    def _lower_instruction(
        self,
        solver_module: SolverModule,
        state: _LoweringState,
        block: SolverBlock,
        inst: TVMInstruction,
        current_c3: str,
    ) -> SolverInstruction:
        opcode = inst.opcode.upper()
        lowered = SolverInstruction(
            index=inst.index,
            opcode=inst.opcode,
            block_id=block.id,
            context_id=block.context_id,
        )

        reg_idx = self._extract_control_register_index(inst)
        if opcode == "PUSHCTR" and reg_idx == self._CTRL_REG_C3:
            lowered.reads.append(current_c3)
            lowered.metadata["ctrl_reg"] = self._CTRL_REG_C3
            lowered.metadata["ctrl_reg_symbol"] = current_c3
            return lowered

        if opcode == "POPCTR" and reg_idx == self._CTRL_REG_C3:
            lowered.reads.append(current_c3)
            new_c3 = self._fresh_symbol_name(f"c3_i{inst.index}")
            stack_cont = self._fresh_symbol_name(f"stack_cont_i{inst.index}")
            self._declare_symbol(solver_module, new_c3, "ContRef")
            self._declare_symbol(solver_module, stack_cont, "ContRef")
            lowered.writes.append(new_c3)
            lowered.metadata["ctrl_reg"] = self._CTRL_REG_C3
            lowered.metadata["stack_source_symbol"] = stack_cont
            self._append_constraint(
                solver_module,
                SolverConstraint(
                    expression=f"(= {new_c3} {stack_cont})",
                    instruction_index=inst.index,
                    metadata={"kind": "popctr_c3_store"},
                ),
            )
            return lowered

        if opcode in DICT_DISPATCH_OPCODES:
            method_id = self._extract_method_id(inst)
            target_sym = self._fresh_symbol_name(f"target_i{inst.index}")
            self._declare_symbol(solver_module, target_sym, "ContRef")
            lowered.reads.append(current_c3)
            lowered.writes.append(target_sym)
            lowered.metadata.update(
                {
                    "dispatch_kind": "dict_lookup",
                    "ctrl_reg": self._CTRL_REG_C3,
                    "ctrl_reg_symbol": current_c3,
                    "target_symbol": target_sym,
                    "method_id": method_id,
                    "candidate_method_ids": (
                        [method_id]
                        if method_id is not None and method_id in state.known_method_ids
                        else []
                    ),
                }
            )
            if method_id is None:
                solver_module.obligations.append(
                    DynamicTargetObligation(
                        instruction_index=inst.index,
                        opcode=inst.opcode,
                        context_id=block.context_id,
                        ctrl_reg_index=self._CTRL_REG_C3,
                        ctrl_reg_symbol=current_c3,
                        method_id=None,
                        reason="missing_method_id_immediate",
                    )
                )
                return lowered

            if method_id not in state.known_method_ids:
                solver_module.obligations.append(
                    DynamicTargetObligation(
                        instruction_index=inst.index,
                        opcode=inst.opcode,
                        context_id=block.context_id,
                        ctrl_reg_index=self._CTRL_REG_C3,
                        ctrl_reg_symbol=current_c3,
                        method_id=method_id,
                        reason="method_id_not_in_module",
                    )
                )

            self._append_constraint(
                solver_module,
                SolverConstraint(
                    expression=f"(= {target_sym} (dict_lookup {current_c3} {method_id}))",
                    instruction_index=inst.index,
                    metadata={"kind": "dict_dispatch"},
                ),
            )
            return lowered

        # No special lowering; keep for traceability.
        return lowered

    def _extract_control_register_index(self, inst: TVMInstruction) -> Optional[int]:
        for loc in list(inst.inputs) + list(inst.outputs):
            if isinstance(loc, RegisterLocation):
                return int(loc.index)
        for imm in inst.immediates:
            if isinstance(imm, int):
                return imm
        return None

    def _extract_method_id(self, inst: TVMInstruction) -> Optional[int]:
        for imm in inst.immediates:
            if isinstance(imm, int):
                return imm
        return None

    def _declare_symbol(self, module: SolverModule, name: str, sort: str) -> None:
        existing = module.symbols.get(name)
        if existing is not None:
            return
        module.symbols[name] = SolverSymbol(name=name, sort=sort)

    def _append_constraint(self, module: SolverModule, constraint: SolverConstraint) -> None:
        module.constraints.append(constraint)

    def _fresh_symbol_name(self, stem: str) -> str:
        # ``stem`` already contains stable context (block/instruction).
        # Counter ensures uniqueness even for fallback/merge-generated symbols.
        if stem:
            return f"{stem}_{self._next_id()}"
        return f"sym_{self._next_id()}"

    def _next_id(self) -> int:
        # local static counter via closure-like class attr emulation
        if not hasattr(self, "_sym_counter"):
            self._sym_counter = 0
        self._sym_counter += 1
        return self._sym_counter

    def _set_summary_metadata(self, module: SolverModule) -> None:
        module.metadata["symbol_count"] = len(module.symbols)
        module.metadata["constraint_count"] = len(module.constraints)
        module.metadata["obligation_count"] = len(module.obligations)
        module.metadata["instruction_count"] = len(module.instructions)
        module.metadata["block_count"] = len(module.blocks)
