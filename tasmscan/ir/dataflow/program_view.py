"""
Unified program view adapters for dataflow analyzers.

This module provides a small compatibility layer that normalizes input
representations (legacy AnalysisFacts and TASIR TVMModule) into a shared
shape consumed by dataflow analyzers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ...analyzer.facts import AnalysisFacts


@dataclass(frozen=True)
class ProgramInstruction:
    """Instruction shape consumed by dataflow analyzers."""

    index: int
    opcode: str
    arguments: Any
    continuation_id: Optional[str] = None


@dataclass(frozen=True)
class ProgramBlock:
    """Basic block shape consumed by dataflow analyzers."""

    id: int
    instruction_indices: List[int]
    successors: List[int]
    successor_edge_kinds: Dict[int, List[str]] = field(default_factory=dict)
    context: str = "main"
    has_unknown_successor: bool = False


@runtime_checkable
class ProgramView(Protocol):
    """Common program interface for dataflow analysis."""

    instructions: Sequence[ProgramInstruction]
    basic_blocks: Sequence[ProgramBlock]
    metadata: Dict[str, Any]

    def entry_block_ids(self) -> List[int]:
        """Return entry block IDs for CFG traversal."""


class FactsProgramView:
    """Adapter from AnalysisFacts to ProgramView."""

    def __init__(self, facts: AnalysisFacts) -> None:
        self._facts = facts
        ordered_instructions = sorted(facts.instructions, key=lambda inst: inst.index)
        self.instructions: List[ProgramInstruction] = [
            ProgramInstruction(
                index=inst.index,
                opcode=inst.opcode,
                arguments=inst.arguments,
                continuation_id=inst.continuation_id,
            )
            for inst in ordered_instructions
        ]
        ordered_blocks = sorted(facts.basic_blocks, key=lambda block: block.id)
        instr_to_block: Dict[int, int] = {}
        for block in ordered_blocks:
            for inst_idx in block.instruction_indices:
                instr_to_block[inst_idx] = block.id

        edge_kind_sets: Dict[int, Dict[int, set]] = {}
        for edge in facts.cfg_edges:
            if edge.target is None:
                continue
            src_block = instr_to_block.get(edge.source)
            dst_block = instr_to_block.get(edge.target)
            if src_block is None or dst_block is None:
                continue
            edge_kind_sets.setdefault(src_block, {}).setdefault(dst_block, set()).add(edge.kind)

        self.basic_blocks: List[ProgramBlock] = [
            ProgramBlock(
                id=block.id,
                instruction_indices=list(block.instruction_indices),
                successors=list(block.successors),
                successor_edge_kinds={
                    succ_id: sorted(
                        edge_kind_sets.get(block.id, {}).get(succ_id, {"fallthrough"})
                    )
                    for succ_id in block.successors
                },
                context=block.context,
                has_unknown_successor=block.has_unknown_successor,
            )
            for block in ordered_blocks
        ]
        self.metadata: Dict[str, Any] = dict(facts.metadata or {})

    def entry_block_ids(self) -> List[int]:
        return self._facts.entry_block_ids()


class ModuleProgramView:
    """Adapter from TASIR TVMModule to ProgramView."""

    def __init__(self, module: Any) -> None:
        self._facts = getattr(module, "_facts", None)

        # Prefer richer source metadata from Facts when available, then overlay
        # TASIR metadata so IR-specific fields (dynamic targets, linking stats)
        # are preserved.
        metadata: Dict[str, Any] = {}
        facts_meta = getattr(self._facts, "metadata", None)
        if isinstance(facts_meta, dict):
            metadata.update(facts_meta)
        module_meta = getattr(module, "analysis_metadata", None)
        if isinstance(module_meta, dict):
            metadata.update(module_meta)
            # Preserve and merge incompleteness reasons from both layers.
            facts_reasons = facts_meta.get("analysis_incomplete_reasons", []) if isinstance(facts_meta, dict) else []
            module_reasons = module_meta.get("analysis_incomplete_reasons", [])
            if facts_reasons or module_reasons:
                metadata["analysis_incomplete_reasons"] = list(
                    dict.fromkeys(list(facts_reasons) + list(module_reasons))
                )
            if (
                bool(facts_meta.get("analysis_incomplete")) if isinstance(facts_meta, dict) else False
            ) or bool(module_meta.get("analysis_incomplete")):
                metadata["analysis_incomplete"] = True
        self.metadata = metadata

        block_list = sorted(module.all_blocks(), key=lambda block: (block.id, block.context_id))
        inst_context: Dict[int, str] = {}
        self.basic_blocks: List[ProgramBlock] = []
        for block in block_list:
            inst_indices = [inst.index for inst in block.instructions]
            for inst in block.instructions:
                inst_context[inst.index] = block.context_id
            succ_kind_sets: Dict[int, set] = {}
            for edge in block.successors:
                succ_kind_sets.setdefault(edge.target_block, set()).add(edge.edge_kind)
            successors = sorted(succ_kind_sets.keys())
            self.basic_blocks.append(
                ProgramBlock(
                    id=block.id,
                    instruction_indices=inst_indices,
                    successors=successors,
                    successor_edge_kinds={
                        succ_id: sorted(succ_kind_sets.get(succ_id, {"fallthrough"}))
                        for succ_id in successors
                    },
                    context=block.context_id,
                    has_unknown_successor=bool(block.has_unknown_successor),
                )
            )

        self.instructions: List[ProgramInstruction] = []
        for inst in module.all_instructions():
            raw_args = list(inst.original_args) if inst.original_args else list(inst.immediates)
            self.instructions.append(
                ProgramInstruction(
                    index=inst.index,
                    opcode=inst.opcode,
                    arguments=raw_args,
                    continuation_id=inst_context.get(inst.index),
                )
            )

        self._entry_ids = self._compute_entry_ids()

    def _compute_entry_ids(self) -> List[int]:
        if self._facts is not None and hasattr(self._facts, "entry_block_ids"):
            try:
                ids = list(self._facts.entry_block_ids())
                if ids:
                    return ids
            except Exception:
                pass

        if not self.basic_blocks:
            return []
        incoming = {block.id: 0 for block in self.basic_blocks}
        for block in self.basic_blocks:
            for succ in block.successors:
                if succ in incoming:
                    incoming[succ] += 1
        entry_ids = [bid for bid, count in incoming.items() if count == 0]
        if not entry_ids and incoming:
            entry_ids = [min(incoming)]
        return sorted(entry_ids)

    def entry_block_ids(self) -> List[int]:
        return list(self._entry_ids)


def ensure_program_view(program: Any) -> ProgramView:
    """Normalize supported program representations into ProgramView."""
    if isinstance(program, AnalysisFacts):
        return FactsProgramView(program)
    if hasattr(program, "all_blocks") and hasattr(program, "all_instructions"):
        return ModuleProgramView(program)
    if isinstance(program, ProgramView):
        return program
    raise TypeError(f"Unsupported program type for dataflow analysis: {type(program)!r}")
