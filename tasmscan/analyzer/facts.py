"""Data models for analysis facts from disassembled instructions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..ir.dataflow import DataFlowGraph


@dataclass
class Continuation:
    """Represents a continuation (code block that can be called)."""

    id: str  # Unique identifier (e.g., "cont_0", "cont_1")
    instructions: List[Any]  # List of Instruction objects
    entry_index: int  # Local entry index within this continuation's instruction list
    parent_context: Optional[str]  # "main" or parent continuation id
    parent_local_index: int  # Local index within the parent context
    kind: str = "continuation"  # "continuation" | "method"
    method_id: Optional[int] = None  # Method selector for dictionary entries
    parent_arg_index: Optional[int] = None  # Arg index within parent instruction
    parent_instruction_index: Optional[int] = None  # Global instruction index (assigned later)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_context": self.parent_context,
            "parent_local_index": self.parent_local_index,
            "kind": self.kind,
            "method_id": self.method_id,
            "parent_arg_index": self.parent_arg_index,
            "parent_instruction_index": self.parent_instruction_index,
            "entry_index": self.entry_index,
            "instruction_count": len(self.instructions),
        }


@dataclass
class InstructionFact:
    """Wrapper for Instruction with analysis metadata."""

    instruction: Any  # Instruction object
    index: int
    # RESERVED FOR FUTURE USE: offset and length are not currently populated.
    # These fields are intended for byte-level tracking when the disassembler provides
    # instruction byte positions. Currently always 0 and safe to ignore.
    # When implemented, offset will represent the byte offset within the cell,
    # and length will be the instruction's encoded byte length.
    offset: int = 0  # Byte offset in cell (reserved, always 0)
    length: int = 0  # Byte length of instruction (reserved, always 0)
    hash: str = ""
    # Continuation context
    continuation_id: Optional[str] = None  # Which continuation this belongs to (None = main)
    parent_index: Optional[int] = None  # For cont instructions: where to return after completion

    @property
    def opcode(self) -> str:
        """Get opcode name from instruction."""
        return self.instruction.name

    @property
    def arguments(self):
        """Get instruction arguments."""
        return self.instruction.args

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opcode": self.opcode,
            "offset": self.offset,
            "length": self.length,
            "hash": self.hash,
            "index": self.index,
            "continuation_id": self.continuation_id,
            "parent_index": self.parent_index,
        }


@dataclass
class Event:
    """Semantic event (accept, send, sender_read, guard, etc.)."""

    type: str
    instruction: InstructionFact

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.instruction.to_dict()}


@dataclass
class ControlFlowEdge:
    """CFG edge between instructions."""

    source: int
    target: Optional[int]
    kind: str  # fallthrough, branch, jump, call, call_cont, return_cont, call_return
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"source": self.source, "target": self.target, "kind": self.kind}
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class BasicBlock:
    """Basic block with instruction indices."""

    id: int
    instruction_indices: List[int]
    successors: List[int] = field(default_factory=list)
    context: str = "main"  # "main" or continuation_id (e.g., "cont_0")
    # Flag indicating this block has unresolved continuation/jump targets.
    # When True, control flow analysis is incomplete for this block.
    # Detectors should handle this conservatively (e.g., assume any behavior possible).
    has_unknown_successor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "instructions": self.instruction_indices,
            "successors": self.successors,
            "context": self.context,
            "has_unknown_successor": self.has_unknown_successor,
        }


@dataclass
class StackState:
    """Stack state at instruction.

    For uncertain instructions (unknown=True), height_after may be None if we can't
    track exact height. In this case, height_min and height_max provide the range
    of possible stack heights after the instruction. If height_max is None, the
    upper bound is unknown/unbounded.
    """

    instruction_index: int
    height_before: Optional[int]
    height_after: Optional[int]
    delta: Optional[int]
    unknown: bool = False
    # Range tracking for uncertain states
    height_min: Optional[int] = None
    height_max: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_index": self.instruction_index,
            "height_before": self.height_before,
            "height_after": self.height_after,
            "delta": self.delta,
            "unknown": self.unknown,
            "height_min": self.height_min,
            "height_max": self.height_max,
        }


@dataclass
class CallSite:
    """Call instruction with target info."""

    instruction: InstructionFact
    target_type: str  # method, reference, inline_block, cell, unknown
    target: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction.to_dict(),
            "target_type": self.target_type,
            "target": self.target,
        }


@dataclass
class CallGraphEdge:
    """Caller -> Callee relationship."""

    caller: str
    callee: str
    target_type: str
    instruction_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "target_type": self.target_type,
            "instruction_index": self.instruction_index,
        }


@dataclass
class BlockSummary:
    """High-level block summary."""

    context_id: str
    hash: str
    event_counts: Dict[str, int]
    call_targets: List[str]
    register_reads: Dict[str, int]
    register_writes: Dict[str, int]
    caller_hash: Optional[str] = None
    caller_offset: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "hash": self.hash,
            "event_counts": self.event_counts,
            "call_targets": self.call_targets,
            "register_reads": self.register_reads,
            "register_writes": self.register_writes,
            "caller_hash": self.caller_hash,
            "caller_offset": self.caller_offset,
        }


@dataclass
class AnalysisFacts:
    """Container for all analysis facts."""

    instructions: List[InstructionFact]
    events: List[Event] = field(default_factory=list)
    cfg_edges: List[ControlFlowEdge] = field(default_factory=list)
    basic_blocks: List[BasicBlock] = field(default_factory=list)
    stack_states: List[StackState] = field(default_factory=list)
    call_sites: List[CallSite] = field(default_factory=list)
    call_graph_edges: List[CallGraphEdge] = field(default_factory=list)
    entry_points: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summaries: List[BlockSummary] = field(default_factory=list)
    # Optional dataflow analysis result
    dataflow_graph: Optional["DataFlowGraph"] = None

    def __post_init__(self) -> None:
        # Build indices for fast lookup
        self._events_by_type: Dict[str, List[Event]] = {}
        for event in self.events:
            self._events_by_type.setdefault(event.type, []).append(event)

        self._opcode_index: Dict[str, List[InstructionFact]] = {}
        for fact in self.instructions:
            self._opcode_index.setdefault(fact.opcode, []).append(fact)

    def events_of(self, event_type: str) -> List[Event]:
        """Get all events of a specific type."""
        return self._events_by_type.get(event_type, [])

    def opcodes(self, opcode_name: str) -> List[InstructionFact]:
        """Get all instructions with specific opcode."""
        return self._opcode_index.get(opcode_name, [])

    def first_event(self, event_type: str) -> Optional[Event]:
        """Get first event of a specific type, ordered by instruction index."""
        events = self.events_of(event_type)
        if not events:
            return None
        return min(events, key=lambda e: e.instruction.index)

    def entry_block_ids(self) -> List[int]:
        """Get entry block IDs, preferring explicit entry_points when available."""
        if self.entry_points:
            entry_blocks = {
                entry.get("block_id")
                for entry in self.entry_points
                if entry.get("block_id") is not None
            }
            return sorted(entry_blocks)

        if not self.basic_blocks:
            return []

        incoming = {block.id: 0 for block in self.basic_blocks}
        for block in self.basic_blocks:
            for succ in block.successors:
                # Note: successors now only contain valid block IDs.
                # has_unknown_successor flag is used instead of -1 for unresolved targets.
                if succ in incoming:
                    incoming[succ] += 1

        entry_ids = [bid for bid, count in incoming.items() if count == 0]
        if not entry_ids and incoming:
            entry_ids = [min(incoming)]

        return sorted(entry_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instructions": [f.to_dict() for f in self.instructions],
            "events": [e.to_dict() for e in self.events],
            "cfg_edges": [e.to_dict() for e in self.cfg_edges],
            "basic_blocks": [b.to_dict() for b in self.basic_blocks],
            "stack_states": [s.to_dict() for s in self.stack_states],
            "call_sites": [c.to_dict() for c in self.call_sites],
            "call_graph_edges": [e.to_dict() for e in self.call_graph_edges],
            "entry_points": self.entry_points,
            "metadata": self.metadata,
            "summaries": [s.to_dict() for s in self.summaries],
        }

    @classmethod
    def merge(cls, *facts_objects: "AnalysisFacts") -> "AnalysisFacts":
        """Merge multiple AnalysisFacts with deduplication."""
        # Instructions: deduplicate by index
        instructions_dict: Dict[int, InstructionFact] = {}
        for facts in facts_objects:
            for inst in facts.instructions:
                if inst.index not in instructions_dict:
                    instructions_dict[inst.index] = inst
        instructions = sorted(instructions_dict.values(), key=lambda x: x.index)

        # Events: deduplicate by (type, instruction.index)
        events_seen: set = set()
        events: List[Event] = []
        for facts in facts_objects:
            for event in facts.events:
                key = (event.type, event.instruction.index)
                if key not in events_seen:
                    events_seen.add(key)
                    events.append(event)

        # CFG edges: deduplicate by (source, target, kind), merge metadata on collision
        cfg_edges_dict: Dict[tuple, ControlFlowEdge] = {}
        for facts in facts_objects:
            for edge in facts.cfg_edges:
                key = (edge.source, edge.target, edge.kind)
                if key not in cfg_edges_dict:
                    cfg_edges_dict[key] = edge
                else:
                    # Merge metadata: combine dicts, prefer True for boolean flags
                    existing = cfg_edges_dict[key]
                    if edge.metadata and existing.metadata:
                        merged_meta = dict(existing.metadata)
                        for k, v in edge.metadata.items():
                            if k in merged_meta:
                                # For boolean flags, prefer True (conservative)
                                if isinstance(v, bool) and isinstance(merged_meta[k], bool):
                                    merged_meta[k] = merged_meta[k] or v
                                # For lists, extend
                                elif isinstance(v, list) and isinstance(merged_meta[k], list):
                                    merged_meta[k] = list(set(merged_meta[k] + v))
                            else:
                                merged_meta[k] = v
                        cfg_edges_dict[key] = ControlFlowEdge(
                            source=existing.source, target=existing.target,
                            kind=existing.kind, metadata=merged_meta,
                        )
                    elif edge.metadata and not existing.metadata:
                        cfg_edges_dict[key] = edge
        cfg_edges = list(cfg_edges_dict.values())

        # Basic blocks: deduplicate by id
        blocks_dict: Dict[int, BasicBlock] = {}
        for facts in facts_objects:
            for block in facts.basic_blocks:
                if block.id not in blocks_dict:
                    blocks_dict[block.id] = block
        basic_blocks = sorted(blocks_dict.values(), key=lambda x: x.id)

        # Stack states: deduplicate by instruction_index
        stack_states_dict: Dict[int, StackState] = {}
        for facts in facts_objects:
            for state in facts.stack_states:
                if state.instruction_index not in stack_states_dict:
                    stack_states_dict[state.instruction_index] = state
        stack_states = sorted(
            stack_states_dict.values(), key=lambda x: x.instruction_index
        )

        # Call sites: deduplicate by instruction.index
        call_sites_dict: Dict[int, CallSite] = {}
        for facts in facts_objects:
            for site in facts.call_sites:
                if site.instruction.index not in call_sites_dict:
                    call_sites_dict[site.instruction.index] = site
        call_sites = sorted(
            call_sites_dict.values(), key=lambda x: x.instruction.index
        )

        # Call graph edges: deduplicate by (caller, callee, instruction_index)
        cg_edges_seen: set = set()
        call_graph_edges: List[CallGraphEdge] = []
        for facts in facts_objects:
            for edge in facts.call_graph_edges:
                key = (edge.caller, edge.callee, edge.instruction_index)
                if key not in cg_edges_seen:
                    cg_edges_seen.add(key)
                    call_graph_edges.append(edge)

        # Entry points: deduplicate by block_id (if present)
        entry_points_seen: set = set()
        entry_points: List[Dict[str, Any]] = []
        for facts in facts_objects:
            for entry in facts.entry_points:
                block_id = entry.get("block_id")
                if block_id is not None:
                    if block_id not in entry_points_seen:
                        entry_points_seen.add(block_id)
                        entry_points.append(entry)
                else:
                    # No block_id, include as-is
                    entry_points.append(entry)

        # Metadata: merge (later values override)
        metadata: Dict[str, Any] = {}
        for facts in facts_objects:
            metadata.update(facts.metadata)

        # Summaries: deduplicate by context_id
        summaries_dict: Dict[str, BlockSummary] = {}
        for facts in facts_objects:
            for summary in facts.summaries:
                if summary.context_id not in summaries_dict:
                    summaries_dict[summary.context_id] = summary
        summaries = list(summaries_dict.values())

        # Use first non-None dataflow_graph since graphs cannot be trivially merged
        # Callers should re-run dataflow analysis if merged result needs unified graph
        dataflow_graph = None
        for facts in facts_objects:
            if facts.dataflow_graph is not None:
                dataflow_graph = facts.dataflow_graph
                break

        return cls(
            instructions=instructions,
            events=events,
            cfg_edges=cfg_edges,
            basic_blocks=basic_blocks,
            stack_states=stack_states,
            call_sites=call_sites,
            call_graph_edges=call_graph_edges,
            entry_points=entry_points,
            metadata=metadata,
            summaries=summaries,
            dataflow_graph=dataflow_graph,
        )
