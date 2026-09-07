"""
Solver IR types for post-TASIR constraint-based reasoning.

This module provides a minimal, explicit representation for:
- SSA-like control-register state (currently c3-focused)
- Dynamic continuation target obligations
- Solver-friendly constraints and symbols
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SolverSymbol:
    """Named symbol with a logical sort used by solver backends."""

    name: str
    sort: str  # e.g. "ContRef", "Bool", "Int"


@dataclass
class ContinuationState:
    """Explicit continuation-related register state at a CFG boundary."""

    ctrl_regs: Dict[int, str] = field(default_factory=dict)


@dataclass
class SolverInstruction:
    """Instruction lowered from TASIR with symbolic read/write sets."""

    index: int
    opcode: str
    block_id: int
    context_id: str
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverBlock:
    """Solver-level basic block with explicit entry/exit continuation state."""

    id: int
    context_id: str
    instruction_indices: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    entry_state: ContinuationState = field(default_factory=ContinuationState)
    exit_state: ContinuationState = field(default_factory=ContinuationState)


@dataclass
class SolverConstraint:
    """Single logical constraint expression (assertion body)."""

    expression: str
    kind: str = "assert"
    instruction_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicTargetObligation:
    """
    Proof obligation for unresolved or partially modeled dynamic target resolution.
    """

    instruction_index: int
    opcode: str
    context_id: str
    ctrl_reg_index: Optional[int]
    ctrl_reg_symbol: Optional[str]
    method_id: Optional[int]
    reason: str


@dataclass
class SolverModule:
    """Complete lowered module for solver ingestion."""

    blocks: Dict[int, SolverBlock] = field(default_factory=dict)
    instructions: Dict[int, SolverInstruction] = field(default_factory=dict)
    symbols: Dict[str, SolverSymbol] = field(default_factory=dict)
    constraints: List[SolverConstraint] = field(default_factory=list)
    obligations: List[DynamicTargetObligation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

