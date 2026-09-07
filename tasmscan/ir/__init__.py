"""
IR (Intermediate Representation) module for TVM bytecode analysis.

This module provides high-level semantic representation of TVM bytecode:
- InstructionKind semantic categories (vs 895 raw opcodes)
- Data flow analysis with taint tracking
- Path-sensitive analysis
- Cross-continuation data flow via SaveList
"""

from .ir_types import IRType
from .dataflow import DataFlowAnalyzer, DataFlowGraph

# TASIR types (structured IR)
from .tasir_types import (
    TVMModule,
    TVMFunction,
    TVMBasicBlock,
    TVMInstruction,
    ContinuationDescriptor,
    SaveList,
    InstructionKind,
    SemanticLabel,
    TVMType,
    TVMLocation,
    StackLocation,
    RegisterLocation,
    GlobalLocation,
    InterProceduralEdge,
    InterProceduralView,
)

# IR lifter
from .ir_lifter import IRLifter

# Continuation linker
from .continuation_linker import ContinuationLinker

# Opcode kind mapper
from .opcode_kind_mapper import get_instruction_kind, get_instruction_kind_from_opcode

# Opcode specification
from .opcode_spec import OpcodeSpec, OpcodeSpecDatabase, get_opcode_spec, get_opcode_spec_db

try:
    from .ir_builder import IRBuilder
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "pytoniq_core":
        raise
    IRBuilder = None  # type: ignore[assignment]

from .solver_ir import (
    ContinuationState,
    DynamicTargetObligation,
    SMTLibExporter,
    SolverBlock,
    SolverConstraint,
    SolverInstruction,
    SolverIRLowering,
    SolverModule,
    SolverSymbol,
)

__all__ = [
    # Core types
    'IRType',
    'IRBuilder',
    'DataFlowAnalyzer',
    'DataFlowGraph',
    # TASIR types
    'TVMModule',
    'TVMFunction',
    'TVMBasicBlock',
    'TVMInstruction',
    'ContinuationDescriptor',
    'SaveList',
    'InstructionKind',
    'SemanticLabel',
    'TVMType',
    'TVMLocation',
    'StackLocation',
    'RegisterLocation',
    'GlobalLocation',
    'InterProceduralEdge',
    'InterProceduralView',
    # IR lifter
    'IRLifter',
    # Continuation linker
    'ContinuationLinker',
    # Opcode kind mapper
    'get_instruction_kind',
    'get_instruction_kind_from_opcode',
    # Opcode specification
    'OpcodeSpec',
    'OpcodeSpecDatabase',
    'get_opcode_spec',
    'get_opcode_spec_db',
    # Solver IR
    'ContinuationState',
    'DynamicTargetObligation',
    'SMTLibExporter',
    'SolverBlock',
    'SolverConstraint',
    'SolverInstruction',
    'SolverIRLowering',
    'SolverModule',
    'SolverSymbol',
]
