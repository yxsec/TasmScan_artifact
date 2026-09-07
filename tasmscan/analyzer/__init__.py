"""Program analyzer module

This package provides TVM program analysis capabilities:
- ProgramAnalyzer: Main coordinator for analyzing disassembled instructions
- ContinuationResolver: Extracts and resolves continuations
- CFGBuilder: Builds control flow graphs and basic blocks
- StackAnalyzer: Analyzes stack heights

The analysis produces AnalysisFacts containing:
- Instructions with metadata (InstructionFact)
- Semantic events (Event)
- Control flow graph edges (ControlFlowEdge)
- Basic blocks (BasicBlock)
- Stack states (StackState)
- Call sites and call graph
"""

from .program_analyzer import ProgramAnalyzer
from .facts import (
    AnalysisFacts,
    BasicBlock,
    BlockSummary,
    CallGraphEdge,
    CallSite,
    Continuation,
    ControlFlowEdge,
    Event,
    InstructionFact,
    StackState,
)
from .constants import (
    CALL_OPCODES,
    CONDITIONAL_BRANCHES,
    CONDITIONAL_GUARDS,
    CONDITIONAL_RETURNS,
    DEFAULT_EVENT_MAP,
    MAIN_CONTEXT,
    RETURNING_CONDITIONALS,
    TERMINATORS,
    UNCONDITIONAL_BRANCHES,
)
from .cfg_builder import CFGBuilder
from .continuation_resolver import ContinuationResolver
from .stack_analyzer import StackAnalyzer

__all__ = [
    # Main analyzer
    'ProgramAnalyzer',
    # Component classes
    'CFGBuilder',
    'ContinuationResolver',
    'StackAnalyzer',
    # Data classes
    'AnalysisFacts',
    'BasicBlock',
    'BlockSummary',
    'CallGraphEdge',
    'CallSite',
    'Continuation',
    'ControlFlowEdge',
    'Event',
    'InstructionFact',
    'StackState',
    # Constants (commonly used by detectors)
    'CALL_OPCODES',
    'CONDITIONAL_BRANCHES',
    'CONDITIONAL_GUARDS',
    'CONDITIONAL_RETURNS',
    'DEFAULT_EVENT_MAP',
    'MAIN_CONTEXT',
    'RETURNING_CONDITIONALS',
    'TERMINATORS',
    'UNCONDITIONAL_BRANCHES',
]
