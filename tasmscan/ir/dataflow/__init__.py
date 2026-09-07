"""
Data Flow Analysis module for TVM bytecode.

Provides:
- ValueSource: Enum for tracking value origins
- DataFlowValue: Represents a tracked value
- DataFlowState: Program state at a specific point
- DataFlowEdge: Edge in data flow graph
- DataFlowGraph: Complete data flow graph
- DataFlowAnalyzer: Main analyzer class

Submodules:
- types: Core data structures (ValueSource, DataFlowValue, etc.)
- taint_registry: Taint source configuration and opcode classifications
- stack_simulator: Stack simulation methods
- guard_analyzer: Guard detection and marking logic
- taint_propagation: Taint tracking and propagation
"""

from .types import (
    ValueSource,
    DataFlowValue,
    DataFlowState,
    DataFlowEdge,
    DataFlowGraph,
)
from .analyzer import DataFlowAnalyzer

# Re-export taint registry constants for backward compatibility
from .taint_registry import (
    CONTINUATION_OPCODES,
    MULTI_OUTPUT_TAINT_RULES,
    TAINT_SOURCES,
    CONDITIONAL_TAINT_SOURCES,
    MESSAGE_SLICE_LOADERS,
    CONTEXT_DEPENDENT_LOADERS,
    SENSITIVE_OPCODES,
    DYNAMIC_CALL_OPCODES,
    CALL_VARARGS_OPCODES,
    CONTINUATION_PUSH_OPCODES,
    DYNAMIC_SHUFFLE_OPCODES,
)

__all__ = [
    # Core types
    'ValueSource',
    'DataFlowValue',
    'DataFlowState',
    'DataFlowEdge',
    'DataFlowGraph',
    # Main analyzer
    'DataFlowAnalyzer',
    # Taint registry constants (backward compatibility)
    'CONTINUATION_OPCODES',
    'MULTI_OUTPUT_TAINT_RULES',
    'TAINT_SOURCES',
    'CONDITIONAL_TAINT_SOURCES',
    'MESSAGE_SLICE_LOADERS',
    'CONTEXT_DEPENDENT_LOADERS',
    'SENSITIVE_OPCODES',
    'DYNAMIC_CALL_OPCODES',
    'CALL_VARARGS_OPCODES',
    'CONTINUATION_PUSH_OPCODES',
    'DYNAMIC_SHUFFLE_OPCODES',
]
