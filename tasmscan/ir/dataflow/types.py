"""
Data Flow Analysis types for TVM bytecode.

Contains core data structures for tracking value sources, data flow states,
and the data flow graph representation.
"""
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ValueSource(Enum):
    """Where a value comes from"""

    MESSAGE_SENDER = "message_sender"
    MESSAGE_VALUE = "message_value"
    MESSAGE_BODY = "message_body"
    MESSAGE_FLAGS = "message_flags"
    STORAGE = "storage"
    CONSTANT = "constant"
    COMPUTATION = "computation"
    UNKNOWN = "unknown"


@dataclass
class DataFlowValue:
    """Represents a value in data flow analysis"""

    source: ValueSource
    definition_site: int  # Instruction index where value is defined
    tainted: bool = False  # Is this value from untrusted input?
    checked: bool = False  # Has this value been validated?
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "DataFlowValue":
        """Create a deep copy of this value"""
        return DataFlowValue(
            source=self.source,
            definition_site=self.definition_site,
            tainted=self.tainted,
            checked=self.checked,
            metadata=copy.deepcopy(self.metadata) if self.metadata else {},
        )

    def __repr__(self):
        status = []
        if self.tainted:
            status.append("tainted")
        if self.checked:
            status.append("checked")
        status_str = f" [{', '.join(status)}]" if status else ""
        return f"{self.source.value}@{self.definition_site}{status_str}"


@dataclass
class DataFlowState:
    """Program state at a specific point"""

    # Stack: index 0 is TOS (top of stack)
    stack: List[Optional[DataFlowValue]]

    # Control registers (c0-c7)
    registers: Dict[int, Optional[DataFlowValue]]

    # Which values have been used in guards (IF/THROW)
    # Higher access_count means more recently/frequently used guard
    guarded_values: Dict[int, int]  # definition_site -> access_count

    def copy(self) -> "DataFlowState":
        """Deep copy of the state.

        Note: DataFlowValue.copy() uses copy.deepcopy for its metadata dict,
        ensuring mutable objects within metadata are properly cloned.
        """
        return DataFlowState(
            stack=[v.copy() if v else None for v in self.stack],
            registers={k: v.copy() if v else None for k, v in self.registers.items()},
            guarded_values=self.guarded_values.copy(),
        )


@dataclass
class DataFlowEdge:
    """Edge in data flow graph showing value dependencies"""

    from_instruction: int
    to_instruction: int
    value: DataFlowValue
    edge_type: str  # "stack", "register", "memory"


@dataclass
class DataFlowGraph:
    """Complete data flow graph"""

    values: Dict[int, List[DataFlowValue]]  # instruction_idx -> values defined
    edges: List[DataFlowEdge]
    tainted_propagation: List[Tuple[int, int]]  # (from_inst, to_inst) for tainted values
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)  # Analysis completeness info
    # Values: "path_sensitive" (more accurate, uses CFG paths) or "path_insensitive" (simple forward pass)
    analysis_type: str = "path_insensitive"
    _taint_dedup: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
