"""Types for dynamic continuation target solving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DynamicTargetSolveResult:
    """Result of solving one dynamic continuation target."""

    instruction_index: int
    opcode: str
    status: str  # sat | unsat | unknown | timeout
    strategy: str
    candidate_continuations: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    model: Dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

