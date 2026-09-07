"""Result types for detectors"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Confidence(Enum):
    """Confidence level for vulnerability findings.

    This enum should be used for the Vulnerability.confidence field.
    For additional metadata in Vulnerability.extra dict, string literals
    ("high", "medium", "low") may be used for backward compatibility with
    external tools that parse the JSON output.
    """
    HIGH = "high"      # Strong evidence, likely true positive
    MEDIUM = "medium"  # Moderate evidence, needs verification
    LOW = "low"        # Weak evidence, may be false positive


@dataclass
class Vulnerability:
    """A detected vulnerability."""

    detector: str
    severity: str  # critical, high, medium, low, info
    message: str
    instruction: Any  # InstructionFact
    remediation: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    confidence: Confidence = Confidence.MEDIUM  # Confidence level for the finding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "severity": self.severity,
            "message": self.message,
            "instruction": self.instruction.to_dict() if self.instruction else None,
            "remediation": self.remediation,
            "extra": self.extra,
            "confidence": self.confidence.value,
        }
