"""Detector registry and configuration support"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

from .base import Detector
from .no_accept import NoAcceptBeforeSendDetector
from .unchecked_sender import UncheckedSenderDetector
from .bounced_message import BouncedMessageDetector, BouncedHandlerPatternDetector
from .stack_underflow import StackUnderflowDetector
from .dynamic_call import DynamicCallDetector
from .summary_accept import SummarySendRequiresAcceptDetector
from .summary_sender_guard import SummarySenderGuardDetector
from .bad_randomness import BadRandomnessDetector
from .lack_end_parse import LackEndParseDetector
from .incompatible_modes import IncompatibleMessageModesDetector
from .dict_type_mismatch import DictTypeMismatchDetector
from .tlb_structure import TLBStructureViolationDetector
from .bad_destination import BadDestinationAddressDetector
from .precision_loss import PrecisionLossDetector
from .inconsistent_data import InconsistentDataDetector
from .global_var_redefined import GlobalVarRedefinedDetector
from .improper_modifier import ImproperModifierDetector
from .unhandled_bounced_op import UnhandledBouncedOpDetector
from .unused_variable import UnusedVariableDetector

DETECTOR_CLASSES: Dict[str, type] = {
    NoAcceptBeforeSendDetector.name: NoAcceptBeforeSendDetector,
    UncheckedSenderDetector.name: UncheckedSenderDetector,
    BouncedMessageDetector.name: BouncedMessageDetector,
    BouncedHandlerPatternDetector.name: BouncedHandlerPatternDetector,
    StackUnderflowDetector.name: StackUnderflowDetector,
    DynamicCallDetector.name: DynamicCallDetector,
    SummarySendRequiresAcceptDetector.name: SummarySendRequiresAcceptDetector,
    SummarySenderGuardDetector.name: SummarySenderGuardDetector,
    BadRandomnessDetector.name: BadRandomnessDetector,
    LackEndParseDetector.name: LackEndParseDetector,
    IncompatibleMessageModesDetector.name: IncompatibleMessageModesDetector,
    DictTypeMismatchDetector.name: DictTypeMismatchDetector,
    TLBStructureViolationDetector.name: TLBStructureViolationDetector,
    BadDestinationAddressDetector.name: BadDestinationAddressDetector,
    PrecisionLossDetector.name: PrecisionLossDetector,
    InconsistentDataDetector.name: InconsistentDataDetector,
    GlobalVarRedefinedDetector.name: GlobalVarRedefinedDetector,
    ImproperModifierDetector.name: ImproperModifierDetector,
    UnhandledBouncedOpDetector.name: UnhandledBouncedOpDetector,
    UnusedVariableDetector.name: UnusedVariableDetector,
}

# Detector categories
DETECTOR_CATEGORIES: Dict[str, List[Type[Detector]]] = {
    "security": [],      # Security vulnerability detectors
    "code_quality": [],  # Code quality detectors
}


def _categorize_detectors() -> None:
    """Categorize detectors based on their category attribute."""
    for detector_cls in DETECTOR_CLASSES.values():
        cat = getattr(detector_cls, 'category', 'security')
        if cat not in DETECTOR_CATEGORIES:
            cat = 'security'
        DETECTOR_CATEGORIES[cat].append(detector_cls)


# Initialize categories on module load
_categorize_detectors()


def default_detectors(include_code_quality: bool = False) -> List[Detector]:
    """
    Get default detectors.

    Args:
        include_code_quality: Whether to include code quality detectors

    Returns:
        List of detector instances
    """
    detectors = [d() for d in DETECTOR_CATEGORIES["security"]
                 if getattr(d, 'enabled_by_default', True)]
    if include_code_quality:
        detectors += [d() for d in DETECTOR_CATEGORIES["code_quality"]
                      if getattr(d, 'enabled_by_default', True)]
    return detectors


def load_detectors_from_config(config: Optional[Dict]) -> List[Detector]:
    """
    Load detectors from configuration dictionary.

    Config format:
    {
        "detectors": [
            {
                "name": "bounced_message",
                "severity": "critical",  # Optional: override default severity
                "options": {}  # Optional: detector-specific options
            }
        ]
    }

    Args:
        config: Configuration dictionary

    Returns:
        List of configured detectors, or default detectors if config is empty
    """
    if not config:
        return default_detectors()

    detectors_config = config.get("detectors")
    if not detectors_config:
        return default_detectors()

    detectors: List[Detector] = []
    for entry in detectors_config:
        name = entry.get("name")
        options = entry.get("options")
        cls = DETECTOR_CLASSES.get(name)
        if not cls:
            logger.warning(f"Unknown detector name in config: '{name}'. Available: {list(DETECTOR_CLASSES.keys())}")
            continue
        detector = cls(options=options) if options else cls()
        severity = entry.get("severity")
        if severity:
            detector.default_severity = severity
        detectors.append(detector)

    return detectors or default_detectors()


# Maximum config file size (1MB)
MAX_CONFIG_FILE_SIZE = 1 * 1024 * 1024

# Allowed config file extensions
ALLOWED_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}


def load_config_file(path: Optional[str]) -> Optional[Dict]:
    """
    Load configuration from YAML or JSON file.

    Supports:
    - JSON files (.json)
    - YAML files (.yaml, .yml) - requires PyYAML

    Security validations:
    - File extension must be .json, .yaml, or .yml
    - File size must be <= 1MB
    - Path traversal protection (resolves symlinks)

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary, or None if path is not provided

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not allowed or file is too large
        RuntimeError: If YAML is used but PyYAML is not installed
    """
    if not path:
        return None

    file_path = Path(path).resolve()

    suffix = file_path.suffix.lower()
    if suffix not in ALLOWED_CONFIG_EXTENSIONS:
        raise ValueError(
            f"Invalid config file extension '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONFIG_EXTENSIONS))}"
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")

    file_size = file_path.stat().st_size
    if file_size > MAX_CONFIG_FILE_SIZE:
        raise ValueError(
            f"Configuration file too large ({file_size} bytes). "
            f"Maximum allowed: {MAX_CONFIG_FILE_SIZE} bytes (1MB)"
        )

    text = file_path.read_text()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to parse YAML config files. "
                "Install it with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text)

    return json.loads(text)


def list_detector_metadata() -> List[Dict[str, str]]:
    """Get metadata for all detectors."""
    metadata = []
    for name, cls in sorted(DETECTOR_CLASSES.items()):
        entry = {
            "name": name,
            "severity": cls.default_severity,
            "description": getattr(cls, "description", ""),
            "category": getattr(cls, "category", "security"),
        }
        # Include experimental flag if present
        if getattr(cls, "experimental", False):
            entry["experimental"] = True
        # Include tags if present
        tags = getattr(cls, "tags", [])
        if tags:
            entry["tags"] = tags
        # Include depends_on if present
        depends_on = getattr(cls, "depends_on", [])
        if depends_on:
            entry["depends_on"] = depends_on
        metadata.append(entry)
    return metadata


def get_detectors_by_tag(tag: str) -> List[Type[Detector]]:
    """Get all detectors with a specific tag."""
    return [cls for cls in DETECTOR_CLASSES.values() if tag in getattr(cls, "tags", [])]


def get_detector_tags() -> Set[str]:
    """Get all unique tags from registered detectors."""
    tags: Set[str] = set()
    for cls in DETECTOR_CLASSES.values():
        tags.update(getattr(cls, "tags", []))
    return tags
