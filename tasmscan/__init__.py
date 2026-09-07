"""
tasmscan - Independent TVM Security Scanner

Lazy exports to avoid importing optional dependencies during test collection.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

__version__ = "1.0.0"

# Configure tasmscan logger
logger = logging.getLogger("tasmscan")
logger.setLevel(logging.WARNING)

# Only add handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)

_CORE_EXPORTS: Dict[str, str] = {
    # Core scanning
    "scan_contract": "tasmscan.scanner:scan_contract",
    "SecurityScanner": "tasmscan.scanner:SecurityScanner",
    # Disassembler
    "parse_boc": "tasmscan.disassembler:parse_boc",
    "decompile_cell": "tasmscan.disassembler:decompile_cell",
    "DisassemblyResult": "tasmscan.disassembler:DisassemblyResult",
    "TvmDisassembler": "tasmscan.disassembler:TvmDisassembler",
    # Analyzer
    "ProgramAnalyzer": "tasmscan.analyzer:ProgramAnalyzer",
    # Detectors
    "default_detectors": "tasmscan.detectors:default_detectors",
    "list_detector_metadata": "tasmscan.detectors:list_detector_metadata",
    "Detector": "tasmscan.detectors:Detector",
}

_OPTIONAL_EXPORTS: Dict[str, str] = {
    # IR
    "IRBuilder": "tasmscan.ir:IRBuilder",
    "DataFlowAnalyzer": "tasmscan.ir:DataFlowAnalyzer",
    "IRType": "tasmscan.ir:IRType",
    "TVMModule": "tasmscan.ir:TVMModule",
}

__all__ = list(_CORE_EXPORTS.keys()) + list(_OPTIONAL_EXPORTS.keys())


def _load_attr(path: str) -> Any:
    module_name, attr = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __getattr__(name: str) -> Any:
    if name in _CORE_EXPORTS:
        return _load_attr(_CORE_EXPORTS[name])
    if name in _OPTIONAL_EXPORTS:
        try:
            return _load_attr(_OPTIONAL_EXPORTS[name])
        except ImportError as exc:  # pragma: no cover - optional deps
            # ImportError: Missing optional dependency (e.g., graphviz)
            raise AttributeError(
                f"optional feature '{name}' requires missing dependency ({exc})"
            ) from exc
        except AttributeError as exc:  # pragma: no cover - optional deps
            # AttributeError: Module exists but attribute not found
            raise AttributeError(
                f"optional feature '{name}' is unavailable: attribute not found ({exc})"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(__all__)))
