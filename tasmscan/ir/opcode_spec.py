"""Unified opcode specification model.

This module provides a unified data model for TVM opcode specifications,
consolidating information from multiple data sources:
1. cp0.json
2. stack-signatures-data.json
3. instruction_table.json
4. MANUAL_STACK_EFFECTS (200+ instructions)
5. opcode_mappings.py (1113 mappings)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path

from .dataflow.taint_registry import TAINT_SOURCES, CONDITIONAL_TAINT_SOURCES, SENSITIVE_OPCODES as REGISTRY_SENSITIVE_OPCODES


logger = logging.getLogger(__name__)


@dataclass
class OpcodeSpec:
    """Unified opcode specification.

    Attributes:
        mnemonic: The opcode name (e.g., "ADD", "PUSHINT")
        stack_inputs: Minimum number of stack inputs
        stack_outputs: Minimum number of stack outputs
        is_dynamic: True if stack effect depends on runtime/immediate parameters
        is_taint_source: True if this opcode produces tainted data (e.g., external input)
        is_sensitive: True if this is a security-sensitive operation (e.g., SENDRAWMSG)
        category: Functional category (e.g., "arithmetic", "stack", "cell", "dict")
        aliases: List of alias names for this opcode
    """
    mnemonic: str
    stack_inputs: int = 0
    stack_outputs: int = 0
    is_dynamic: bool = False
    is_taint_source: bool = False
    is_sensitive: bool = False
    category: str = "misc"
    aliases: List[str] = field(default_factory=list)

    @property
    def net_effect(self) -> int:
        """Net change to stack depth (outputs - inputs)."""
        return self.stack_outputs - self.stack_inputs


class OpcodeSpecDatabase:
    """Runtime database for opcode specifications.

    This database lazily loads opcode specifications from JSON or generates
    them from existing data sources (stack_effects, opcode_mappings).
    """

    def __init__(self):
        self._specs: Dict[str, OpcodeSpec] = {}
        self._loaded = False

    def load(self, json_path: Optional[Path] = None):
        """Load specs from JSON file or generate from sources.

        Args:
            json_path: Optional path to opcode_specs.json. If None, uses default path.
        """
        if json_path is None:
            json_path = Path(__file__).parent.parent / "data" / "opcode_specs.json"

        if json_path.exists():
            self._load_from_json(json_path)
        else:
            self._generate_from_sources()

        self._loaded = True

    def _load_from_json(self, json_path: Path):
        """Load specifications from JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for mnemonic, spec_data in data.items():
                self._specs[mnemonic] = OpcodeSpec(
                    mnemonic=mnemonic,
                    stack_inputs=spec_data.get("stack_inputs", 0),
                    stack_outputs=spec_data.get("stack_outputs", 0),
                    is_dynamic=spec_data.get("is_dynamic", False),
                    is_taint_source=spec_data.get("is_taint_source", False),
                    is_sensitive=spec_data.get("is_sensitive", False),
                    category=spec_data.get("category", "misc"),
                    aliases=spec_data.get("aliases", []),
                )
            logger.debug(f"Loaded {len(self._specs)} opcode specs from {json_path}")
        except Exception as e:
            logger.warning(f"Failed to load opcode specs from {json_path}: {e}")
            self._generate_from_sources()

    def _generate_from_sources(self):
        """Generate specs from existing data sources."""
        # Import existing sources
        try:
            from .stack_effects import get_stack_effect, get_stack_effect_db
            from .opcode_mappings import OPCODE_TO_IR_TYPE
        except ImportError as e:
            logger.warning(f"Cannot import data sources: {e}")
            return

        # Common opcodes that may appear as generic names in analysis
        # but exist in opcode_mappings with suffixes (e.g., PUSHINT_4, PUSHINT_8)
        common_opcodes = {
            # Push operations (0 inputs, 1 output)
            "PUSHINT": (0, 1, False, "stack"),
            "PUSHSLICE": (0, 1, False, "stack"),
            "PUSHCONT": (0, 1, False, "stack"),
            "PUSHREF": (0, 1, False, "cell"),
            "PUSHNULL": (0, 1, False, "stack"),
            "PUSHNAN": (0, 1, False, "stack"),
            "PUSHPOW2": (0, 1, False, "stack"),
            "PUSHNEGPOW2": (0, 1, False, "stack"),
            # Common arithmetic
            "ADD": (2, 1, False, "arithmetic"),
            "SUB": (2, 1, False, "arithmetic"),
            "MUL": (2, 1, False, "arithmetic"),
            "DIV": (2, 1, False, "arithmetic"),
            "MOD": (2, 1, False, "arithmetic"),
            "INC": (1, 1, False, "arithmetic"),
            "DEC": (1, 1, False, "arithmetic"),
            "NEGATE": (1, 1, False, "arithmetic"),
            "ABS": (1, 1, False, "arithmetic"),
            # Common stack operations
            "SWAP": (2, 2, False, "stack"),
            "DUP": (1, 2, False, "stack"),
            "DROP": (1, 0, False, "stack"),
            "NIP": (2, 1, False, "stack"),
            "OVER": (2, 3, False, "stack"),
            "ROT": (3, 3, False, "stack"),
            "ROTREV": (3, 3, False, "stack"),
            "TUCK": (2, 3, False, "stack"),
            # Comparisons
            "EQUAL": (2, 1, False, "compare"),
            "LESS": (2, 1, False, "compare"),
            "GREATER": (2, 1, False, "compare"),
            "LEQ": (2, 1, False, "compare"),
            "GEQ": (2, 1, False, "compare"),
            "NEQ": (2, 1, False, "compare"),
            "CMP": (2, 1, False, "compare"),
            "ISNULL": (1, 1, False, "compare"),
            "ISNAN": (1, 1, False, "compare"),
            # Control flow
            "RET": (0, 0, False, "control"),
            "RETFALSE": (0, 0, False, "control"),
            "RETTRUE": (0, 0, False, "control"),
            "JMPX": (1, 0, False, "control"),
            "CALLX": (1, 0, True, "control"),
            # Cell operations
            "NEWC": (0, 1, False, "cell"),
            "ENDC": (1, 1, False, "cell"),
            "CTOS": (1, 1, False, "cell"),
            "ENDS": (1, 0, False, "cell"),
        }

        # Taint sources - derived from taint_registry (single source of truth)
        # Includes unconditional sources + conditional sources + spec-level extras
        taint_sources = set(TAINT_SOURCES.keys()) | set(CONDITIONAL_TAINT_SOURCES.keys()) | {
            # Spec-level taint sources not in dataflow taint_registry
            # (these are classified at the OpcodeSpec level for metadata,
            # but the actual taint decision is made by taint_registry at runtime)
            "INMSGPARAM", "INMSGPARAMS", "INMSG_VALUE", "INMSG_ORIGVALUE",
            "INMSG_VALUEEXTRA", "INMSG_FWDFEE", "INMSG_LT", "INMSG_UTIME",
            "INMSG_BOUNCE", "INMSG_BOUNCED", "INMSG_STATEINIT",
            "INCOMINGVALUE", "BALANCE",
            "CONFIGROOT", "GETPARAMLONG", "GETPARAMLONG2",
        }

        # Sensitive operations - derived from taint_registry (single source of truth)
        # Plus spec-level extras for THROW variants (which are sensitive at OpcodeSpec level)
        sensitive_ops = set(REGISTRY_SENSITIVE_OPCODES) | {
            "THROW", "THROWANY", "THROWIF", "THROWIFNOT",
            "THROWARG", "THROWARGIF", "THROWARGIFNOT",
        }

        # Category mapping based on IR types
        category_map = {
            # Arithmetic
            "ADD": "arithmetic", "SUB": "arithmetic", "MUL": "arithmetic",
            "DIV": "arithmetic", "MOD": "arithmetic", "INC": "arithmetic",
            "DEC": "arithmetic", "NEGATE": "arithmetic", "ABS": "arithmetic",
            # Stack
            "SWAP": "stack", "DUP": "stack", "DROP": "stack", "OVER": "stack",
            "ROT": "stack", "ROTREV": "stack", "NIP": "stack", "TUCK": "stack",
            "PICK": "stack", "ROLL": "stack", "XCHG": "stack", "PUSH": "stack",
            "POP": "stack", "BLKDROP": "stack", "BLKPUSH": "stack", "BLKSWAP": "stack",
            # Cell operations
            "NEWC": "cell", "ENDC": "cell", "CTOS": "cell", "ENDS": "cell",
            # Comparison
            "EQUAL": "compare", "LESS": "compare", "GREATER": "compare",
            "LEQ": "compare", "GEQ": "compare", "NEQ": "compare", "CMP": "compare",
            # Control flow
            "JMPX": "control", "CALLX": "control", "RET": "control",
            "IFELSE": "control", "IF": "control", "IFNOT": "control",
            "WHILE": "control", "UNTIL": "control", "REPEAT": "control",
            # Dictionary
            "DICTGET": "dict", "DICTSET": "dict", "DICTDEL": "dict",
        }

        # Get stack effect database for coverage
        db = get_stack_effect_db()

        # Build specs from opcode mappings
        for opcode in OPCODE_TO_IR_TYPE:
            effect = get_stack_effect(opcode)

            # Determine category
            category = "misc"
            for prefix, cat in category_map.items():
                if opcode.startswith(prefix) or opcode == prefix:
                    category = cat
                    break

            # Infer category from opcode patterns
            if category == "misc":
                if any(opcode.startswith(p) for p in ["LD", "PLD", "ST"]):
                    category = "cell"
                elif any(opcode.startswith(p) for p in ["DICT"]):
                    category = "dict"
                elif any(opcode.startswith(p) for p in ["PUSH", "POP", "XCHG", "DUP", "DROP", "SWAP", "ROLL", "PICK", "BLK"]):
                    category = "stack"
                elif any(opcode.startswith(p) for p in ["JMP", "CALL", "RET", "IF", "WHILE", "UNTIL", "REPEAT", "AGAIN"]):
                    category = "control"
                elif any(opcode.startswith(p) for p in ["ADD", "SUB", "MUL", "DIV", "MOD", "NEG", "ABS", "INC", "DEC", "LSHIFT", "RSHIFT"]):
                    category = "arithmetic"
                elif any(opcode.startswith(p) for p in ["EQ", "LESS", "GREAT", "LEQ", "GEQ", "NEQ", "CMP", "ISNAN", "ISNULL"]):
                    category = "compare"

            self._specs[opcode] = OpcodeSpec(
                mnemonic=opcode,
                stack_inputs=effect.min_inputs if effect else 0,
                stack_outputs=effect.min_outputs if effect else 0,
                is_dynamic=effect.is_dynamic if effect else False,
                is_taint_source=opcode in taint_sources,
                is_sensitive=opcode in sensitive_ops,
                category=category,
            )

        # Also add opcodes from stack effects that aren't in opcode mappings
        for mnemonic in db.effects:
            if mnemonic not in self._specs:
                effect = db.effects[mnemonic]

                # Determine category
                category = "misc"
                if any(mnemonic.startswith(p) for p in ["LD", "PLD", "ST"]):
                    category = "cell"
                elif any(mnemonic.startswith(p) for p in ["DICT", "PFXDICT"]):
                    category = "dict"
                elif any(mnemonic.startswith(p) for p in ["PUSH", "POP", "XCHG", "DUP", "DROP", "SWAP", "ROLL", "PICK", "BLK", "NIP", "TUCK", "ROT", "OVER"]):
                    category = "stack"
                elif any(mnemonic.startswith(p) for p in ["JMP", "CALL", "RET", "IF", "WHILE", "UNTIL", "REPEAT", "AGAIN"]):
                    category = "control"
                elif any(mnemonic.startswith(p) for p in ["ADD", "SUB", "MUL", "DIV", "MOD", "NEG", "ABS", "INC", "DEC", "LSHIFT", "RSHIFT"]):
                    category = "arithmetic"

                self._specs[mnemonic] = OpcodeSpec(
                    mnemonic=mnemonic,
                    stack_inputs=effect.min_inputs,
                    stack_outputs=effect.min_outputs,
                    is_dynamic=effect.is_dynamic,
                    is_taint_source=mnemonic in taint_sources,
                    is_sensitive=mnemonic in sensitive_ops,
                    category=category,
                )

        # Add common opcodes that may not be in opcode_mappings or stack_effects
        for opcode, (inputs, outputs, is_dynamic, category) in common_opcodes.items():
            if opcode not in self._specs:
                self._specs[opcode] = OpcodeSpec(
                    mnemonic=opcode,
                    stack_inputs=inputs,
                    stack_outputs=outputs,
                    is_dynamic=is_dynamic,
                    is_taint_source=opcode in taint_sources,
                    is_sensitive=opcode in sensitive_ops,
                    category=category,
                )

        logger.debug(f"Generated {len(self._specs)} opcode specs from sources")

    def get(self, mnemonic: str) -> Optional[OpcodeSpec]:
        """Get spec for an opcode.

        Args:
            mnemonic: The opcode name

        Returns:
            OpcodeSpec if found, None otherwise
        """
        if not self._loaded:
            self.load()
        return self._specs.get(mnemonic)

    def __len__(self) -> int:
        """Return number of loaded specs."""
        if not self._loaded:
            self.load()
        return len(self._specs)

    def __contains__(self, mnemonic: str) -> bool:
        """Check if an opcode is in the database."""
        if not self._loaded:
            self.load()
        return mnemonic in self._specs

    def all_specs(self) -> Dict[str, OpcodeSpec]:
        """Return all loaded specs."""
        if not self._loaded:
            self.load()
        return self._specs.copy()

    def export_to_json(self, path: Path) -> None:
        """Export current specs to JSON file.

        Args:
            path: Output path for JSON file
        """
        if not self._loaded:
            self.load()

        data = {}
        for mnemonic, spec in self._specs.items():
            data[mnemonic] = {
                "stack_inputs": spec.stack_inputs,
                "stack_outputs": spec.stack_outputs,
                "is_dynamic": spec.is_dynamic,
                "is_taint_source": spec.is_taint_source,
                "is_sensitive": spec.is_sensitive,
                "category": spec.category,
                "aliases": spec.aliases,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        logger.info(f"Exported {len(data)} opcode specs to {path}")


# Global singleton
_db: Optional[OpcodeSpecDatabase] = None


def get_opcode_spec(mnemonic: str) -> Optional[OpcodeSpec]:
    """Get opcode specification.

    Args:
        mnemonic: The opcode name (e.g., "PUSHINT", "ADD")

    Returns:
        OpcodeSpec if found, None otherwise

    Example:
        spec = get_opcode_spec("PUSHINT")
        if spec:
            print(f"PUSHINT: {spec.stack_inputs} inputs, {spec.stack_outputs} outputs")
    """
    global _db
    if _db is None:
        _db = OpcodeSpecDatabase()
    return _db.get(mnemonic)


def get_opcode_spec_db() -> OpcodeSpecDatabase:
    """Get the global OpcodeSpecDatabase instance.

    Returns:
        The singleton OpcodeSpecDatabase instance
    """
    global _db
    if _db is None:
        _db = OpcodeSpecDatabase()
        _db.load()
    return _db
