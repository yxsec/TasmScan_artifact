"""
Instruction metadata loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ArgSpec:
    type: str
    len: int | None = None
    pad: int | None = None
    delta: int | None = None
    arg: "ArgSpec" | None = None
    refs: "ArgSpec" | None = None
    bits: "ArgSpec" | None = None


@dataclass(frozen=True)
class InstructionSpec:
    name: str
    min: int
    max: int
    opcode_bits: int
    prefix: int
    kind: str
    args: Sequence[ArgSpec]


def _load_specs() -> Dict[str, InstructionSpec]:
    table_path = Path(__file__).with_name("instruction_table.json")
    try:
        data = json.loads(table_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in instruction table: {e}") from e
    except FileNotFoundError:
        raise ValueError(f"Instruction table not found: {table_path}") from None

    if "instructions" not in data:
        raise ValueError("Instruction table missing 'instructions' key")

    instructions_data = data["instructions"]
    if not isinstance(instructions_data, list):
        raise ValueError("Instruction table 'instructions' must be a list")
    for i, instr in enumerate(instructions_data):
        if not isinstance(instr, dict):
            raise ValueError(f"Instruction at index {i} must be an object")
        if "name" not in instr:
            raise ValueError(f"Instruction at index {i} missing 'name' field")

    def build_arg(spec: Dict) -> ArgSpec:
        return ArgSpec(
            type=spec["type"],
            len=spec.get("len"),
            pad=spec.get("pad"),
            delta=spec.get("delta"),
            arg=build_arg(spec["arg"]) if spec.get("arg") else None,
            refs=build_arg(spec["refs"]) if spec.get("refs") else None,
            bits=build_arg(spec["bits"]) if spec.get("bits") else None,
        )

    instructions: Dict[str, InstructionSpec] = {}
    for entry in data["instructions"]:
        try:
            specs = [build_arg(arg) for arg in entry.get("args", [])]
            instructions[entry["name"]] = InstructionSpec(
                name=entry["name"],
                min=entry["min"],
                max=entry["max"],
                opcode_bits=entry["opcodeBits"],
                prefix=entry["prefix"],
                kind=entry["kind"],
                args=specs,
            )
        except KeyError as e:
            raise ValueError(f"Invalid instruction entry, missing field {e}: {entry.get('name', 'unknown')}") from e
    return instructions


@lru_cache(maxsize=1)
def get_instruction_specs() -> Dict[str, InstructionSpec]:
    return _load_specs()

