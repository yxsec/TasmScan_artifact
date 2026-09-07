"""
Integrated TVM disassembler module for tasmscan.

This module contains the core disassembly functionality adapted from pytasm (now built-in),
making tasmscan a fully self-contained security scanner.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pytoniq_core import Cell

from .boc_parser import parse_boc
from .instruction_decoder import Instruction, decompile_cell


@dataclass
class DisassemblyResult:
    instructions: List[Instruction]
    root_cell: Cell


class TvmDisassembler:
    def disassemble(self, source: Union[str, Path, bytes, bytearray, Cell]) -> DisassemblyResult:
        if isinstance(source, Cell):
            root_cell = source
        elif isinstance(source, (bytes, bytearray)):
            root_cell = parse_boc(bytes(source))
        else:
            path = Path(source)
            root_cell = parse_boc(path.read_bytes())

        instructions = decompile_cell(root_cell)
        return DisassemblyResult(instructions=instructions, root_cell=root_cell)


__all__ = ['parse_boc', 'decompile_cell', 'DisassemblyResult', 'TvmDisassembler']
