"""TVM Instruction Decoder backed by the TypeScript metadata."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_OPCODE_BITS = 8   # Minimum bits required for any TVM opcode
MAX_OPCODE_BITS = 24  # Maximum bits used for opcode lookup (aligned to 24)

from pytoniq_core import Cell, Slice, begin_cell

from ..config import MAX_INSTRUCTION_DECODE_DEPTH, MAX_DICT_PARSE_DEPTH, MAX_UNARY_LENGTH
from .instruction_table import ArgSpec, InstructionSpec, get_instruction_specs


@dataclass
class InstructionArg:
    """Represents a decoded argument together with its type."""

    type: str
    value: Any


@dataclass
class Instruction:
    """Decoded TVM instruction."""

    name: str
    args: List[InstructionArg] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        if not self.args:
            return self.name
        args_repr = ", ".join(f"{arg.type}={arg.value}" for arg in self.args)
        return f"{self.name}({args_repr})"


@dataclass
class CodeArgument:
    """Holds either nested instructions or a raw cell for code-like arguments."""

    cell: Cell
    instructions: Optional[List[Instruction]] = None


@dataclass
class SliceArgument:
    """Represents an arbitrary slice payload."""

    cell: Cell


@dataclass
class DictionaryEntry:
    key: int
    code: CodeArgument


@dataclass
class DictionaryArgument:
    cell: Cell
    entries: Optional[List[DictionaryEntry]] = None
    key_length: Optional[int] = None


@dataclass
class ExoticArgument:
    cell: Cell
    kind: str  # "library" | "default"
    data_slice: Optional[Slice] = None


def _bitarray_to_string(bits: Any) -> str:
    """Best-effort conversion of TvmBitarray into a bit-string."""
    if hasattr(bits, "to01"):
        return bits.to01()
    if hasattr(bits, "__len__"):
        return "".join("1" if bits[i] else "0" for i in range(len(bits)))
    raise TypeError("Unsupported bit array representation")


def _store_bitstring(builder, bitstring: str) -> None:
    """Stores an arbitrary bit-string into the builder."""
    if not bitstring:
        return
    chunk = 0
    while chunk < len(bitstring):
        window = bitstring[chunk : chunk + 32]
        builder.store_uint(int(window, 2), len(window))
        chunk += 32


class InstructionDecoder:
    """Decodes TVM instructions from bytecode slices."""
    def __init__(self) -> None:
        self._specs_by_name: Dict[str, InstructionSpec] = get_instruction_specs()
        self._ranges: List[Tuple[int, int, str]] = sorted(
            ((spec.min, spec.max, spec.name) for spec in self._specs_by_name.values()),
            key=lambda item: item[0],
        )

    def _find_spec(self, opcode_aligned: int) -> InstructionSpec:
        left, right = 0, len(self._ranges)

        while right - left > 1:
            mid = (left + right) // 2
            min_val, _, _ = self._ranges[mid]
            if min_val <= opcode_aligned:
                left = mid
            else:
                right = mid

        if left < len(self._ranges):
            min_val, max_val, name = self._ranges[left]
            if min_val <= opcode_aligned < max_val:
                spec = self._specs_by_name.get(name)
                if spec is None:
                    raise ValueError(f"Missing spec for instruction {name}")
                return spec

        raise ValueError(f"Unknown opcode: 0x{opcode_aligned:X}")

    def decode_instruction(self, slice_obj: Slice) -> Instruction:
        if slice_obj.remaining_bits < MIN_OPCODE_BITS: 
            raise ValueError(
                f"Invalid opcode: not enough bits (need ≥ {MIN_OPCODE_BITS}, got {slice_obj.remaining_bits})",
            )

        bits_available = min(slice_obj.remaining_bits, MAX_OPCODE_BITS)
        opcode = slice_obj.preload_uint(bits_available)
        opcode_aligned = opcode << (MAX_OPCODE_BITS - bits_available)

        spec = self._find_spec(opcode_aligned)
        opcode_bits = spec.opcode_bits
        if bits_available < opcode_bits:
            raise ValueError(
                f"Invalid opcode length: prefix needs {opcode_bits} bits, got {bits_available}",
            )

        slice_obj.skip_bits(opcode_bits)

        arg_reader = _ArgumentReader(self, slice_obj)
        args = arg_reader.decode_all(spec.args)

        return Instruction(name=spec.name, args=args)

    def decompile_slice(
        self, slice_obj: Slice, depth: int = 0, max_depth: int = MAX_INSTRUCTION_DECODE_DEPTH
    ) -> List[Instruction]:
        """Decompile a slice into instructions.

        Args:
            slice_obj: The slice to decompile
            depth: Current recursion depth (internal use)
            max_depth: Maximum recursion depth (default MAX_INSTRUCTION_DECODE_DEPTH, prevents DoS)

        Returns:
            List of decoded instructions

        Raises:
            RecursionError: If max_depth is exceeded
        """
        if depth >= max_depth:
            raise RecursionError(f"Decompilation exceeded max depth {max_depth}")

        instructions: List[Instruction] = []
        work_slice = slice_obj.copy()
        # Save original slice for fallback (matching tasm's sliceBackup behavior)
        original_cell = slice_obj.to_cell()

        while work_slice.remaining_bits > 0:
            try:
                instr = self.decode_instruction(work_slice) # decode one instruction
                instructions.append(instr) # append the instruction
            except Exception as e:
                # on ANY decode failure, discard all decoded instructions and return the entire original slice as raw data
                logger.debug("Failed to decode instruction at depth %d: %s", depth, e)
                fallback_arg = InstructionArg(
                    type="slice",
                    value=SliceArgument(cell=original_cell),
                )
                return [Instruction(name="PSEUDO_PUSHSLICE", args=[fallback_arg])]

        # All bits successfully decoded - now walk remaining refs as implicit continuations
        while work_slice.remaining_refs > 0:
            ref_cell = work_slice.load_ref()
            nested = self.decompile_cell(ref_cell, depth + 1, max_depth)
            code_arg = InstructionArg(
                type="code",
                value=CodeArgument(cell=ref_cell, instructions=nested),
            )
            instructions.append(Instruction(name="PSEUDO_PUSHREF", args=[code_arg]))

        return instructions

    def decompile_cell(
        self, cell: Cell, depth: int = 0, max_depth: int = MAX_INSTRUCTION_DECODE_DEPTH
    ) -> List[Instruction]:
        """Decompile a cell into instructions.

        Args:
            cell: The cell to decompile
            depth: Current recursion depth (internal use)
            max_depth: Maximum recursion depth (default MAX_INSTRUCTION_DECODE_DEPTH, prevents DoS)

        Returns:
            List of decoded instructions

        Raises:
            RecursionError: If max_depth is exceeded
        """
        if depth >= max_depth:
            raise RecursionError(f"Decompilation exceeded max depth {max_depth}")

        if cell.is_exotic:
            slice_view = cell.begin_parse()
            cell_type = slice_view.load_uint(8)
            if cell_type == 2:
                library_slice = slice_view
                exotic_arg = InstructionArg(
                    type="exotic",
                    value=ExoticArgument(cell=cell, kind="library", data_slice=library_slice),
                )
            else:
                exotic_arg = InstructionArg(
                    type="exotic",
                    value=ExoticArgument(cell=cell, kind="default"),
                )
            return [Instruction("PSEUDO_EXOTIC", args=[exotic_arg])]

        slice_obj = cell.begin_parse()
        return self.decompile_slice(slice_obj, depth, max_depth)


class _ArgumentReader:
    """Stateful helper for decoding argument sequences."""

    def __init__(self, decoder: InstructionDecoder, slice_obj: Slice) -> None:
        self._decoder = decoder
        self._slice = slice_obj
        self._decoded: List[InstructionArg] = []

    def decode_all(self, specs: Sequence[ArgSpec]) -> List[InstructionArg]:
        results: List[InstructionArg] = []
        for spec in specs:
            arg = self._decode_spec(spec)
            results.append(arg)
            self._decoded.append(arg)
        self._resolve_pending_dicts(results)
        return results

    def _decode_spec(self, spec: ArgSpec) -> InstructionArg:
        type_name = spec.type
        if type_name == "uint":
            bits = spec.len or 0
            value = 0 if bits == 0 else self._slice.load_uint(bits)
            return InstructionArg(type_name, value)
        if type_name == "int":
            value = self._slice.load_int(spec.len or 0)
            return InstructionArg(type_name, value)
        if type_name == "stack":
            bits = spec.len or 0
            value = 0 if bits == 0 else self._slice.load_uint(bits)
            return InstructionArg(type_name, value)
        if type_name == "control":
            value = self._slice.load_uint(4)
            if value == 6:
                raise ValueError("Invalid control register index c6")
            return InstructionArg(type_name, value)
        if type_name == "plduzArg":
            # PLDUZ argument: raw 0-7 maps to bit counts 32, 64, 96, ..., 256, 0
            raw = self._slice.load_uint(3)
            value = ((raw + 1) & 7) << 5  # & 7 needed here for wraparound: 7+1=8 & 7=0
            return InstructionArg(type_name, value)
        if type_name == "tinyInt":
            raw = self._slice.load_uint(4)
            value = ((raw + 5) & 15) - 5
            return InstructionArg(type_name, value)
        if type_name == "largeInt":
            count = self._slice.load_uint(5) & 31
            bit_length = 3 + (count + 2) * 8
            if self._slice.remaining_bits < bit_length:
                raise ValueError(
                    f"largeInt requires {bit_length} bits but only {self._slice.remaining_bits} available"
                )
            bits = self._slice.load_bits(bit_length)
            bitstring = _bitarray_to_string(bits)
            value = _bits_to_signed_int(bitstring)
            return InstructionArg(type_name, value)
        if type_name == "minusOne":
            return InstructionArg(type_name, -1)
        if type_name == "s1":
            return InstructionArg(type_name, 1)
        if type_name == "delta":
            inner = self._decode_spec(spec.arg)  # type: ignore[arg-type]
            if not isinstance(inner.value, int):
                raise ValueError("Delta can only be applied to numeric args")
            adjusted = InstructionArg(inner.type, inner.value + (spec.delta or 0))
            return adjusted
        if type_name == "codeSlice":
            arg = self._decode_code_slice(spec)
            return InstructionArg("code", arg)
        if type_name == "inlineCodeSlice":
            arg = self._decode_inline_code_slice(spec)
            return InstructionArg("code", arg)
        if type_name == "refCodeSlice":
            arg = self._decode_ref_code_slice()
            return InstructionArg("code", arg)
        if type_name == "slice":
            arg = self._decode_slice(spec)
            return InstructionArg("slice", arg)
        if type_name == "dict":
            arg = self._decode_dict()
            return InstructionArg("dict", arg)
        if type_name == "debugstr":
            arg = self._decode_debug_slice()
            return InstructionArg("slice", arg)
        if type_name == "exoticCell":
            arg = self._decode_exotic_cell()
            return InstructionArg("exotic", arg)

        raise ValueError(f"Unsupported argument type: {type_name}")

    def _decode_code_slice(self, spec: ArgSpec) -> CodeArgument:
        count_refs = self._read_scalar(spec.refs)
        byte_len = self._read_scalar(spec.bits)
        real_length = byte_len * 8
        if self._slice.remaining_bits < real_length:
            raise ValueError(f"Code slice requires {real_length} bits but only {self._slice.remaining_bits} available")
        bits = self._slice.load_bits(real_length)
        refs = [self._slice.load_ref() for _ in range(count_refs)]

        builder = begin_cell()
        if real_length:
            builder.store_bits(bits)
        for ref in refs:
            builder.store_ref(ref)
        cell = builder.end_cell()

        nested = self._try_decompile_cell(cell)
        return CodeArgument(cell=cell, instructions=nested)

    def _decode_inline_code_slice(self, spec: ArgSpec) -> CodeArgument:
        byte_len = self._read_scalar(spec.bits)
        real_length = byte_len * 8
        if self._slice.remaining_bits < real_length:
            raise ValueError(f"Inline code slice requires {real_length} bits but only {self._slice.remaining_bits} available")
        bits = self._slice.load_bits(real_length)

        builder = begin_cell()
        if real_length:
            builder.store_bits(bits)
        cell = builder.end_cell()

        nested = self._try_decompile_cell(cell)
        return CodeArgument(cell=cell, instructions=nested)

    def _decode_ref_code_slice(self) -> CodeArgument:
        cell = self._slice.load_ref()
        nested = self._try_decompile_cell(cell)
        return CodeArgument(cell=cell, instructions=nested)

    def _decode_slice(self, spec: ArgSpec) -> SliceArgument:
        count_refs = self._read_scalar(spec.refs)
        byte_len = self._read_scalar(spec.bits)
        pad = spec.pad or 0
        total_bits = byte_len * 8 + pad
        if self._slice.remaining_bits < total_bits:
            raise ValueError(f"Slice requires {total_bits} bits but only {self._slice.remaining_bits} available")
        bits = self._slice.load_bits(total_bits)
        bitstring = _bitarray_to_string(bits)
        sentinel = bitstring.rfind("1")
        trimmed = bitstring[: sentinel] if sentinel >= 0 else ""

        builder = begin_cell()
        _store_bitstring(builder, trimmed)
        refs = [self._slice.load_ref() for _ in range(count_refs)]
        for ref in refs:
            builder.store_ref(ref)
        cell = builder.end_cell()
        return SliceArgument(cell=cell)

    def _decode_dict(self) -> DictionaryArgument:
        dict_cell = self._slice.load_ref()
        key_length: Optional[int] = None
        try:
            key_length = self._previous_numeric_value()
        except ValueError:
            key_length = None

        entries: Optional[List[DictionaryEntry]] = None
        if key_length is not None:
            try:
                entries = _parse_dictionary(dict_cell, key_length, self._decoder)
            except Exception as e:
                logger.debug("Failed to parse dictionary with key_length=%d: %s", key_length, e)
                entries = None

        return DictionaryArgument(cell=dict_cell, entries=entries, key_length=key_length)

    def _decode_debug_slice(self) -> SliceArgument:
        length_marker = self._slice.load_uint(4)
        real_length = (length_marker + 1) * 8
        # Boundary check before loading bits
        if self._slice.remaining_bits < real_length:
            raise ValueError(
                f"Debug slice requires {real_length} bits but only "
                f"{self._slice.remaining_bits} available"
            )
        bits = self._slice.load_bits(real_length)
        builder = begin_cell()
        builder.store_bits(bits)
        cell = builder.end_cell()
        return SliceArgument(cell=cell)

    def _decode_exotic_cell(self) -> ExoticArgument:
        cell = self._slice.load_ref()
        slice_view = cell.begin_parse()
        cell_type = slice_view.load_uint(8)
        if cell_type == 2:
            return ExoticArgument(cell=cell, kind="library", data_slice=slice_view)
        return ExoticArgument(cell=cell, kind="default")

    def _read_scalar(self, spec: Optional[ArgSpec]) -> int:
        if spec is None:
            return 0
        if spec.type in {"uint", "stack"}:
            bits = spec.len or 0
            return 0 if bits == 0 else self._slice.load_uint(bits)
        if spec.type == "int":
            return self._slice.load_int(spec.len or 0)
        if spec.type == "delta":
            nested = self._read_scalar(spec.arg)
            return nested + (spec.delta or 0)
        raise ValueError(f"Unsupported scalar type inside composite arg: {spec.type}")

    def _previous_numeric_value(self) -> int:
        for arg in reversed(self._decoded):
            if isinstance(arg.value, int):
                return arg.value
        raise ValueError("Dictionary argument requires preceding numeric key length")

    def _resolve_pending_dicts(self, decoded: List[InstructionArg]) -> None:
        """Bind dict arguments to following numeric key lengths when needed."""
        for idx, arg in enumerate(decoded):
            if arg.type != "dict":
                continue
            dict_arg = arg.value
            if getattr(dict_arg, "key_length", None) is not None:
                continue

            for next_arg in decoded[idx + 1:]:
                if not isinstance(next_arg.value, int):
                    continue
                dict_arg.key_length = next_arg.value
                try:
                    dict_arg.entries = _parse_dictionary(
                        dict_arg.cell,
                        next_arg.value,
                        self._decoder,
                    )
                except Exception:
                    dict_arg.entries = None
                break

    def _try_decompile_cell(self, cell: Cell) -> Optional[List[Instruction]]:
        """Attempt to decompile a Cell; return None on failure."""
        try:
            return self._decoder.decompile_cell(cell)
        except Exception:
            return None


def _bits_to_signed_int(bitstring: str, width: Optional[int] = None) -> int:
    """Convert a binary string to a signed integer using two's complement.

    Args:
        bitstring: Binary string (e.g., "1010" for 10 or -6 depending on width)
        width: Bit width for sign extension. If None, uses len(bitstring).

    Returns:
        Signed integer value

    Edge cases:
        - Empty string: Returns 0 (guard clause prevents int('', 2) ValueError)
        - width=0: Returns the unsigned value (no sign bit)
        - width > len(bitstring): Treats value as unsigned (positive)
        - Single bit "1" with width=1: Returns -1 (0b1 in 1-bit signed = -1)

    Examples:
        >>> _bits_to_signed_int("1111", 4)  # -1 in 4-bit signed
        -1
        >>> _bits_to_signed_int("0111", 4)  # 7 in 4-bit signed
        7
        >>> _bits_to_signed_int("")  # Empty string edge case
        0
    """
    if not bitstring:
        return 0
    if width is None:
        width = len(bitstring)
    value = int(bitstring, 2)
    if width > 0 and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def _parse_dictionary(
    dict_cell: Cell,
    key_length: int,
    decoder: InstructionDecoder,
    max_depth: int = MAX_DICT_PARSE_DEPTH,
) -> List[DictionaryEntry]:
    """Parse TVM dictionary into list of entries.

    Args:
        dict_cell: The dictionary cell to parse
        key_length: Bit length of dictionary keys
        decoder: Instruction decoder for nested code
        max_depth: Maximum recursion depth (default MAX_DICT_PARSE_DEPTH, prevents DoS via malformed BOC)

    Returns:
        List of dictionary entries

    Raises:
        RecursionError: If max_depth is exceeded
    """
    entries: List[DictionaryEntry] = []
    if key_length <= 0:
        return entries

    def read_unary_length(sl: Slice) -> int:
        count = 0
        while sl.load_bit():
            count += 1
            if count > MAX_UNARY_LENGTH:
                raise ValueError(f"Unary length exceeds maximum ({MAX_UNARY_LENGTH})")
        return count

    def do_parse(prefix: str, sl: Slice, remaining: int, depth: int = 0) -> None:
        if depth > max_depth:
            raise RecursionError(f"Dictionary parsing exceeded max depth {max_depth}")
        # TVM dictionary node encoding uses label bits to indicate format
        label_bit_0 = 1 if sl.load_bit() else 0
        prefix_length = 0
        current_prefix = prefix

        if label_bit_0 == 0:
            # hml_short: unary-encoded length followed by bits
            prefix_length = read_unary_length(sl)
            for _ in range(prefix_length):
                current_prefix += "1" if sl.load_bit() else "0"
        else:
            label_bit_1 = 1 if sl.load_bit() else 0
            log_bits = 0 if remaining + 1 <= 1 else max(0, math.ceil(math.log2(remaining + 1)))
            if label_bit_1 == 0:
                # hml_long: explicit length followed by bits
                prefix_length = sl.load_uint(log_bits) if log_bits > 0 else 0
                for _ in range(prefix_length):
                    current_prefix += "1" if sl.load_bit() else "0"
            else:
                # hml_same: single bit repeated prefix_length times
                repeated_bit = "1" if sl.load_bit() else "0"
                prefix_length = sl.load_uint(log_bits) if log_bits > 0 else 0
                current_prefix += repeated_bit * prefix_length

        if prefix_length > remaining:
            logger.debug(
                "Invalid dictionary node: prefix_length=%d exceeds remaining=%d, skipping",
                prefix_length, remaining
            )
            return

        new_remaining = remaining - prefix_length
        if new_remaining == 0:
            # Leaf node: extract the value
            child_slice = sl.copy()
            value_cell = child_slice.to_cell()
            # This matches TVM's internal representation where keys can be negative
            # (e.g., method_id -1 for recv_external). For key_length=1, prefix "1" = -1.
            # If unsigned interpretation is needed, callers should use: key & ((1 << key_length) - 1)
            key = _bits_to_signed_int(current_prefix, key_length)
            nested = decoder.decompile_cell(value_cell)
            code_arg = CodeArgument(cell=value_cell, instructions=nested)
            entries.append(DictionaryEntry(key=key, code=code_arg))
        elif new_remaining > 0:
            # Fork node: recurse into left (0) and right (1) branches
            left = sl.load_ref()
            right = sl.load_ref()
            fork_remaining = new_remaining - 1
            if fork_remaining >= 0:
                if not left.is_exotic:
                    do_parse(current_prefix + "0", left.begin_parse(), fork_remaining, depth + 1)
                if not right.is_exotic:
                    do_parse(current_prefix + "1", right.begin_parse(), fork_remaining, depth + 1)

    do_parse("", dict_cell.begin_parse(), key_length, 0)
    return entries


_DEFAULT_DECODER = InstructionDecoder() # Module-level singleton to avoid rebuilding the instruction spec index on every call.


def decompile_cell(cell: Cell) -> List[Instruction]:
    """Module-level helper mirroring the legacy API."""
    return _DEFAULT_DECODER.decompile_cell(cell)
