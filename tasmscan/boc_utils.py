"""
BOC decoding utilities shared by CLI, library, and scripts.
"""
from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE64_ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "+/=_-"
)
BOC_MAGIC = b'\xb5\xee\x9c\x72'


def load_boc_file(path: Path) -> bytes:
    """Read a BOC file, auto-detecting binary vs text-encoded (base64/hex)."""
    raw = path.read_bytes()
    if raw[:4] == BOC_MAGIC:
        return raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Not a valid BOC file (bad magic) and not text-decodable: {path}") from exc
    return detect_and_decode_boc(text)


def _normalize_base64(data_clean: str) -> str:
    if not data_clean:
        raise ValueError("Empty base64 data")
    if any(c not in _BASE64_ALLOWED for c in data_clean):
        raise ValueError("Invalid base64: contains non-base64 characters")

    stripped = data_clean.rstrip("=")
    if not stripped:
        raise ValueError("Invalid base64: no data")
    if "=" in stripped:
        raise ValueError("Invalid base64: padding character in middle")
    if len(data_clean) - len(stripped) > 2:
        raise ValueError("Invalid base64: too much padding")

    mod = len(stripped) % 4
    if mod == 1:
        raise ValueError("Invalid base64: incorrect length")
    if mod:
        stripped += "=" * (4 - mod)
    return stripped


def _decode_base64_strict(data_clean: str) -> bytes:
    normalized = _normalize_base64(data_clean)
    normalized = normalized.replace("-", "+").replace("_", "/")
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc


def detect_and_decode_boc(data: str) -> bytes:
    """
    Auto-detect encoding format (base64 or hex) and decode to BOC bytes.

    Priority: magic-verified results first, then fallback without magic check.
    1. hex with BOC magic → return
    2. base64 with BOC magic → return
    3. base64 without magic → return (fallback)
    4. hex without magic → return (fallback)
    5. all failed → raise
    """
    data_clean = "".join(data.split())
    if not data_clean:
        raise ValueError("Empty input data")

    has_0x_prefix = data_clean.lower().startswith("0x")
    hex_candidate = data_clean
    if has_0x_prefix:
        hex_candidate = data_clean[2:]
        if not hex_candidate:
            raise ValueError("Invalid hex: missing data after 0x prefix")

    # Try hex decode once
    hex_result: bytes | None = None
    is_likely_hex = bool(hex_candidate) and all(c in "0123456789abcdefABCDEF" for c in hex_candidate)
    if is_likely_hex:
        try:
            hex_result = bytes.fromhex(hex_candidate)
        except ValueError:
            logger.debug("hex decode failed")

    # Try base64 decode once
    b64_result: bytes | None = None
    try:
        b64_result = _decode_base64_strict(data_clean)
    except ValueError as exc:
        logger.debug("base64 decode failed: %s", exc)

    # Priority 1: hex with magic
    if hex_result is not None and hex_result[:4] == BOC_MAGIC:
        return hex_result

    # Priority 2: base64 with magic
    if b64_result is not None and b64_result[:4] == BOC_MAGIC:
        return b64_result

    # Fallback: prefer hex when 0x prefix signals hex intent
    first, second = (hex_result, b64_result) if has_0x_prefix else (b64_result, hex_result)
    first_label, second_label = ("hex", "base64") if has_0x_prefix else ("base64", "hex")

    if first is not None:
        logger.debug("returning %s result without BOC magic verification", first_label)
        return first

    if second is not None:
        logger.debug("returning %s result without BOC magic verification", second_label)
        return second

    raise ValueError(
        "Cannot decode data: not valid base64 or hex format. "
        f"Data preview: {data_clean[:40]}..."
    )
