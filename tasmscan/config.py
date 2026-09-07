"""Default configuration values for tasmscan."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Optional, Union

logger = logging.getLogger("tasmscan.config")


def validate_config_value(name: str, value: int, min_val: int, max_val: int) -> int:
    """
    Validate and clamp configuration value to acceptable range.

    This function provides safe configuration handling by:
    1. Validating that values fall within defined bounds
    2. Clamping out-of-range values to the nearest bound
    3. Logging a warning when clamping occurs

    Args:
        name: Configuration parameter name (for logging)
        value: The value to validate
        min_val: Minimum acceptable value (inclusive)
        max_val: Maximum acceptable value (inclusive)

    Returns:
        The original value if within range, otherwise the clamped value

    Example:
        >>> validate_config_value("MAX_PATHS", 500, 1, 100)
        WARNING: Config MAX_PATHS=500 out of range [1, 100], clamping to bounds
        100
    """
    if value < min_val or value > max_val:
        logger.warning(
            f"Config {name}={value} out of range [{min_val}, {max_val}], "
            f"clamping to bounds"
        )
        return max(min_val, min(max_val, value))
    return value


# Configuration bounds for runtime validation
# Used by PathSensitiveDataFlowAnalyzer to validate user-provided config values.
# Format: parameter_name -> (min_value, max_value)
# See path_sensitive_dataflow.py for usage with validate_config_value().
CONFIG_BOUNDS = {
    "MAX_PATHS": (1, 10000),
    "MAX_LOOP_UNROLL": (1, 100),
    "MAX_ANALYSIS_DEPTH": (10, 500),
    "MAX_BLOCK_VISITS": (1, 100),
    "MAX_SEARCH_ITERATIONS": (100, 100000),
    "MAX_WORKLIST_SIZE": (100, 100000),
}


class AnalysisMode(Enum):
    """Analysis mode affecting precision/soundness tradeoff."""
    SECURITY_AUDIT = "security_audit"  # Prioritize soundness, more false positives OK
    CI_CHECK = "ci_check"              # Prioritize precision, fewer false positives
    BALANCED = "balanced"              # Default balanced mode


DEFAULT_ANALYSIS_MODE = AnalysisMode.BALANCED
STRICT_COMPLETENESS_DEFAULT = False

# Mode-specific settings
# Note: currently only max_paths and max_block_visits are consumed.
ANALYSIS_MODE_SETTINGS = {
    AnalysisMode.SECURITY_AUDIT: {
        "max_paths": 500,
        "max_block_visits": 20,
    },
    AnalysisMode.CI_CHECK: {
        "max_paths": 50,
        "max_block_visits": 5,
    },
    AnalysisMode.BALANCED: {
        "max_paths": 100,
        "max_block_visits": 10,
    },
}


def resolve_analysis_mode(mode: Optional[Union[AnalysisMode, str]]) -> AnalysisMode:
    """Resolve analysis mode from string or enum value.

    Raises:
        ValueError: If mode string is not a valid AnalysisMode value
    """
    if mode is None:
        return DEFAULT_ANALYSIS_MODE
    if isinstance(mode, AnalysisMode):
        return mode
    try:
        return AnalysisMode(mode)
    except ValueError:
        valid_modes = [m.value for m in AnalysisMode]
        raise ValueError(
            f"Invalid analysis mode '{mode}'. Valid modes: {valid_modes}"
        ) from None


def build_path_config(
    analysis_mode: Optional[Union[AnalysisMode, str]] = None,
    overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Build path-sensitive analysis configuration.

    Args:
        analysis_mode: Optional analysis mode (string or enum) to seed defaults
        overrides: Optional explicit overrides for path-sensitive parameters

    Returns:
        Dict containing path-sensitive configuration values
    """
    config: Dict[str, int] = {}

    if analysis_mode is not None:
        mode = resolve_analysis_mode(analysis_mode)
        settings = ANALYSIS_MODE_SETTINGS.get(mode, {})
        for key in (
            "max_paths",
            "max_block_visits",
            "max_analysis_depth",
            "max_loop_unroll",
            "max_worklist_size",
        ):
            if key in settings:
                config[key] = settings[key]

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    return config

MAX_PATHS = 100
MAX_LOOP_UNROLL = 3
MAX_ANALYSIS_DEPTH = 50

MIN_COMPLEXITY_THRESHOLD = 10

MAX_SEARCH_ITERATIONS = 1000

# Path-sensitive completeness materiality threshold.
# Tiny truncation counts can occur around defensive widening/merge points and
# often do not materially reduce security findings quality.
MIN_MATERIAL_TRUNCATION_EVENTS = 5

# Stack underflow detection
# Threshold for cumulative stack consumption before flagging potential underflow.
# Set to 3 to reduce false positives - many normal sequences consume 2-3 elements.
# Lower values increase sensitivity, higher values reduce false positives.
CUMULATIVE_STACK_CONSUMPTION_THRESHOLD = 3

# Path-sensitive analysis block visit limit
# Each block can be visited at most this many times across all paths
MAX_BLOCK_VISITS = 10

# Prevents unbounded memory growth in pathological CFGs with exponential branching
# 10000 entries is sufficient for most real-world contracts while providing DoS protection
MAX_WORKLIST_SIZE = 10000

# Bounced message detection
BOUNCED_SEARCH_RANGE = 30

# INMSGPARAM index for sender (alias INMSG_SRC in cp0.json)
INMSGPARAM_SENDER_INDEX = 2

# Guarded values set maximum size (for memory management)
MAX_GUARDED_VALUES = 1000

# Message context detection stack depth
# Checking 5 positions covers typical message parsing patterns where
# intermediate values may be pushed between loading message data.
# Configurable to adjust sensitivity of message context detection.
MESSAGE_CONTEXT_STACK_DEPTH = 5

# Default stack check depth for sensitive operations
# When stack effect is unknown, check this many top stack positions
# for tainted unchecked values. Value of 3 covers most TVM operations
# which typically consume 1-3 stack elements.
DEFAULT_SENSITIVE_CHECK_DEPTH = 3

# Maximum abstract taint-stack depth retained during fixpoint analysis.
# This is the K_s bound used by the evaluation configuration.
TAINT_STACK_DEPTH = 64

# =============================================================================
# Centralized Guard Opcodes Definition
# =============================================================================
# Opcodes that validate values through control flow (IF/THROW patterns).
# This centralized definition ensures consistency across all modules:
# - dataflow/analyzer.py: Taint analysis guard detection
# - detectors/cfg_utils.py: CFG traversal guard detection
# - detectors/bounced_message.py: Bounced flag guard detection
#
# Adding new guard opcodes here automatically updates all detection logic.
GUARD_OPCODES = frozenset({
    # Conditional branches (basic)
    "IF", "IFNOT", "IFJMP", "IFNOTJMP", "IFRET", "IFNOTRET", "IFELSE",
    # Conditional branches (ref variants) - execute continuation from cell reference
    "IFREF", "IFNOTREF", "IFJMPREF", "IFNOTJMPREF",
    # Conditional branches (mixed ref/stack)
    "IFREFELSE", "IFELSEREF", "IFREFELSEREF",
    # Conditional returns (alt variants)
    "IFRETALT", "IFNOTRETALT",
    # Bit-conditional jumps - test specific bit and jump
    "IFBITJMP", "IFNBITJMP", "IFBITJMPREF", "IFNBITJMPREF",
    # Throw instructions (standard)
    "THROW", "THROWIF", "THROWIFNOT",
    # Throw instructions (short forms)
    "THROWIF_SHORT", "THROWIFNOT_SHORT",
    # Throw instructions (with argument)
    "THROWARG", "THROWARGIF", "THROWARGIFNOT",
    # Throw instructions (any exception code)
    "THROWANY", "THROWANYIF", "THROWANYIFNOT",
    # Throw instructions (with argument + any exception code)
    "THROWARGANY", "THROWARGANYIF", "THROWARGANYIFNOT",
})

# =============================================================================
# Centralized Sensitive Opcodes Definition
# =============================================================================
# Core set of security-sensitive opcodes. Individual detectors may extend
# this with domain-specific additions, but the core set ensures consistency.
#
# Adding new sensitive opcodes here automatically updates:
# - taint_registry.py: Taint analysis sensitive operation detection
# - stack_underflow.py: Stack underflow near sensitive ops
# - bad_randomness.py: Randomness near sensitive ops
# - opcode_spec.py: OpcodeSpec.is_sensitive classification
CORE_SENSITIVE_OPCODES = frozenset({
    "SENDRAWMSG",   # Send raw message - must validate sender/value
    "SENDMSG",      # TVM v4+ send message
    "RAWRESERVE",   # Reserve funds - must check amount
    "RAWRESERVEX",  # Extended reserve variant
    "ACCEPT",       # Accept gas payment
    "SETGASLIMIT",  # Set gas limit
    "SETCODE",      # Change contract code
    "COMMIT",       # Commit state changes
    "POPCTR",       # Write to control register
})

# Maximum unary-encoded length for dictionary parsing
# TVM dictionary keys are limited to 1023 bits maximum
MAX_UNARY_LENGTH = 1023

# =============================================================================
# =============================================================================
# Centralized recursion depth limits to prevent stack overflow DoS attacks.
# These values are tuned for typical TVM contract structures while providing
# protection against maliciously crafted inputs.

# For instruction decoder (decompile_cell/decompile_slice)
# Lower limit (64) because instruction decoding creates deeper call stacks
MAX_INSTRUCTION_DECODE_DEPTH = 64

# For dictionary parsing (_parse_dictionary, parse_code_dictionary)
# Higher limit (256) to accommodate deeply nested dictionary structures
MAX_DICT_PARSE_DEPTH = 256

# For program analysis (continuation extraction)
# Higher limit (256) to handle complex contract control flow graphs
MAX_PROGRAM_ANALYZE_DEPTH = 256


# TON smart contract entry point initialization
# See: https://docs.ton.org/v3/documentation/tvm/tvm-initialization
# When a message is received, TVM initializes the stack with 5 items:
#   s0: selector (method_id: 0=recv_internal, -1=recv_external, -2=run_ticktock)
#   s1: msg_body (Slice)
#   s2: msg_cell (Cell)
#   s3: msg_value (Int, in nanotons)
#   s4: balance (Tuple: [current_balance, extra_currencies])
CONTRACT_ENTRY_STACK_HEIGHT = 5

# run_ticktock entry point stack height
# See: https://docs.ton.org/v3/documentation/tvm/tvm-initialization
# Stack layout: s0=selector(-2), s1=is_tock, s2=account_addr, s3=balance
RUN_TICKTOCK_STACK_HEIGHT = 4

# Stack underflow detection thresholds
# STACK_UNCERTAINTY_THRESHOLD: Maximum uncertainty range to report
# If height_max - height_min > this value, skip reporting (too uncertain)
# Value of 15 chosen based on empirical analysis: dynamic instructions like DROPX,
# BLKDROP with unknown arguments can accumulate conservative estimates that
# produce large uncertainty ranges. Beyond 15 elements, the false positive rate
# exceeds acceptable levels for most codebases.
STACK_UNCERTAINTY_THRESHOLD = 15

# STACK_SHALLOW_UNDERFLOW_THRESHOLD: Minimum depth for shallow underflow
# If height_min > this value with uncertainty, skip (likely false positive)
# Value of -5 chosen because: small negative heights (-1 to -4) with uncertainty
# often arise from instructions like TUPLE/UNTUPLE where the analyzer uses
# worst-case estimates. True underflows typically have more significant negative
# depths. At -5 or below, even uncertain reports warrant attention.
STACK_SHALLOW_UNDERFLOW_THRESHOLD = -5

# =============================================================================
# Shared Slice Data Load Opcodes
# =============================================================================
# Opcodes that read/load data from slices. Used by multiple detectors:
# - lack_end_parse.py: Detecting unvalidated slice reads
# - tlb_structure.py: Detecting TLB structure violations
SLICE_DATA_LOAD_OPCODES = frozenset({
    # Integer loads
    "LDI", "LDIX", "LDU", "LDUX",
    "PLDI", "PLDIX", "PLDU", "PLDUX",
    "LDIQ", "LDUQ",
    # Slice loads
    "LDSLICE", "LDSLICEX", "PLDSLICE", "PLDSLICEX",
    "LDSLICEQ",
    # Variable integer loads
    "LDVARINT16", "LDVARINT32", "LDVARUINT16", "LDVARUINT32",
    "LDVARINTQ", "LDVARUINTQ",
    # Special loads
    "LDDICT", "LDMSGADDR", "LDGRAMS", "LDCOINS",
    "LDOPTREF", "LDZEROES", "LDONES", "LDSAME",
    "LDMSGADDRQ",
    "REWRITESTDADDR", "REWRITEVARADDR",
})

SLICE_REF_LOAD_OPCODES = frozenset({
    "LDREF", "LDREFRTOS", "PLDREF", "PLDREFIDX",
})
