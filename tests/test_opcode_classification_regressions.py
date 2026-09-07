from tasmscan.analyzer.constants import (
    BRANCH_CONT_ARITY,
    INFINITE_LOOP_OPCODES,
    LOOP_CALL_OPCODES,
    TERMINATORS,
)
from tasmscan.ir.ir_types import IRType
from tasmscan.ir.mappings.control_flow import CONTROL_FLOW_MAPPINGS


def test_throw_short_and_againend_are_terminators():
    assert "THROW_SHORT" in TERMINATORS
    assert "AGAINEND" in TERMINATORS


def test_end_loop_variants_are_classified_for_cfg():
    end_variants = {
        "REPEATEND",
        "REPEATENDBRK",
        "UNTILEND",
        "UNTILENDBRK",
        "WHILEEND",
        "WHILEENDBRK",
        "AGAINEND",
        "AGAINENDBRK",
    }
    assert end_variants.issubset(LOOP_CALL_OPCODES)
    assert "AGAINEND" in INFINITE_LOOP_OPCODES
    assert "AGAINENDBRK" in INFINITE_LOOP_OPCODES
    assert BRANCH_CONT_ARITY["REPEATEND"] == 0
    assert BRANCH_CONT_ARITY["WHILEEND"] == 1
    assert BRANCH_CONT_ARITY["AGAINEND"] == 0


def test_non_return_control_flow_variants_map_to_set_continuation():
    assert CONTROL_FLOW_MAPPINGS["RETURNARGS"] == IRType.SET_CONTINUATION
    assert CONTROL_FLOW_MAPPINGS["RETURNVARARGS"] == IRType.SET_CONTINUATION
    assert CONTROL_FLOW_MAPPINGS["THENRET"] == IRType.SET_CONTINUATION
    assert CONTROL_FLOW_MAPPINGS["THENRETALT"] == IRType.SET_CONTINUATION
