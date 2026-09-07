"""
Tests for BouncedMessageDetector and BouncedHandlerPatternDetector.

Tests cover:
- _check_bounced_extraction: LDU 4 + various bit extraction patterns
- _check_bounced_extraction_32bit: LDU 32 + mask patterns
- _has_guard_after: Guard detection logic
- _is_trivial_contract: Boundary cases
- CFG-based detection path
- Fallback detection without CFG
"""

from tasmscan.analyzer.facts import AnalysisFacts, BasicBlock, Event, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.bounced_message import (
    BouncedMessageDetector,
    BouncedHandlerPatternDetector,
)
from tasmscan.ir.ir_builder import IRBuilder
from tasmscan.ir.tasir_types import (
    InstructionKind,
    TVMBasicBlock,
    TVMFunction,
    TVMInstruction,
    TVMModule,
)


class MockInstruction:
    """Mock instruction."""

    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class MockArg:
    """Mock instruction argument with value."""

    def __init__(self, value):
        self.value = value


def make_instruction_fact(opcode: str, index: int, arg_value=None) -> InstructionFact:
    """Create an InstructionFact with optional argument value."""
    args = [MockArg(arg_value)] if arg_value is not None else []
    return InstructionFact(
        instruction=MockInstruction(opcode, args),
        index=index,
    )


def make_facts(instructions: list, events=None, basic_blocks=None) -> AnalysisFacts:
    """Create AnalysisFacts with given instructions and events."""
    return AnalysisFacts(
        instructions=instructions,
        events=events or [],
        basic_blocks=basic_blocks or [],
    )


class TestBouncedMessageDetector:
    """Tests for BouncedMessageDetector."""

    def test_trivial_contract_skipped(self):
        """Trivial contracts (< min_complexity) should be skipped."""
        detector = BouncedMessageDetector()
        # Less than default MIN_COMPLEXITY_THRESHOLD (10)
        instructions = [make_instruction_fact("NOP", i) for i in range(5)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_empty_contract_skipped(self):
        """Contracts with < 2 instructions should be skipped."""
        detector = BouncedMessageDetector(min_complexity_threshold=1)
        instructions = [make_instruction_fact("NOP", 0)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_no_bounced_check_reports_vulnerability(self):
        """Contract without bounced check should report vulnerability."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        # Instructions without LDU 4 pattern
        instructions = [
            make_instruction_fact("SETCP0", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "bounced flag" in findings[0].message.lower()
        assert findings[0].severity == "critical"

    def test_ldu4_with_and_guard_passes(self):
        """LDU 4 + PUSHINT 1 + AND + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("PUSHINT", 1, arg_value=1),
            make_instruction_fact("AND", 2),
            make_instruction_fact("IFNOT", 3),  # Guard
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_ldu4_with_modpow2_guard_passes(self):
        """LDU 4 + MODPOW2 1 + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("MODPOW2", 1, arg_value=1),
            make_instruction_fact("THROWIFNOT", 2),  # Guard
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_ldu4_with_rshift3_guard_passes(self):
        """LDU 4 + RSHIFT 3 + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("RSHIFT", 1, arg_value=3),
            make_instruction_fact("IF", 2),  # Guard
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_pldu4_pattern_passes(self):
        """PLDU 4 + bit extraction + guard should be recognized.

        Note: TVM has no ISNONZERO opcode; use ISZERO instead.
        """
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("PLDU", 0, arg_value=4),
            make_instruction_fact("ISZERO", 1),
            make_instruction_fact("THROWIFNOT", 2),  # ISZERO + THROWIFNOT = throw if non-zero
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_ldu4_without_guard_fails(self):
        """LDU 4 + bit extraction WITHOUT guard should fail."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("PUSHINT", 1, arg_value=1),
            make_instruction_fact("AND", 2),
            make_instruction_fact("DROP", 3),  # Consuming, not guard
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1

    def test_ldu32_with_mask_guard_passes(self):
        """LDU 32 + PUSHINT 0x80000000 + AND + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=32),
            make_instruction_fact("PUSHINT", 1, arg_value=0x80000000),
            make_instruction_fact("AND", 2),
            make_instruction_fact("IFNOT", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_ldu32_with_rshift31_guard_passes(self):
        """LDU 32 + RSHIFT 31 + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=32),
            make_instruction_fact("RSHIFT", 1, arg_value=31),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_pldu_1_opcode_pattern_passes(self):
        """LDU 4 + PLDU 1 + guard should be recognized (single bit extraction)."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("PLDU", 1, arg_value=1),  # Extract single bit
            make_instruction_fact("IFJMP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_booleval_pattern_passes(self):
        """LDU 4 + BOOLEVAL + guard should be recognized.

        Note: TVM has BOOLEVAL (not BOOLVAL) which converts integer to 0/-1.
        """
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("BOOLEVAL", 1),
            make_instruction_fact("IFRET", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_comparison_opcodes_pass(self):
        """LDU 4 + comparison opcode + guard should be recognized.

        Note: TVM has no ISNONZERO opcode; non-zero check is done via ISZERO + NOT.
        """
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        for cmp_opcode in ["ISZERO", "EQINT", "NEQINT"]:
            instructions = [
                make_instruction_fact("LDU", 0, arg_value=4),
                make_instruction_fact(cmp_opcode, 1),
                make_instruction_fact("IF", 2),
                make_instruction_fact("NOP", 3),
                make_instruction_fact("NOP", 4),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert findings == [], f"Failed for {cmp_opcode}"

    def test_gtint_lessint_patterns_pass(self):
        """LDU 4 + GTINT/LESSINT with boundary values + guard should pass."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        for opcode, arg in [("GTINT", 0), ("LESSINT", 1), ("LEQINT", 0), ("GEQINT", 0)]:
            instructions = [
                make_instruction_fact("LDU", 0, arg_value=4),
                make_instruction_fact(opcode, 1, arg_value=arg),
                make_instruction_fact("THROWIF", 2),
                make_instruction_fact("NOP", 3),
                make_instruction_fact("NOP", 4),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert findings == [], f"Failed for {opcode} {arg}"

    def test_shift_chain_pattern_passes(self):
        """LDU 4 + LSHIFT + RSHIFT + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=6)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("LSHIFT", 1, arg_value=28),
            make_instruction_fact("RSHIFT", 2, arg_value=31),
            make_instruction_fact("IF", 3),
            make_instruction_fact("NOP", 4),
            make_instruction_fact("NOP", 5),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_pushpow2_pattern_passes(self):
        """LDU 4 + PUSHPOW2 0 (=1) + AND + guard should be recognized."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("PUSHPOW2", 1, arg_value=0),  # 2^0 = 1
            make_instruction_fact("AND", 2),
            make_instruction_fact("IFNOT", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_pushint_4_and_far_throwifnot_passes(self):
        """PUSHINT immediate specializations should still feed AND-pattern matching."""
        detector = BouncedMessageDetector(min_complexity_threshold=13)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("SWAP", 1),
            make_instruction_fact("PUSHINT_4", 2, arg_value=1),
            make_instruction_fact("AND", 3),
            make_instruction_fact("NEGATE", 4),
            make_instruction_fact("SWAP", 5),
            make_instruction_fact("LDMSGADDR", 6),
            make_instruction_fact("SWAP", 7),
            make_instruction_fact("DUP", 8),
            make_instruction_fact("SBITS", 9),
            make_instruction_fact("PUSHINT_16", 10, arg_value=267),
            make_instruction_fact("EQUAL", 11),
            make_instruction_fact("THROWIFNOT", 12),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_search_range_option(self):
        """Custom search_range option should be respected."""
        detector = BouncedMessageDetector(
            options={"search_range": 5},
            min_complexity_threshold=10,
        )
        assert detector.search_range == 5

    def test_stops_on_unrelated_operations(self):
        """Should stop pattern matching on unrelated operations like ADD."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("ADD", 1),  # Unrelated, should break
            make_instruction_fact("PUSHINT", 2, arg_value=1),
            make_instruction_fact("AND", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1  # No bounced check found due to ADD breaking chain


class TestBouncedMessageDetectorTASIRScope:
    """Regression tests for recv_internal-scoped TASIR analysis."""

    def test_scopes_search_to_recv_internal(self):
        """
        Bounced check in recv_internal should be found even when earlier global
        instructions exceed the detector search range.
        """
        detector = BouncedMessageDetector(
            options={"search_range": 5},
            min_complexity_threshold=0,
        )

        instructions = [
            make_instruction_fact("NOP", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
            make_instruction_fact("LDU", 5, arg_value=4),
            make_instruction_fact("PUSHINT", 6, arg_value=1),
            make_instruction_fact("AND", 7),
            make_instruction_fact("IFNOT", 8),
        ]
        facts = make_facts(instructions)

        other_block = TVMBasicBlock(
            id=0,
            context_id="method_1",
            instructions=[
                TVMInstruction(index=i, kind=InstructionKind.NOP, opcode="NOP")
                for i in range(5)
            ],
        )
        recv_block = TVMBasicBlock(
            id=1,
            context_id="recv_internal",
            instructions=[
                TVMInstruction(index=5, kind=InstructionKind.CELL_LOAD, opcode="LDU"),
                TVMInstruction(index=6, kind=InstructionKind.STACK_PUSH, opcode="PUSHINT"),
                TVMInstruction(index=7, kind=InstructionKind.BITWISE, opcode="AND"),
                TVMInstruction(index=8, kind=InstructionKind.BRANCH_CONDITIONAL, opcode="IFNOT"),
            ],
        )
        module = TVMModule(
            functions={
                1: TVMFunction(method_id=1, blocks={0: other_block}),
                0: TVMFunction(method_id=0, blocks={1: recv_block}, is_recv_internal=True),
            }
        )
        facts.metadata["tasir_module"] = module

        findings = detector.detect(facts)
        assert findings == []

    def test_scope_includes_recv_internal_reachable_continuations(self):
        detector = BouncedMessageDetector(
            options={"search_range": 20},
            min_complexity_threshold=0,
        )

        class _Cont:
            def __init__(self, instructions):
                self.instructions = instructions

        class _Arg:
            def __init__(self, value):
                self.value = value

        class _Inst:
            def __init__(self, name, args=None):
                self.name = name
                self.args = args or []

        facts = ProgramAnalyzer().analyze(
            [
                _Inst(
                    "PUSHCONT",
                    [
                        _Arg(
                            _Cont(
                                [
                                    _Inst("LDU", [_Arg(4)]),
                                    _Inst("PUSHINT", [_Arg(1)]),
                                    _Inst("AND"),
                                    _Inst("IFNOT"),
                                ]
                            )
                        )
                    ],
                ),
                _Inst("EXECUTE"),
            ],
            cell=None,
        )

        fallback_findings = detector.detect(facts)

        module = IRBuilder().build_tasir(facts)
        facts.metadata["tasir_module"] = module
        tasir_findings = detector.detect(facts)

        assert fallback_findings == []
        assert tasir_findings == []

    def test_tasir_bounced_check_with_send_reports_medium(self):
        detector = BouncedMessageDetector(
            options={"search_range": 20},
            min_complexity_threshold=0,
        )

        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("SWAP", 1),
            make_instruction_fact("PUSHINT_4", 2, arg_value=1),
            make_instruction_fact("AND", 3),
            make_instruction_fact("NEGATE", 4),
            make_instruction_fact("SWAP", 5),
            make_instruction_fact("LDMSGADDR", 6),
            make_instruction_fact("SWAP", 7),
            make_instruction_fact("DUP", 8),
            make_instruction_fact("SBITS", 9),
            make_instruction_fact("PUSHINT_16", 10, arg_value=267),
            make_instruction_fact("EQUAL", 11),
            make_instruction_fact("THROWIFNOT", 12),
            make_instruction_fact("DUP", 13),
            make_instruction_fact("PLDU", 14, arg_value=11),
            make_instruction_fact("DUP", 15),
            make_instruction_fact("PUSHINT_16", 16, arg_value=1279),
            make_instruction_fact("EQUAL", 17),
            make_instruction_fact("THROWIF", 18),
            make_instruction_fact("SENDRAWMSG", 19),
        ]
        facts = make_facts(instructions, events=[Event(type="send", instruction=instructions[19])])

        recv_block = TVMBasicBlock(
            id=1,
            context_id="recv_internal",
            instructions=[
                TVMInstruction(index=0, kind=InstructionKind.CELL_LOAD, opcode="LDU"),
                TVMInstruction(index=1, kind=InstructionKind.STACK_SHUFFLE, opcode="SWAP"),
                TVMInstruction(index=2, kind=InstructionKind.STACK_PUSH, opcode="PUSHINT_4"),
                TVMInstruction(index=3, kind=InstructionKind.BITWISE, opcode="AND"),
                TVMInstruction(index=4, kind=InstructionKind.ARITHMETIC, opcode="NEGATE"),
                TVMInstruction(index=5, kind=InstructionKind.STACK_SHUFFLE, opcode="SWAP"),
                TVMInstruction(index=6, kind=InstructionKind.CELL_LOAD, opcode="LDMSGADDR"),
                TVMInstruction(index=7, kind=InstructionKind.STACK_SHUFFLE, opcode="SWAP"),
                TVMInstruction(index=8, kind=InstructionKind.STACK_SHUFFLE, opcode="DUP"),
                TVMInstruction(index=9, kind=InstructionKind.CELL_LOAD, opcode="SBITS"),
                TVMInstruction(index=10, kind=InstructionKind.STACK_PUSH, opcode="PUSHINT_16"),
                TVMInstruction(index=11, kind=InstructionKind.COMPARISON, opcode="EQUAL"),
                TVMInstruction(index=12, kind=InstructionKind.THROW, opcode="THROWIFNOT"),
                TVMInstruction(index=13, kind=InstructionKind.STACK_SHUFFLE, opcode="DUP"),
                TVMInstruction(index=14, kind=InstructionKind.CELL_LOAD, opcode="PLDU"),
                TVMInstruction(index=15, kind=InstructionKind.STACK_SHUFFLE, opcode="DUP"),
                TVMInstruction(index=16, kind=InstructionKind.STACK_PUSH, opcode="PUSHINT_16"),
                TVMInstruction(index=17, kind=InstructionKind.COMPARISON, opcode="EQUAL"),
                TVMInstruction(index=18, kind=InstructionKind.THROW, opcode="THROWIF"),
                TVMInstruction(index=19, kind=InstructionKind.SEND_MESSAGE, opcode="SENDRAWMSG"),
            ],
        )
        module = TVMModule(
            functions={
                0: TVMFunction(method_id=0, blocks={1: recv_block}, is_recv_internal=True),
            }
        )
        facts.metadata["tasir_module"] = module

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "medium"


class TestBouncedHandlerPatternDetector:
    """Tests for BouncedHandlerPatternDetector."""

    def test_trivial_contract_skipped(self):
        """Trivial contracts should be skipped."""
        detector = BouncedHandlerPatternDetector()
        instructions = [make_instruction_fact("NOP", i) for i in range(5)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_no_bounced_load_returns_empty(self):
        """No bounced load instruction should return empty findings."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("NOP", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # No findings because no LDU 4/32 at all - handled by BouncedMessageDetector
        assert findings == []

    def test_flag_load_without_guard_high_severity(self):
        """Flag load without subsequent guard should be high severity."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("DROP", 1),  # No guard
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        # Create send event
        send_event = Event(type="send", instruction=instructions[4])
        facts = make_facts(instructions, events=[send_event])

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"
        assert "guard" in findings[0].message.lower()

    def test_tasir_no_guard_uses_send_events_when_kind_missing(self):
        """Should still report when TASIR kind misses SEND_MESSAGE classification."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("DROP", 1),
            make_instruction_fact("SENDRAWMSG", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        send_event = Event(type="send", instruction=instructions[2])
        facts = make_facts(instructions, events=[send_event])

        recv_block = TVMBasicBlock(
            id=1,
            context_id="recv_internal",
            instructions=[
                TVMInstruction(index=0, kind=InstructionKind.CELL_LOAD, opcode="LDU"),
                TVMInstruction(index=1, kind=InstructionKind.STACK_POP, opcode="DROP"),
                # Intentionally misclassified kind to simulate incomplete TASIR labeling.
                TVMInstruction(index=2, kind=InstructionKind.NOP, opcode="SENDRAWMSG"),
                TVMInstruction(index=3, kind=InstructionKind.NOP, opcode="NOP"),
                TVMInstruction(index=4, kind=InstructionKind.NOP, opcode="NOP"),
            ],
        )
        module = TVMModule(
            functions={
                0: TVMFunction(method_id=0, blocks={1: recv_block}, is_recv_internal=True),
            }
        )
        facts.metadata["tasir_module"] = module

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_with_cfg_analysis(self):
        """With CFG available, should use CFG-based detection."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("AND", 1),
            make_instruction_fact("IFNOT", 2),  # Guard
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]

        # Create basic blocks
        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3, 4],
            successors=[],
        )

        send_event = Event(type="send", instruction=instructions[3])
        facts = make_facts(instructions, events=[send_event], basic_blocks=[block])

        findings = detector.detect(facts)
        # With guard before send, should have no findings
        assert findings == []

    def test_send_before_guard_fallback(self):
        """Without CFG, send before guard should be flagged."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=6)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("SENDRAWMSG", 1),  # Send BEFORE guard
            make_instruction_fact("AND", 2),
            make_instruction_fact("IFNOT", 3),  # Guard after send
            make_instruction_fact("NOP", 4),
            make_instruction_fact("NOP", 5),
        ]

        send_event = Event(type="send", instruction=instructions[1])
        facts = make_facts(instructions, events=[send_event])

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_ldu32_also_supported(self):
        """LDU 32 (alternate flag loading) should also be detected."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=32),
            make_instruction_fact("AND", 1),
            make_instruction_fact("IFNOT", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        # find_bounced_load should find it
        idx = detector._find_bounced_load(instructions)
        assert idx == 0

    def test_long_throwifnot_guard_with_send_passes(self):
        """Longer compiler-generated guard sequences should be accepted."""
        detector = BouncedHandlerPatternDetector(min_complexity_threshold=14)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("SWAP", 1),
            make_instruction_fact("PUSHINT_4", 2, arg_value=1),
            make_instruction_fact("AND", 3),
            make_instruction_fact("NEGATE", 4),
            make_instruction_fact("SWAP", 5),
            make_instruction_fact("LDMSGADDR", 6),
            make_instruction_fact("SWAP", 7),
            make_instruction_fact("DUP", 8),
            make_instruction_fact("SBITS", 9),
            make_instruction_fact("PUSHINT_16", 10, arg_value=267),
            make_instruction_fact("EQUAL", 11),
            make_instruction_fact("THROWIFNOT", 12),
            make_instruction_fact("SENDRAWMSG", 13),
        ]
        send_event = Event(type="send", instruction=instructions[13])
        facts = make_facts(instructions, events=[send_event])

        findings = detector.detect(facts)
        assert findings == []



class TestBouncedDetectorHelpers:
    """Tests for helper methods."""

    def test_get_arg_value_with_value(self):
        """_get_arg_value should extract integer argument."""
        detector = BouncedMessageDetector()
        inst = make_instruction_fact("LDU", 0, arg_value=4)

        result = detector._get_arg_value(inst)
        assert result == 4

    def test_get_arg_value_without_args(self):
        """_get_arg_value should return None for instructions without args."""
        detector = BouncedMessageDetector()
        inst = make_instruction_fact("NOP", 0)

        result = detector._get_arg_value(inst)
        assert result is None

    def test_has_guard_after_with_immediate_guard(self):
        """_has_guard_after should find immediate guard."""
        detector = BouncedMessageDetector()
        instructions = [
            make_instruction_fact("AND", 0),
            make_instruction_fact("IF", 1),
            make_instruction_fact("NOP", 2),
        ]

        result = detector._has_guard_after(instructions, 0)
        assert result is True

    def test_has_guard_after_with_stack_manipulation(self):
        """_has_guard_after should allow stack manipulation before guard."""
        detector = BouncedMessageDetector()
        instructions = [
            make_instruction_fact("AND", 0),
            make_instruction_fact("DUP", 1),
            make_instruction_fact("SWAP", 2),
            make_instruction_fact("THROWIF", 3),
        ]

        result = detector._has_guard_after(instructions, 0)
        assert result is True

    def test_has_guard_after_fails_on_consuming_op(self):
        """_has_guard_after should fail if consuming op before guard."""
        detector = BouncedMessageDetector()
        instructions = [
            make_instruction_fact("AND", 0),
            make_instruction_fact("DROP", 1),  # Consuming without guard
            make_instruction_fact("IF", 2),
        ]

        result = detector._has_guard_after(instructions, 0)
        assert result is False

    def test_has_guard_after_respects_lookahead(self):
        """_has_guard_after should respect GUARD_SEARCH_LOOKAHEAD limit."""
        detector = BouncedMessageDetector()
        # Guard is beyond the lookahead window
        instructions = [make_instruction_fact("NOP", i) for i in range(10)]
        instructions.append(make_instruction_fact("IF", 10))

        result = detector._has_guard_after(instructions, 0)
        assert result is False


class TestFindBouncedGuard:
    """Tests for _find_bounced_guard in BouncedHandlerPatternDetector."""

    def test_finds_guard_after_bit_extraction(self):
        """Should find guard after bit extraction."""
        detector = BouncedHandlerPatternDetector()
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("AND", 1),
            make_instruction_fact("IFNOT", 2),
            make_instruction_fact("NOP", 3),
        ]

        result = detector._find_bounced_guard(instructions, 0)
        assert result == 2

    def test_finds_guard_immediately_after_load(self):
        """Guard immediately after flag load should be found (some compilers)."""
        detector = BouncedHandlerPatternDetector()
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("IF", 1),  # Immediate guard
            make_instruction_fact("NOP", 2),
        ]

        result = detector._find_bounced_guard(instructions, 0)
        assert result == 1

    def test_returns_none_without_guard(self):
        """Should return None if no guard found."""
        detector = BouncedHandlerPatternDetector()
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("ADD", 1),  # No bit extraction
            make_instruction_fact("NEWC", 2),  # Unrelated
        ]

        result = detector._find_bounced_guard(instructions, 0)
        assert result is None

    def test_stops_on_unrelated_operations(self):
        """Should stop searching at unrelated operations like LDREF."""
        detector = BouncedHandlerPatternDetector()
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("LDREF", 1),  # Breaks search
            make_instruction_fact("AND", 2),
            make_instruction_fact("IF", 3),
        ]

        result = detector._find_bounced_guard(instructions, 0)
        assert result is None


class TestEdgeCases:
    """Edge case tests."""

    def test_negative_mask_value(self):
        """32-bit mask might be negative in Python - should still work."""
        detector = BouncedMessageDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=32),
            make_instruction_fact("PUSHINT", 1, arg_value=-0x80000000),  # Negative
            make_instruction_fact("AND", 2),
            make_instruction_fact("IFNOT", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_pushint_tracking_reset(self):
        """PUSHINT tracking should reset after non-stack operations."""
        detector = BouncedMessageDetector(min_complexity_threshold=6)
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("PUSHINT", 1, arg_value=1),
            make_instruction_fact("NEWC", 2),  # Resets tracking
            make_instruction_fact("AND", 3),  # Should NOT recognize as pattern
            make_instruction_fact("SENDRAWMSG", 4),
            make_instruction_fact("NOP", 5),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # NEWC should have broken the detection chain
        assert len(findings) == 1

    def test_empty_instructions(self):
        """Empty instruction list should be handled gracefully."""
        detector = BouncedMessageDetector(min_complexity_threshold=0)
        facts = make_facts([])

        findings = detector.detect(facts)
        # Should return empty, not crash
        assert findings == []

    def test_experimental_flag(self):
        """BouncedHandlerPatternDetector should be marked experimental."""
        detector = BouncedHandlerPatternDetector()
        assert detector.experimental is True

    def test_default_severities(self):
        """Default severities should be correct."""
        assert BouncedMessageDetector().default_severity == "critical"
        assert BouncedHandlerPatternDetector().default_severity == "medium"


class TestExtensiblePatterns:
    """Tests for extensible bounced pattern matching."""

    def test_default_patterns_initialized(self):
        """Default patterns should be initialized from class constants."""
        detector = BouncedMessageDetector()
        assert detector.bounced_patterns == detector.DEFAULT_BOUNCED_PATTERNS
        assert detector.bounced_patterns_32bit == detector.DEFAULT_BOUNCED_PATTERNS_32BIT

    def test_additional_patterns_merged(self):
        """additional_patterns option should be merged with defaults."""
        custom_pattern = {
            "custom_test": {
                "description": "Custom test pattern",
                "opcodes": ["CUSTOM_OP"],
                "arg_value": 42,
            }
        }
        detector = BouncedMessageDetector(options={
            "additional_patterns": custom_pattern
        })

        # Default patterns should still exist
        assert "push_one_and" in detector.bounced_patterns
        assert "modpow2_1" in detector.bounced_patterns
        # Custom pattern should be added
        assert "custom_test" in detector.bounced_patterns
        assert detector.bounced_patterns["custom_test"]["arg_value"] == 42

    def test_additional_patterns_32bit_merged(self):
        """additional_patterns_32bit option should be merged with defaults."""
        custom_pattern = {
            "custom_32bit": {
                "description": "Custom 32-bit pattern",
                "opcodes": ["CUSTOM32_OP"],
                "arg_value": 31,
            }
        }
        detector = BouncedMessageDetector(options={
            "additional_patterns_32bit": custom_pattern
        })

        # Default 32-bit patterns should still exist
        assert "push_mask_and" in detector.bounced_patterns_32bit
        assert "rshift_31" in detector.bounced_patterns_32bit
        # Custom pattern should be added
        assert "custom_32bit" in detector.bounced_patterns_32bit

    def test_custom_pattern_detection(self):
        """Custom patterns should be used in detection."""
        # Add a custom pattern for XCHG opcode (just for testing)
        custom_pattern = {
            "xchg_test": {
                "description": "Test pattern with XCHG",
                "opcodes": ["XCHG"],
            }
        }
        detector = BouncedMessageDetector(
            options={"additional_patterns": custom_pattern},
            min_complexity_threshold=5
        )
        instructions = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("XCHG", 1),  # Custom pattern
            make_instruction_fact("IFNOT", 2),  # Guard
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # Should pass because custom XCHG pattern is matched with guard
        assert findings == []

    def test_pattern_override(self):
        """Custom patterns can override default patterns."""
        # Override modpow2_1 to require arg_value=2 instead of 1
        override_pattern = {
            "modpow2_1": {
                "description": "Override: MODPOW2 2 pattern",
                "opcodes": ["MODPOW2"],
                "arg_value": 2,  # Changed from 1 to 2
            }
        }
        detector = BouncedMessageDetector(
            options={"additional_patterns": override_pattern},
            min_complexity_threshold=5
        )

        # MODPOW2 1 should no longer match
        instructions1 = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("MODPOW2", 1, arg_value=1),  # Original value
            make_instruction_fact("IFNOT", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts1 = make_facts(instructions1)
        findings1 = detector.detect(facts1)
        # Should fail because modpow2_1 now requires arg_value=2
        assert len(findings1) == 1  # Vulnerability reported

        # MODPOW2 2 should now match
        instructions2 = [
            make_instruction_fact("LDU", 0, arg_value=4),
            make_instruction_fact("MODPOW2", 1, arg_value=2),  # New value
            make_instruction_fact("IFNOT", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts2 = make_facts(instructions2)
        findings2 = detector.detect(facts2)
        # Should pass with new pattern
        assert findings2 == []

    def test_class_constants_unchanged(self):
        """Class-level DEFAULT_BOUNCED_PATTERNS should not be modified."""
        # Create detector with custom patterns
        detector = BouncedMessageDetector(options={
            "additional_patterns": {"test": {"opcodes": ["TEST"]}}
        })

        # Instance patterns should have custom
        assert "test" in detector.bounced_patterns

        # But class constant should be unchanged
        assert "test" not in BouncedMessageDetector.DEFAULT_BOUNCED_PATTERNS

    def test_preserves_prev_push_for_pushint_specializations(self):
        """Immediate PUSHINT variants should preserve tracked constant values."""
        for opcode in [
            "PUSHINT",
            "PUSHINT_LONG",
            "PUSHINT_SHORT",
            "PUSHINT_4",
            "PUSHINT_16",
            "PUSHPOW2",
            "DUP",
            "OVER",
            "SWAP",
            "ROT",
            "DUP2",
            "OVER2",
        ]:
            assert BouncedMessageDetector._preserves_prev_push(opcode) is True

    def test_unrelated_opcodes_constant(self):
        """UNRELATED_OPCODES should contain expected opcodes."""
        expected = {"ADD", "MUL", "SUB", "NEWC", "ENDC", "STREF", "LDREF", "SENDRAWMSG"}
        assert BouncedMessageDetector.UNRELATED_OPCODES == expected
