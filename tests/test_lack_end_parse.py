"""
Tests for LackEndParseDetector.

Tests cover:
- Slice creation detection (CTOS, LDSLICE, etc.)
- Slice read detection (LDI, LDU, LDREF, etc.)
- ENDS validation detection
- Simple opcode-based fallback detection
- Trivial contract skipping
"""

from tasmscan.analyzer.facts import AnalysisFacts, InstructionFact
from tasmscan.detectors.lack_end_parse import LackEndParseDetector


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


def make_facts(instructions: list, metadata=None) -> AnalysisFacts:
    """Create AnalysisFacts with given instructions."""
    return AnalysisFacts(
        instructions=instructions,
        events=[],
        metadata=metadata or {},
    )


class TestLackEndParseDetector:
    """Tests for LackEndParseDetector."""

    def test_trivial_contract_skipped(self):
        """Trivial contracts (< min_complexity) should be skipped."""
        detector = LackEndParseDetector()
        # Less than default MIN_COMPLEXITY_THRESHOLD (10)
        instructions = [make_instruction_fact("NOP", i) for i in range(5)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_no_slice_operations_returns_empty(self):
        """Contract without slice operations should return empty."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("PUSHINT", 0, arg_value=42),
            make_instruction_fact("ADD", 1),
            make_instruction_fact("SUB", 2),
            make_instruction_fact("MUL", 3),
            make_instruction_fact("DIV", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_slice_create_with_ends_passes(self):
        """Slice creation followed by ENDS should pass."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),       # Slice creation
            make_instruction_fact("LDU", 1, arg_value=32),  # Slice read
            make_instruction_fact("ENDS", 2),       # Validation
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_slice_create_read_no_ends_reports(self):
        """Slice creation with reads but no ENDS should report vulnerability."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),       # Slice creation
            make_instruction_fact("LDU", 1, arg_value=32),  # Slice read
            make_instruction_fact("DROP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "end_parse" in findings[0].message.lower()
        assert findings[0].severity == "medium"
        assert findings[0].extra["slice_create_opcode"] == "CTOS"

    def test_ldslice_creates_slice_detection(self):
        """LDSLICE should be detected as slice creation."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDSLICE", 0, arg_value=100),  # Creates slice
            make_instruction_fact("LDI", 1, arg_value=8),        # Read
            make_instruction_fact("DROP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "LDSLICE" in findings[0].message

    def test_multiple_read_operations_counted(self):
        """Multiple slice read operations should be counted."""
        detector = LackEndParseDetector(min_complexity_threshold=8)
        instructions = [
            make_instruction_fact("CTOS", 0),
            make_instruction_fact("LDU", 1, arg_value=32),   # Read 1
            make_instruction_fact("LDU", 2, arg_value=64),   # Read 2
            make_instruction_fact("LDREF", 3),               # Read 3
            make_instruction_fact("LDDICT", 4),              # Read 4
            make_instruction_fact("DROP", 5),
            make_instruction_fact("NOP", 6),
            make_instruction_fact("NOP", 7),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].extra["read_count"] >= 4

    def test_sdepth_counts_as_validation(self):
        """SDEPTH should count as slice validation."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),
            make_instruction_fact("LDU", 1, arg_value=32),
            make_instruction_fact("SDEPTH", 2),  # Validation alternative
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_only_slice_create_no_read_passes(self):
        """Slice creation without read operations should pass (no finding)."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),
            make_instruction_fact("DROP", 1),  # Drop without reading
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # No reads, so no finding (the slice might be intentionally dropped)
        assert findings == []

    def test_pldslice_creates_slice(self):
        """PLDSLICE should be detected as slice creation."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("PLDSLICE", 0, arg_value=100),
            make_instruction_fact("PLDU", 1, arg_value=32),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "PLDSLICE" in findings[0].message

    def test_various_read_opcodes(self):
        """Various slice read opcodes should be detected."""
        detector = LackEndParseDetector(min_complexity_threshold=5)

        read_opcodes = ["LDI", "LDIX", "LDU", "LDUX", "LDREF", "LDDICT", "LDMSGADDR", "LDGRAMS"]

        for read_op in read_opcodes:
            instructions = [
                make_instruction_fact("CTOS", 0),
                make_instruction_fact(read_op, 1),
                make_instruction_fact("NOP", 2),
                make_instruction_fact("NOP", 3),
                make_instruction_fact("NOP", 4),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert len(findings) == 1, f"Failed for read opcode: {read_op}"

    def test_detector_metadata(self):
        """Detector should have correct metadata."""
        detector = LackEndParseDetector()
        assert detector.name == "lack_end_parse"
        assert detector.category == "security"
        assert detector.enabled_by_default is True
        assert detector.default_severity == "medium"
        assert "slice" in detector.tags

    def test_remediation_message(self):
        """Finding should include remediation message."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),
            make_instruction_fact("LDU", 1, arg_value=32),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].remediation is not None
        assert "end_parse" in findings[0].remediation.lower()

    def test_extra_metadata_in_finding(self):
        """Finding should include useful extra metadata."""
        detector = LackEndParseDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("CTOS", 0),
            make_instruction_fact("LDU", 1, arg_value=32),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "slice_create_opcode" in findings[0].extra
        assert "slice_create_index" in findings[0].extra
        assert "read_count" in findings[0].extra
        assert "detection_method" in findings[0].extra

    def test_empty_instructions(self):
        """Empty instruction list should be handled gracefully."""
        detector = LackEndParseDetector(min_complexity_threshold=0)
        facts = make_facts([])

        findings = detector.detect(facts)
        # Should return empty, not crash
        assert findings == []


class TestSliceOpcodeClassification:
    """Tests for opcode classification constants."""

    def test_slice_create_opcodes(self):
        """Verify slice creation opcodes are correctly classified."""
        detector = LackEndParseDetector()
        expected = {"CTOS", "LDSLICE", "LDSLICEX", "PLDSLICE", "PLDSLICEX"}
        assert detector.SLICE_CREATE_OPCODES == expected

    def test_ends_opcodes(self):
        """Verify ENDS opcodes are correctly classified."""
        detector = LackEndParseDetector()
        assert "ENDS" in detector.ENDS_OPCODES
        assert "SDEPTH" in detector.ENDS_OPCODES

    def test_common_read_opcodes_included(self):
        """Verify common read opcodes are included."""
        detector = LackEndParseDetector()
        common_reads = {"LDI", "LDU", "LDREF", "LDDICT", "LDMSGADDR", "LDGRAMS"}
        for op in common_reads:
            assert op in detector.SLICE_READ_OPCODES, f"{op} should be in SLICE_READ_OPCODES"


class TestRegistration:
    """Tests for detector registration."""

    def test_detector_in_registry(self):
        """Detector should be registered in the registry."""
        from tasmscan.detectors.registry import DETECTOR_CLASSES
        assert "lack_end_parse" in DETECTOR_CLASSES
        assert DETECTOR_CLASSES["lack_end_parse"] == LackEndParseDetector

    def test_detector_in_default_detectors(self):
        """Detector should be included in default detectors."""
        from tasmscan.detectors.registry import default_detectors
        detectors = default_detectors()
        detector_names = [d.name for d in detectors]
        assert "lack_end_parse" in detector_names
