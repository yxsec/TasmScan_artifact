from tasmscan.analyzer.facts import AnalysisFacts, InstructionFact
from tasmscan.detectors.improper_modifier import ImproperModifierDetector


class MockArg:
    def __init__(self, value):
        self.value = value


class MockInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class FakeTasirInstruction:
    def __init__(self, opcode: str, index: int, *, original_args=None, kind=None):
        self.opcode = opcode
        self.index = index
        self.original_args = original_args or []
        self.kind = kind


class FakeFunction:
    def __init__(self, name: str, instructions):
        self.name = name
        self._instructions = list(instructions)

    def all_instructions(self):
        return list(self._instructions)


class FakeModule:
    def __init__(self, functions, call_graph=None):
        self.functions = functions
        self.call_graph = call_graph or []


def make_instruction_fact(opcode: str, index: int, *args) -> InstructionFact:
    return InstructionFact(
        instruction=MockInstruction(opcode, [MockArg(arg) for arg in args]),
        index=index,
    )


def make_facts(instructions, module=None) -> AnalysisFacts:
    metadata = {"tasir_module": module} if module is not None else {}
    return AnalysisFacts(instructions=instructions, events=[], metadata=metadata)


def test_improper_modifier_skips_opcode_only_noise_without_tasir():
    detector = ImproperModifierDetector(min_complexity_threshold=0)
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("DROP", 1),
        ]
    )

    findings = detector.detect(facts)
    assert findings == []


def test_improper_modifier_ignores_non_effectful_parser_helpers():
    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            1: FakeFunction(
                "caller",
                [
                    FakeTasirInstruction("CALLDICT", 0, original_args=[2]),
                    FakeTasirInstruction("DROP", 1),
                ],
            ),
            2: FakeFunction(
                "parser_helper",
                [
                    FakeTasirInstruction("LDU", 10),
                    FakeTasirInstruction("LDMSGADDR", 11),
                ],
            ),
        },
        call_graph=[(1, 2)],
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("DROP", 1),
        ],
        module=module,
    )

    findings = detector.detect(facts)
    assert findings == []


def test_improper_modifier_skips_global_only_tasir_when_analysis_is_incomplete():
    from tasmscan.ir.tasir_types import InstructionKind

    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            1: FakeFunction(
                "caller",
                [
                    FakeTasirInstruction("CALLDICT", 0, original_args=[2]),
                    FakeTasirInstruction("DROP", 1),
                ],
            ),
            2: FakeFunction(
                "global_helper",
                [
                    FakeTasirInstruction("SETGLOB", 20, kind=InstructionKind.GLOBAL_STORE),
                ],
            ),
        },
        call_graph=[(1, 2)],
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("DROP", 1),
        ],
        module=module,
    )
    facts.metadata["analysis_incomplete"] = True

    findings = detector.detect(facts)
    assert findings == []


def test_improper_modifier_reports_transitive_broad_effects_when_result_dropped():
    from tasmscan.ir.tasir_types import InstructionKind

    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            1: FakeFunction(
                "caller",
                [
                    FakeTasirInstruction("CALLDICT", 0, original_args=[2]),
                    FakeTasirInstruction("DROP", 1),
                ],
            ),
            2: FakeFunction(
                "intermediate_helper",
                [
                    FakeTasirInstruction("CALLDICT", 20, original_args=[3]),
                ],
            ),
            3: FakeFunction(
                "storage_writer",
                [
                    FakeTasirInstruction("POPCTR", 30, original_args=[4]),
                    FakeTasirInstruction("THROWIFNOT", 31, kind=InstructionKind.THROW),
                ],
            ),
        },
        call_graph=[(1, 2), (2, 3)],
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("DROP", 1),
        ],
        module=module,
    )

    findings = detector.detect(facts)
    assert len(findings) == 1
    assert findings[0].extra["callee"] == "intermediate_helper"
    assert set(findings[0].extra["effects"]) == {"storage", "throw"}


def test_improper_modifier_skips_noisy_raw_calldict_blkdrop2_shape():
    from tasmscan.ir.tasir_types import InstructionKind

    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            2: FakeFunction(
                "effectful_helper",
                [
                    FakeTasirInstruction("SENDRAWMSG", 21, kind=InstructionKind.SEND_MESSAGE),
                    FakeTasirInstruction("THROWIFNOT", 22, kind=InstructionKind.THROW),
                ],
            ),
        },
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("BLKDROP2", 1),
        ],
        module=module,
    )

    findings = detector.detect(facts)
    assert findings == []


def test_improper_modifier_keeps_distance_three_raw_calldict_drop_shape():
    from tasmscan.ir.tasir_types import InstructionKind

    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            2: FakeFunction(
                "effectful_helper",
                [
                    FakeTasirInstruction("SENDRAWMSG", 21, kind=InstructionKind.SEND_MESSAGE),
                    FakeTasirInstruction("THROWIFNOT", 22, kind=InstructionKind.THROW),
                ],
            ),
        },
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("SWAP", 1),
            make_instruction_fact("LDU", 2, 160),
            make_instruction_fact("DROP", 3),
        ],
        module=module,
    )

    findings = detector.detect(facts)
    assert len(findings) == 1
    assert findings[0].extra["detection_method"] == "raw_call_drop"
    assert findings[0].extra["drop_index"] == 3


def test_improper_modifier_falls_back_to_raw_calldict_drop_when_tasir_hides_callsite():
    from tasmscan.ir.tasir_types import InstructionKind

    detector = ImproperModifierDetector(min_complexity_threshold=0)
    module = FakeModule(
        functions={
            2: FakeFunction(
                "effectful_helper",
                [
                    FakeTasirInstruction("SENDRAWMSG", 21, kind=InstructionKind.SEND_MESSAGE),
                    FakeTasirInstruction("THROWIFNOT", 22, kind=InstructionKind.THROW),
                ],
            ),
        },
    )
    facts = make_facts(
        [
            make_instruction_fact("CALLDICT", 0, 2),
            make_instruction_fact("DROP", 1),
        ],
        module=module,
    )

    findings = detector.detect(facts)
    assert len(findings) == 1
    assert findings[0].extra["callee"] == "effectful_helper"
    assert findings[0].extra["detection_method"] == "raw_call_drop"
    assert set(findings[0].extra["effects"]) == {"send", "throw"}
