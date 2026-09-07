from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.incompatible_modes import IncompatibleMessageModesDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockArg:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


def _build_facts(instructions):
    return ProgramAnalyzer().analyze(instructions, cell=None)


def _issues(findings):
    return [f.extra.get("issue") for f in findings if isinstance(f.extra, dict)]


def test_incompatible_modes_cfg_fallback_resolves_pushint_from_mock_args():
    facts = _build_facts(
        [
            MockInstruction("PUSHINT", [MockArg("uint", 64)]),
            MockInstruction("SENDRAWMSG"),
            MockInstruction("PUSHINT", [MockArg("uint", 64)]),
            MockInstruction("SENDRAWMSG"),
        ]
    )

    findings = IncompatibleMessageModesDetector().detect(facts)
    assert "double_send_remaining_value" in _issues(findings)


def test_incompatible_modes_tasir_decodes_pushpow2_constant_mode():
    facts = _build_facts(
        [
            MockInstruction("PUSHPOW2", [MockArg("uint", 6)]),  # 2^6 = 64
            MockInstruction("SENDRAWMSG"),
            MockInstruction("PUSHPOW2", [MockArg("uint", 6)]),  # 2^6 = 64
            MockInstruction("SENDRAWMSG"),
        ]
    )
    facts.metadata["tasir_module"] = IRBuilder().build_tasir(facts)

    findings = IncompatibleMessageModesDetector().detect(facts)
    assert "double_send_remaining_value" in _issues(findings)
