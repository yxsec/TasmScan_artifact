from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


def test_ldmsgaddrq_is_collected_as_sender_read_event():
    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(
        [
            MockInstruction("LDMSGADDRQ"),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    sender_indices = {event.instruction.index for event in facts.events_of("sender_read")}
    assert 0 in sender_indices
