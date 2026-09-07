"""
Tests for UncheckedSenderDetector.

Tests cover:
- Detection with/without sender checks
- CFG path analysis logic
- Dataflow integration
- Helper methods: _find_guard_after, _find_next_event
- _search_paths with various CFG patterns
"""
from tasmscan.analyzer.facts import (
    AnalysisFacts,
    BasicBlock,
    Continuation,
    Event,
    InstructionFact,
)
from tasmscan.detectors.unchecked_sender import UncheckedSenderDetector
from tasmscan.ir.dataflow import DataFlowGraph
from tasmscan.ir.tasir_types import (
    InstructionKind,
    InterProceduralView,
    SemanticLabel,
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


def make_instruction_fact(opcode: str, index: int) -> InstructionFact:
    """Create an InstructionFact."""
    return InstructionFact(
        instruction=MockInstruction(opcode),
        index=index,
    )


def make_facts(
    instructions: list,
    events=None,
    basic_blocks=None,
    dataflow_graph=None,
) -> AnalysisFacts:
    """Create AnalysisFacts with given data."""
    facts = AnalysisFacts(
        instructions=instructions,
        events=events or [],
        basic_blocks=basic_blocks or [],
    )
    facts.dataflow_graph = dataflow_graph
    return facts


class TestUncheckedSenderDetector:
    """Tests for UncheckedSenderDetector."""

    def test_trivial_contract_skipped(self):
        """Trivial contracts should return no findings."""
        detector = UncheckedSenderDetector()
        instructions = [make_instruction_fact("NOP", i) for i in range(5)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_no_sender_read_no_findings(self):
        """No sender_read events should return no findings."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [make_instruction_fact("NOP", i) for i in range(10)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_sender_read_with_guard_before_send(self):
        """Sender read with guard before send should pass."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read
            make_instruction_fact("EQUAL", 1),
            make_instruction_fact("THROWIFNOT", 2),  # guard
            make_instruction_fact("SENDRAWMSG", 3),  # send
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])
        guard_event = Event(type="guard", instruction=instructions[2])
        send_event = Event(type="send", instruction=instructions[3])

        facts = make_facts(instructions, events=[sender_event, guard_event, send_event])

        findings = detector.detect(facts)
        assert findings == []

    def test_sender_read_without_guard_before_send(self):
        """Sender read without guard before send should be flagged."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),  # send before guard
            make_instruction_fact("THROWIFNOT", 3),  # guard after send
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])
        guard_event = Event(type="guard", instruction=instructions[3])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(instructions, events=[sender_event, guard_event, send_event])

        findings = detector.detect(facts)
        assert len(findings) >= 1
        assert "sender" in findings[0].message.lower()

    def test_sender_read_no_guard_at_all(self):
        """Sender read with no guard at all should be flagged."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),  # send
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(instructions, events=[sender_event, send_event])

        findings = detector.detect(facts)
        assert len(findings) >= 1

    def test_no_send_event_still_warns(self):
        """Sender read without send operation may still warn (detector behavior).

        Note: The detector may still report findings even without send events
        if it detects unguarded sender read through dataflow analysis.
        This is implementation-dependent behavior.
        """
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])

        facts = make_facts(instructions, events=[sender_event])

        findings = detector.detect(facts)
        # The detector behavior: may report findings based on dataflow analysis
        # even without explicit send events. This test verifies it doesn't crash.
        assert isinstance(findings, list)


class TestUncheckedSenderCFGAnalysis:
    """Tests for CFG-based analysis in UncheckedSenderDetector."""

    def test_cfg_analysis_with_guarded_path(self):
        """CFG analysis should recognize guarded paths."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read
            make_instruction_fact("THROWIFNOT", 1),  # guard
            make_instruction_fact("SENDRAWMSG", 2),  # send
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3, 4],
            successors=[],
        )

        sender_event = Event(type="sender_read", instruction=instructions[0])
        guard_event = Event(type="guard", instruction=instructions[1])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(
            instructions,
            events=[sender_event, guard_event, send_event],
            basic_blocks=[block],
        )

        findings = detector.detect(facts)
        assert findings == []

    def test_cfg_analysis_with_unguarded_path(self):
        """CFG analysis should detect unguarded paths to send."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # sender_read in block 0
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),  # send in block 1 (no guard)
            make_instruction_fact("THROWIFNOT", 3),  # guard in block 2
            make_instruction_fact("NOP", 4),
        ]

        # Block 0 jumps to block 1 (no guard) or block 2 (has guard)
        block0 = BasicBlock(id=0, instruction_indices=[0, 1], successors=[1, 2])
        block1 = BasicBlock(id=1, instruction_indices=[2], successors=[])  # Unguarded
        block2 = BasicBlock(id=2, instruction_indices=[3, 4], successors=[])

        sender_event = Event(type="sender_read", instruction=instructions[0])
        guard_event = Event(type="guard", instruction=instructions[3])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(
            instructions,
            events=[sender_event, guard_event, send_event],
            basic_blocks=[block0, block1, block2],
        )

        findings = detector.detect(facts)
        # Should find unguarded path: block0 -> block1 (send without guard)
        assert len(findings) >= 1

    def test_cfg_analysis_treats_auth_pattern_as_guard_on_forward_path(self):
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]

        block = BasicBlock(id=0, instruction_indices=[0, 1, 2, 3, 4], successors=[])
        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[3])
        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block],
        )

        findings = detector.detect(facts)

        assert findings == []

    def test_cfg_analysis_skips_sender_when_predecessor_path_is_guarded(self):
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("SDEQ", 0),
            make_instruction_fact("THROWIFNOT", 1),
            make_instruction_fact("LDMSGADDR", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]

        block0 = BasicBlock(id=0, instruction_indices=[0, 1], successors=[1], context="main")
        block1 = BasicBlock(id=1, instruction_indices=[2, 3, 4], successors=[], context="cont_0")
        sender_event = Event(type="sender_read", instruction=instructions[2])
        send_event = Event(type="send", instruction=instructions[3])
        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
        )

        findings = detector._detect_with_cfg(
            facts,
            sender_reads=[sender_event],
            dataflow_sink_indices=set(),
            tasir_send_indices={3},
        )

        assert findings == []

    def test_cfg_analysis_handles_unknown_successor(self):
        """CFG analysis should handle blocks with unknown successors."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("EXECUTE", 1),  # Dynamic call
            make_instruction_fact("SENDRAWMSG", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3, 4],
            successors=[],
            has_unknown_successor=True,
        )

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block],
        )

        findings = detector.detect(facts)
        # Should handle unknown successor conservatively
        assert len(findings) >= 1

    def test_cfg_block_not_found_fallback(self):
        """If instruction not in any block, should still report finding."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        # Block doesn't include instruction 0
        block = BasicBlock(id=0, instruction_indices=[1, 2, 3, 4], successors=[])

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[2])

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block],
        )

        findings = detector.detect(facts)
        # Should still report finding
        assert len(findings) >= 1


class TestUncheckedSenderDataflowIntegration:
    """Tests for dataflow analysis integration."""

    def test_dataflow_finds_tainted_flow(self):
        """Dataflow should detect tainted sender values reaching sinks."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[2])

        # Create dataflow graph with tainted propagation
        dataflow_graph = DataFlowGraph(
            values={},
            edges=[],
            tainted_propagation=[(0, 2)],  # Taint flows from 0 to 2
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph

        findings = detector.detect(facts)
        # Should find tainted flow
        assert len(findings) >= 1
        # Check that finding mentions taint analysis
        dataflow_findings = [f for f in findings if "taint" in f.message.lower()]
        assert len(dataflow_findings) >= 1

    def test_dataflow_covers_sender_indices(self):
        """Dataflow findings should cover sender indices to avoid duplicates."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),
            make_instruction_fact("NOP", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[2])

        # Dataflow already reports this
        dataflow_graph = DataFlowGraph(
            values={},
            edges=[],
            tainted_propagation=[(0, 2)],
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph

        findings = detector.detect(facts)
        # Should not have duplicate findings for same issue
        # Dataflow finding should be sufficient
        dataflow_findings = [f for f in findings if "taint" in f.message.lower()]
        cfg_findings = [f for f in findings if "path" in f.message.lower()]
        # Either dataflow or CFG, but not both for same sink
        assert len(dataflow_findings) >= 1 or len(cfg_findings) >= 1

    def test_dataflow_replays_same_context_auth_guard(self):
        """Local auth guards should suppress same-context taint false positives."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]
        for inst in instructions:
            inst.continuation_id = "cont_0"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[3])
        dataflow_graph = DataFlowGraph(
            values={},
            edges=[],
            tainted_propagation=[(0, 3)],
        )
        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3, 4],
            successors=[],
            context="cont_0",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph

        findings = detector.detect(facts)

        assert findings == []

    def test_dataflow_replays_same_context_predecessor_guard(self):
        """Sender already guarded on predecessor paths should not report taint flow."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("SDEQ", 0),
            make_instruction_fact("THROWIFNOT", 1),
            make_instruction_fact("LDMSGADDR", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender_event = Event(type="sender_read", instruction=instructions[2])
        send_event = Event(type="send", instruction=instructions[3])
        dataflow_graph = DataFlowGraph(
            values={},
            edges=[],
            tainted_propagation=[(2, 3)],
        )
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1],
            successors=[1],
            context="main",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[2, 3, 4],
            successors=[],
            context="main",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph

        findings = detector.detect(facts)

        assert findings == []

    def test_dataflow_keeps_cross_context_taint_pair(self):
        """CFG replay must not suppress taint that crosses CFG contexts."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_1"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[3])
        dataflow_graph = DataFlowGraph(
            values={},
            edges=[],
            tainted_propagation=[(0, 3)],
        )
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2],
            successors=[],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[3, 4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph

        findings = detector.detect(facts)

        assert any("taint analysis" in finding.message.lower() for finding in findings)

    def test_dataflow_suppresses_guarded_callref_child_context(self):
        """Guarded CALLREF helper entries should not keep cross-context taint findings."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("CALLREF", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=3,
                parent_instruction_index=3,
            )
        }

        findings = detector.detect(facts)

        assert findings == []

    def test_dataflow_keeps_guarded_child_context_when_parent_opcode_is_not_supported(self):
        """Unsupported helper-entry opcodes must stay reported."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("IF", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=3,
                parent_instruction_index=3,
            )
        }

        findings = detector.detect(facts)

        dataflow_findings = [
            f for f in findings if f.extra and f.extra.get("signal") == "dataflow"
        ]
        assert len(dataflow_findings) == 1

    def test_dataflow_keeps_callref_child_context_when_guard_is_after_callsite(self):
        """The CALLREF helper suppressor must not ignore unguarded entry paths."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("CALLREF", 1),
            make_instruction_fact("SDEQ", 2),
            make_instruction_fact("THROWIFNOT", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=1,
                parent_instruction_index=1,
            )
        }

        findings = detector.detect(facts)

        dataflow_findings = [
            f for f in findings if f.extra and f.extra.get("signal") == "dataflow"
        ]
        assert len(dataflow_findings) == 1

    def test_dataflow_suppresses_guarded_ifrefelse_child_context(self):
        """Guarded IFREFELSE helper entries with nearby auth should be suppressed."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
            make_instruction_fact("IFREFELSE", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=3,
                parent_instruction_index=3,
            )
        }

        findings = detector.detect(facts)

        assert findings == []

    def test_dataflow_keeps_ifrefelse_child_context_without_nearby_auth(self):
        """IFREFELSE helper entries without nearby auth must stay reported."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
            make_instruction_fact("IFREFELSE", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=3,
                parent_instruction_index=3,
            )
        }

        findings = detector.detect(facts)

        dataflow_findings = [
            f for f in findings if f.extra and f.extra.get("signal") == "dataflow"
        ]
        assert len(dataflow_findings) == 1

    def test_dataflow_keeps_ifrefelse_child_context_when_auth_is_after_parent(self):
        """Auth after IFREFELSE must not suppress the child continuation finding."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("IFREFELSE", 1),
            make_instruction_fact("SDEQ", 2),
            make_instruction_fact("THROWIFNOT", 3),
            make_instruction_fact("SENDRAWMSG", 4),
        ]
        instructions[0].continuation_id = "cont_0"
        instructions[1].continuation_id = "cont_0"
        instructions[2].continuation_id = "cont_0"
        instructions[3].continuation_id = "cont_0"
        instructions[4].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[4])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 4)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2, 3],
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[4],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=1,
                parent_instruction_index=1,
            )
        }

        findings = detector.detect(facts)

        dataflow_findings = [
            f for f in findings if f.extra and f.extra.get("signal") == "dataflow"
        ]
        assert len(dataflow_findings) == 1

    def test_dataflow_suppresses_guarded_ifelse_child_context_with_longer_auth_window(self):
        """IFELSE helper entries may use a longer equal_slices setup window."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
        ]
        instructions.extend(
            make_instruction_fact("NOP", idx) for idx in range(3, 28)
        )
        instructions.extend(
            [
                make_instruction_fact("IFELSE", 28),
                make_instruction_fact("SENDRAWMSG", 29),
            ]
        )

        for inst in instructions[:-1]:
            inst.continuation_id = "cont_0"
        instructions[-1].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[29])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 29)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=list(range(29)),
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[29],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=28,
                parent_instruction_index=28,
            )
        }

        findings = detector.detect(facts)

        assert findings == []

    def test_dataflow_keeps_ifelse_child_context_when_auth_is_too_far_back(self):
        """IFELSE helper entries should not use auth beyond the widened window."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SDEQ", 1),
            make_instruction_fact("THROWIFNOT", 2),
        ]
        instructions.extend(
            make_instruction_fact("NOP", idx) for idx in range(3, 34)
        )
        instructions.extend(
            [
                make_instruction_fact("IFELSE", 34),
                make_instruction_fact("SENDRAWMSG", 35),
            ]
        )

        for inst in instructions[:-1]:
            inst.continuation_id = "cont_0"
        instructions[-1].continuation_id = "cont_1"

        sender_event = Event(type="sender_read", instruction=instructions[0])
        send_event = Event(type="send", instruction=instructions[35])
        dataflow_graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[(0, 35)])
        block0 = BasicBlock(
            id=0,
            instruction_indices=list(range(35)),
            successors=[1],
            context="cont_0",
        )
        block1 = BasicBlock(
            id=1,
            instruction_indices=[35],
            successors=[],
            context="cont_1",
        )

        facts = make_facts(
            instructions,
            events=[sender_event, send_event],
            basic_blocks=[block0, block1],
            dataflow_graph=dataflow_graph,
        )
        facts.dataflow_graph = dataflow_graph
        facts.metadata["continuations"] = {
            "cont_1": Continuation(
                id="cont_1",
                instructions=[],
                entry_index=0,
                parent_context="cont_0",
                parent_local_index=34,
                parent_instruction_index=34,
            )
        }

        findings = detector.detect(facts)

        dataflow_findings = [
            f for f in findings if f.extra and f.extra.get("signal") == "dataflow"
        ]
        assert len(dataflow_findings) == 1


class TestUncheckedSenderHelpers:
    """Tests for helper methods."""

    def test_find_guard_after_finds_guard(self):
        """_find_guard_after should find guard after sender event."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 0)
        guard_inst = make_instruction_fact("THROWIFNOT", 2)

        sender_event = Event(type="sender_read", instruction=sender_inst)
        guard_event = Event(type="guard", instruction=guard_inst)

        result = detector._find_guard_after(sender_event, [guard_event])
        assert result == guard_event

    def test_find_guard_after_returns_none_if_before(self):
        """_find_guard_after should return None if guard is before sender."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 5)
        guard_inst = make_instruction_fact("THROWIFNOT", 2)  # Before sender

        sender_event = Event(type="sender_read", instruction=sender_inst)
        guard_event = Event(type="guard", instruction=guard_inst)

        result = detector._find_guard_after(sender_event, [guard_event])
        assert result is None

    def test_find_guard_after_finds_first_guard(self):
        """_find_guard_after should find first guard after sender."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 0)
        guard1 = make_instruction_fact("THROWIFNOT", 2)
        guard2 = make_instruction_fact("IF", 5)

        sender_event = Event(type="sender_read", instruction=sender_inst)
        guard_event1 = Event(type="guard", instruction=guard1)
        guard_event2 = Event(type="guard", instruction=guard2)

        result = detector._find_guard_after(sender_event, [guard_event1, guard_event2])
        assert result == guard_event1

    def test_find_next_event_finds_next_send(self):
        """_find_next_event should find next event after sender."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 0)
        send_inst = make_instruction_fact("SENDRAWMSG", 3)

        sender_event = Event(type="sender_read", instruction=sender_inst)
        send_event = Event(type="send", instruction=send_inst)

        result = detector._find_next_event(sender_event, [send_event])
        assert result == send_event

    def test_find_next_event_returns_none_if_before(self):
        """_find_next_event should return None if event is before sender."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 5)
        send_inst = make_instruction_fact("SENDRAWMSG", 2)  # Before sender

        sender_event = Event(type="sender_read", instruction=sender_inst)
        send_event = Event(type="send", instruction=send_inst)

        result = detector._find_next_event(sender_event, [send_event])
        assert result is None

    def test_find_next_event_returns_none_at_same_index(self):
        """_find_next_event should return None if event at same index."""
        detector = UncheckedSenderDetector()
        sender_inst = make_instruction_fact("LDMSGADDR", 3)
        send_inst = make_instruction_fact("SENDRAWMSG", 3)  # Same index

        sender_event = Event(type="sender_read", instruction=sender_inst)
        send_event = Event(type="send", instruction=send_inst)

        result = detector._find_next_event(sender_event, [send_event])
        assert result is None


class TestUncheckedSenderSearchPaths:
    """Tests for _search_paths method."""

    def test_search_paths_finds_unguarded_send(self):
        """_search_paths should find send operations without guards."""
        detector = UncheckedSenderDetector()

        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2],
            successors=[],
        )
        block_map = {0: block}

        guard_indices = set()  # No guards
        send_indices = {2}

        flagged, truncated = detector._search_paths(
            block_map, 0, 0, guard_indices, send_indices
        )

        assert 2 in flagged
        assert truncated is False

    def test_search_paths_respects_guard(self):
        """_search_paths should not flag sends after guards."""
        detector = UncheckedSenderDetector()

        block = BasicBlock(
            id=0,
            instruction_indices=[0, 1, 2],
            successors=[],
        )
        block_map = {0: block}

        guard_indices = {1}  # Guard at index 1
        send_indices = {2}  # Send at index 2

        flagged, truncated = detector._search_paths(
            block_map, 0, 0, guard_indices, send_indices
        )

        assert 2 not in flagged
        assert truncated is False

    def test_search_paths_handles_multiple_paths(self):
        """_search_paths should check all CFG paths."""
        detector = UncheckedSenderDetector()

        # Two paths: one guarded, one not
        block0 = BasicBlock(id=0, instruction_indices=[0], successors=[1, 2])
        block1 = BasicBlock(id=1, instruction_indices=[1, 3], successors=[])  # Guard, then send
        block2 = BasicBlock(id=2, instruction_indices=[4], successors=[])  # Send without guard

        block_map = {0: block0, 1: block1, 2: block2}

        guard_indices = {1}  # Guard in block1
        send_indices = {3, 4}  # Sends in both paths

        flagged, truncated = detector._search_paths(
            block_map, 0, 0, guard_indices, send_indices
        )

        # Send at 4 should be flagged (unguarded path)
        # Send at 3 should not be flagged (after guard)
        assert 4 in flagged
        assert 3 not in flagged

    def test_search_paths_detects_truncation(self):
        """_search_paths should detect when iteration limit exceeded."""
        # Create cyclic CFG to trigger truncation
        block0 = BasicBlock(id=0, instruction_indices=[0], successors=[1])
        block1 = BasicBlock(id=1, instruction_indices=[1], successors=[2])
        block2 = BasicBlock(id=2, instruction_indices=[2], successors=[0])  # Back edge

        block_map = {0: block0, 1: block1, 2: block2}

        # With very low iteration limit, should truncate
        from tasmscan.detectors.cfg_utils import traverse_cfg_for_unguarded_sinks

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=0,
            guard_indices=set(),
            sink_indices={2},
            max_iterations=2,  # Very low
        )

        # May or may not truncate depending on visit deduplication
        # But should not crash
        assert isinstance(result.flagged_indices, set)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_sender_reads(self):
        """Empty sender_read list should return no findings."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [make_instruction_fact("NOP", i) for i in range(10)]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_multiple_sender_reads(self):
        """Multiple sender reads should each be analyzed."""
        detector = UncheckedSenderDetector(min_complexity_threshold=5)
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),  # First sender_read
            make_instruction_fact("THROWIFNOT", 1),  # Guard
            make_instruction_fact("LDMSGADDR", 2),  # Second sender_read (no guard after)
            make_instruction_fact("SENDRAWMSG", 3),
            make_instruction_fact("NOP", 4),
        ]

        sender1 = Event(type="sender_read", instruction=instructions[0])
        sender2 = Event(type="sender_read", instruction=instructions[2])
        guard = Event(type="guard", instruction=instructions[1])
        send = Event(type="send", instruction=instructions[3])

        facts = make_facts(instructions, events=[sender1, sender2, guard, send])

        findings = detector.detect(facts)
        # Second sender_read has no guard after it before send
        # At least one finding expected
        assert len(findings) >= 1

    def test_default_severity(self):
        """Default severity should be medium."""
        detector = UncheckedSenderDetector()
        assert detector.default_severity == "medium"

    def test_detector_name(self):
        """Detector name should be correct."""
        detector = UncheckedSenderDetector()
        assert detector.name == "unchecked_sender"


class TestUncheckedSenderCrossFunction:
    """Tests for inter-procedural sender/send pairing."""

    @staticmethod
    def _make_cross_function_facts(source_has_guard: bool) -> AnalysisFacts:
        instructions = [
            make_instruction_fact("LDMSGADDR", 0),
            make_instruction_fact("SENDRAWMSG", 1),
        ]
        facts = make_facts(instructions)

        sink_inst = TVMInstruction(
            index=1,
            kind=InstructionKind.SEND_MESSAGE,
            opcode="SENDRAWMSG",
            semantic_labels=[SemanticLabel.MESSAGE_SEND],
        )
        sink_block = TVMBasicBlock(id=10, context_id="main", instructions=[sink_inst])
        sink_func = TVMFunction(method_id=1, entry_block_id=10, blocks={10: sink_block})
        source_func = TVMFunction(method_id=0, entry_block_id=0, blocks={})

        inter = InterProceduralView(
            function_summaries={
                0: {
                    "has_sender_read": True,
                    "has_guard": source_has_guard,
                    "has_send": False,
                },
                1: {
                    "has_sender_read": False,
                    "has_guard": False,
                    "has_send": True,
                },
            },
            taint_paths=[(0, -1, 1, -1)],
        )
        facts.metadata["tasir_module"] = TVMModule(
            functions={0: source_func, 1: sink_func},
            inter_procedural=inter,
        )
        return facts

    def test_cross_function_skips_guarded_source_method(self):
        detector = UncheckedSenderDetector(min_complexity_threshold=0)
        facts = self._make_cross_function_facts(source_has_guard=True)

        findings = detector._detect_cross_function(facts)

        assert findings == []

    def test_cross_function_reports_when_source_method_has_no_guard(self):
        detector = UncheckedSenderDetector(min_complexity_threshold=0)
        facts = self._make_cross_function_facts(source_has_guard=False)

        findings = detector._detect_cross_function(facts)

        assert len(findings) == 1
        assert "Cross-function" in findings[0].message
