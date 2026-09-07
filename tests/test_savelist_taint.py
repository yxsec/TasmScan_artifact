"""
Tests for Cross-continuation taint propagation: cross-continuation taint propagation via SaveLists.

Verifies that _propagate_savelist_taint correctly adds taint propagation edges
when a SaveList contains registers whose definition sites are tainted in the
DataFlowGraph, and correctly skips untainted or empty SaveLists.
"""
from tasmscan.ir.dataflow.analyzer import DataFlowAnalyzer
from tasmscan.ir.dataflow.types import (
    DataFlowGraph,
    DataFlowValue,
    ValueSource,
)
from tasmscan.ir.tasir_types import (
    SaveList,
    TVMAbstractValue,
)


def _make_graph_with_tainted_site(definition_site: int) -> DataFlowGraph:
    """Create a DataFlowGraph where `definition_site` has a tainted value."""
    return DataFlowGraph(
        values={
            definition_site: [
                DataFlowValue(
                    source=ValueSource.MESSAGE_BODY,
                    definition_site=definition_site,
                    tainted=True,
                )
            ]
        },
        edges=[],
        tainted_propagation=[],
    )


def _make_graph_with_untainted_site(definition_site: int) -> DataFlowGraph:
    """Create a DataFlowGraph where `definition_site` has an untainted value."""
    return DataFlowGraph(
        values={
            definition_site: [
                DataFlowValue(
                    source=ValueSource.CONSTANT,
                    definition_site=definition_site,
                    tainted=False,
                )
            ]
        },
        edges=[],
        tainted_propagation=[],
    )


def _make_savelist(registers: dict[int, TVMAbstractValue]) -> SaveList:
    """Create a SaveList with the given register mappings."""
    return SaveList(registers=registers)


class TestSavelistTaintPropagationBasic:
    """Cross-continuation taint propagation: tainted definition_site in graph -> edge added."""

    def test_savelist_taint_propagation_basic(self):
        """When a SaveList register's definition_site IS tainted in the graph,
        _propagate_savelist_taint must add a taint propagation edge."""
        definition_site = 5
        call_site = 20

        graph = _make_graph_with_tainted_site(definition_site)

        savelist = _make_savelist({
            0: TVMAbstractValue(
                definition_site=definition_site,
                tainted=False,  # stale linking-phase value; graph lookup should win
                source="message_body",
            ),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added > 0, "Expected at least one taint propagation edge"
        assert (definition_site, call_site) in graph.tainted_propagation
        # Verify a DataFlowValue was recorded at the call site
        assert call_site in graph.values
        assert any(
            v.tainted and v.metadata.get("savelist_source") is True
            for v in graph.values[call_site]
        )

    def test_savelist_taint_propagation_multiple_registers(self):
        """Multiple tainted registers each produce an edge."""
        call_site = 30
        graph = DataFlowGraph(
            values={
                2: [DataFlowValue(source=ValueSource.MESSAGE_SENDER, definition_site=2, tainted=True)],
                7: [DataFlowValue(source=ValueSource.MESSAGE_VALUE, definition_site=7, tainted=True)],
            },
            edges=[],
            tainted_propagation=[],
        )

        savelist = _make_savelist({
            0: TVMAbstractValue(definition_site=2, source="msg_sender"),
            1: TVMAbstractValue(definition_site=7, source="msg_value"),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 2
        assert (2, call_site) in graph.tainted_propagation
        assert (7, call_site) in graph.tainted_propagation

    def test_savelist_taint_via_abstract_value_flag(self):
        """Even if the graph has no entry for the definition_site, a tainted
        TVMAbstractValue flag should still trigger edge creation."""
        call_site = 40
        definition_site = 99

        # Graph has NO entry for definition_site 99
        graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[])

        savelist = _make_savelist({
            0: TVMAbstractValue(
                definition_site=definition_site,
                tainted=True,  # This flag alone should suffice
                source="external",
            ),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 1
        assert (definition_site, call_site) in graph.tainted_propagation


class TestSavelistTaintPropagationUntainted:
    """SL-C-01 negative case: untainted definition_site -> no edge."""

    def test_savelist_taint_propagation_untainted(self):
        """When a SaveList register's definition_site is NOT tainted,
        _propagate_savelist_taint must NOT add any edge."""
        definition_site = 10
        call_site = 25

        graph = _make_graph_with_untainted_site(definition_site)

        savelist = _make_savelist({
            0: TVMAbstractValue(
                definition_site=definition_site,
                tainted=False,
                source="constant",
            ),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 0
        assert len(graph.tainted_propagation) == 0
        assert call_site not in graph.values

    def test_savelist_taint_propagation_definition_site_absent_from_graph(self):
        """When the definition_site is not in the graph AND the abstract value
        is not tainted, no edge should be added."""
        call_site = 50
        # Graph is completely empty
        graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[])

        savelist = _make_savelist({
            0: TVMAbstractValue(
                definition_site=42,
                tainted=False,
                source="unknown",
            ),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 0
        assert len(graph.tainted_propagation) == 0

    def test_savelist_no_definition_site(self):
        """When a SaveList register has definition_site=None,
        no edge can be formed."""
        call_site = 60
        graph = _make_graph_with_tainted_site(5)

        savelist = _make_savelist({
            0: TVMAbstractValue(
                definition_site=None,
                tainted=True,
                source="orphan",
            ),
        })

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 0


class TestSavelistTaintPropagationEmptySavelist:
    """SL-C-01 edge case: empty SaveList -> no edges."""

    def test_savelist_taint_propagation_empty_savelist(self):
        """An empty SaveList must produce zero edges."""
        call_site = 15
        graph = _make_graph_with_tainted_site(5)

        savelist = _make_savelist({})

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 0
        assert len(graph.tainted_propagation) == 0

    def test_savelist_default_construction(self):
        """A default-constructed SaveList (no args) is empty and safe."""
        call_site = 70
        graph = _make_graph_with_tainted_site(5)

        savelist = SaveList()

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert edges_added == 0


class TestSavelistTaintIdempotency:
    """Duplicate edges are not re-added."""

    def test_duplicate_edge_not_added(self):
        """Calling _propagate_savelist_taint twice with the same SaveList
        should not duplicate the taint propagation edge."""
        definition_site = 5
        call_site = 20
        graph = _make_graph_with_tainted_site(definition_site)

        savelist = _make_savelist({
            0: TVMAbstractValue(definition_site=definition_site, tainted=False),
        })

        analyzer = DataFlowAnalyzer()
        first = analyzer._propagate_savelist_taint(graph, call_site, savelist)
        second = analyzer._propagate_savelist_taint(graph, call_site, savelist)

        assert first == 1
        assert second == 0
        assert graph.tainted_propagation.count((definition_site, call_site)) == 1


class TestSaveListDisjunctiveDomain:
    """Disjunctive SaveList candidates are preserved and widened deterministically."""

    def test_savelist_disjunctive_candidates_propagate_all_taint_origins(self):
        graph = DataFlowGraph(
            values={
                3: [DataFlowValue(source=ValueSource.MESSAGE_BODY, definition_site=3, tainted=True)],
                4: [DataFlowValue(source=ValueSource.MESSAGE_SENDER, definition_site=4, tainted=True)],
            },
            edges=[],
            tainted_propagation=[],
        )
        savelist = SaveList()
        savelist = savelist.save(0, TVMAbstractValue(definition_site=3), max_candidates=4)
        savelist = savelist.save(0, TVMAbstractValue(definition_site=4), max_candidates=4)

        analyzer = DataFlowAnalyzer()
        edges_added = analyzer._propagate_savelist_taint(graph, call_site=12, savelist=savelist)

        assert edges_added == 2
        assert (3, 12) in graph.tainted_propagation
        assert (4, 12) in graph.tainted_propagation
        assert any(
            value.metadata.get("savelist_disjunctive") is True
            for value in graph.values.get(12, [])
        )

    def test_savelist_candidate_cap_widens_to_single_value(self):
        savelist = SaveList()
        savelist = savelist.save(0, TVMAbstractValue(definition_site=10), max_candidates=1)
        savelist = savelist.save(0, TVMAbstractValue(definition_site=20), max_candidates=1)

        values = savelist.restore_all(0)
        assert len(values) == 1
        assert values[0].metadata.get("savelist_widened") is True


class TestScanSensitiveOpsInLoop:
    """Loop truncation coverage: Test coverage for truncated loop scanning."""

    def test_scan_sensitive_ops_placeholder(self):
        """Placeholder: _scan_sensitive_ops_in_block requires full PathSensitiveAnalyzer
        setup with CFG blocks. This test verifies the method exists and is callable."""
        from tasmscan.ir.path_sensitive_dataflow import PathSensitiveDataFlowAnalyzer
        assert hasattr(PathSensitiveDataFlowAnalyzer, '_scan_sensitive_ops_in_block')
