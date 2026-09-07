"""Tests for completeness metadata normalization utilities."""

from tasmscan.completeness import build_completeness_report, collect_incomplete_reasons


def test_collect_incomplete_reasons_infers_dynamic_and_truncation():
    metadata = {
        "analysis_incomplete": True,
        "dynamic_target_count": 2,
        "truncated_count": 1,
        "truncated_loop_count": 0,
    }

    reasons = collect_incomplete_reasons(metadata)

    assert "dynamic_continuation_targets" in reasons
    assert "path_truncation" in reasons


def test_build_completeness_report_complete():
    report = build_completeness_report({"analysis_incomplete": False})

    assert report["analysis_incomplete"] is False
    assert report["analysis_incomplete_reasons"] == []
    assert report["summary"] == "complete"


def test_build_completeness_report_uses_reason_descriptions():
    report = build_completeness_report(
        {
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["dynamic_continuation_targets"],
        }
    )

    assert report["analysis_incomplete"] is True
    assert report["analysis_incomplete_reasons"] == ["dynamic_continuation_targets"]
    assert report["reason_descriptions"] == ["dynamic continuation targets were not solved"]


def test_collect_incomplete_reasons_respects_non_material_truncation_flag():
    metadata = {
        "analysis_incomplete": False,
        "material_truncation": False,
        "truncated": True,
        "truncated_count": 3,
        "truncated_loop_count": 1,
        "resource_truncation_count": 1,
    }

    reasons = collect_incomplete_reasons(metadata)

    assert "path_truncation" not in reasons


def test_collect_incomplete_reasons_infers_stack_non_convergence():
    metadata = {
        "analysis_incomplete": True,
        "stack_analysis": {
            "global_converged": False,
            "global_iterations": 128,
            "global_iteration_limit": 128,
        },
    }

    reasons = collect_incomplete_reasons(metadata)

    assert "stack_analysis_non_converged" in reasons


def test_collect_incomplete_reasons_infers_savelist_unresolved_and_fixpoint_cap():
    metadata = {
        "cont_ref_resolution": {"unresolved": 2},
        "cont_stack_fixpoint_capped": True,
    }

    reasons = collect_incomplete_reasons(metadata)

    assert "savelist_target_unresolved" in reasons
    assert "savelist_fixpoint_capped" in reasons
