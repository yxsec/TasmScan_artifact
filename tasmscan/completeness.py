"""Utilities for normalizing analysis completeness metadata."""

from typing import Any, Dict, List, Optional


REASON_DESCRIPTIONS: Dict[str, str] = {
    "path_truncation": "path-sensitive dataflow hit exploration limits",
    "cfg_unknown_successor": "CFG contains blocks with unknown successors",
    "dynamic_continuation_targets": "dynamic continuation targets were not solved",
    "continuation_extraction_failed": "continuation extraction failed",
    "cfg_entry_shape_widened": "interprocedural CFG entry-shape was conservatively widened",
    "cfg_entry_shape_non_converged": "interprocedural CFG entry-shape did not converge",
    "stack_analysis_non_converged": "stack analysis did not converge within iteration budget",
    "savelist_target_unresolved": "SaveList target continuation could not be resolved precisely",
    "savelist_fixpoint_capped": "SaveList continuation-stack fixpoint hit iteration budget",
    "unspecified_incomplete": "analysis reported incomplete without a specific reason",
}


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect_incomplete_reasons(metadata: Optional[Dict[str, Any]]) -> List[str]:
    """Collect normalized incomplete reasons from metadata."""
    if not isinstance(metadata, dict):
        return []

    reasons: List[str] = []

    raw_reasons = metadata.get("analysis_incomplete_reasons", [])
    if isinstance(raw_reasons, list):
        for reason in raw_reasons:
            if isinstance(reason, str) and reason and reason not in reasons:
                reasons.append(reason)

    if metadata.get("continuation_extraction_failed"):
        if "continuation_extraction_failed" not in reasons:
            reasons.append("continuation_extraction_failed")

    truncated_count = (
        _to_int(metadata.get("truncated_count"))
        + _to_int(metadata.get("truncated_loop_count"))
        + _to_int(metadata.get("resource_truncation_count"))
    )
    material_truncation = metadata.get("material_truncation")
    if material_truncation is True:
        if "path_truncation" not in reasons:
            reasons.append("path_truncation")
    elif material_truncation is None:
        # Backward compatibility for legacy metadata that did not report
        # truncation materiality explicitly.
        if bool(metadata.get("truncated")) or truncated_count > 0:
            if "path_truncation" not in reasons:
                reasons.append("path_truncation")

    if _to_int(metadata.get("unknown_successor_count")) > 0:
        if "cfg_unknown_successor" not in reasons:
            reasons.append("cfg_unknown_successor")

    if _to_int(metadata.get("dynamic_target_count")) > 0:
        if "dynamic_continuation_targets" not in reasons:
            reasons.append("dynamic_continuation_targets")

    cont_ref_resolution = metadata.get("cont_ref_resolution")
    if isinstance(cont_ref_resolution, dict) and _to_int(cont_ref_resolution.get("unresolved")) > 0:
        if "savelist_target_unresolved" not in reasons:
            reasons.append("savelist_target_unresolved")

    if metadata.get("cont_stack_fixpoint_capped"):
        if "savelist_fixpoint_capped" not in reasons:
            reasons.append("savelist_fixpoint_capped")

    stack_meta = metadata.get("stack_analysis")
    if metadata.get("stack_analysis_non_converged") or (
        isinstance(stack_meta, dict) and not bool(stack_meta.get("global_converged", True))
    ):
        if "stack_analysis_non_converged" not in reasons:
            reasons.append("stack_analysis_non_converged")

    if metadata.get("analysis_incomplete") and not reasons:
        reasons.append("unspecified_incomplete")

    return reasons


def build_completeness_report(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a machine-readable completeness report from analysis metadata."""
    meta: Dict[str, Any] = metadata if isinstance(metadata, dict) else {}
    reasons = collect_incomplete_reasons(meta)

    truncated_count = _to_int(meta.get("truncated_count"))
    truncated_loop_count = _to_int(meta.get("truncated_loop_count"))
    resource_truncation_count = _to_int(meta.get("resource_truncation_count"))
    dynamic_target_count = _to_int(meta.get("dynamic_target_count"))
    unknown_successor_count = _to_int(meta.get("unknown_successor_count"))

    incomplete = bool(meta.get("analysis_incomplete")) or bool(reasons)
    reason_descriptions = [REASON_DESCRIPTIONS.get(reason, reason) for reason in reasons]
    summary = (
        "complete"
        if not incomplete
        else ", ".join(reason_descriptions) if reason_descriptions else REASON_DESCRIPTIONS["unspecified_incomplete"]
    )

    return {
        "analysis_incomplete": incomplete,
        "analysis_incomplete_reasons": reasons,
        "reason_descriptions": reason_descriptions,
        "truncated": bool(meta.get("truncated")) or (truncated_count + truncated_loop_count + resource_truncation_count) > 0,
        "truncated_count": truncated_count,
        "truncated_loop_count": truncated_loop_count,
        "resource_truncation_count": resource_truncation_count,
        "dynamic_target_count": dynamic_target_count,
        "unknown_successor_count": unknown_successor_count,
        "summary": summary,
    }
