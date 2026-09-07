from scripts.benchmark_test import ContractResult, evaluate_release_gate


def test_release_gate_go_when_all_checks_pass():
    results = [
        ContractResult(ipfs_hash="a", network="mainnet", compiler="func", status="success"),
        ContractResult(ipfs_hash="b", network="mainnet", compiler="func", status="success"),
    ]
    gate = evaluate_release_gate(
        results,
        max_incomplete_rate=0.5,
        max_dynamic_targets=0,
    )

    assert gate["go"] is True
    assert gate["failed_checks"] == []
    assert gate["metrics"]["incomplete_rate"] == 0.0
    assert gate["metrics"]["dynamic_continuation_targets"] == 0


def test_release_gate_no_go_when_budgets_exceeded():
    results = [
        ContractResult(
            ipfs_hash="a",
            network="mainnet",
            compiler="func",
            status="success",
            analysis_incomplete=True,
            analysis_incomplete_reasons=["path_truncation", "dynamic_continuation_targets"],
        ),
        ContractResult(ipfs_hash="b", network="mainnet", compiler="func", status="crash"),
    ]
    gate = evaluate_release_gate(
        results,
        max_incomplete_rate=0.1,
        max_dynamic_targets=0,
    )

    assert gate["go"] is False
    assert "no_crashes" in gate["failed_checks"]
    assert "incomplete_rate_within_budget" in gate["failed_checks"]
    assert "dynamic_targets_within_budget" in gate["failed_checks"]
