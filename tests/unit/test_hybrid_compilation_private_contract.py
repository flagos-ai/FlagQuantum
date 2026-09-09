from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "hybrid-compilation-private-v0-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_private_hybrid_contract_preserves_current_authorities() -> None:
    contract = _contract()

    assert contract["public_api_change"] is False
    assert contract["default_path_change"] is False
    assert contract["implementation_language"] == "python"
    assert contract["external_compiler"] is None
    assert contract["first_execution_classification"] == "single_device_fast_path"
    assert contract["existing_authorities"] == {
        "public_program": "flagquantum.core.ir.CircuitIR",
        "circuit_compiler": "flagquantum.compiler",
        "execution": "flagquantum.runtime",
        "numerics": "flagquantum.simulation",
        "plugin_registry": "flagquantum.ecosystem.extensions.ExtensionRegistry",
    }


def test_private_hybrid_contract_has_structured_control_and_quantum_effects() -> None:
    contract = _contract()
    operations = set(contract["phase1_operations"])
    invariants = set(contract["phase1_invariants"])

    assert {"program.func", "scf.if", "scf.for", "scf.yield"} <= operations
    assert {"tensor.extract", "arith.cmp"} <= operations
    assert {"quantum.rx", "quantum.ry", "quantum.cx"} <= operations
    assert "quantum.expectation" in operations
    assert "linear_ordered_quantum_effect" in invariants
    assert "matching_branch_signatures" in invariants
    assert "matching_loop_carried_signatures" in invariants


def test_phase1_cannot_claim_execution_or_public_integration() -> None:
    contract = _contract()
    exclusions = set(contract["phase1_exclusions"])
    rules = contract["rules"]

    assert {
        "public_exports",
        "default_compiler_integration",
        "runtime_execution",
        "gradient_execution",
        "measurement_control_flow",
        "catalyst",
        "flagquantum_authored_cpp",
    } <= exclusions
    assert rules["candidate_is_public_contract"] is False
    assert rules["may_update_public_api_snapshots"] is False
    assert rules["may_restore_removed_private_tree_wholesale"] is False
    assert rules["unsupported_behavior_fails_closed"] is True


def test_phase1_entry_remains_blocked_until_compiler_worktree_is_reconciled() -> None:
    contract = _contract()
    gates = contract["entry_gates"]

    assert gates["integration_worktree_has_only_this_change"] is True
    assert gates["compiler_worktree_on_assigned_branch"] is False
    assert (
        gates["compiler_worktree_synchronized_to_approved_integration_commit"] is False
    )
    assert gates["team_scope_preflight"] is False
    assert contract["implementation_started"] is False
