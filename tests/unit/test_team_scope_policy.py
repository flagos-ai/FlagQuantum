from __future__ import annotations

from tools.check_team_scope import load_policy, owner_for, policy_errors, scope_errors


def test_team_ownership_policy_is_valid_and_complete() -> None:
    policy = load_policy()
    assert policy_errors(policy) == ()
    assert set(policy["teams"]) == {
        "integration",
        "core",
        "compiler",
        "runtime",
        "simulation",
        "platform",
        "execution",
        "ecosystem",
        "agent",
        "docs",
    }


def test_domain_owner_resolution() -> None:
    policy = load_policy()
    assert owner_for("flagquantum/runtime/execution.py", policy) == "runtime"
    assert (
        owner_for("flagquantum/runtime/executors/mps/core.py", policy) == "simulation"
    )
    assert owner_for("flagquantum/providers/platform/flagos.py", policy) == "platform"
    assert owner_for("flagquantum/ecosystem/extensions/sdk.py", policy) == "ecosystem"


def test_team_can_change_owned_and_shared_test_paths() -> None:
    policy = load_policy()
    assert (
        scope_errors(
            "compiler",
            ["flagquantum/compiler/pipeline.py", "tests/unit/test_compiler.py"],
            policy,
        )
        == ()
    )


def test_protected_and_cross_team_changes_fail_closed() -> None:
    policy = load_policy()
    errors = scope_errors(
        "runtime",
        [
            "contracts/new-contract.json",
            "flagquantum/runtime/executors/mps/core.py",
            "unknown-root-file.txt",
        ],
        policy,
    )
    assert len(errors) == 3
    assert "protected integration surface" in errors[0]
    assert "owned by team 'simulation'" in errors[1]
    assert "unowned path" in errors[2]


def test_integration_team_can_coordinate_any_path() -> None:
    policy = load_policy()
    assert (
        scope_errors(
            "integration", ["contracts/core.json", "unknown-root-file.txt"], policy
        )
        == ()
    )
