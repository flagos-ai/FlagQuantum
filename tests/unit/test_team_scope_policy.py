from __future__ import annotations

import fnmatch
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

from tools.check_architecture import CONFIG as ARCHITECTURE
from tools.check_team_scope import load_policy, owner_for, policy_errors, scope_errors

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _existing_paths() -> tuple[str, ...]:
    """Repository-relative paths that currently exist, excluding VCS state."""

    ignored = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    return tuple(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not ignored.intersection(path.parts)
    )


def _tracked_paths() -> tuple[str, ...]:
    """Every tracked path, which is the set the ownership policy has to cover."""

    executable = shutil.which("git")
    if executable is None:
        pytest.skip("git is unavailable; the tracked-file invariant needs a checkout")
    completed = subprocess.run(
        [executable, "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def test_team_ownership_policy_is_valid_and_complete() -> None:
    policy = load_policy()
    assert policy_errors(policy) == ()
    assert set(policy["teams"]) == {
        "integration",
        "core",
        "compiler",
        "runtime",
        "simulation",
        "compute",
        "remote",
        "ecosystem",
        "services",
        "algorithms",
        "verification",
        "docs",
    }


def test_every_tracked_file_is_classified() -> None:
    """Every tracked file must be protected, shared, or owned by exactly one team.

    `scope_errors` depends on this. A file that matches no pattern resolves to
    `None` and is reported as "unowned path", so no team is accountable for it
    and any change to it is unassignable. A file matching equally specific
    patterns in two teams is reported as ambiguous. Both make the ownership
    policy advisory for that path.
    """

    policy = load_policy()
    protected = policy["policy"]["protected_paths"]
    shared = policy["policy"]["shared_paths"]

    unresolved: dict[str, str] = {}
    for path in _tracked_paths():
        if _matches(path, protected) or _matches(path, shared):
            continue
        owner = owner_for(path, policy)
        if owner is None or owner.startswith("AMBIGUOUS:"):
            unresolved[path] = owner or "unowned"
        if len(unresolved) >= 20:
            break

    assert unresolved == {}


def test_domain_owner_resolution() -> None:
    policy = load_policy()
    assert owner_for("flagquantum/runtime/execution.py", policy) == "runtime"
    assert (
        owner_for("flagquantum/runtime/executors/mps/core.py", policy) == "simulation"
    )
    assert owner_for("flagquantum/compute/flagos.py", policy) == "compute"
    assert owner_for("flagquantum/ecosystem/extensions/sdk.py", policy) == "ecosystem"
    assert owner_for("flagquantum/services/preflight.py", policy) == "services"


def test_every_module_domain_has_a_team_owner() -> None:
    """`architecture.toml` declares the allowed domains; all of them need an owner.

    A domain listed as allowed but owned by no team resolves to `None`, which
    `scope_errors` reports as "unowned path". That silently makes every change in
    the domain unassignable and leaves the module without an accountable team.
    """

    policy = load_policy()
    domains = ARCHITECTURE["package_layout"]["allowed_top_level_directories"]

    unresolved = {
        name: owner_for(f"flagquantum/{name}/__init__.py", policy) for name in domains
    }
    unresolved = {
        name: owner
        for name, owner in unresolved.items()
        if owner is None or owner.startswith("AMBIGUOUS:")
    }
    assert unresolved == {}


def test_the_ir_single_source_of_truth_is_protected() -> None:
    """Every existing IR artifact must be a protected integration surface.

    `AGENTS.md` names FlagQuantum IR as the one source of truth. A pattern that
    ends in `/**` does not match a same-named module, so a flat `core/ir.py` was
    unprotected even though `flagquantum/core/ir/**` was listed.
    """

    protected = load_policy()["policy"]["protected_paths"]
    core = ROOT / "flagquantum" / "core"
    targets: list[Path] = []
    for path in sorted(core.iterdir()):
        if not path.name.startswith("ir"):
            continue
        if path.is_file():
            targets.append(path)
        else:
            targets.extend(child for child in path.rglob("*") if child.is_file())

    assert targets, "flagquantum/core/ir artifacts are missing; update this test"
    unprotected = [
        path.relative_to(ROOT).as_posix()
        for path in targets
        if not _matches(path.relative_to(ROOT).as_posix(), protected)
    ]
    assert unprotected == []


def test_protected_patterns_match_existing_paths() -> None:
    """A protected pattern that matches nothing gives false confidence."""

    policy = load_policy()
    existing = _existing_paths()
    dead = [
        pattern
        for pattern in policy["policy"]["protected_paths"]
        # `flagquantum/core/ir/**` is a documented forward reservation for a
        # package split, so it is exempt from this check.
        if pattern != "flagquantum/core/ir/**"
        and not any(_matches(path, [pattern]) for path in existing)
    ]
    assert dead == []


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
