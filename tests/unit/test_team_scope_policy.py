from __future__ import annotations

import fnmatch
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

from tools.check_architecture import CONFIG as ARCHITECTURE
from tools.check_team_scope import (
    classification_errors,
    load_policy,
    owner_for,
    owners_for,
    policy_errors,
    scope_errors,
)

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


def test_owned_patterns_match_existing_paths() -> None:
    """An `owns` pattern that matches nothing assigns nothing.

    The sibling check above covers `protected_paths`. `owns` needs the same
    treatment and did not have it: `policy_errors` reads the schema only, so a
    team could be named as the owner of a directory that does not exist, and no
    check reported it. The pattern is invisible in review because it still reads
    as a claim about the architecture, and it is invisible to
    `scope_errors`, which only ever asks about paths that were actually changed.
    """

    policy = load_policy()
    existing = _existing_paths()
    dead = [
        f"{team}: {pattern}"
        for team, config in policy["teams"].items()
        for pattern in config["owns"]
        if not any(_matches(path, [pattern]) for path in existing)
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


def test_classification_accepts_protected_shared_and_owned_paths() -> None:
    policy = load_policy()
    assert (
        classification_errors(
            [
                "team-ownership.toml",
                "tests/unit/test_team_scope_policy.py",
                "tools/check_team_scope.py",
                "flagquantum/compiler/pipeline.py",
                "docs/development/MULTI_TEAM_DEVELOPMENT.md",
            ],
            policy,
        )
        == ()
    )


def test_classification_rejects_unowned_and_ambiguous_paths() -> None:
    policy = load_policy()
    errors = classification_errors(["CHANGELOG.md", "setup.cfg"], policy)
    assert len(errors) == 2
    assert all("no rule classifies this path" in error for error in errors)

    # Simulate an ambiguity the policy cannot currently produce: two teams
    # claiming the same pattern equally specifically.
    ambiguous = {
        **policy,
        "teams": {
            **policy["teams"],
            "core": {
                **policy["teams"]["core"],
                "owns": [*policy["teams"]["core"]["owns"], "flagquantum/x.py"],
            },
        },
    }
    ambiguous["teams"]["runtime"] = {
        **ambiguous["teams"]["runtime"],
        "owns": [*ambiguous["teams"]["runtime"]["owns"], "flagquantum/x.py"],
    }
    errors = classification_errors(["flagquantum/x.py"], ambiguous)
    assert len(errors) == 1
    assert "ambiguous ownership" in errors[0]


def test_classification_stays_team_independent() -> None:
    """The same path is rejected for every team, so no team can be blamed for it.

    A gate built only on `scope_errors` cannot catch an unowned path until a
    team has already been named, which is why the changed-path check does not
    take `--team`.
    """

    policy = load_policy()
    assert scope_errors("core", ["CHANGELOG.md"], policy)
    assert scope_errors("integration", ["CHANGELOG.md"], policy) == ()
    assert classification_errors(["CHANGELOG.md"], policy)


def test_owner_summary_counts_teams_and_marks_shared_paths() -> None:
    policy = load_policy()
    counts = owners_for(
        [
            "flagquantum/compiler/pipeline.py",
            "flagquantum/runtime/execution.py",
            "flagquantum/runtime/contracts.py",
            "tests/unit/test_team_scope_policy.py",
            "team-ownership.toml",
            "docs/development/MULTI_TEAM_DEVELOPMENT.md",
        ],
        policy,
    )
    assert counts == {"compiler": 1, "runtime": 2, "(shared)": 2, "(protected)": 1}
