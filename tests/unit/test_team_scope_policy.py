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
    protected_path_errors,
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

    # The exemption this test used to carry for `flagquantum/core/ir/**` is
    # gone. That reservation matched no file, so it protected nothing while
    # reading as a guarantee. The property is now enforced by the checker as
    # well, so `tools/check_team_scope.py --validate` refuses a protected
    # entry in that state instead of leaving it to whoever runs this file.
    assert protected_path_errors(load_policy(), ROOT) == ()


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


def _normalized(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def test_the_manual_does_not_require_a_team_branch_or_worktree() -> None:
    """`AGENTS.md` must not instruct a flow that CI cannot merge.

    The manual is the highest-authority instruction a session receives. It said
    sessions "must use the branch and linked worktree assigned in
    `team-ownership.toml`", while no branch with the `refactor/` prefix and no
    `FlagQuantum-vNext*` worktree has ever existed here. A session that obeyed
    produced a branch and a handoff record that could not merge.
    """

    manual = _normalized("AGENTS.md")
    assert "must use the branch and linked worktree assigned" not in manual
    assert "reserved names for that mode rather than current instructions" in manual
    assert "do not cut one for ordinary work" in manual


def test_the_manual_points_at_the_flow_that_reaches_main() -> None:
    """The declared flow must be the one document that measures it."""

    manual = _normalized("AGENTS.md")
    assert "docs/development/INTEGRATION_WORKFLOW.md" in manual
    assert "is a short-lived branch cut from `main` and merged by pull request" in (
        manual
    )


def test_the_multi_team_document_marks_its_own_mode_optional() -> None:
    """The optional mode may not present its startup checks as mandatory.

    The document carries a correction header, but its body is long enough that a
    reader arriving at the imperative sees only the instruction. The branch check
    in particular named a branch that does not exist, so it must be scoped to the
    optional mode rather than listed among the unconditional checks.
    """

    text = _normalized("docs/development/MULTI_TEAM_DEVELOPMENT.md")
    assert "This document does not describe how changes reach `main` today." in text
    assert "In this optional mode, also verify that the current branch matches" in text
    assert (
        "2. Verify that the current branch matches the team's entry in "
        "`team-ownership.toml`." not in text
    )


def test_a_protected_pattern_that_matches_nothing_is_rejected(tmp_path: Path) -> None:
    """The check is mechanical; reading the policy's comments cannot replace it."""

    (tmp_path / "present.toml").write_text("", encoding="utf-8")
    policy = load_policy()
    policy["policy"] = dict(policy["policy"])
    policy["policy"]["protected_paths"] = ["present.toml", "absent/**"]
    errors = protected_path_errors(policy, tmp_path)
    assert len(errors) == 1
    assert "'absent/**'" in errors[0]
    assert "present.toml" not in errors[0]
