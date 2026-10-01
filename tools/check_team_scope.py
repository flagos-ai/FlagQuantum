#!/usr/bin/env python
"""Validate multi-team ownership and changes made by one team branch."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - CPython 3.10 tooling
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "team-ownership.toml"


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _specificity(pattern: str) -> int:
    return len(pattern.split("*", 1)[0].split("?", 1)[0])


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def owner_for(path: str, policy: Mapping[str, Any]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for team, config in policy["teams"].items():
        if team == policy["policy"]["integration_team"]:
            continue
        for pattern in config["owns"]:
            if fnmatch.fnmatchcase(path, pattern):
                candidates.append((_specificity(pattern), team))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_specificity = candidates[0][0]
    best_teams = {team for score, team in candidates if score == best_specificity}
    if len(best_teams) != 1:
        return "AMBIGUOUS:" + ",".join(sorted(best_teams))
    return next(iter(best_teams))


def policy_errors(policy: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if policy.get("schema") != "flagquantum.team_ownership":
        errors.append("team ownership policy has an invalid schema")
    teams = policy.get("teams", {})
    integration = policy.get("policy", {}).get("integration_team")
    if integration not in teams:
        errors.append("team ownership policy must define its integration team")

    branches: dict[str, str] = {}
    worktrees: dict[str, str] = {}
    for team, config in teams.items():
        for field, seen in (("branch", branches), ("worktree", worktrees)):
            value = config.get(field)
            if not value:
                errors.append(f"team {team!r} is missing {field}")
            elif value in seen:
                errors.append(
                    f"teams {seen[value]!r} and {team!r} share {field} {value!r}"
                )
            else:
                seen[value] = team
        if not config.get("owns"):
            errors.append(f"team {team!r} has no owned path")

    for field in ("protected_paths", "shared_paths"):
        if not policy.get("policy", {}).get(field):
            errors.append(f"team ownership policy has no {field}")
    return tuple(errors)


def scope_errors(
    team: str, paths: Iterable[str], policy: Mapping[str, Any]
) -> tuple[str, ...]:
    teams = policy["teams"]
    if team not in teams:
        return (f"unknown team {team!r}",)
    if team == policy["policy"]["integration_team"]:
        return ()

    protected = policy["policy"]["protected_paths"]
    shared = policy["policy"]["shared_paths"]
    errors: list[str] = []
    for raw_path in sorted(set(paths)):
        path = Path(raw_path).as_posix().removeprefix("./")
        if path.startswith("../") or path.startswith("/"):
            errors.append(f"{raw_path}: path must be repository-relative")
            continue
        if _matches(path, protected):
            errors.append(
                f"{path}: protected integration surface; submit a contract/ADR change"
            )
            continue
        owner = owner_for(path, policy)
        if owner == team or _matches(path, shared):
            continue
        if owner is None:
            errors.append(f"{path}: unowned path; integration assignment is required")
        elif owner.startswith("AMBIGUOUS:"):
            errors.append(
                f"{path}: ambiguous ownership ({owner.removeprefix('AMBIGUOUS:')})"
            )
        else:
            errors.append(f"{path}: owned by team {owner!r}, not {team!r}")
    return tuple(errors)


def classification_errors(
    paths: Iterable[str], policy: Mapping[str, Any]
) -> tuple[str, ...]:
    """Report changed paths that no rule classifies.

    This is the team-independent half of `scope_errors`. A path that matches no
    rule resolves to no owner for every team, so the per-team check can only
    report it after someone has already named the team that should have caught
    it. Running this over the changed paths instead fails on the first pull
    request that introduces such a path, rather than on the first one that
    happens to declare a team.
    """

    protected = policy["policy"]["protected_paths"]
    shared = policy["policy"]["shared_paths"]
    errors: list[str] = []
    for raw_path in sorted(set(paths)):
        path = Path(raw_path).as_posix().removeprefix("./")
        if path.startswith("../") or path.startswith("/"):
            errors.append(f"{raw_path}: path must be repository-relative")
            continue
        if _matches(path, protected) or _matches(path, shared):
            continue
        owner = owner_for(path, policy)
        if owner is None:
            errors.append(
                f"{path}: no rule classifies this path; add it to a team's `owns`,"
                " or to `protected_paths` or `shared_paths` in team-ownership.toml"
            )
        elif owner.startswith("AMBIGUOUS:"):
            errors.append(
                f"{path}: ambiguous ownership ({owner.removeprefix('AMBIGUOUS:')})"
            )
    return tuple(errors)


def owners_for(paths: Iterable[str], policy: Mapping[str, Any]) -> dict[str, int]:
    """Count changed paths per team, for the review summary.

    Paths that are protected or shared are not attributed to a team: they belong
    to Integration or to everyone, so counting them would make every change look
    cross-team.
    """

    protected = policy["policy"]["protected_paths"]
    shared = policy["policy"]["shared_paths"]
    counts: dict[str, int] = {}
    for raw_path in sorted(set(paths)):
        path = Path(raw_path).as_posix().removeprefix("./")
        if _matches(path, protected):
            counts["(protected)"] = counts.get("(protected)", 0) + 1
            continue
        if _matches(path, shared):
            counts["(shared)"] = counts.get("(shared)", 0) + 1
            continue
        owner = owner_for(path, policy)
        if owner is not None:
            counts[owner] = counts.get(owner, 0) + 1
    return counts


def _git_changed_files(base: str) -> tuple[str, ...]:
    executable = os.environ.get("FLAGQUANTUM_GIT") or shutil.which("git")
    if not executable:
        raise RuntimeError("git is unavailable; pass paths with --files")

    commands = (
        [executable, "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"],
        [executable, "diff", "--name-only", "--diff-filter=ACMR"],
        [executable, "ls-files", "--others", "--exclude-standard"],
    )
    paths: set[str] = set()
    for command in commands:
        try:
            completed = subprocess.run(
                command, cwd=ROOT, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"{' '.join(command)} failed with exit code {error.returncode};"
                " fetch the base revision, or pass paths with --files"
            ) from error
        paths.update(line for line in completed.stdout.splitlines() if line)
    return tuple(sorted(paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate policy only")
    parser.add_argument(
        "--require-classified",
        action="store_true",
        help="check that every changed path is protected, shared, or owned",
    )
    parser.add_argument("--team", help="team key from team-ownership.toml")
    parser.add_argument("--base", help="base revision for committed change discovery")
    parser.add_argument("--files", nargs="*", help="explicit repository-relative paths")
    return parser


def _paths(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        tuple(args.files) if args.files is not None else _git_changed_files(args.base)
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_policy()
    errors = list(policy_errors(policy))
    try:
        if args.validate:
            if args.team or args.require_classified:
                errors.append(
                    "--validate cannot be combined with --team or --require-classified"
                )
        elif args.require_classified:
            if args.team:
                errors.append("--require-classified does not take --team")
            if args.files is None and not args.base:
                errors.append("provide --files or --base")
            if not errors:
                paths = _paths(args)
                errors.extend(classification_errors(paths, policy))
                if not errors:
                    counts = owners_for(paths, policy)
                    summary = ", ".join(
                        f"{team} {count}" for team, count in sorted(counts.items())
                    )
                    print(f"{len(set(paths))} changed paths: {summary or 'none'}")
        else:
            if not args.team:
                errors.append("--team is required unless --validate is used")
            if args.files is None and not args.base:
                errors.append("provide --files or --base")
            if not errors:
                errors.extend(scope_errors(args.team, _paths(args), policy))
    except RuntimeError as error:
        errors.append(str(error))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if not args.require_classified:
        print("team ownership policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
