#!/usr/bin/env python
"""Validate multi-team ownership and changes made by one team branch."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

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
        completed = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
        paths.update(line for line in completed.stdout.splitlines() if line)
    return tuple(sorted(paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate policy only")
    parser.add_argument("--team", help="team key from team-ownership.toml")
    parser.add_argument("--base", help="base revision for committed change discovery")
    parser.add_argument("--files", nargs="*", help="explicit repository-relative paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_policy()
    errors = list(policy_errors(policy))
    if not args.validate:
        if not args.team:
            errors.append("--team is required unless --validate is used")
        if args.files is None and not args.base:
            errors.append("provide --files or --base")
        if not errors:
            paths = (
                tuple(args.files)
                if args.files is not None
                else _git_changed_files(args.base)
            )
            errors.extend(scope_errors(args.team, paths, policy))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("team ownership policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
