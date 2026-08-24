#!/usr/bin/env python3
"""Validate install extras and externally managed platform boundaries."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "dependency-policy.toml"
PYPROJECT = ROOT / "pyproject.toml"
EXPECTED_SCHEMA = "flagquantum_dependency_policy_v2"
EXPECTED_INTEROP_EXTRAS = ("braket", "pennylane", "quafu", "qiskit")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def distribution_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"invalid requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def requirement_names(requirements: object) -> tuple[str, ...] | None:
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) for item in requirements
    ):
        return None
    try:
        return tuple(distribution_name(item) for item in requirements)
    except ValueError:
        return None


def policy_errors(policy: dict[str, Any], pyproject: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if policy.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"dependency policy schema must be {EXPECTED_SCHEMA}")

    project = pyproject.get("project")
    if not isinstance(project, dict):
        return tuple(errors + ["pyproject is missing [project]"])
    project_core = project.get("dependencies")
    policy_core = policy.get("core")
    if policy_core != project_core:
        errors.append("policy core dependencies do not exactly match pyproject")
    core_names = requirement_names(project_core)
    if core_names is None:
        errors.append("core dependencies must be a list of valid requirements")
        core_names = ()
    if core_names != ("torch",):
        errors.append(f"core dependencies must contain only torch, got {core_names}")

    project_extras = project.get("optional-dependencies")
    policy_extras = policy.get("extras")
    if not isinstance(project_extras, dict) or not isinstance(policy_extras, dict):
        return tuple(errors + ["extras must be tables in policy and pyproject"])
    project_names = set(project_extras)
    policy_names = set(policy_extras)
    if project_names != policy_names:
        missing = sorted(project_names - policy_names)
        unexpected = sorted(policy_names - project_names)
        errors.append(
            "policy extras differ from pyproject: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(project_names & policy_names):
        if policy_extras[name] != project_extras[name]:
            errors.append(f"extra {name!r} does not exactly match pyproject")

    classes = policy.get("classes")
    classified: dict[str, str] = {}
    if not isinstance(classes, dict):
        errors.append("dependency policy is missing [classes]")
    else:
        for class_name, names in classes.items():
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                errors.append(f"class {class_name!r} must be a list of extra names")
                continue
            for name in names:
                if name in classified:
                    errors.append(
                        f"extra {name!r} is classified by both "
                        f"{classified[name]!r} and {class_name!r}"
                    )
                classified[name] = class_name
        unclassified = sorted(project_names - set(classified))
        unknown = sorted(set(classified) - project_names)
        if unclassified or unknown:
            errors.append(
                "extra classification is incomplete: "
                f"unclassified={unclassified}, unknown={unknown}"
            )
        interop = classes.get("interop")
        interop_names = tuple(interop) if isinstance(interop, list) else ()
        if interop_names != EXPECTED_INTEROP_EXTRAS:
            errors.append(
                "interop class must be exactly braket, pennylane, quafu, and qiskit"
            )

    aggregates = policy.get("aggregates")
    if not isinstance(aggregates, dict):
        errors.append("dependency policy is missing [aggregates]")
    else:
        declared_aggregates = (
            set(classes.get("aggregate", [])) if isinstance(classes, dict) else set()
        )
        if set(aggregates) != declared_aggregates:
            errors.append(
                "aggregate declarations differ from the aggregate class: "
                f"declared={sorted(aggregates)}, "
                f"classified={sorted(declared_aggregates)}"
            )
        for aggregate, components in aggregates.items():
            if aggregate not in policy_extras:
                errors.append(f"aggregate {aggregate!r} is not a declared extra")
                continue
            if not isinstance(components, list) or not components:
                errors.append(f"aggregate {aggregate!r} must list component extras")
                continue
            expanded: list[str] = []
            for component in components:
                if component == aggregate:
                    errors.append(f"aggregate {aggregate!r} cannot include itself")
                    continue
                requirements = policy_extras.get(component)
                if not isinstance(requirements, list):
                    errors.append(
                        f"aggregate {aggregate!r} references unknown extra {component!r}"
                    )
                    continue
                expanded.extend(requirements)
            if policy_extras[aggregate] != expanded:
                errors.append(
                    f"aggregate {aggregate!r} does not equal its component extras"
                )

    platforms = policy.get("external_platforms")
    flagos = platforms.get("flagos") if isinstance(platforms, dict) else None
    if not isinstance(flagos, dict):
        errors.append("dependency policy is missing [external_platforms.flagos]")
    else:
        distribution = flagos.get("distribution")
        if distribution != "torch-fl":
            errors.append("FlagOS external distribution must be torch-fl")
        if flagos.get("installation") != "managed_outside_flagquantum":
            errors.append("Torch-FL installation must remain externally managed")
        for field in ("core_dependency_allowed", "optional_extra_allowed"):
            if flagos.get(field) is not False:
                errors.append(f"FlagOS {field} must be false")
        if "flagos" in project_names:
            errors.append("FlagOS must not be a FlagQuantum install extra")
        all_optional_names: set[str] = set()
        for extra, requirements in project_extras.items():
            names = requirement_names(requirements)
            if names is None:
                errors.append(f"extra {extra!r} must be a list of valid requirements")
                continue
            all_optional_names.update(names)
        if "torch-fl" in set(core_names) | all_optional_names:
            errors.append("Torch-FL must not be managed by FlagQuantum packaging")

    import_policy = policy.get("import_policy")
    if not isinstance(import_policy, dict):
        errors.append("dependency policy is missing [import_policy]")
    else:
        allowed = import_policy.get("core_allowed_distributions")
        if allowed != list(core_names):
            errors.append("core_allowed_distributions must match core dependencies")
        forbidden = import_policy.get("core_forbidden_imports")
        if not isinstance(forbidden, list) or not forbidden:
            errors.append("core_forbidden_imports must be a non-empty list")
        elif len(forbidden) != len(set(forbidden)):
            errors.append("core_forbidden_imports contains duplicates")
        elif isinstance(flagos, dict):
            for name in flagos.get("import_names", []):
                if name not in forbidden:
                    errors.append(
                        f"externally managed import {name!r} is not forbidden at core import"
                    )

    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    args = parser.parse_args(argv)
    errors = policy_errors(load_toml(args.policy), load_toml(args.pyproject))
    if errors:
        print("\n".join(errors))
        return 1
    print("dependency policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
