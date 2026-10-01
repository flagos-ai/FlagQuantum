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
EXPECTED_INTEROP_EXTRAS = ("braket", "cirq", "pennylane", "quafu", "qiskit")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
# An environment marker such as ``python_version < '3.11'`` is not a bound, so a
# version must start with a digit for the specifier to count.
REQUIREMENT_BOUND = re.compile(
    r"(?P<operator><=|>=|<|>|==)\s*(?P<version>[0-9][0-9A-Za-z.!+*-]*)"
)


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def distribution_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"invalid requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def requirement_bounds(requirement: str) -> dict[str, str | None]:
    """Return the lower and upper bound one requirement declares.

    An exact pin sets both bounds to the same value, which is an upper bound as
    well, so it satisfies the requirement for a bounded development entry.
    """
    bounds: dict[str, str | None] = {"lower": None, "upper": None}
    for match in REQUIREMENT_BOUND.finditer(requirement):
        operator, version = match.group("operator"), match.group("version")
        if operator == ">=":
            bounds["lower"] = version
        elif operator == "<":
            bounds["upper"] = version
        elif operator == "==":
            bounds["lower"] = bounds["upper"] = version
    return bounds


def version_numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"[0-9]+", version))


def compare_versions(left: str, right: str) -> int:
    """Compare release numbers, treating ``1.117`` and ``1.117.0`` as equal."""
    width = max(len(version_numbers(left)), len(version_numbers(right)))
    padded = []
    for version in (left, right):
        parts = version_numbers(version)
        padded.append(parts + (0,) * (width - len(parts)))
    return (padded[0] > padded[1]) - (padded[0] < padded[1])


def declared_requirements(policy: dict[str, Any]) -> dict[str, str]:
    """Map every declared distribution name to its requirement string."""
    extras = policy.get("extras")
    groups: list[object] = [policy.get("core")]
    if isinstance(extras, dict):
        groups.extend(extras.values())
    declared: dict[str, str] = {}
    for requirements in groups:
        if not isinstance(requirements, list):
            continue
        for requirement in requirements:
            if not isinstance(requirement, str) or not requirement.strip():
                continue
            try:
                name = distribution_name(requirement)
            except ValueError:
                continue
            declared.setdefault(name, requirement)
    return declared


def recorded_requirement(policy: dict[str, Any], name: str) -> str | None:
    """Resolve one ``[tested]`` key to the requirement it records versions of.

    A key is either the distribution it records (``torch``) or the name of the
    extra that installs it under a different distribution (``cirq`` installs
    ``cirq-core``).
    """
    declared = declared_requirements(policy)
    if name in declared:
        return declared[name]
    extras = policy.get("extras")
    requirements = extras.get(name) if isinstance(extras, dict) else None
    if isinstance(requirements, list) and len(requirements) == 1:
        requirement = requirements[0]
        return requirement if isinstance(requirement, str) else None
    return None


def requirement_names(requirements: object) -> tuple[str, ...] | None:
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) for item in requirements
    ):
        return None
    try:
        return tuple(distribution_name(item) for item in requirements)
    except ValueError:
        return None


def development_bound_errors(
    policy: dict[str, Any], policy_extras: dict[str, Any]
) -> tuple[str, ...]:
    """Require every development-class requirement to declare an upper bound.

    The development extra is the build and test environment this repository
    installs itself, so an unbounded entry lets an upstream release change the
    toolchain and break `main` with no commit and no diff.
    """
    classes = policy.get("classes")
    development = classes.get("development") if isinstance(classes, dict) else None
    if not isinstance(development, list) or not development:
        return ("dependency policy must classify a development extra",)
    errors: list[str] = []
    for extra in development:
        requirements = policy_extras.get(extra)
        if not isinstance(requirements, list):
            errors.append(f"development extra {extra!r} is not declared")
            continue
        for requirement in requirements:
            if not isinstance(requirement, str) or not requirement.strip():
                continue
            if requirement_bounds(requirement)["upper"] is None:
                errors.append(
                    f"development requirement {requirement!r} must declare an "
                    "upper bound"
                )
    return tuple(errors)


def recorded_version_errors(policy: dict[str, Any]) -> tuple[str, ...]:
    """Check that every ``[tested]`` version lies inside the declared range.

    The recorded floor is what the compatibility lanes install, so a recorded
    version outside the range would name an environment the packaging metadata
    does not allow.
    """
    recorded = policy.get("tested")
    if not isinstance(recorded, dict):
        return ("dependency policy is missing [tested]",)
    errors: list[str] = []
    for name, versions in recorded.items():
        requirement = recorded_requirement(policy, name)
        if requirement is None:
            errors.append(f"recorded dependency {name!r} is not declared")
            continue
        if not isinstance(versions, list) or not versions:
            errors.append(f"recorded dependency {name!r} must list exact versions")
            continue
        if not all(isinstance(version, str) and version for version in versions):
            errors.append(f"recorded dependency {name!r} must list exact versions")
            continue
        bounds = requirement_bounds(requirement)
        lower, upper = bounds["lower"], bounds["upper"]
        if lower is None:
            errors.append(f"recorded dependency {name!r} declares no lower bound")
            continue
        if upper is None:
            errors.append(f"recorded dependency {name!r} declares no upper bound")
            continue
        if lower == upper:
            # An exact pin admits one release, so every recorded version has to
            # be that release; the range comparison below would call the pin
            # itself out of range.
            outside = [
                version for version in versions if compare_versions(version, lower) != 0
            ]
            if outside:
                errors.append(
                    f"recorded dependency {name!r} is pinned to {lower!r} but "
                    f"records {outside[0]!r}"
                )
            continue
        if compare_versions(versions[0], lower) != 0:
            errors.append(
                f"recorded dependency {name!r} starts at {versions[0]!r} but "
                f"{requirement!r} declares {lower!r}"
            )
        if compare_versions(versions[-1], upper) >= 0:
            errors.append(
                f"recorded dependency {name!r} records {versions[-1]!r}, which the "
                f"declared range {requirement!r} does not allow"
            )
    return tuple(errors)


def development_floor_pins(policy: dict[str, Any]) -> tuple[str, ...]:
    """Return the ``distribution==version`` pins of the development floor.

    The CI floor lane installs exactly these pins, so the versions this
    repository claims as its oldest supported toolchain are exercised instead of
    trusted.
    """
    extras = policy.get("extras")
    classes = policy.get("classes")
    recorded = policy.get("tested")
    if not isinstance(extras, dict) or not isinstance(recorded, dict):
        return ()
    development = classes.get("development") if isinstance(classes, dict) else None
    if not isinstance(development, list):
        return ()
    pins: list[str] = []
    for extra in development:
        requirements = extras.get(extra)
        if not isinstance(requirements, list):
            continue
        for requirement in requirements:
            if not isinstance(requirement, str) or not requirement.strip():
                continue
            try:
                name = distribution_name(requirement)
            except ValueError:
                continue
            versions = recorded.get(name)
            if isinstance(versions, list) and versions:
                pins.append(f"{name}=={versions[0]}")
    return tuple(pins)


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
                "interop class must be exactly braket, cirq, pennylane, quafu, and qiskit"
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

    errors.extend(development_bound_errors(policy, policy_extras))
    errors.extend(recorded_version_errors(policy))

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
    parser.add_argument(
        "--print-floor",
        action="store_true",
        help="print the pinned development floor for the CI floor lane",
    )
    args = parser.parse_args(argv)
    policy = load_toml(args.policy)
    errors = policy_errors(policy, load_toml(args.pyproject))
    if errors:
        print("\n".join(errors))
        return 1
    if args.print_floor:
        pins = development_floor_pins(policy)
        if not pins:
            print("the development class records no pinned floor")
            return 1
        print(" ".join(pins))
        return 0
    print("dependency policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
