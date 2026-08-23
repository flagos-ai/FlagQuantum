#!/usr/bin/env python
"""Fail when FlagQuantum package boundaries or size budgets regress."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - CPython 3.10 development tools
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "flagquantum"
MAINTAINED_ENTRYPOINTS = ("benchmarks", "examples", "tools")
DEVELOPMENT_MILESTONE_TOKENS = ("phase4", "phase5", "phase_4", "phase_5")
CONFIG = tomllib.loads((ROOT / "architecture.toml").read_text(encoding="utf-8"))


def _imports(path: Path) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                imports.append((module, name.name))
        elif isinstance(node, ast.Import):
            imports.extend((item.name, "") for item in node.names)
    return tuple(imports)


def _torch_cuda_use_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return sum(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "cuda"
        for node in ast.walk(tree)
    )


def architecture_errors() -> tuple[str, ...]:
    errors: list[str] = []
    boundaries = CONFIG["boundaries"]
    exceptions = CONFIG.get("legacy_exceptions", {})
    legacy_subsystems = CONFIG.get("legacy_subsystems", {})
    default_ceiling = int(boundaries["default_module_line_ceiling"])
    root_ceiling = int(boundaries["root_init_line_ceiling"])
    accelerator_boundaries = CONFIG.get("accelerator_boundaries", {})
    direct_cuda_allowed = set(accelerator_boundaries.get("direct_cuda_allowed", ()))
    direct_cuda_call_ceiling = {
        str(path): int(count)
        for path, count in accelerator_boundaries.get(
            "direct_cuda_call_ceiling", {}
        ).items()
    }
    torch_fl_import_allowed = set(
        accelerator_boundaries.get("torch_fl_import_allowed", ())
    )

    removed_runtime_stack = PACKAGE / "runtime_stack"
    if removed_runtime_stack.exists():
        errors.append(
            "flagquantum/runtime_stack: removed compatibility package must not return"
        )
    removed_core_circuit = PACKAGE / "core" / "circuit.py"
    if removed_core_circuit.exists():
        errors.append(
            "flagquantum/core/circuit.py: removed compatibility shim must not return"
        )
    removed_algorithms_stack = PACKAGE / "algorithms_stack"
    if removed_algorithms_stack.exists():
        errors.append(
            "flagquantum/algorithms_stack: removed compatibility package must not return"
        )

    for directory in ("flagquantum", "tests", *MAINTAINED_ENTRYPOINTS):
        for path in (ROOT / directory).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and any(
                    token in path.name.lower() for token in DEVELOPMENT_MILESTONE_TOKENS
                )
            ):
                relative = path.relative_to(ROOT).as_posix()
                errors.append(
                    f"{relative}: file names must describe capabilities, "
                    "not development milestones"
                )

    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        imports = _imports(path)
        cuda_use_count = _torch_cuda_use_count(path)
        cuda_ceiling = direct_cuda_call_ceiling.get(relative, 0)
        if relative not in direct_cuda_allowed and cuda_use_count > cuda_ceiling:
            errors.append(
                f"{relative}: {cuda_use_count} direct torch.cuda uses exceed "
                f"the migration ceiling {cuda_ceiling}; use PlatformRuntime"
            )
        for module, _ in imports:
            if (
                module == "torch_fl" or module.startswith("torch_fl.")
            ) and relative not in torch_fl_import_allowed:
                errors.append(
                    f"{relative}: torch_fl imports are isolated to the FlagOS adapter"
                )
        for subsystem_name, policy in legacy_subsystems.items():
            allowed_importers = set(policy.get("allowed_importers", ()))
            if relative in allowed_importers:
                continue
            legacy_roots = {
                item.removeprefix("flagquantum/")
                .removesuffix("/__init__.py")
                .removesuffix(".py")
                .replace("/", ".")
                for item in policy.get("modules", ())
            }
            for module, name in imports:
                imported = {module, name}
                if any(
                    candidate == root
                    or candidate.endswith(f".{root}")
                    or root.startswith(f"{candidate}.")
                    for candidate in imported
                    if candidate
                    for root in legacy_roots
                ):
                    errors.append(
                        f"{relative}: new dependency on closed legacy subsystem "
                        f"{subsystem_name}: {module or name}"
                    )
        for module, name in imports:
            if name == "*":
                errors.append(f"{relative}: star imports are forbidden")
            if "runtime_stack" in module.split("."):
                errors.append(
                    f"{relative}: removed runtime namespace imported: {module}"
                )
            if "algorithms_stack" in module.split("."):
                errors.append(
                    f"{relative}: removed algorithms namespace imported: {module}"
                )

        if relative.startswith("flagquantum/core/"):
            forbidden = tuple(boundaries["core_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(f"{relative}: core imports forbidden layer {module}")

        if relative.startswith("flagquantum/noise/"):
            forbidden = tuple(boundaries["noise_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: noise semantics import forbidden layer {module}"
                    )

        if relative.startswith("flagquantum/compilation/noise/"):
            forbidden = tuple(boundaries["compilation_noise_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: noise compilation imports forbidden layer {module}"
                    )

        if relative.startswith("flagquantum/runtime/backends/density_matrix/"):
            forbidden = tuple(boundaries["density_backend_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: density backend imports forbidden layer {module}"
                    )

        if relative.startswith("flagquantum/runtime/trajectories/"):
            forbidden = tuple(boundaries["trajectory_runtime_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: trajectory runtime imports forbidden layer {module}"
                    )

        if relative in {
            "flagquantum/runtime/distributed/protocols.py",
        }:
            forbidden = tuple(boundaries["executor_policy_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: executor imports classification policy {module}"
                    )

        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if relative == "flagquantum/__init__.py":
            ceiling = root_ceiling
        else:
            ceiling = int(
                exceptions.get(relative, {}).get("line_ceiling", default_ceiling)
            )
        if line_count > ceiling:
            errors.append(f"{relative}: {line_count} lines exceeds ceiling {ceiling}")

    for directory in MAINTAINED_ENTRYPOINTS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            for module, _ in _imports(path):
                if "runtime_stack" in module.split("."):
                    errors.append(
                        f"{relative}: maintained entrypoints must import "
                        f"flagquantum.runtime, not legacy {module}"
                    )
                if "algorithms_stack" in module.split("."):
                    errors.append(
                        f"{relative}: maintained entrypoints must import "
                        f"flagquantum.algorithms, not deprecated {module}"
                    )

    for path, policy in exceptions.items():
        if not policy.get("owner") or not policy.get("removal_version"):
            errors.append(
                f"{path}: compatibility exception needs owner and removal_version"
            )
    return tuple(errors)


def main() -> int:
    errors = architecture_errors()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("architecture boundaries passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
