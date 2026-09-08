#!/usr/bin/env python
"""Fail when FlagQuantum package boundaries or size budgets regress."""

from __future__ import annotations

import ast
import json
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
NON_JAX_EXECUTORS = (
    "flagquantum/runtime/executors/statevector/",
    "flagquantum/runtime/executors/mps/",
    "flagquantum/runtime/executors/tensor_network/",
)
CONFIG = tomllib.loads((ROOT / "architecture.toml").read_text(encoding="utf-8"))
LONG_HORIZON_CONTRACT = ROOT / "contracts" / "long-horizon-architecture-v1.json"


def _long_horizon_contract_errors() -> tuple[str, ...]:
    """Validate invariants that keep the target architecture unambiguous."""

    payload = json.loads(LONG_HORIZON_CONTRACT.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("shared_contract_owner") != "core":
        errors.append("long-horizon architecture: Core must own shared contracts")

    domains = payload.get("domains", {})
    if domains.get("runtime", {}).get("may_depend_on") != ["core"]:
        errors.append(
            "long-horizon architecture: Runtime may depend only on Core contracts"
        )

    provider_layers = payload.get("provider_layers", {})
    if set(provider_layers.get("execution", ())) != {
        "simulation",
        "qpu",
        "remote_service",
    }:
        errors.append(
            "long-horizon architecture: execution providers must remain distinct"
        )
    if set(provider_layers.get("platform", ())) != {
        "cpu",
        "accelerator",
        "communication",
    }:
        errors.append(
            "long-horizon architecture: platform providers must remain distinct"
        )

    required_track_fields = {
        "name",
        "owner",
        "target_milestone",
        "current_authority",
        "target_authority",
        "compatibility_adapter",
        "completion_evidence",
        "retirement_condition",
        "status",
    }
    tracks = payload.get("migration_tracks", ())
    names = [track.get("name") for track in tracks]
    if not tracks or len(names) != len(set(names)):
        errors.append(
            "long-horizon architecture: migration tracks must be present and unique"
        )
    for index, track in enumerate(tracks):
        missing = required_track_fields - set(track)
        if missing:
            errors.append(
                "long-horizon architecture: migration track "
                f"{index} is missing {', '.join(sorted(missing))}"
            )
        if track.get("status") not in {"planned", "in_progress", "complete"}:
            errors.append(
                "long-horizon architecture: migration track "
                f"{track.get('name', index)!r} has an invalid status"
            )
    return tuple(errors)


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


def _module_name(path: Path, package_root: Path) -> str:
    parts = path.relative_to(package_root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package_root.name, *parts))


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


def _eager_import_candidates(path: Path, package_root: Path) -> tuple[str, ...]:
    """Return imports evaluated while a module is initialized.

    Function bodies and ``TYPE_CHECKING`` blocks are deliberately excluded:
    those references do not create Python initialization cycles.
    """

    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    module = _module_name(path, package_root)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    candidates: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_If(self, node: ast.If) -> None:
            if _is_type_checking_guard(node.test):
                for statement in node.orelse:
                    self.visit(statement)
                return
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            candidates.extend(item.name for item in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level:
                package_parts = package.split(".") if package else []
                keep = max(0, len(package_parts) - (node.level - 1))
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
            candidates.extend(
                f"{base}.{item.name}" if base else item.name
                for item in node.names
                if item.name != "*"
            )

    Visitor().visit(tree)
    return tuple(candidates)


def eager_import_cycles(package_root: Path = PACKAGE) -> tuple[tuple[str, ...], ...]:
    """Return strongly connected module-initialization dependencies."""

    paths = tuple(sorted(package_root.rglob("*.py")))
    modules = {_module_name(path, package_root): path for path in paths}
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        for candidate in _eager_import_candidates(path, package_root):
            target = candidate
            while target and target not in modules:
                target = target.rpartition(".")[0]
            if target and target != module:
                graph[module].add(target)

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = lowlinks[module] = index
        index += 1
        stack.append(module)
        active.add(module)
        for dependency in graph[module]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in active:
                lowlinks[module] = min(lowlinks[module], indices[dependency])
        if lowlinks[module] != indices[module]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            active.remove(member)
            component.append(member)
            if member == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in graph:
        if module not in indices:
            visit(module)
    return tuple(sorted(components))


def _torch_cuda_use_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return sum(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "cuda"
        for node in ast.walk(tree)
    )


def _imports_jax_executor(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    module_parts = path.relative_to(ROOT).with_suffix("").parts[:-1]
    target = "flagquantum.runtime.executors.jax"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                item.name == target or item.name.startswith(f"{target}.")
                for item in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if node.level:
                keep = max(0, len(module_parts) - (node.level - 1))
                imported = ".".join((*module_parts[:keep], *imported.split(".")))
            if imported == target or imported.startswith(f"{target}."):
                return True
            if imported == "flagquantum.runtime.executors" and any(
                item.name == "jax" for item in node.names
            ):
                return True
    return False


def architecture_errors() -> tuple[str, ...]:
    errors = list(_long_horizon_contract_errors())
    for cycle in eager_import_cycles():
        errors.append("eager import cycle: " + " -> ".join((*cycle, cycle[0])))
    allowed_package_directories = set(
        CONFIG.get("package_layout", {}).get("allowed_top_level_directories", ())
    )
    actual_package_directories = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir()
        and any(
            candidate.is_file() and "__pycache__" not in candidate.parts
            for candidate in path.rglob("*.py")
        )
    }
    for name in sorted(actual_package_directories - allowed_package_directories):
        errors.append(
            f"flagquantum/{name}: unreviewed top-level package directory is forbidden"
        )
    for name in sorted(allowed_package_directories - actual_package_directories):
        errors.append(f"flagquantum/{name}: stale allowed top-level package directory")

    boundaries = CONFIG["boundaries"]
    exceptions = CONFIG.get("legacy_exceptions", {})
    legacy_subsystems = CONFIG.get("legacy_subsystems", {})
    default_ceiling = int(boundaries["default_module_line_ceiling"])
    root_ceiling = int(boundaries["root_init_line_ceiling"])
    accelerator_boundaries = CONFIG.get("accelerator_boundaries", {})
    interop_boundaries = CONFIG.get("interop_boundaries", {})
    simulation_boundaries = CONFIG.get("simulation_boundaries", {})
    northbound_boundaries = CONFIG.get("northbound_boundaries", {})
    long_horizon_boundaries = CONFIG.get("long_horizon_boundaries", {})
    runtime_compiler_import_allowed = set(
        long_horizon_boundaries.get("runtime_compiler_import_allowed", ())
    )
    observed_runtime_compiler_imports: set[str] = set()
    simulation_runtime_import_allowed = set(
        simulation_boundaries.get("runtime_import_allowed", ())
    )
    qiskit_import_allowed_prefixes = tuple(
        interop_boundaries.get("qiskit_import_allowed_prefixes", ())
    )
    pennylane_import_allowed_prefixes = tuple(
        interop_boundaries.get("pennylane_import_allowed_prefixes", ())
    )
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
    removed_runtime_platforms = PACKAGE / "runtime" / "platforms"
    if removed_runtime_platforms.exists() and any(
        removed_runtime_platforms.glob("*.py")
    ):
        errors.append(
            "flagquantum/runtime/platforms: migrated platform package must not return; "
            "use flagquantum/providers/platform"
        )
    for removed_compiler_path in (
        PACKAGE / "compiler.py",
        PACKAGE / "compilation" / "compiler.py",
        PACKAGE / "compilation" / "routing.py",
    ):
        if removed_compiler_path.exists():
            errors.append(
                f"{removed_compiler_path.relative_to(ROOT).as_posix()}: migrated "
                "compiler authority must not return; use flagquantum/compiler"
            )
    removed_plan_adapter = PACKAGE / "compilation" / "contract_adapter.py"
    if removed_plan_adapter.exists():
        errors.append(
            "flagquantum/compilation/contract_adapter.py: merged single-use plan "
            "projection must not return; use execution_plan_contract.py"
        )
    removed_plan_builder = PACKAGE / "compilation" / "execution_plan_builder.py"
    if removed_plan_builder.exists():
        errors.append(
            "flagquantum/compilation/execution_plan_builder.py: Runtime plan "
            "assembly must not return; use runtime/planner"
        )
    for removed_runtime_planning_name in (
        "backend_selection.py",
        "candidate_plans.py",
        "candidates.py",
        "estimates.py",
        "execution_policy.py",
        "metadata_projection.py",
        "planner.py",
        "providers.py",
        "selection_context.py",
        "selection_result.py",
        "tn_calibration.py",
        "topology.py",
        "training_preflight.py",
    ):
        removed_path = PACKAGE / "compilation" / removed_runtime_planning_name
        if removed_path.exists():
            errors.append(
                f"{removed_path.relative_to(ROOT).as_posix()}: Runtime planning "
                "authority must not return; use flagquantum/runtime/planner"
            )
    removed_noise_compilation = PACKAGE / "compilation" / "noise"
    if removed_noise_compilation.exists() and any(
        removed_noise_compilation.glob("*.py")
    ):
        errors.append(
            "flagquantum/compilation/noise: migrated noise compilation package "
            "must not return; use flagquantum/compiler/noise.py for lowering and "
            "the existing execution-plan product for planning"
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
        if relative.startswith(NON_JAX_EXECUTORS) and _imports_jax_executor(path):
            errors.append(
                f"{relative}: non-JAX execution backends must not import the JAX "
                "backend; cross-backend comparison belongs in Runtime Planner"
            )
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
            if (
                module == "qiskit"
                or module.startswith("qiskit.")
                or module == "qiskit_aer"
                or module.startswith("qiskit_aer.")
            ) and not relative.startswith(qiskit_import_allowed_prefixes):
                errors.append(
                    f"{relative}: Qiskit imports are isolated to "
                    "flagquantum.ecosystem.qiskit"
                )
            if (
                module == "pennylane" or module.startswith("pennylane.")
            ) and not relative.startswith(pennylane_import_allowed_prefixes):
                errors.append(
                    f"{relative}: PennyLane imports are isolated to "
                    "flagquantum.ecosystem.pennylane"
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

        if relative.startswith("flagquantum/compiler/"):
            forbidden = tuple(boundaries["compiler_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: compiler imports forbidden layer {module}"
                    )

        if relative.startswith("flagquantum/runtime/"):
            imports_compiler = any(
                any(
                    part in {"_compiler", "compilation", "compiler"}
                    for part in module.split(".")
                )
                for module, _ in imports
            )
            if imports_compiler:
                observed_runtime_compiler_imports.add(relative)
                if relative not in runtime_compiler_import_allowed:
                    errors.append(
                        f"{relative}: new Runtime dependency on Compiler is "
                        "forbidden; move the shared contract to Core"
                    )

        if relative.startswith("flagquantum/agent/"):
            forbidden = tuple(northbound_boundaries.get("agent_forbidden", ()))
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: agent service imports forbidden implementation "
                        f"layer {module}"
                    )

        if relative.startswith("flagquantum/noise/"):
            forbidden = tuple(boundaries["noise_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: noise semantics import forbidden layer {module}"
                    )

        if relative.startswith("flagquantum/runtime/trajectories/"):
            forbidden = tuple(boundaries["trajectory_runtime_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: trajectory runtime imports forbidden layer {module}"
                    )

        if (
            relative.startswith("flagquantum/simulation/")
            and relative not in simulation_runtime_import_allowed
        ):
            forbidden = tuple(boundaries["simulation_forbidden"])
            for module, _ in imports:
                if any(part in module.split(".") for part in forbidden):
                    errors.append(
                        f"{relative}: simulation primitive imports forbidden layer {module}"
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

    stale_runtime_compiler_allowances = (
        runtime_compiler_import_allowed - observed_runtime_compiler_imports
    )
    for relative in sorted(stale_runtime_compiler_allowances):
        errors.append(
            f"{relative}: stale Runtime -> Compiler migration allowance must be removed"
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
