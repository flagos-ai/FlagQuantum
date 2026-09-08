from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
INNER_LAYERS = (
    ROOT / "flagquantum/core",
    ROOT / "flagquantum/runtime",
    ROOT / "flagquantum/simulation",
)
EXTERNAL_OBJECT_FRAMEWORKS = {
    "braket",
    "cirq",
    "cuda_quantum",
    "cudaq",
    "pennylane",
    "qiskit",
    "qiskit_aer",
    "quark",
}


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(path for root in INNER_LAYERS for path in root.rglob("*.py")))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            imported.append(node.args[0].value)
    return tuple(imported)


def test_external_sdk_imports_do_not_enter_inner_layers() -> None:
    leaks = []
    for path in _python_files():
        for module in _imports(path):
            if module.split(".", 1)[0] in EXTERNAL_OBJECT_FRAMEWORKS:
                leaks.append(f"{path.relative_to(ROOT)} -> {module}")

    assert leaks == []


def test_jax_objects_are_confined_to_the_optional_kernel_boundary() -> None:
    leaks = []
    runtime_boundary = ROOT / "flagquantum/runtime/backends/jax"
    simulation_boundary = ROOT / "flagquantum/simulation/jax"
    for path in _python_files():
        for module in _imports(path):
            if (
                module.split(".", 1)[0] in {"jax", "jaxlib"}
                and runtime_boundary not in path.parents
                and simulation_boundary not in path.parents
            ):
                leaks.append(f"{path.relative_to(ROOT)} -> {module}")

    assert leaks == []


def test_runtime_does_not_import_ecosystem_adapters() -> None:
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in _python_files()
        if any(
            module.startswith("flagquantum.ecosystem")
            or module.lstrip(".").startswith("ecosystem")
            for module in _imports(path)
        )
    }

    assert importers == set()


def test_external_framework_names_are_not_part_of_owned_ir_type_annotations() -> None:
    owned_contracts = (
        ROOT / "flagquantum/core",
        ROOT / "flagquantum/runtime/execution_plan.py",
        ROOT / "flagquantum/runtime/contracts.py",
        ROOT / "flagquantum/runtime/result.py",
    )
    offenders = []
    for candidate in owned_contracts:
        paths = (candidate,) if candidate.is_file() else tuple(candidate.rglob("*.py"))
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    annotation = ast.unparse(node.annotation).lower()
                elif isinstance(node, ast.arg) and node.annotation is not None:
                    annotation = ast.unparse(node.annotation).lower()
                elif (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.returns
                ):
                    annotation = ast.unparse(node.returns).lower()
                else:
                    continue
                if any(name in annotation for name in EXTERNAL_OBJECT_FRAMEWORKS):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}"
                    )

    assert offenders == []
