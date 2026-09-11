import ast
from pathlib import Path

from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime import dynamic
from flagquantum.runtime.dynamic import circuit, deployment, execution, result, routing
from flagquantum.runtime.dynamic.dialects import braket_iqm, openqasm3


def test_layered_dynamic_imports_preserve_public_builder_identity() -> None:
    assert dynamic.DynamicCircuit is circuit.DynamicCircuit
    assert dynamic.DynamicCircuit is DynamicCircuit
    assert dynamic.DynamicExecutionResult is result.DynamicExecutionResult
    assert dynamic.run_dynamic is execution.run_dynamic
    assert dynamic.route_dynamic_circuit is routing.route_dynamic_circuit
    assert (
        dynamic.create_dynamic_deployment_package
        is deployment.create_dynamic_deployment_package
    )
    assert dynamic.export_dynamic_qasm3 is openqasm3.export_dynamic_qasm3
    assert (
        dynamic.export_braket_iqm_dynamic_qasm3
        is braket_iqm.export_braket_iqm_dynamic_qasm3
    )


def test_dynamic_dialects_do_not_import_provider_or_execution_layers() -> None:
    dialect_root = (
        Path(__file__).parents[2] / "flagquantum" / "runtime" / "dynamic" / "dialects"
    )
    forbidden = {"providers", "deployment", "execution"}
    for path in dialect_root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(
            part in forbidden for imported in imports for part in imported.split(".")
        ), path


def test_dynamic_implementations_live_in_layer_modules() -> None:
    runtime_root = Path(__file__).parents[2] / "flagquantum" / "runtime" / "dynamic"

    def functions(path: Path) -> set[str]:
        return {
            node.name
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    standard = functions(runtime_root / "dialects" / "openqasm3.py")
    iqm = functions(runtime_root / "dialects" / "braket_iqm.py")
    execution_functions = functions(runtime_root / "execution.py")
    routing_functions = functions(runtime_root / "routing.py")
    deployment_functions = functions(runtime_root / "deployment.py")

    assert "export_dynamic_qasm3" in standard
    assert "export_dynamic_qasm3_for_backend" in standard
    assert "export_braket_iqm_dynamic_qasm3" in iqm
    assert "run_dynamic" in execution_functions
    assert "route_dynamic_circuit" in routing_functions
    assert "create_dynamic_deployment_package" in deployment_functions
    assert not (runtime_root / "_implementation.py").exists()


def test_dynamic_layers_do_not_reference_removed_monolith() -> None:
    runtime_root = Path(__file__).parents[2] / "flagquantum" / "runtime" / "dynamic"
    for relative in (
        "circuit.py",
        "result.py",
        "_conditions.py",
        "execution.py",
        "routing.py",
        "deployment.py",
        "dialects/openqasm3.py",
        "dialects/braket_iqm.py",
    ):
        source = (runtime_root / relative).read_text()
        assert "_implementation" not in source, relative


def test_dynamic_execution_layer_has_no_transport_or_dialect_dependencies() -> None:
    path = (
        Path(__file__).parents[2]
        / "flagquantum"
        / "runtime"
        / "dynamic"
        / "execution.py"
    )
    source = path.read_text()
    for forbidden in ("deployment", "providers", "dialects", "routing"):
        assert forbidden not in source
