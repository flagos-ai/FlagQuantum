import ast
from pathlib import Path

import flagquantum as fq
from flagquantum.runtime import dynamic
from flagquantum.runtime.dynamic import circuit, deployment, execution, result, routing
from flagquantum.runtime.dynamic.dialects import braket_iqm, openqasm3


def test_layered_dynamic_imports_preserve_experimental_api_identity() -> None:
    assert dynamic.DynamicCircuit is circuit.DynamicCircuit
    assert dynamic.DynamicCircuit is fq.experimental.DynamicCircuit
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
        Path(__file__).parents[2]
        / "flagquantum"
        / "runtime"
        / "dynamic"
        / "dialects"
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
            part in forbidden
            for imported in imports
            for part in imported.split(".")
        ), path


def test_dialect_implementations_live_outside_internal_monolith() -> None:
    runtime_root = Path(__file__).parents[2] / "flagquantum" / "runtime" / "dynamic"

    def functions(path: Path) -> set[str]:
        return {
            node.name
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    implementation = functions(runtime_root / "_implementation.py")
    standard = functions(runtime_root / "dialects" / "openqasm3.py")
    iqm = functions(runtime_root / "dialects" / "braket_iqm.py")

    assert "export_dynamic_qasm3" in standard
    assert "export_dynamic_qasm3_for_backend" in standard
    assert "export_braket_iqm_dynamic_qasm3" in iqm
    assert not {
        "export_dynamic_qasm3",
        "export_dynamic_qasm3_for_backend",
        "export_braket_iqm_dynamic_qasm3",
    } & implementation


def test_core_dynamic_types_and_conditions_do_not_depend_on_monolith() -> None:
    runtime_root = Path(__file__).parents[2] / "flagquantum" / "runtime" / "dynamic"
    for relative in (
        "circuit.py",
        "result.py",
        "_conditions.py",
        "dialects/openqasm3.py",
        "dialects/braket_iqm.py",
    ):
        source = (runtime_root / relative).read_text()
        assert "_implementation" not in source, relative

    implementation = ast.parse((runtime_root / "_implementation.py").read_text())
    class_names = {
        node.name for node in ast.walk(implementation) if isinstance(node, ast.ClassDef)
    }
    function_names = {
        node.name
        for node in ast.walk(implementation)
        if isinstance(node, ast.FunctionDef)
    }
    assert "DynamicCircuit" not in class_names
    assert "DynamicExecutionResult" not in class_names
    assert "_instruction_conditions" not in function_names
    assert "_classical_width" not in function_names
