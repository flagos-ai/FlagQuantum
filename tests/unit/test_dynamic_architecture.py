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
