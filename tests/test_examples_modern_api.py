from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
LEGACY_EXAMPLE_PATTERNS = {
    "legacy quantum device": re.compile(r"\b(?:Distributed)?QuantumDevice\b"),
    "legacy encoder": re.compile(r"\bGeneralEncoder\b"),
    "legacy measurement helper": re.compile(r"\bmeasure_allZ\b"),
    "legacy module gate": re.compile(r"\bfq\.(?:CX|RY)\b"),
    "legacy PyTorch adapter": re.compile(r"\bQuantumTorchLayer\b"),
    "internal runtime import": re.compile(r"\bfrom\s+flagquantum\.runtime\."),
    "legacy user CLI": re.compile(r"['\"]--n-wires(?:=|['\"])"),
    "legacy Circuit size keyword": re.compile(
        r"\bfq\.Circuit\s*\([^)]*\bn_wires\s*=", re.DOTALL
    ),
}


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


@pytest.mark.unit
def test_examples_do_not_reintroduce_legacy_device_api() -> None:
    violations: list[str] = []
    for path in sorted(EXAMPLES.rglob("*")):
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8")
        elif path.suffix == ".ipynb":
            source = _notebook_code(path)
        else:
            continue
        for label, pattern in LEGACY_EXAMPLE_PATTERNS.items():
            if pattern.search(source):
                violations.append(f"{path.relative_to(ROOT)}: {label}")
    assert not violations, "\n".join(violations)


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        "00_understanding_states.ipynb",
        "01_basic_operations.ipynb",
        "02_measurement.ipynb",
        "03_parameterized_gates.ipynb",
        "04_quantum_circuit_builder.ipynb",
        "05_quantum_machine_learning.ipynb",
        "06_vqe_statevector.ipynb",
        "07_runtime_selection_statevector_mps_tn.ipynb",
        "08_pytorch_jax_qml_layer.ipynb",
        "09_gradient_precision_speed_benchmark.ipynb",
    ],
)
def test_modern_intro_notebook_executes(name: str) -> None:
    if (
        name == "08_pytorch_jax_qml_layer.ipynb"
        and importlib.util.find_spec("jax") is None
    ):
        pytest.skip("the JAX tutorial requires the optional jax dependency")
    namespace = {"__name__": "__notebook__"}
    path = EXAMPLES / "tutorials" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", ()))
        exec(compile(code, f"{path}:cell-{index}", "exec"), namespace)


@pytest.mark.unit
def test_tutorial_notebook_inventory_and_outputs_are_curated() -> None:
    tutorial_dir = EXAMPLES / "tutorials"
    expected = {
        f"{index:02d}_{suffix}.ipynb"
        for index, suffix in enumerate(
            (
                "understanding_states",
                "basic_operations",
                "measurement",
                "parameterized_gates",
                "quantum_circuit_builder",
                "quantum_machine_learning",
                "vqe_statevector",
                "runtime_selection_statevector_mps_tn",
                "pytorch_jax_qml_layer",
                "gradient_precision_speed_benchmark",
            )
        )
    }
    assert {path.name for path in tutorial_dir.glob("*.ipynb")} == expected
    for path in sorted(tutorial_dir.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        stored_outputs = [
            output for cell in notebook["cells"] for output in cell.get("outputs", ())
        ]
        assert not stored_outputs, f"{path.relative_to(ROOT)} stores stale output"


@pytest.mark.integration
def test_quantum_transformer_feature_layer_is_differentiable() -> None:
    from examples.quantum_transformer import QuantumFeatureLayer

    layer = QuantumFeatureLayer(n_qubits=3, n_layers=1)
    inputs = torch.randn(2, 3, requires_grad=True)
    output = layer(inputs)
    assert output.shape == (2, 3)
    output.sum().backward()
    assert inputs.grad is not None
    assert layer.angles.grad is not None


@pytest.mark.unit
def test_experimental_mps_example_boundary_is_importable() -> None:
    removed = {
        "execute_torch_distributed_mps_forward",
        "execute_torch_distributed_mps_reverse",
        "site_sharded_z_zz_observations",
        "reset_mps_site_kernel_stats",
        "mps_site_kernel_stats",
    }
    assert removed.isdisjoint(fq.experimental.distributed.__all__)
    for name in removed:
        with pytest.raises(AttributeError):
            getattr(fq.experimental.distributed, name)
