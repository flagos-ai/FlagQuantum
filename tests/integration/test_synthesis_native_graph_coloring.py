"""The complete native coloring model matches frozen source and Circuit states."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.min_graph_coloring import (
    main as coloring,
)
from examples.synthesis.applications.optimization.min_graph_coloring.comparisons import (
    circuit as reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from tools.validate_qmod_phase_factors import ArithmeticOracle

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/applications/optimization/min_graph_coloring/min_graph_coloring.qmod"
)


@pytest.mark.parametrize("binding", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_coloring_original_and_native(
    binding: int, objective: Literal["width", "depth"]
) -> None:
    params = reference.BINDINGS[binding]
    native = synth.synthesize(
        coloring.main,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    compare_states(native, reference.reference_circuit(params), ("colors",))
    compare_states(
        source, reference.source_reference_circuit(params), reference.SOURCE_FIELDS
    )
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 15
    assert native.report["workspace_qubits"] == 0


def test_every_assignment_preserves_source_arithmetic_and_constraint_factors() -> None:
    oracle = ArithmeticOracle(SOURCE.read_text())
    factors = reference.cost_factors()
    for basis in range(32768):
        bits = tuple((basis >> bit) & 1 for bit in range(15))
        expected = reference.cost(bits)
        assert oracle.cost(bits) == expected
        actual = sum(
            table[sum(bits[bit] << i for i, bit in enumerate(support))]
            for support, table in factors
        )
        assert actual == expected


def test_twice_counted_edges_and_absolute_constant_are_retained() -> None:
    assert reference.cost((0,) * 15) == -50
    # One color shared by all five vertices: five conflicts are doubled then squared.
    assert reference.cost((1,) * 5 + (0,) * 10) == -1001
    assert reference.cost((1,) * 15) == -9203


def test_zero_angles_and_binding_validation() -> None:
    native = coloring.build_program((0.0,) * 12)
    circuit = fq.Circuit(15, dtype=torch.complex128)
    for wire in range(15):
        circuit.h(wire)
    compare_states(native, circuit, ("colors",))
    with pytest.raises(ValueError, match="parameters"):
        coloring.build_program((0.1,))
    with pytest.raises(ValueError, match="finite"):
        coloring.build_program((float("nan"),) * 12)


@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_coloring_respects_actual_resource_budgets(budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            coloring.main,
            classical_inputs={"params": reference.BINDINGS[0]},
            max_qubits=14 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )
