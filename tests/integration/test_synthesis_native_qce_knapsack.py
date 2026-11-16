"""QCE Knapsack keeps unsigned numeric semantics and complete helper composition."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit import (
    cost,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii import (
    main as tutorial,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.comparisons import (
    circuit as reference,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II.qmod"
)


@pytest.mark.parametrize("binding", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_original_native_and_independent_numeric_circuit(
    binding: int, objective: Literal["width", "depth"]
) -> None:
    params = reference.BINDINGS[binding]
    native = synth.synthesize(
        tutorial.main,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    circuit = reference.reference_circuit(params)
    compare_states(native, circuit, ("a", "b"))
    compare_states(source, circuit, reference.SOURCE_FIELDS)
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 5
    assert native.report["workspace_qubits"] == 0


def test_all_numeric_inputs_and_inclusive_boundary() -> None:
    for a in range(8):
        for b in range(4):
            bits = tuple((((a << 2) | b) >> bit) & 1 for bit in range(5))
            assert cost(bits) == (-3 * a - 5 * b if 2 * a + 3 * b <= 12 else 0)
    for a, b, expected in ((6, 0, -18), (3, 2, -19), (4, 2, 0)):
        bits = tuple((((a << 2) | b) >> bit) & 1 for bit in range(5))
        assert cost(bits) == expected


def test_zero_angles_and_real_binding_validation() -> None:
    program = tutorial.build_program((0.0,) * 6)
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    compare_states(program, circuit, ("a", "b"))
    with pytest.raises(ValueError, match="parameters"):
        tutorial.build_program((0.1,))
    with pytest.raises(ValueError, match="finite"):
        tutorial.build_program((float("nan"),) * 6)


@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_numeric_program_budget_refusals(budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            tutorial.main,
            classical_inputs={"params": reference.BINDINGS[0]},
            max_qubits=4 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )
