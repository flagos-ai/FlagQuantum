"""Concrete native Grover programs retain original oracle and reflection phases."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.algorithms.search_and_optimization.grover.grover_max_cut import (
    main as cut,
)
from examples.synthesis.algorithms.search_and_optimization.grover.grover_max_cut.comparisons import (
    circuit as cut_ref,
)
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover import (
    main as sat,
)
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.comparisons import (
    circuit as sat_ref,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
CORPUS = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/classiq_qmod/corpus/original/algorithms/search_and_optimization/grover"
)


@pytest.mark.parametrize("model", ["sat", "cut"])
@pytest.mark.parametrize("r", [0, 1, 2])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_native_and_original_grover_states(
    model: str, r: int, objective: Literal["width", "depth"]
) -> None:
    entry = sat.main if model == "sat" else cut.main
    source = CORPUS / ("3_sat_grover.qmod" if model == "sat" else "grover_max_cut.qmod")
    reference = (
        sat_ref.reference_circuit(r) if model == "sat" else cut_ref.reference_circuit(r)
    )
    native = synth.synthesize(entry, classical_inputs={"r": r}, objective=objective)
    original = synth.synthesize(source, classical_inputs={"r": r}, objective=objective)
    compare_states(native, reference, ("q",))
    compare_states(original, reference, ("x",) if model == "sat" else ("v",))
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits >= (4 if model == "sat" else 5)


def test_all_sat_assignments_and_maxcut_threshold_boundary() -> None:
    satisfying = [
        basis
        for basis in range(16)
        if sat_ref.marked(tuple((basis >> bit) & 1 for bit in range(4)))
    ]
    assert satisfying == [8, 12]
    observed_counts = set()
    for basis in range(32):
        bits = tuple((basis >> bit) & 1 for bit in range(5))
        edge_count = (
            (bits[0] ^ bits[1])
            + (bits[0] ^ bits[2])
            + (bits[1] ^ bits[2])
            + (bits[1] ^ bits[3])
            + (bits[2] ^ bits[4])
            + (bits[3] ^ bits[4])
        )
        assert cut_ref.cut_size(bits) == edge_count
        assert cut_ref.marked(bits) == (edge_count <= 4)
        observed_counts.add(edge_count)
    assert {4, 5} <= observed_counts


@pytest.mark.parametrize("model", ["sat", "cut"])
def test_grover_iteration_validation_and_explicit_zero_workspace(model: str) -> None:
    build = sat.build_program if model == "sat" else cut.build_program
    entry = sat.main if model == "sat" else cut.main
    reference = (
        sat_ref.reference_circuit(2) if model == "sat" else cut_ref.reference_circuit(2)
    )
    for invalid in (-1, 33, True):
        with pytest.raises(ValueError, match="iteration"):
            build(invalid)
    program = synth.synthesize(entry, classical_inputs={"r": 2}, max_workspace=0)
    compare_states(program, reference, ("q",))
    assert program.report["workspace_qubits"] == 0
    with pytest.raises(ValueError):
        synth.synthesize(entry, classical_inputs={"r": 1.5})


@pytest.mark.parametrize("model", ["sat", "cut"])
@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_grover_actual_resource_ceiling_refusals(model: str, budget: str) -> None:
    entry = sat.main if model == "sat" else cut.main
    width = 4 if model == "sat" else 5
    with pytest.raises(ValueError):
        synth.synthesize(
            entry,
            classical_inputs={"r": 2},
            max_qubits=width - 1 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )
