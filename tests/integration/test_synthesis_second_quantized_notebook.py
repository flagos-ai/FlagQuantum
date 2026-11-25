"""Independent acceptance of the complete native fermionic energy workflow."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
    build_program,
)
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.overview.analysis import (
    Workflow,
    run_workflow,
)
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.overview.matrix import (
    occupation_matrix,
    reference_circuit,
)
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.overview.optimization import (
    hamiltonian,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/chemistry/second_quantized_hamiltonian/overview"
)


@pytest.mark.parametrize("word", range(16))
def test_independent_ordered_fermions_and_mapped_expectation(word: int) -> None:
    matrix = occupation_matrix()
    occupied = tuple((word >> (3 - mode)) & 1 for mode in range(4))
    expected = (
        0.2 * occupied[0]
        - 0.1 * occupied[0] * occupied[1]
        + 0.3 * occupied[2] * occupied[3]
    )
    vector = torch.zeros(16, dtype=torch.complex128)
    vector[word] = 1
    torch.testing.assert_close(matrix @ vector, expected * vector, atol=1e-14, rtol=0)
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for mode, bit in enumerate(occupied):
        if bit:
            circuit.x(mode)
    result = fq.run(circuit, outputs=[fq.expectation(hamiltonian(), name="energy")])
    assert float(result.expectation("energy").reshape(())) == pytest.approx(
        expected, abs=1e-14
    )


@pytest.mark.parametrize(
    "parameters",
    [
        (0.0,) * 12,
        tuple(0.05 * (i + 1) for i in range(12)),
        tuple((-1) ** i * (0.13 + 0.11 * i) for i in range(12)),
    ],
)
def test_full_state_absolute_phase_mode_order_and_zero_workspace(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    actual = fq.run(program).statevector().flatten()
    reference = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
    result = fq.run(
        program.circuit, outputs=[fq.expectation(hamiltonian(), name="energy")]
    )
    expected = (reference.conj() @ (occupation_matrix() @ reference)).real
    torch.testing.assert_close(
        result.expectation("energy").reshape(()), expected, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0 and actual.numel() == 16


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual_native_optimization_best_reexecution_and_budget(
    workflow: Workflow,
) -> None:
    result = workflow.optimization
    assert len(result.parameters) == 12 and len(result.trace) == 200
    assert result.initial_energy == pytest.approx(0.3, abs=1e-14)
    assert result.best_energy == min(result.trace) < result.initial_energy
    assert 0 <= workflow.energy < 0.001
    assert workflow.energy == pytest.approx(result.best_energy, abs=1e-14)
    assert "budget exhausted" in result.termination
    vector = fq.run(reference_circuit(result.parameters)).statevector().flatten()
    expected = float((vector.conj() @ (occupation_matrix() @ vector)).real)
    assert workflow.energy == pytest.approx(expected, abs=1e-12)
    actual = fq.run(workflow.program).statevector().flatten()
    torch.testing.assert_close(actual, vector, atol=1e-12, rtol=0)
    sectors = dict(workflow.sectors)
    assert sum(sectors.values()) == pytest.approx(1, abs=1e-12)
    assert sectors[2] < 1 - 1e-3
    independent = {
        particles: sum(
            float(vector[word].abs().square())
            for word in range(16)
            if word.bit_count() == particles
        )
        for particles in range(5)
    }
    assert sectors == pytest.approx(independent, abs=1e-12)


def test_actual_four_cell_notebook_and_source_outputs() -> None:
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            assert "from classiq" not in "".join(cell["source"])
    namespace, count = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert count == 4
    workflow = namespace["workflow"]
    assert isinstance(workflow, Workflow)
    assert workflow.energy < workflow.optimization.initial_energy
    from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
        main as native_main,
    )
    from flagquantum import synth

    entry = namespace["variational_program"]
    assert isinstance(entry, type(native_main))
    for parameters in (
        (0.0,) * 12,
        tuple(0.05 * (i + 1) for i in range(12)),
        tuple((-1) ** i * (0.13 + 0.11 * i) for i in range(12)),
    ):
        inline = synth.synthesize(
            entry,
            classical_inputs={"params": list(parameters)},
            max_qubits=4,
            max_workspace=0,
        )
        actual = fq.run(inline).statevector().flatten()
        expected = fq.run(reference_circuit(parameters)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_resource_budgets_refuse(budget: str) -> None:
    from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
        main,
    )
    from flagquantum import synth
    from flagquantum.errors import PlanningError

    parameters: dict[str, float | list[float]] = {"params": [0.0] * 12}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=parameters, max_qubits=3)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=parameters, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=parameters, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=parameters, memory_limit_bytes=16)
