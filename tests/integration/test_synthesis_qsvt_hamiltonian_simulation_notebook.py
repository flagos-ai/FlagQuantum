"""Check both original QSVT parity sequences and full evolution states."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
    even_phases,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", [132, 247])
def test_original_evolution_and_full_state(seed: int) -> None:
    value = workflow.run(seed)
    assert value.model.pauli_counts == (4, 4)
    assert value.model.alpha == 0.75
    assert value.resources.n_qubits == 6
    assert value.cosine.phases.numel() == 33
    assert value.sine.phases.numel() == 34
    assert max(value.cosine.residual, value.sine.residual) < 1e-10
    assert max(value.block_error, value.full_state_error) < 1e-10
    assert value.evolution_error < workflow.EPS
    assert 0.24 < value.success_probability < 0.26
    torch.testing.assert_close(value.state, value.expected, atol=1e-7, rtol=0)
    # Z on the source's bit one contributes to the two high bits.
    diagonal = value.model.target.diag().real
    torch.testing.assert_close(
        diagonal,
        torch.tensor([0.7, 0.3, 0.1, 0.5], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )


def test_even_fit_derivative_at_center() -> None:
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
        response,
        signal,
    )

    x = torch.tensor([0.13, 0.51, 0.88], dtype=torch.float64)
    half = torch.tensor([0.12, -0.16, 0.21], dtype=torch.float64)
    values, jacobian = even_phases.evaluate(signal(x), half)
    torch.testing.assert_close(
        values, response(x, torch.cat((half, half[:-1].flip((0,))))), atol=1e-12, rtol=0
    )
    perturbation = torch.tensor([0, 0, 1e-6], dtype=torch.float64)
    plus, _ = even_phases.evaluate(signal(x), half + perturbation)
    minus, _ = even_phases.evaluate(signal(x), half - perturbation)
    torch.testing.assert_close(
        jacobian[:, -1], (plus - minus) / 2e-6, atol=1e-9, rtol=0
    )


def test_source_stages() -> None:
    directory = Path(workflow.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    proof = json.loads((directory / "provenance.json").read_text())
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 12
    assert all(c["execution_count"] is None for c in cells)


@pytest.fixture(scope="module")
def fitted_source() -> workflow.Result:
    return workflow.run()


@pytest.mark.parametrize("seed", [132, 247, None])
def test_native_sequence_preserves_absolute_state_and_clean_work(
    fitted_source: workflow.Result, seed: int | None
) -> None:
    import flagquantum as fq
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
        quantum,
    )
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
        unitary,
    )

    amplitudes = (
        workflow.input_state(seed)
        if seed is not None
        else torch.tensor([1, -2, 3, -4], dtype=torch.float64) / 30**0.5
    )
    program = quantum.build_program(
        fitted_source.cosine, fitted_source.sine, amplitudes
    )
    result = fq.run(program)
    full = result.statevector().reshape(64, 16)
    independent = workflow.full_reference(
        unitary(fitted_source.model),
        fitted_source.cosine,
        fitted_source.sine,
        amplitudes,
    )
    torch.testing.assert_close(full[:, 0], independent, atol=1e-10, rtol=0)
    assert float(full[:, 1:].abs().square().sum()) < 1e-24
    # Work bits are distinct from the source's postselected response flags.
    assert program.circuit.n_wires == 10
    assert program.report["workspace_qubits"] == 4
    assert program.report["resources"].n_operations <= 3500
    selected = full[:4, 0] * (2 / workflow.SCALE)
    exact = torch.linalg.matrix_exp(-1j * workflow.TIME * fitted_source.model.target)
    assert (
        float((selected - exact @ amplitudes.to(torch.complex128)).norm())
        < workflow.EPS
    )
    named = program.probabilities(result, registers=("parity", "branch", "selection"))
    assert named[(0, 0, 0)] == pytest.approx(float(full[:4, 0].abs().square().sum()))


@pytest.mark.parametrize("column", range(16))
def test_native_signal_all_columns_and_bit_order(column: int) -> None:
    import flagquantum as fq
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.quantum import (
        signal,
    )
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
        unitary,
    )
    from flagquantum import synth
    from flagquantum.synth import (
        Output,
        QArray,
        QBit,
        QNum,
        allocate,
        inplace_prepare_amplitudes,
        prepare_amplitudes,
        qfunc,
    )

    @qfunc
    def probe(
        selector_values: list[float],
        values: list[float],
        weights: list[float],
        selection: Output[QNum[2]],
        data: Output[QArray[QBit]],
    ) -> None:
        allocate(selection)
        inplace_prepare_amplitudes(selector_values, 0, selection)
        prepare_amplitudes(values, 0, data)
        signal(weights, selection, data)

    program = synth.synthesize(
        probe,
        classical_inputs={
            "selector_values": torch.eye(4)[column // 4].tolist(),
            "values": torch.eye(4)[column % 4].tolist(),
            "weights": [(value / workflow.ALPHA) ** 0.5 for value in workflow.WEIGHTS],
        },
        max_qubits=8,
        max_workspace=4,
    )
    full = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        full[:, 0], unitary(workflow.encoding())[:, column], atol=1e-12, rtol=0
    )
    assert float(full[:, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize(
    "constraints", [{"max_workspace": 0}, {"max_qubits": 6}, {"max_operations": 3000}]
)
def test_native_sequence_refuses_insufficient_budget(
    fitted_source: workflow.Result, constraints: dict[str, int]
) -> None:
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
        quantum,
    )
    from flagquantum import synth

    with pytest.raises(synth.SynthesisError):
        quantum.build_program(
            fitted_source.cosine,
            fitted_source.sine,
            workflow.input_state(),
            **constraints,
        )


def test_native_joint_samples_keep_named_order_and_conditional_denominator(
    fitted_source: workflow.Result,
) -> None:
    import flagquantum as fq
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
        quantum,
    )

    program = quantum.build_program(
        fitted_source.cosine, fitted_source.sine, workflow.input_state()
    )
    names = ("parity", "branch", "selection", "data")
    wires = tuple(wire for name in names for wire in program.registers[name])
    result = fq.run(
        program.circuit,
        outputs=fq.samples(qubits=wires, name="joint"),
        options=fq.ExecutionOptions(shots=4096, seed=37),
    )
    rows = result.measurement("joint").value[0]
    accepted = rows[:, :4].sum(dim=1) == 0
    count = int(accepted.sum())
    assert count / len(rows) == pytest.approx(
        fitted_source.success_probability, abs=0.04
    )
    # Data[0] has the high tensor-product significance in the named view.
    values = 2 * rows[accepted, 4] + rows[accepted, 5]
    histogram = torch.bincount(values, minlength=4).double() / count
    expected = fitted_source.expected.abs().square()
    torch.testing.assert_close(histogram, expected, atol=0.07, rtol=0)
    assert float(histogram.sum()) == pytest.approx(1)
    assert float((torch.bincount(values, minlength=4) / len(rows)).sum()) < 0.3
    # Reverse the requested port order on a deterministic basis state to detect
    # a convention error that a symmetric Bell distribution cannot reveal.
    basis = fq.Circuit(3).x(0)
    ordered = fq.run(
        basis,
        outputs=fq.counts(qubits=(2, 0, 1), name="reordered"),
        options=fq.ExecutionOptions(shots=7, seed=37),
    )
    assert ordered.measurement("reordered").value == [{"010": 7}]


@pytest.mark.parametrize("stream", ["cosine", "sine"])
def test_native_sequence_rejects_missing_source_phase(
    fitted_source: workflow.Result, stream: str
) -> None:
    from dataclasses import replace

    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
        quantum,
    )

    cosine, sine = fitted_source.cosine, fitted_source.sine
    if stream == "cosine":
        cosine = replace(cosine, phases=cosine.phases[:-1])
    else:
        sine = replace(sine, phases=sine.phases[:-1])
    with pytest.raises(ValueError, match="33/34"):
        quantum.build_program(cosine, sine, workflow.input_state())
