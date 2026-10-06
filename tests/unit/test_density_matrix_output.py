"""A density-matrix output keeps the named qubits and traces the rest out.

The reductions here are checked against a formulation written independently of
the implementation: the state is contracted with its own conjugate through
``einsum``, with each traced qubit's row and column index identified so the
pair is summed together. That is a different code path from the implementation's
``torch.diagonal`` contraction and would disagree with it if either mishandled
which index belongs to which qubit.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.noise import NoiseModel, ReadoutError, depolarizing_channel
from flagquantum.runtime.noise_registry import (
    noisy_density_matrix as registry_noisy_density_matrix,
)

pytestmark = pytest.mark.unit

_ROW_LETTERS = "abcdefghij"
_COLUMN_LETTERS = "ABCDEFGHIJ"


def _reference_reduction(state: torch.Tensor, keep: tuple[int, ...]) -> torch.Tensor:
    """Reduce a statevector to ``keep``, in that order, by explicit contraction."""

    vector = state.reshape(-1)
    n_qubits = int(round(math.log2(vector.numel())))
    traced = [qubit for qubit in range(n_qubits) if qubit not in keep]
    row = list(_ROW_LETTERS[:n_qubits])
    column = list(_COLUMN_LETTERS[:n_qubits])
    for qubit in traced:
        column[qubit] = row[qubit]
    output = "".join(row[qubit] for qubit in keep) + "".join(
        column[qubit] for qubit in keep
    )
    return torch.einsum(
        f"{''.join(row)},{''.join(column)}->{output}",
        vector.reshape([2] * n_qubits),
        vector.conj().reshape([2] * n_qubits),
    ).reshape(2 ** len(keep), 2 ** len(keep))


def _asymmetric_circuit(n_qubits: int = 3) -> fq.Circuit:
    circuit = fq.Circuit(n_qubits, dtype=torch.complex128)
    for qubit, angle in enumerate((0.7, 1.1, 0.4)[:n_qubits]):
        circuit = circuit.ry(qubit, theta=angle)
    for qubit in range(n_qubits - 1):
        circuit = circuit.cx(qubit, qubit + 1)
    return circuit


@pytest.mark.parametrize("keep", [(0,), (1,), (2,), (0, 1), (1, 0), (2, 1), (0, 2)])
def test_every_reduction_matches_an_independent_contraction(
    keep: tuple[int, ...],
) -> None:
    circuit = _asymmetric_circuit()

    result = fq.run(circuit, outputs=fq.density_matrix(qubits=keep))
    expected = _reference_reduction(circuit.state()[0], keep)

    torch.testing.assert_close(result.density_matrix[0], expected)


@pytest.mark.parametrize("keep", [None, (0,), (2, 1)])
def test_every_execution_mode_returns_the_same_matrix(
    keep: tuple[int, ...] | None,
) -> None:
    """The reduction is a property of the state, not of the engine that held it."""

    circuit = _asymmetric_circuit()
    request = fq.density_matrix(qubits=keep)
    reference = fq.run(circuit, outputs=request).density_matrix

    for mode in ("auto", "statevector", "density_matrix", "mps"):
        options = fq.ExecutionOptions(mode=mode, precision="complex128")
        actual = fq.run(circuit, options=options, outputs=request).density_matrix
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=0)


def test_an_empty_selection_keeps_every_qubit_as_it_does_for_the_other_outputs() -> (
    None
):
    """An empty selection is the family's spelling of "no selection", not of "none".

    ``probabilities(qubits=())`` already returns every qubit's probabilities, so
    the density-matrix request inherits that reading rather than inventing a
    second one for itself. This is pinned here because a silently promoted
    empty selection is the kind of reading that is easy to change by accident.
    """

    circuit = _asymmetric_circuit(2)

    omitted = fq.run(circuit, outputs=fq.density_matrix()).density_matrix
    empty = fq.run(circuit, outputs=fq.density_matrix(())).density_matrix

    assert fq.probabilities(()).qubits == ()
    assert empty.shape == (1, 4, 4)
    torch.testing.assert_close(empty, omitted)


def test_a_reduction_is_a_trace_and_not_a_marginal() -> None:
    """Summing one index instead of contracting two gives a purity, not a state.

    A maximally entangled pair reduces to the maximally mixed qubit. Reporting
    the marginal of one qubit's index would instead return a pure-looking
    matrix whose trace is the purity of the whole state.
    """

    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)

    single = fq.run(circuit, outputs=fq.density_matrix(qubits=0)).density_matrix

    torch.testing.assert_close(
        single[0], torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.complex128)
    )
    assert single[0].diagonal().sum().real == pytest.approx(1.0)


def test_the_request_records_the_order_the_caller_named() -> None:
    circuit = _asymmetric_circuit(2)
    request = fq.density_matrix(qubits=(1, 0))

    result = fq.run(circuit, outputs=request)

    assert request.qubits == (1, 0)
    assert result.measurement(0).qubits == (1, 0)
    torch.testing.assert_close(
        result.density_matrix[0], _reference_reduction(circuit.state()[0], (1, 0))
    )


def test_an_observable_is_not_a_density_matrix_selection() -> None:
    with pytest.raises(TypeError, match="not an Observable"):
        fq.density_matrix(fq.Z(0))
    with pytest.raises(TypeError, match="does not accept an observable"):
        fq.OutputRequest("density_matrix", observable=fq.Z(0))


def test_a_qubit_outside_the_program_is_refused_by_name() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="outside a 2-qubit circuit"):
        fq.run(circuit, outputs=fq.density_matrix(qubits=2))
    with pytest.raises(ValueError, match="must be unique"):
        fq.run(circuit, outputs=fq.density_matrix(qubits=(0, 0)))


def test_readout_confusion_is_refused_rather_than_folded_in() -> None:
    """Readout acts on outcomes after the state, so a matrix cannot carry it."""

    circuit = fq.Circuit(1).x(0)
    model = NoiseModel().add_readout(0, ReadoutError(((0.9, 0.1), (0.1, 0.9))))

    with pytest.raises(ValueError, match="cannot include readout confusion"):
        fq.run(circuit, noise_model=model, outputs=fq.density_matrix())
    # The same model still governs a measurement of outcomes.
    assert fq.run(
        circuit, noise_model=model, outputs=fq.probabilities()
    ).probabilities.shape == (1, 2)


def test_a_statevector_to_matrix_step_is_bounded_and_says_what_to_do() -> None:
    circuit = fq.Circuit(12).x(0)

    with pytest.raises(ValueError, match="exceeds the supported limit of 10 qubits"):
        fq.run(circuit, outputs=fq.density_matrix())
    with pytest.raises(ValueError, match="mode='density_matrix'"):
        fq.run(circuit, outputs=fq.density_matrix())


def test_the_bound_does_not_apply_to_a_run_that_already_holds_the_matrix() -> None:
    """A density-mode run paid for the matrix, so reducing it is one pass over it."""

    circuit = fq.Circuit(12, dtype=torch.complex128).x(0)
    options = fq.ExecutionOptions(mode="density_matrix", precision="complex128")

    reduced = fq.run(circuit, options=options, outputs=fq.density_matrix(qubits=0))

    torch.testing.assert_close(
        reduced.density_matrix[0],
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128),
    )


def test_a_shot_value_without_a_sampled_output_is_still_refused() -> None:
    with pytest.raises(ValueError, match="requires fq.samples"):
        fq.run(fq.Circuit(1), outputs=fq.density_matrix(), shots=16)


def test_the_result_reports_an_absent_or_ambiguous_matrix() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(RuntimeError, match="does not contain density_matrix"):
        _ = fq.run(circuit, outputs=fq.probabilities()).density_matrix

    two = fq.run(
        circuit,
        outputs=(
            fq.density_matrix(qubits=0, name="first"),
            fq.density_matrix(qubits=1, name="second"),
        ),
    )
    with pytest.raises(RuntimeError, match="multiple density_matrix outputs"):
        _ = two.density_matrix
    assert two.measurement("first").qubits == (0,)
    assert two.measurement("second").qubits == (1,)


def test_a_named_matrix_is_selected_by_its_name() -> None:
    circuit = _asymmetric_circuit(2)

    result = fq.run(circuit, outputs=fq.density_matrix(qubits=0, name="reduced"))

    assert result.measurement("reduced").kind == "density_matrix"
    torch.testing.assert_close(
        result.measurement("reduced").value[0],
        _reference_reduction(circuit.state()[0], (0,)),
    )


def test_a_matrix_keeps_the_gradient_of_its_parameter() -> None:
    theta = torch.tensor(0.3, requires_grad=True)

    result = fq.run(fq.Circuit(1).ry(0, theta=theta), outputs=fq.density_matrix())

    result.density_matrix.real.sum().backward()
    torch.testing.assert_close(theta.grad, torch.cos(theta))


def test_a_noisy_matrix_agrees_between_the_two_routes() -> None:
    circuit = _asymmetric_circuit(3)
    model = NoiseModel().add("cx", depolarizing_channel(0.05, dtype=torch.complex128))
    options = fq.ExecutionOptions(mode="density_matrix", precision="complex128")

    analytic = fq.run(
        circuit, noise_model=model, outputs=fq.density_matrix(qubits=(2, 0))
    )
    held = fq.run(
        circuit,
        options=options,
        noise_model=model,
        outputs=fq.density_matrix(qubits=(2, 0)),
    )

    torch.testing.assert_close(
        analytic.density_matrix, held.density_matrix, atol=1e-10, rtol=0
    )
    assert analytic.density_matrix[0].diagonal().sum().real == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_a_circuit_keeps_its_precision_on_both_density_matrix_paths(
    dtype: torch.dtype,
) -> None:
    """The noisy path reads the circuit's precision, as the noiseless one does.

    The two paths previously disagreed: the noisy matrix was built at the
    process-wide runtime default, so at ``complex128`` it came back at
    ``complex64`` and no ``complex128`` operator could be contracted against it.
    """

    circuit = fq.Circuit(2, dtype=dtype).h(0).cx(0, 1)
    model = NoiseModel().add("cx", depolarizing_channel(0.05, dtype=dtype))

    assert circuit.density_matrix().dtype == dtype
    assert circuit.noisy_density_matrix(model).dtype == dtype

    # The method is the object-oriented spelling of the registry function, and
    # after the repair the two agree at the circuit's dtype instead of the method
    # silently answering at the process-wide default.
    torch.testing.assert_close(
        circuit.noisy_density_matrix(model),
        registry_noisy_density_matrix(circuit, model, dtype=dtype),
    )

    # An operator at the circuit's dtype contracts against the matrix, which the
    # silent downgrade made a hard error rather than a rounding difference.
    zz = torch.kron(
        torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64),
    ).to(dtype)
    correlation = torch.einsum("ij,ji->", zz, circuit.noisy_density_matrix(model)[0])
    assert correlation.real.item() == pytest.approx(0.8712, abs=1e-3)
