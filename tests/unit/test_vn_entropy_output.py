"""A user asks an execution for the von Neumann entropy of a named subsystem.

These are the workflows the contract's recorded numbers describe, written the way
a user writes them: build a program, ask for the entropy of a qubit selection, and
read it off the result. The reductions are checked against a formulation written
independently of the implementation, so the two would disagree if either mishandled
which qubit belongs to which side of the cut.

The shortest path -- build a Bell pair, ask for one qubit, read the number -- is
pinned by name in ``test_vn_entropy_output_contract.py``, where it sits beside the
tamper tests; it is not repeated here.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.noise import (
    NoiseModel,
    amplitude_damping_channel,
    depolarizing_channel,
)

pytestmark = pytest.mark.unit

_MODES = ("statevector", "mps", "tensor_network", "density_matrix")
_ROW_LETTERS = "abcdefghij"
_COLUMN_LETTERS = "ABCDEFGHIJ"


def _chain(n_qubits: int, *, dtype: torch.dtype = torch.complex128) -> fq.Circuit:
    """A program whose reduced states are asymmetric, so an order mistake shows."""

    circuit = fq.Circuit(n_qubits, dtype=dtype)
    for qubit in range(n_qubits):
        circuit = circuit.ry(qubit, theta=0.5 + 0.4 * qubit)
    for qubit in range(n_qubits - 1):
        circuit = circuit.cx(qubit, qubit + 1)
    return circuit


def _reference_entropy(state: torch.Tensor, keep: tuple[int, ...]) -> float:
    """The entropy of ``keep``, by explicit reduction and diagonalisation."""

    vector = state.reshape(-1)
    n_qubits = int(round(math.log2(vector.numel())))
    traced = [qubit for qubit in range(n_qubits) if qubit not in keep]
    row = list(_ROW_LETTERS[:n_qubits])
    column = list(_COLUMN_LETTERS[:n_qubits])
    for qubit in traced:
        column[qubit] = row[qubit]
    output = "".join(row[q] for q in keep) + "".join(column[q] for q in keep)
    reduced = torch.einsum(
        f"{''.join(row)},{''.join(column)}->{output}",
        vector.reshape([2] * n_qubits),
        vector.conj().reshape([2] * n_qubits),
    ).reshape(2 ** len(keep), 2 ** len(keep))
    spectrum = torch.linalg.eigvalsh(reduced).real
    positive = spectrum[spectrum > 1e-30]
    return float(-(positive * torch.log(positive)).sum())


def test_a_base_of_two_answers_in_bits_and_the_default_answers_in_nats() -> None:
    """``log_base`` scales the answer, and the request keeps what the caller wrote."""

    circuit = _chain(3)
    request = fq.vn_entropy([0, 1], log_base=2)
    assert request.log_base == 2

    natural = fq.run(circuit, outputs=fq.vn_entropy([0, 1]))
    bits = fq.run(circuit, outputs=request)
    assert float(bits.vn_entropy[0]) == pytest.approx(
        float(natural.vn_entropy[0]) / math.log(2.0), abs=1e-12
    )

    # The frozen request keeps `None` as `None` rather than folding it to a number, so a
    # result's request can be compared with the call that produced it.
    assert fq.vn_entropy([0]).log_base is None


def test_the_same_number_comes_back_from_every_mode_that_holds_a_state() -> None:
    """A user who changes the execution mode does not change the answer."""

    circuit = _chain(3)
    amplitudes = fq.run(circuit).to_statevector().reshape(-1).to(torch.complex128)
    expected = _reference_entropy(amplitudes, (1, 2))

    readings = {
        mode: float(
            fq.run(
                circuit,
                outputs=fq.vn_entropy([1, 2]),
                options=fq.ExecutionOptions(mode=mode),
            ).vn_entropy[0]
        )
        for mode in _MODES
    }
    for mode, reading in readings.items():
        assert reading == pytest.approx(expected, abs=1e-12), mode


def test_the_selection_is_the_side_that_was_asked_about() -> None:
    """On a globally pure state the two sides agree, so the mixed state is the real test."""

    bell = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    # A Bell pair is pure, so one qubit and its complement carry the same entropy; this
    # row cannot tell which side was asked about, which is why it is not the only one.
    assert float(
        fq.run(bell, outputs=fq.vn_entropy([0])).vn_entropy[0]
    ) == pytest.approx(
        float(fq.run(bell, outputs=fq.vn_entropy([1])).vn_entropy[0]), abs=1e-12
    )

    # A damping channel makes the state mixed, so the two sides separate and the reading
    # says which one was named. The gate builds this channel the same way, including the
    # explicit complex128: the channel's rate is widened through the default complex
    # dtype otherwise, so a rate that is not exactly representable would be narrowed.
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0)
    model = NoiseModel().add(
        "h", amplitude_damping_channel(0.25, dtype=torch.complex128), qubits=0
    )
    damped = fq.run(circuit, outputs=fq.vn_entropy([0]), noise_model=model)
    undamped = fq.run(circuit, outputs=fq.vn_entropy([1]), noise_model=model)
    assert float(damped.vn_entropy[0]) == pytest.approx(0.19646697846469818, abs=1e-12)
    assert float(undamped.vn_entropy[0]) == pytest.approx(
        2.2204460492503126e-16, abs=1e-12
    )
    assert float(damped.vn_entropy[0]) - float(undamped.vn_entropy[0]) > 1e-6


def test_the_entropy_is_available_where_the_matrix_is_not() -> None:
    """The compressed route answers a selection the dense route refuses by capacity."""

    circuit = fq.Circuit(11).h(0)
    for qubit in range(10):
        circuit = circuit.cx(qubit, qubit + 1)

    with pytest.raises(ValueError, match="exceeds the supported limit"):
        fq.run(circuit, outputs=fq.density_matrix([0]))

    compressed = fq.run(
        circuit,
        outputs=fq.vn_entropy([0]),
        options=fq.ExecutionOptions(mode="mps"),
    )
    assert float(compressed.vn_entropy[0]) == pytest.approx(
        0.6931471805599454, abs=1e-6
    )


def test_a_descending_or_interior_selection_still_answers() -> None:
    """The cheap route needs a leading run; the selection itself may be any subset."""

    circuit = _chain(4)
    amplitudes = fq.run(circuit).to_statevector().reshape(-1).to(torch.complex128)

    for selection in ((3, 2), (1, 2), (0, 3)):
        expected = _reference_entropy(amplitudes, selection)
        for mode in _MODES:
            reading = float(
                fq.run(
                    circuit,
                    outputs=fq.vn_entropy(list(selection)),
                    options=fq.ExecutionOptions(mode=mode),
                ).vn_entropy[0]
            )
            assert reading == pytest.approx(expected, abs=1e-12), (selection, mode)


def test_a_user_sees_which_request_the_package_will_not_answer() -> None:
    """Every refusal a user can meet, asserted on the sentence rather than on the class alone."""

    bell = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    with pytest.raises(ValueError, match="requires at least one qubit"):
        fq.vn_entropy([])
    with pytest.raises(ValueError, match="must not be 1"):
        fq.vn_entropy([0], log_base=1)
    with pytest.raises(ValueError, match="must be positive"):
        fq.vn_entropy([0], log_base=-2)
    with pytest.raises(TypeError, match="must be a number"):
        fq.vn_entropy([0], log_base="2")
    with pytest.raises(TypeError, match="does not accept a log_base"):
        fq.OutputRequest("probabilities", (0,), log_base=2)
    with pytest.raises(ValueError, match="shots requires"):
        fq.run(bell, outputs=fq.vn_entropy([0]), options=fq.ExecutionOptions(shots=100))
    with pytest.raises(ValueError, match="outside a 2-qubit circuit"):
        fq.run(bell, outputs=fq.vn_entropy([5]))


def test_a_noisy_program_refuses_the_modes_that_cannot_hold_a_mixed_state() -> None:
    """A user who asks a noisy program for one of those modes is told which ones work."""

    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    model = NoiseModel().add(["h"], depolarizing_channel(0.1))

    for mode in ("statevector", "mps", "tensor_network"):
        with pytest.raises(
            ValueError, match="supports mode='auto' or mode='density_matrix'"
        ):
            fq.run(
                circuit,
                outputs=fq.vn_entropy([0]),
                noise_model=model,
                options=fq.ExecutionOptions(mode=mode),
            )

    for mode in ("auto", "density_matrix"):
        answer = fq.run(
            circuit,
            outputs=fq.vn_entropy([0]),
            noise_model=model,
            options=fq.ExecutionOptions(mode=mode),
        )
        assert float(answer.vn_entropy[0]) > 0.0
