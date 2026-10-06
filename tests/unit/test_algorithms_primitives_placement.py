"""The primitive append forms are placements, and the placement is ``Circuit.compose``.

Every append form in :mod:`flagquantum.algorithms.primitives` writes one sub-program into a
circuit the caller owns. This file pins what that sub-program is: for each of the four
primitives the plan's construction slice names -- ``append_qft``, ``append_arbitrary_state``,
``append_bit_oracle``, and ``append_phase_estimation`` -- the instructions the append form
emits are exactly the instructions the matching standalone builder emits once
:meth:`Circuit.compose <flagquantum.circuit.Circuit.compose>` places it on the same qubits.

The equivalence is checked rather than assumed, because it is the whole claim: a user can
reach the same program either way, so the append form adds a signature and not a second
construction. For ``append_qft`` the placement *is* the implementation. For the other three
the standalone builder already derives from the append body, so the test states the direction
the single construction runs in rather than a second path that was removed.

``append_qft`` additionally became atomic here: a placement that names a qubit outside the
circuit is refused before any instruction is written, where the previous emission raised
part-way and left the transform half-applied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.primitives import (
    append_arbitrary_state,
    append_bit_oracle,
    append_phase_estimation,
    append_qft,
    arbitrary_state,
    bit_oracle,
    phase_estimation_circuit,
    qft,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


class _PhaseGate:
    """A one-qubit operator multiplying the ``|1>`` component by ``exp(2*pi*1j*phase)``."""

    def __init__(self, phase: float) -> None:
        self._phase = phase

    @property
    def n_qubits(self) -> int:
        return 1

    def apply(self, circuit: Circuit, qubits: Sequence[int]) -> None:
        circuit.gate("p", qubits[0], theta=2 * math.pi * self._phase)

    def apply_controlled(
        self, circuit: Circuit, control: int, qubits: Sequence[int]
    ) -> None:
        circuit.gate("cphase", (control, qubits[0]), theta=2 * math.pi * self._phase)

    def apply_power_controlled(
        self, circuit: Circuit, control: int, qubits: Sequence[int], power: int
    ) -> None:
        circuit.gate(
            "cphase", (control, qubits[0]), theta=2 * math.pi * self._phase * power
        )


def _signature(circuit: Circuit) -> tuple[tuple[str, tuple[int, ...], dict], ...]:
    """Return the emitted instructions as a comparable, index-carrying record."""
    return tuple(
        (instruction.name, tuple(instruction.wires), dict(instruction.params))
        for instruction in circuit.to_ir().instructions
    )


def _marked(value: int) -> bool:
    """A predicate with an irregular truth table, so a wrong ladder cannot pass by luck."""
    return value in (1, 2, 5)


@pytest.mark.parametrize(
    "qubits", [(0,), (0, 1), (0, 1, 2), (1, 2, 3), (3, 1, 4), (2, 0), (0, 2, 5)]
)
@pytest.mark.parametrize("inverse", [False, True])
def test_append_qft_places_the_standalone_transform(qubits, inverse):
    appended = Circuit(max(qubits) + 2).x(0).x(max(qubits) + 1)
    prefix = _signature(appended)
    append_qft(appended, qubits, inverse=inverse)

    placed = Circuit(max(qubits) + 2).x(0).x(max(qubits) + 1)
    placed.compose(qft(len(qubits), inverse=inverse), qubits=qubits)

    assert _signature(appended) == _signature(placed)
    assert _signature(appended)[: len(prefix)] == prefix
    width = len(qubits)
    emitted = width + width * (width - 1) // 2 + width // 2
    assert len(_signature(appended)) == len(prefix) + emitted


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5, 6])
def test_the_inverse_transform_is_the_forward_transform_adjoint(n_qubits):
    forward = qft(n_qubits)
    assert _signature(qft(n_qubits, inverse=True)) == _signature(forward.adjoint())
    assert forward.adjoint().n_qubits == n_qubits


def test_a_qft_round_trip_leaves_the_register_where_it_started():
    circuit = Circuit(3).x(1)
    before = circuit.state()[0].clone()
    append_qft(circuit, (0, 1, 2))
    append_qft(circuit, (0, 1, 2), inverse=True)
    assert torch.allclose(circuit.state()[0], before, atol=1e-6)


@pytest.mark.parametrize("basis_state", [0, 1, 2, 3])
def test_the_transform_maps_a_basis_state_onto_its_discrete_fourier_transform(
    basis_state: int,
):
    circuit = Circuit(2)
    for position in range(2):
        if basis_state >> (1 - position) & 1:
            circuit.x(position)
    append_qft(circuit, (0, 1))

    width = 2**2
    expected = torch.tensor(
        [
            complex(
                math.cos(2 * math.pi * basis_state * k / width),
                math.sin(2 * math.pi * basis_state * k / width),
            )
            / math.sqrt(width)
            for k in range(width)
        ],
        dtype=torch.complex64,
    )
    assert torch.allclose(circuit.state()[0], expected, atol=1e-6)


@pytest.mark.parametrize("qubits", [(0, 1), (2, 3), (0, 4, 7)])
def test_append_arbitrary_state_places_the_standalone_preparation(qubits):
    amplitudes = torch.tensor([0.5, -0.5, 0.5j, -0.5j, 0.5, 0.5, -0.5, -0.5])[
        : 2 ** len(qubits)
    ]
    width = max(qubits) + 1

    appended = Circuit(width)
    append_arbitrary_state(appended, amplitudes, qubits)

    placed = Circuit(width).compose(
        arbitrary_state(amplitudes), qubit_map=dict(enumerate(qubits))
    )

    assert _signature(appended) == _signature(placed)


@pytest.mark.parametrize("qubits", [(0, 1, 2), (1, 2, 3, 4)])
def test_append_bit_oracle_places_the_standalone_oracle(qubits):
    n_qubits = len(qubits)
    spare = max(0, n_qubits - 2)
    target = max(qubits) + 1
    ancillas = list(range(target + 1, target + 1 + spare))
    width = target + 1 + spare

    appended = Circuit(width)
    append_bit_oracle(appended, _marked, qubits, target=target, ancillas=ancillas)

    mapping = {position: qubits[position] for position in range(n_qubits)}
    mapping[n_qubits] = target
    mapping.update({n_qubits + 1 + index: ancillas[index] for index in range(spare)})
    placed = Circuit(width).compose(bit_oracle(_marked, n_qubits), qubit_map=mapping)

    assert _signature(appended) == _signature(placed)


@pytest.mark.parametrize("counting,evaluation", [((0, 1), (2,)), ((0, 1, 2), (3,))])
def test_append_phase_estimation_places_the_standalone_circuit(counting, evaluation):
    unitary = _PhaseGate(0.375)
    width = max(counting + evaluation) + 1

    appended = Circuit(width)
    append_phase_estimation(
        appended,
        unitary=unitary,
        counting_qubits=list(counting),
        evaluation_qubits=list(evaluation),
    )

    mapping = {qubit: qubit for qubit in counting}
    mapping.update(
        {len(counting) + index: evaluation[index] for index in range(len(evaluation))}
    )
    placed = Circuit(width).compose(
        phase_estimation_circuit(unitary=unitary, n_counting_qubits=len(counting)),
        qubit_map=mapping,
    )

    assert _signature(appended) == _signature(placed)


def test_a_placement_outside_the_circuit_writes_nothing():
    circuit = Circuit(2).x(0)
    prefix = _signature(circuit)
    with pytest.raises(ValueError, match="outside circuit range"):
        append_qft(circuit, (1, 5))
    assert _signature(circuit) == prefix


def test_a_batched_receiver_is_refused_rather_than_broadcast_into():
    """``compose``'s batch rule is inherited rather than restated: the transform is unbatched."""
    circuit = Circuit(3, bsz=4)
    prefix = _signature(circuit)
    with pytest.raises(ValueError, match="cannot mix batch sizes"):
        append_qft(circuit, (0, 1, 2))
    assert _signature(circuit) == prefix

    plain = Circuit(3)
    append_qft(plain, (0, 1, 2))
    assert len(_signature(plain)) == 7


def test_a_repeated_qubit_and_an_empty_register_are_still_refused():
    circuit = Circuit(4)
    with pytest.raises(ValueError, match="qft qubits must be distinct"):
        append_qft(circuit, (1, 1))
    assert append_qft(circuit, ()) is None
    assert circuit.to_ir().instructions == ()
    with pytest.raises(ValueError, match="qft needs at least one qubit"):
        qft(0)
