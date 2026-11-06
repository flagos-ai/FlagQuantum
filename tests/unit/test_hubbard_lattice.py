"""Verify the periodic Hubbard circuit against fermionic Hamiltonians."""

from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.hubbard_lattice import HubbardSuzukiStep
from flagquantum.circuit import Circuit
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_hubbard_plaquette import _hopping_hamiltonian

pytestmark = pytest.mark.unit
_UP = (0, 1, 2, 3)
_DOWN = (4, 5, 6, 7)
_WORK = (8, 9, 10)


def _operators():
    identity = torch.eye(16, dtype=torch.complex128)
    # In row-major order the two-by-two cycle visits sites 0, 1, 3, 2.
    order = torch.tensor(
        [(index & 12) | ((index & 2) >> 1) | ((index & 1) << 1) for index in range(16)]
    )
    # This is a fermionic change of order, including the doubly occupied sign.
    signs = torch.tensor([(-1 if index & 3 == 3 else 1) for index in range(16)])
    single_spin = _hopping_hamiltonian()[order][:, order] * signs[:, None] * signs
    hopping = torch.kron(single_spin, identity) + torch.kron(identity, single_spin)
    energies = torch.tensor(
        [4 - 2 * ((index >> 4) ^ (index & 15)).bit_count() for index in range(256)]
    )
    return energies, hopping


@pytest.mark.parametrize("strip,adjoint", list(product((False, True), repeat=2)))
def test_full_periodic_unitary(strip, adjoint):
    timestep, u, coupling = 0.123, 2.1, -0.7
    op = HubbardSuzukiStep.make_on(
        length=2,
        up=_UP,
        down=_DOWN,
        hubb_u=u,
        timestep=timestep,
        hubb_t=coupling,
        strip_layer=strip,
        adjoint=adjoint,
    )
    energies, hopping = _operators()
    interaction = torch.diag(torch.exp(0.5j * timestep * u / 4 * energies.double()))
    # Both colors traverse the same square in a two-by-two periodic lattice.
    kinetic = torch.linalg.matrix_exp(2j * timestep * coupling * hopping)
    expected = interaction @ kinetic @ interaction
    if strip:
        expected = interaction @ interaction @ kinetic
    if adjoint:
        expected = expected.mH
    circuit = Circuit(8).append(op)
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128), expected, atol=8e-12, rtol=0
    )
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == report.logical.gates.t == 48
    assert report.logical.peak_logical_qubits == 4
    assert report.emitted.declared_qubits == 8


@pytest.mark.parametrize("strip,adjoint", list(product((False, True), repeat=2)))
def test_measured_step_all_layer_histories_and_cleanup(strip, adjoint):
    kwargs = dict(
        length=2,
        up=_UP,
        down=_DOWN,
        hubb_u=2.1,
        timestep=0.123,
        hubb_t=-0.7,
        strip_layer=strip,
        adjoint=adjoint,
    )
    measured = HubbardSuzukiStep.make_on(**kwargs, hamming_weight=True, workspace=_WORK)
    plain = HubbardSuzukiStep.make_on(**kwargs)
    amps = torch.arange(1, 257, dtype=torch.float64)
    state = torch.complex(amps, amps.remainder(17) - 8)
    state /= state.norm()
    initial_full = torch.zeros(2048, dtype=torch.complex128)
    initial_full[::8] = state
    for actual, expected in zip(measured.layers, plain.layers, strict=True):
        coherent = tuple(expected.emit((), ()))
        from examples.algorithms.binary_field_arithmetic import FieldOperation

        goal = (
            Circuit(8, inputs=state, dtype=torch.complex128)
            .append(FieldOperation(8, coherent))
            .state()
            .flatten()
        )
        initial = torch.zeros(2048, dtype=torch.complex128)
        initial[::8] = state
        output = torch.zeros_like(initial)
        output[::8] = goal
        gates = tuple(actual.emit(_WORK, (0,)))
        assert sum(gate.name == "measure" for gate in gates) == 2
        # Each normalized branch agrees and restores work before the next layer.
        for history in product((0, 1), repeat=2):
            torch.testing.assert_close(
                forced_branch(initial, gates, history), output / 2, atol=6e-12, rtol=0
            )
        state = goal

    emitted = DynamicCircuit(11).append(measured).to_ir().instructions
    torch.testing.assert_close(
        forced_branch(initial_full, emitted, (1, 0) * len(measured.layers)),
        output / 2 ** len(measured.layers),
        atol=6e-12,
        rtol=0,
    )
    circuit = DynamicCircuit(12)
    circuit.measure(11, classical_bit=3)
    circuit.append(measured)
    report = estimate_ftqc_resources(measured, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == 48 + 8 * len(measured.layers)
    assert report.emitted.unconditional["measure"] == 1 + 2 * len(measured.layers)
    assert {
        gate.metadata["classical_bit"]
        for gate in circuit.to_ir().instructions
        if gate.name == "measure"
    } == {3, 4}
    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(11).append(measured)


def test_larger_square_has_two_disjoint_tilings_with_unique_edges():
    from collections import Counter

    from flagquantum.algorithms.primitives.hubbard_lattice import _plaquettes

    length = 4
    edges = Counter()
    for offset in (0, 1):
        squares = _plaquettes(length, offset)
        assert sorted(site for square in squares for site in square) == list(range(16))
        for square in squares:
            edges.update(
                tuple(sorted(pair))
                for pair in zip(square, (*square[1:], square[0]), strict=True)
            )
    expected = {
        tuple(sorted((row * length + col, rr * length + cc)))
        for row in range(length)
        for col in range(length)
        for rr, cc in ((row, (col + 1) % length), ((row + 1) % length, col))
    }
    assert edges == Counter(expected)
