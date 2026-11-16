"""Verify the periodic Hubbard circuit against fermionic Hamiltonians."""

from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.hubbard_lattice import (
    HubbardHoppingTile,
    HubbardInteraction,
    HubbardSuzukiStep,
)
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


@pytest.mark.parametrize("pink,adjoint", list(product((False, True), repeat=2)))
def test_standalone_hopping_tile_and_hamming_weight_branches(pink, adjoint):
    timestep = 0.21
    coupling = -0.63
    _, hopping = _operators()
    signed_time = -timestep if adjoint else timestep
    matrix = torch.linalg.matrix_exp(1j * signed_time * coupling * hopping)
    args = {
        "length": 2,
        "up": _UP,
        "down": _DOWN,
        "timestep": timestep,
        "hubb_t": coupling,
        "pink": pink,
        "adjoint": adjoint,
    }
    ordinary = HubbardHoppingTile.make_on(**args)
    circuit = Circuit(8).append(ordinary)
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128), matrix, atol=8e-12, rtol=0
    )
    report = estimate_ftqc_resources(ordinary, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == 16
    assert report.logical.peak_logical_qubits == 4

    measured = HubbardHoppingTile.make_on(**args, hamming_weight=True, workspace=_WORK)
    circuit = DynamicCircuit(11).append(measured)
    ir = circuit.to_ir()
    amp = torch.arange(1, 257, dtype=torch.float64)
    state = torch.complex(amp, amp.remainder(13) - 6)
    state /= state.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    initial[::8] = state
    expected = torch.zeros_like(initial)
    expected[::8] = matrix @ state
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, ir.instructions, history),
            expected / 2,
            atol=8e-12,
            rtol=0,
        )
    measured_report = estimate_ftqc_resources(measured, emitted_ir=ir)
    assert measured_report.emitted is not None
    assert measured_report.emitted.t_count == 24
    assert measured_report.emitted.unconditional["measure"] == 2
    assert {
        gate.metadata["classical_bit"]
        for gate in ir.instructions
        if gate.name == "measure"
    } == {0}


@pytest.mark.parametrize(
    ("pink", "cycle"), [(True, (0, 1, 5, 4)), (False, (0, 3, 15, 12))]
)
def test_hopping_color_routes_distinct_periodic_cycles(pink, cycle):
    from math import cos, sin, sqrt

    from tests.unit.sparse_register_execution import execute_sparse_register

    time = 0.327
    op = HubbardHoppingTile.make_on(
        length=4,
        up=tuple(range(16)),
        down=tuple(range(16, 32)),
        timestep=time,
        pink=pink,
    )
    circuit = Circuit(32).append(op)
    scale = 1 / sqrt(2)
    initial = {1 << 31: scale, 1 << 15: 1j * scale}
    along = 1j * sin(time) * cos(time)
    amplitudes = (cos(time) ** 2, along, -sin(time) ** 2, along)
    expected = {
        1 << (31 - offset - site): spin * scale * amplitude
        for offset, spin in ((0, 1), (16, 1j))
        for site, amplitude in zip(cycle, amplitudes, strict=True)
    }
    actual = execute_sparse_register(circuit.to_ir(), initial, seed=7)
    assert all(
        abs(actual.get(i, 0) - expected.get(i, 0)) < 8e-12
        for i in actual.keys() | expected.keys()
    )


@pytest.mark.parametrize("adjoint", (False, True))
def test_standalone_interaction_full_matrix_and_erase_histories(adjoint):
    time, coupling = 0.23, -1.7
    angles = dict(
        length=2,
        up=_UP,
        down=_DOWN,
        timestep=time,
        hubb_u=coupling,
        adjoint=adjoint,
    )
    signed = -time if adjoint else time
    energies = torch.tensor(
        [4 - 2 * ((word >> 4) ^ (word & 15)).bit_count() for word in range(256)],
        dtype=torch.float64,
    )
    diagonal = torch.exp(1j * signed * coupling / 4 * energies)
    ordinary = HubbardInteraction.make_on(**angles)
    circuit = Circuit(8).append(ordinary)
    ir = circuit.to_ir()
    pairs = list(zip(_UP, _DOWN, strict=True))
    assert [(gate.name, gate.wires) for gate in ir.instructions] == [
        *(("cx", pair) for pair in pairs),
        *(("rz", (wire,)) for wire in _DOWN),
        *(("cx", pair) for pair in reversed(pairs)),
    ]
    assert all(
        gate.params["theta"] == -signed * coupling / 2
        for gate in ir.instructions
        if gate.name == "rz"
    )
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128),
        torch.diag(diagonal),
        atol=8e-12,
        rtol=0,
    )

    measured = HubbardInteraction.make_on(
        **angles, hamming_weight=True, workspace=_WORK
    )
    circuit = DynamicCircuit(11).append(measured)
    emitted = circuit.to_ir()
    report = estimate_ftqc_resources(measured, emitted_ir=emitted)
    assert report.emitted is not None
    assert report.emitted.t_count == 8
    assert report.emitted.unconditional["measure"] == 2
    assert report.logical.gates.measurement == 2
    assert report.logical.peak_logical_qubits == 5
    amps = torch.arange(1, 257, dtype=torch.float64)
    state = torch.complex(amps.remainder(17) - 5, amps)
    state /= state.norm()
    incoming = torch.zeros(2048, dtype=torch.complex128)
    incoming[::8] = state
    expected = torch.zeros_like(incoming)
    expected[::8] = state * diagonal
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(incoming, emitted.instructions, history),
            expected / 2,
            atol=8e-12,
            rtol=0,
        )


@pytest.mark.parametrize("hamming_weight", (False, True))
def test_odd_lattice_interaction_keeps_the_last_site(hamming_weight):
    from cmath import exp
    from math import sqrt

    from tests.unit.sparse_register_execution import execute_sparse_register

    system = 18
    workspace = tuple(range(system, system + 6)) if hamming_weight else ()
    width = system + len(workspace)
    time, coupling = -0.367, 1.31
    up, down = tuple(range(9)), tuple(range(9, 18))
    op = HubbardInteraction.make_on(
        length=3,
        up=up,
        down=down,
        timestep=time,
        hubb_u=coupling,
        hamming_weight=hamming_weight,
        workspace=workspace,
    )
    ir = DynamicCircuit(width).append(op).to_ir()
    labels = tuple(word << len(workspace) for word in (0, 1, 256, 257, 511, 131073))
    incoming = {word: (1j**j) / sqrt(len(labels)) for j, word in enumerate(labels)}
    expected = {}
    for word, amplitude in incoming.items():
        data = word >> len(workspace)
        parity = ((data >> 9) ^ (data & 511)).bit_count()
        expected[word] = amplitude * exp(1j * time * coupling / 4 * (9 - 2 * parity))
    for seed in (1, 8):
        actual = execute_sparse_register(ir, incoming, seed=seed)
        assert all(
            abs(actual.get(i, 0) - expected.get(i, 0)) < 8e-12
            for i in actual.keys() | expected.keys()
        )
    if hamming_weight:
        assert [
            (gate.wires, gate.params["theta"])
            for gate in ir.instructions
            if gate.name == "rz"
        ] == [((down[-1],), -time * coupling / 2)]


def test_single_site_interaction_uses_one_direct_phase():
    time, coupling = 0.31, 1.5
    arguments = dict(length=1, up=(0,), down=(1,), timestep=time, hubb_u=coupling)
    op = HubbardInteraction.make_on(**arguments)
    circuit = Circuit(2).append(op)
    target = time * coupling / 4
    diagonal = torch.exp(
        1j * torch.tensor([target, -target, -target, target], dtype=torch.float64)
    )
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128),
        torch.diag(diagonal),
        atol=8e-12,
        rtol=0,
    )
    with pytest.raises(ValueError, match="positive concrete width"):
        HubbardInteraction.make_on(**arguments, hamming_weight=True)


@pytest.mark.parametrize("stripped", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("hamming_weight", (False, True))
def test_suzuki_operands_match_ordered_standalone_layers(
    stripped, adjoint, hamming_weight
):
    common = dict(
        length=2,
        up=(2, 0, 6, 4),
        down=(3, 1, 7, 5),
        workspace=_WORK if hamming_weight else (),
        hamming_weight=hamming_weight,
    )
    timestep, hubb_u, hubb_t = -0.34, 1.23, -0.51
    plan = [("pink", 0.5), ("gold", 1.0), ("pink", 0.5)]
    if stripped:
        plan.append(("interaction", 1.0))
    else:
        plan = [("interaction", 0.5), *plan, ("interaction", 0.5)]
    if adjoint:
        plan.reverse()
    expected = []
    for term, coefficient in plan:
        time = coefficient * timestep
        if term == "interaction":
            op = HubbardInteraction.make_on(
                **common, timestep=time, hubb_u=hubb_u, adjoint=adjoint
            )
        else:
            op = HubbardHoppingTile.make_on(
                **common,
                timestep=time,
                hubb_t=hubb_t,
                pink=term == "pink",
                adjoint=adjoint,
            )
        (layer,) = op.layers
        expected.extend(layer.emit(common["workspace"], (0,) if hamming_weight else ()))
    op = HubbardSuzukiStep.make_on(
        **common,
        timestep=timestep,
        hubb_u=hubb_u,
        hubb_t=hubb_t,
        strip_layer=stripped,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(11) if hamming_weight else Circuit(8)
    assert circuit.append(op).to_ir().instructions == tuple(expected)
