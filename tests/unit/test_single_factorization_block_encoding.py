"""Compare the executable SF signal block with the quantized paper matrix."""

from itertools import product

import numpy as np
import pytest

from flagquantum.algorithms.primitives.single_factorization_block_encoding import (
    SingleFactorizationBlockEncoding,
    SingleFactorizationOneBody,
)
from flagquantum.algorithms.primitives.single_factorization_inner_prepare import (
    PrepareSingleFactorizationInner,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _probabilities(weights, bits):
    weights = np.abs(weights)
    counts = len(weights) * (1 << bits)
    endpoints = np.floor(np.r_[0, np.cumsum(weights)] * counts / weights.sum() + 0.5)
    return np.diff(endpoints) / counts


def _paper_pair(orbitals, p, q, spin):
    """Jordan–Wigner form of Eq. A4 on interleaved spin orbitals."""
    i, j = sorted((2 * p + spin, 2 * q + spin))
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    endpoint = x if p < q else y
    result = np.ones((1, 1))
    for site in range(2 * orbitals):
        if i == j == site:
            factor = -z
        elif site in (i, j):
            factor = endpoint
        elif i < site < j:
            factor = z
        else:
            factor = identity
        result = np.kron(result, factor)
    return result


def _paper_body(orbitals, row, bits):
    pairs = tuple((p, q) for q in range(orbitals) for p in range(q + 1))
    terms = []
    for (p, q), value, probability in zip(
        pairs, row, _probabilities(row, bits), strict=True
    ):
        average = (
            sum(
                _paper_pair(orbitals, a, b, spin)
                for (a, b), spin in product(((p, q), (q, p)), (0, 1))
            )
            / 4
        )
        terms.append(np.sign(value) * probability * average)
    return sum(terms)


@pytest.mark.parametrize(
    ("orbitals", "rows", "address"),
    [
        (1, ((-1.0,),), 0),
        (1, ((1.0,), (-2.0,)), 1),
        (2, ((1.0, -2.0, 1.0),), 0),
    ],
)
def test_standalone_one_body_projection_and_clean_work(orbitals, rows, address):
    inner = PrepareSingleFactorizationInner.from_coefficients(
        orbitals, rows[0], rows[1:], probability_bits=1
    )
    body = SingleFactorizationOneBody(inner)
    logical = sum(width for _, width in body.layout)
    work, slots = body.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), body.layout)
    bound = body.make_on(ports, workspace=tuple(range(logical, width)))
    ir = DynamicCircuit(width).append(bound).to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work, slots) == (
        (26, 3, 2, 1) if orbitals == 1 else (148, 23, 4, 3)
    )
    assert {
        gate.metadata["classical_bit"]
        for gate in ir.instructions
        if gate.name == "measure"
    } == set(range(slots))
    system_size = 1 << (2 * orbitals)
    system_mask = (system_size - 1) << work
    success_mask = 1 << (width - 1 - ports["succ_l"][0])
    address_shift = width - 1 - ports["l"][-1] if ports["l"] else 0
    address_mask = ((1 << len(ports["l"])) - 1) << address_shift
    cleaned = (1 << work) - 1
    for name in ports.keys() - {
        "succ_l",
        "l",
        "p",
        "q",
        "sigma",
        "rotation",
        "spin",
        "swap_pq",
        "system",
    }:
        cleaned |= sum(1 << (width - 1 - wire) for wire in ports[name])
    histories = (
        product((0, 1), repeat=events)
        if orbitals == 1
        else ((pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0)))
    )

    full_outputs = {}
    for history in histories:
        for success in (0, 1):
            fixed = (success * success_mask) | (address << address_shift)
            matrix = np.zeros((system_size, system_size), dtype=complex)
            for col in range(system_size):
                state = _branch(
                    ir.instructions,
                    width,
                    history,
                    {fixed | (col << work): 1 + 0j},
                    normalize_measurements=True,
                )
                assert all(not index & cleaned for index in state)
                assert all(
                    index & (success_mask | address_mask) == fixed for index in state
                )
                known = full_outputs.setdefault((success, col), state)
                assert state == pytest.approx(known, abs=4e-12)
                for index, amplitude in state.items():
                    if index & ~system_mask == fixed:
                        matrix[(index & system_mask) >> work, col] = amplitude
            reference = (
                _paper_body(orbitals, rows[address], 1)
                if success
                else np.eye(system_size)
            )
            np.testing.assert_allclose(matrix, reference, atol=4e-12, rtol=0)


@pytest.mark.parametrize(
    "orbitals,rows",
    [
        (1, ((-1.0,),)),
        (1, ((1.0,), (-2.0,))),
        (2, ((1.0, -2.0, 1.0), (2.0, 3.0, -1.0))),
    ],
)
def test_complete_block_and_clean_measurement_feedback(orbitals, rows):
    encoding = SingleFactorizationBlockEncoding.from_coefficients(
        orbitals,
        rows[0],
        rows[1:],
        outer_bits=1,
        inner_bits=1,
    )
    logical = sum(width for _, width in encoding.layout)
    work, slots = encoding.requirements()
    wires = logical + work
    ports = partition_register(tuple(range(logical)), encoding.layout)
    bound = encoding.make_on(ports, workspace=tuple(range(logical, wires)))
    circuit = DynamicCircuit(wires).append(bound)
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work, slots) == {
        (1, 1): (94, 13, 4, 4),
        (1, 2): (108, 13, 4, 4),
        (2, 2): (408, 67, 6, 6),
    }[orbitals, len(rows)]
    system_width = 1 << (2 * orbitals)
    weights = (
        2 * sum(map(abs, rows[0])),
        *(sum(map(abs, row)) ** 2 for row in rows[1:]),
    )
    probabilities = _probabilities(weights, 1)
    bodies = tuple(_paper_body(orbitals, row, 1) for row in rows)
    reference = probabilities[0] * bodies[0]
    for probability, body in zip(probabilities[1:], bodies[1:], strict=True):
        reference += probability * (2 * body @ body - np.eye(system_width))

    signal_mask = (1 << (2 * orbitals + work)) - 1
    cleaned = (1 << work) - 1
    for name in ports.keys() - {
        "l",
        "sigma",
        "inner_p",
        "inner_q",
        "inner_sigma",
        "inner_rotation",
        "spin",
        "swap_pq",
        "system",
    }:
        cleaned |= sum(1 << (wires - 1 - wire) for wire in ports[name])

    expected_states = {}
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        matrix = np.zeros_like(reference, dtype=complex)
        for col in range(system_width):
            initial = {col << work: 1 + 0j}
            state = _branch(
                ir.instructions, wires, history, initial, normalize_measurements=True
            )
            assert all(not index & cleaned for index in state)
            if col not in expected_states:
                expected_states[col] = state
            assert state == pytest.approx(expected_states[col], abs=4e-12)
            for index, amplitude in state.items():
                if not index & ~signal_mask:
                    matrix[index >> work, col] = amplitude
        np.testing.assert_allclose(matrix, reference, atol=4e-12, rtol=0)
