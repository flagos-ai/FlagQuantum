"""Check selected DF rotations, the signed phase and scratch restoration."""

from itertools import product
from math import cos, sin, sqrt, tau

import numpy as np
import pytest

from flagquantum.algorithms.primitives.multiplexers.select_double_factorization import (
    SelectDoubleFactorization,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _basis_matrix(orbitals, angles):
    basis = np.eye(1 << orbitals)
    for left, theta in enumerate(angles):
        cosine, sine = cos(2 * theta), sin(2 * theta)
        pair = np.array(
            [[1, 0, 0, 0], [0, cosine, -sine, 0], [0, sine, cosine, 0], [0, 0, 0, 1]]
        )
        gate = np.kron(
            np.kron(np.eye(1 << left), pair), np.eye(1 << (orbitals - left - 2))
        )
        basis = basis @ gate
    number = np.kron(np.diag([1, -1]), np.eye(1 << (orbitals - 1)))
    return basis.T @ number @ basis


@pytest.mark.parametrize(
    "orbitals,bits,log,extra", [(1, 1, 0, True), (2, 4, 0, False), (3, 4, 1, True)]
)
def test_df_select_matches_signed_rotated_number_operator(orbitals, bits, log, extra):
    rows = tuple(
        tuple((i + j + 1) % (1 << bits) for j in range(orbitals - 1)) for i in range(4)
    )
    select = SelectDoubleFactorization.from_angle_words(
        orbitals,
        rows,
        angle_bits=bits,
        log_block_size=log,
    )
    layout = select.layout(additional_control=extra)
    logical = sum(size for _, size in layout)
    work = select.workspace_size(additional_control=extra)
    width = logical + work
    ports = partition_register(tuple(range(logical)), layout)
    op = select.make_on(ports, workspace=tuple(range(logical, width)))
    circuit = DynamicCircuit(width).append(op)
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work) == {
        1: (22, 2, 2),
        2: (104, 19, 3),
        3: (218, 28, 3),
    }[orbitals]

    def pack(values, system):
        word = 0
        for name, size in layout:
            word = (word << size) | (
                system if name == "system" else values.get(name, 0)
            )
        return word << work

    matrices = tuple(
        _basis_matrix(orbitals, tuple(tau * word / (1 << bits) for word in row))
        for row in rows
    )
    initial, expected = {}, {}
    n = 1 << orbitals
    # Include offset+p distinct from both inputs, both spins and signs, and every failure flag.
    controls = [(1, 1, 1), (0, 1, 1), (1, 0, 1)]
    if extra:
        controls.append((1, 1, 0))
    addresses = {
        1: ((0, 0),),
        2: ((0, 0), (0, 1), (1, 0), (1, 1)),
        3: ((0, 0), (1, 1)),
    }[orbitals]
    basis_states = range(n * n) if orbitals < 3 else (0, 1, 3, 4, 8, 16, 24, 32, 63)
    states = tuple(product(addresses, (0, 1), (0, 1), controls, basis_states))
    scale = 1 / sqrt(2 * len(states))
    for (offset, p), spin, sign, (succ_l, succ_p, nonzero), system in states:
        values = dict(
            offset=offset,
            p=p,
            spin=spin,
            theta=sign,
            succ_l=succ_l,
            succ_p=succ_p,
            l_ne_zero=nonzero,
        )
        basis = pack(values, system)
        coefficient = complex(1, (-1) ** (basis.bit_count())) * scale
        initial[basis] = coefficient
        if not (succ_l and succ_p and nonzero):
            expected[basis] = expected.get(basis, 0) + coefficient
            continue
        local, other = (system % n, system // n) if spin else (system // n, system % n)
        for destination, amplitude in enumerate(matrices[offset + p][:, local]):
            out = other * n + destination if spin else destination * n + other
            target = pack(values, out)
            expected[target] = (
                expected.get(target, 0) + coefficient * (-1) ** sign * amplitude
            )
    expected = {basis: amp for basis, amp in expected.items() if abs(amp) > 1e-12}
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        assert _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        ) == pytest.approx(expected, abs=3e-12)
