"""Verify the executable DF encoding against a quantized block matrix."""

from math import acos, cos, floor, sqrt, tau

import numpy as np
import pytest

from flagquantum.algorithms.primitives.double_factorization_block_encoding import (
    DoubleFactorizationBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_select_double_factorization import _basis_matrix
from tests.unit.test_single_factorization_block_encoding import _probabilities
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _expected_block(rows, angles, alias_bits, angle_bits, rank_bits):
    spatial = len(rows[0])
    identity = np.eye(1 << spatial)
    system_identity = np.eye(1 << (2 * spatial))
    weights = (
        sum(map(abs, rows[0])),
        *(sum(map(abs, row)) ** 2 / 4 for row in rows[1:]),
    )
    block = np.zeros_like(system_identity)
    offset = 0
    for factor, (row, weight) in enumerate(
        zip(rows, _probabilities(weights, alias_bits), strict=True)
    ):
        body = np.zeros_like(system_identity)
        for orbital, (value, probability) in enumerate(
            zip(row, _probabilities(row, alias_bits), strict=True)
        ):
            theta = tuple(
                tau * word / (1 << angle_bits) for word in angles[offset + orbital]
            )
            number = _basis_matrix(spatial, theta)
            spin_average = (np.kron(number, identity) + np.kron(identity, number)) / 2
            body += (
                probability * np.sign(value) * spin_average * (-1 if factor == 0 else 1)
            )
        capacity = 1 << (len(row) - 1).bit_length()
        step = tau / (1 << rank_bits)
        angle = step * floor(2 * acos(sqrt(capacity / (4 * len(row)))) / step + 0.5)
        initial_success = len(row) / capacity * cos(angle / 2) ** 2
        success = initial_success * (3 - 4 * initial_success) ** 2
        body = success * body + (1 - success) * system_identity
        block += weight * (2 * body @ body - system_identity if factor else body)
        offset += len(row)
    return block


@pytest.mark.parametrize(
    "rows,angle_rows,rank_bits",
    [
        (((-1.0,),), ((),), 2),
        (((1.0,), (-2.0,)), ((), ()), 3),
        (((1.0, -3.0), (2.0,)), ((1,), (3,), (2,)), 2),
    ],
)
def test_df_block_projection_with_rank_failure_and_sign(rows, angle_rows, rank_bits):
    encoding = DoubleFactorizationBlockEncoding.from_eigenvalues(
        rows[0],
        rows[1:],
        angle_rows,
        outer_bits=1,
        inner_bits=1,
        rank_rotation_bits=rank_bits,
        basis_rotation_bits=4,
    )
    logical = sum(width for _, width in encoding.layout)
    work, _ = encoding.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), encoding.layout)
    circuit = DynamicCircuit(width).append(
        encoding.make_on(ports, workspace=tuple(range(logical, width)))
    )
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work) == {
        (1, 1): (136, 20, 3),
        (1, 2): (150, 20, 3),
        (2, 2): (450, 81, 4),
    }[len(rows[0]), len(rows)]
    expected = _expected_block(rows, angle_rows, 1, 4, rank_bits)
    system_width = len(ports["system"])
    projection = (1 << (system_width + work)) - 1
    clean = (1 << work) - 1
    for name in ports.keys() - {
        "l",
        "sigma",
        "inner_p",
        "inner_sigma",
        "inner_rotation",
        "spin",
        "system",
    }:
        clean |= sum(1 << (width - wire - 1) for wire in ports[name])

    def run(state, history):
        result = _branch(
            ir.instructions, width, history, state, normalize_measurements=True
        )
        assert all(not value & clean for value in result)
        return result

    def project(state):
        vector = np.zeros(1 << system_width, dtype=complex)
        for value, amplitude in state.items():
            if not value & ~projection:
                vector[value >> work] = amplitude
        return vector

    zero_history = (0,) * events
    matrix = np.column_stack(
        [
            project(run({basis << work: 1 + 0j}, zero_history))
            for basis in range(1 << system_width)
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=6e-12, rtol=0)

    inputs = np.arange(1, (1 << system_width) + 1, dtype=float)
    complex_input = inputs + 1j * inputs[::-1]
    complex_input /= np.linalg.norm(complex_input)
    initial = {basis << work: value for basis, value in enumerate(complex_input)}
    reference_state = run(initial, zero_history)
    for pattern in ((1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        assert run(initial, history) == pytest.approx(reference_state, abs=6e-12)
    np.testing.assert_allclose(
        project(reference_state), expected @ complex_input, atol=6e-12, rtol=0
    )
