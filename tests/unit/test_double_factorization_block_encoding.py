"""Verify the executable DF encoding against a quantized block matrix."""

from math import acos, cos, floor, sin, sqrt, tau

import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.double_factorization_angles import (
    orbital_rotation_words,
)
from flagquantum.algorithms.primitives.double_factorization_block_encoding import (
    DoubleFactorizationBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_select_double_factorization import _basis_matrix
from tests.unit.test_single_factorization_block_encoding import _probabilities
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "vector", [(2 / 3, 1 / 3, -2 / 3), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)]
)
def test_df_eigenvector_words_encode_single_particle_projector(vector):
    bits = 20
    words = orbital_rotation_words(vector, bits)
    angles = tuple(tau * word / (1 << bits) for word in words)
    reflected = _basis_matrix(3, angles)[np.ix_([4, 2, 1], [4, 2, 1])]
    expected = np.eye(3) - 2 * np.outer(vector, vector)
    np.testing.assert_allclose(reflected, expected, atol=4 * tau / (1 << bits), rtol=0)


@pytest.mark.parametrize("vector", [(), (0.0,), (2.0,), (float("nan"),)])
def test_df_eigenvector_words_reject_invalid_vectors(vector):
    with pytest.raises(ValueError, match="DF orbital vectors"):
        orbital_rotation_words(vector, 8)


def test_df_eigensystem_rejects_wrong_orbital_width():
    with pytest.raises(ValueError, match="spatial orbital count"):
        DoubleFactorizationBlockEncoding.from_eigensystems(((1.0, (1.0, 0.0)),), ())


@pytest.mark.parametrize(
    "one_body,factors,match",
    [
        ([[1, 0, 0], [0, 1, 0]], (), "square real"),
        ([[1j]], (), "square real"),
        ([[float("inf")]], (), "finite and symmetric"),
        ([[0, 1], [0, 0]], (), "finite and symmetric"),
        ([[1]], ([[1, 0], [0, 1]],), "share one orbital basis"),
    ],
)
def test_df_coefficient_factory_rejects_invalid_matrices(one_body, factors, match):
    with pytest.raises(ValueError, match=match):
        DoubleFactorizationBlockEncoding.from_coefficients(
            torch.tensor(one_body), tuple(torch.tensor(factor) for factor in factors)
        )


def test_df_coefficient_factory_retains_zero_eigenvalues():
    one_body = torch.diag(torch.tensor([0.0, 1.0]))
    encoding = DoubleFactorizationBlockEncoding.from_coefficients(one_body, (one_body,))
    assert encoding.inner.ranks == (2, 2)
    assert encoding.outer.normalization == 1.25


def test_df_coefficient_factory_projects_spin_summed_fermion_operator():
    one_body = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    factor = torch.diag(torch.tensor([1.5, 0.5]))
    encoding = DoubleFactorizationBlockEncoding.from_coefficients(
        one_body,
        (factor,),
        outer_bits=1,
        inner_bits=1,
        rank_rotation_bits=2,
        basis_rotation_bits=4,
    )
    assert encoding.outer.normalization == 2

    identity = np.eye(16)

    def centered_spin_sum(matrix):
        spinless = np.zeros((4, 4))
        spinless[1:3, 1:3] = matrix.numpy()[::-1, ::-1]
        spinless[3, 3] = matrix.trace().item()
        return (
            np.kron(spinless, np.eye(4))
            + np.kron(np.eye(4), spinless)
            - matrix.trace().item() * identity
        )

    # For either two-orbital row the quantized rank preparation succeeds
    # with probability 1/2, so the remaining half contributes identity.
    one_body_block = (centered_spin_sum(one_body) + identity) / 2
    factor_block = (identity - centered_spin_sum(factor) / 2) / 2
    expected = (one_body_block + 2 * factor_block @ factor_block - identity) / 2
    logical = sum(size for _, size in encoding.layout)
    work, _ = encoding.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), encoding.layout)
    circuit = DynamicCircuit(width).append(
        encoding.make_on(ports, workspace=tuple(range(logical, width)))
    )
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work, width) == (482, 89, 4, 42)
    vector = np.arange(1, 17) + 1j * np.arange(16, 0, -1)
    vector /= np.linalg.norm(vector)
    initial = {state << work: value for state, value in enumerate(vector)}
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        state = _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        )
        assert all(index % (1 << work) == 0 for index in state)
        projected = np.array([state.get(index << work, 0) for index in range(16)])
        np.testing.assert_allclose(projected, expected @ vector, atol=6e-12, rtol=0)


@pytest.mark.parametrize(
    "one_body,factors,angles",
    [
        ((1.0,), ((2.0,), (-2.0,)), ((), (), ())),
        ((1.0, -3.0), ((2.0,),), ((1,), (3,), (2,))),
    ],
)
def test_df_adjoint_on_persistent_signal_and_system(one_body, factors, angles):
    encoding = DoubleFactorizationBlockEncoding.from_eigenvalues(
        one_body,
        factors,
        angles,
        outer_bits=1,
        inner_bits=1,
        rank_rotation_bits=2,
        basis_rotation_bits=4,
    )
    logical = sum(size for _, size in encoding.layout)
    work, _ = encoding.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), encoding.layout)
    workspace = tuple(range(logical, width))
    allowed = (
        tuple(wire for name, _ in encoding.signal_layout for wire in ports[name])
        + ports["system"]
    )
    bit_masks = tuple(1 << (width - 1 - wire) for wire in allowed)
    words = (0, *bit_masks, sum(bit_masks))
    initial = {word: (1j ** (i % 4)) / sqrt(len(words)) for i, word in enumerate(words)}
    forward = encoding.make_on(ports, workspace=workspace)
    backward = encoding.make_on(ports, workspace=workspace, adjoint=True)
    forward_ir = DynamicCircuit(width).append(forward).to_ir()
    backward_ir = DynamicCircuit(width).append(backward).to_ir()
    forward_counts = count_emitted_instructions(forward_ir)
    backward_counts = count_emitted_instructions(backward_ir)
    events = forward_counts.unconditional["measure"]
    assert backward_counts.unconditional["measure"] == events
    assert backward_counts.t_count == forward_counts.t_count

    def run(ir, state, history):
        result = _branch(
            ir.instructions, width, history, state, normalize_measurements=True
        )
        assert all(index & ~sum(bit_masks) == 0 for index in result)
        return result

    reference = run(forward_ir, initial, (0,) * events)
    for pattern in ((1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        assert run(backward_ir, initial, history) == pytest.approx(reference, abs=1e-11)
        assert run(backward_ir, reference, history) == pytest.approx(initial, abs=1e-11)


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
    if len(rows[0]) == 2:
        pairs = []
        offset = 0
        for row in rows:
            eigenpairs = []
            for i, value in enumerate(row, start=offset):
                angle = 2 * tau * angle_rows[i][0] / 16
                eigenpairs.append((value, (cos(angle), sin(angle))))
            pairs.append(eigenpairs)
            offset += len(row)
        inferred = DoubleFactorizationBlockEncoding.from_eigensystems(
            pairs[0],
            pairs[1:],
            outer_bits=1,
            inner_bits=1,
            rank_rotation_bits=rank_bits,
            basis_rotation_bits=4,
        )
        assert inferred == encoding
        encoding = inferred
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
