"""Phase recursion preserves source gauge and existing physical rotations."""

import cirq
import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("length", [2, 3, 5])
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
def test_source_phases_full_gqsp_matrix_and_clean_rotation_execution(
    length: int, real: bool, negative_power: int
) -> None:
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.qsp.generalized_qsp import (
        GeneralizedQSP,
    )
    from qualtran.bloqs.qsp.generalized_qsp import (
        qsp_phase_factors as source,
    )

    p = random_qsp_polynomial(
        length, random_state=np.random.RandomState(42), only_real_coeffs=real
    )
    q = tuple(complex(c) for c in qsp_complementary_polynomial(p, verify=True))
    theta, phi, lambd = qsp_phase_factors(p, q)
    expected_theta, expected_phi, expected_lambda = source(p, q)
    np.testing.assert_array_equal(theta, expected_theta)
    np.testing.assert_array_equal(phi, expected_phi)
    assert lambd == expected_lambda
    unitary = np.diag(np.exp(1j * np.array([-0.37, 0.29])))
    reference = cirq.unitary(
        GeneralizedQSP(
            MatrixGate(bitsize=1, matrix=unitary),
            tuple(p),
            q,
            negative_power=negative_power,
        )
    )
    native = np.eye(4, dtype=np.complex128)
    remaining = negative_power
    for i, (t, f) in enumerate(zip(theta, phi, strict=True)):
        rotation = SU2RotationGate(t, f, lambd if i == 0 else 0).rotation_matrix
        if i:
            if remaining:
                signal = np.block(
                    [
                        [np.eye(2), np.zeros((2, 2))],
                        [np.zeros((2, 2)), unitary.conj().T],
                    ]
                )
                remaining -= 1
            else:
                signal = np.block(
                    [[unitary, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]]
                )
            native = signal @ native
        native = np.kron(rotation, np.eye(2)) @ native
        # The existing Compiler emits these rotations unchanged; no new GQSP emitter here.
        gates = su2_rotation_ir(0, float(t), float(f), float(lambd if i == 0 else 0))
        for bit in (0, 1):
            circuit = Circuit(2, dtype=torch.complex128)
            if bit:
                circuit.x(0)
            circuit.append(FieldOperation(2, gates))
            assert circuit.to_ir().instructions[-len(gates) :] == gates
            state = circuit.state().detach().numpy().reshape(2, 2)
            np.testing.assert_allclose(
                state[:, 0], rotation[:, bit], atol=2e-14, rtol=0
            )
            np.testing.assert_array_equal(state[:, 1], 0)
    for _ in range(remaining):
        native = np.kron(np.eye(2), unitary.conj().T) @ native
    np.testing.assert_allclose(native, reference, atol=2e-14, rtol=0)
    for row, coefficients in ((0, p), (2, q)):
        expected = evaluate_polynomial_of_unitary_matrix(
            tuple(coefficients), unitary, offset=-negative_power
        )
        np.testing.assert_allclose(
            native[row : row + 2, :2], expected, atol=2e-12, rtol=0
        )


@pytest.mark.parametrize(
    "p,q",
    [
        ((), ()),
        ((0j,), (0j,)),
        ((1j,), (0j,)),
        ((0j,), (1j,)),
        ((1e-12,), (1e-12,)),
        ((0.3, 0.2), (0.1, 0.0)),
        ((0.3, 0.2), (0.1, 1e-9)),
    ],
)
def test_phase_boundary_pairs_match_source(
    p: tuple[complex, ...], q: tuple[complex, ...]
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import qsp_phase_factors as source

    before = p, q
    actual = qsp_phase_factors(p, q)
    expected = source(p, q)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert before == (p, q)


def test_unequal_lengths_preserve_source_error() -> None:
    from qualtran.bloqs.qsp.generalized_qsp import qsp_phase_factors as source

    with pytest.raises(ValueError) as original:
        source((0.3,), (0.2, 0.4))
    with pytest.raises(ValueError) as native:
        qsp_phase_factors((0.3,), (0.2, 0.4))
    assert str(native.value) == str(original.value)
