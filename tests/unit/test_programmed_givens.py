"""Compare measured Givens rotations with an independent two-mode matrix."""

from itertools import product
from math import cos, sin, sqrt, tau

import pytest

from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.phase.programmed_givens import programmed_givens_ir
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bits", [1, 2, 4])
@pytest.mark.parametrize("adjoint", [False, True])
def test_quantum_angle_givens_all_measurement_histories(bits, adjoint):
    angle = tuple(range(bits))
    left, right = bits, bits + 1
    gradient = tuple(range(bits + 2, 2 * bits + 2))
    carries = tuple(range(2 * bits + 2, 3 * bits + 1))
    width = 3 * bits + 1
    scratch = len(gradient) + len(carries)
    gates = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *programmed_givens_ir(
            left,
            right,
            angle,
            gradient,
            carries,
            tuple(range(bits - 1)),
            adjoint=adjoint,
        ),
        *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
    )
    initial = {}
    expected = {}
    words = tuple(range(1 << bits)) if bits < 4 else (0, 1, 5, 7, 15)
    for word, value in product(words, range(4)):
        index = ((word << 2) | value) << scratch
        coefficient = complex(1, (-1) ** (word + value)) / sqrt(8 * len(words))
        initial[index] = coefficient
        theta = 2 * tau * word / (1 << bits) * (-1 if adjoint else 1)
        if value in (0, 3):
            expected[index] = expected.get(index, 0) + coefficient
        else:
            other = index ^ (3 << scratch)
            expected[index] = expected.get(index, 0) + coefficient * cos(theta)
            expected[other] = expected.get(other, 0) + coefficient * sin(theta) * (
                1 if value == 1 else -1
            )
    expected = {basis: amp for basis, amp in expected.items() if abs(amp) > 1e-12}
    events = 2 * (bits - 1)
    assert sum(gate.name == "measure" for gate in gates) == events
    assert sum(gate.name in {"t", "tdg"} for gate in gates) == 4 * events
    for history in product((0, 1), repeat=events):
        assert _branch(
            gates, width, history, initial, normalize_measurements=True
        ) == pytest.approx(expected, abs=2e-12)
