"""Original exchange object identities and complete phase-sensitive consumers."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_swap import (
    controlled_swap_classical_values,
    swap_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PAIRS = [
    (0, 1),
    (-5, 2**90),
    (np.int8(-1), np.uint8(255)),
    (np.array([0, 1]), np.array([[2], [3]])),
    ([], [1]),
    (None, object),
    (sympy.Symbol("x"), sympy.Integer(2)),
]
CONTROLS = [
    0,
    1,
    2,
    -1,
    True,
    False,
    np.uint8(0),
    np.int8(1),
    np.array([0]),
    np.array([1]),
    np.array([0, 1]),
    sympy.Integer(1),
    sympy.Symbol("c"),
]


@pytest.mark.parametrize("kind", ["bit", "word"])
@pytest.mark.parametrize("width", [0, 1, 3, -1, np.int64(3), sympy.Symbol("n")])
@pytest.mark.parametrize("left,right", PAIRS)
def test_original_uncontrolled_raw_objects_and_unused_width(kind, width, left, right):
    from qualtran.bloqs.basic_gates.swap import Swap, TwoBitSwap

    source = TwoBitSwap() if kind == "bit" else Swap(width)
    compare_calls(
        lambda x, y: tuple(source.on_classical_vals(x, y).values()),
        swap_classical_values,
        (left, right),
    )
    output = swap_classical_values(left, right)
    assert output[0] is right and output[1] is left


@pytest.mark.parametrize("kind", ["bit", "word", "relative"])
@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("left,right", PAIRS)
def test_original_control_comparison_literal_output_and_invalid_controls(
    kind, control, left, right
):
    from qualtran.bloqs.basic_gates.swap import CSwap, TwoBitCSwap
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    source = {
        "bit": TwoBitCSwap(),
        "word": CSwap(sympy.Symbol("n")),
        "relative": CSwapApprox(sympy.Symbol("n")),
    }[kind]
    compare_calls(
        lambda c, x, y: tuple(source.on_classical_vals(c, x, y).values()),
        controlled_swap_classical_values,
        (control, left, right),
    )


def test_literal_control_and_alias_preservation():
    value = np.array([1, 2])
    for control in (np.uint8(0), np.uint8(1), np.array([1]), True):
        output = controlled_swap_classical_values(control, value, value)
        assert type(output[0]) is int
        assert output[1] is output[2] is value


@pytest.mark.parametrize("controlled", [False, True])
def test_complete_complex_raw_permutation_exact_stream_and_inverse(controlled):
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.controlled_swap import controlled_swap_ir
    from flagquantum.compiler.registers.swap import swap_register_ir

    gates = (
        controlled_swap_ir(0, (1,), (2,))
        if controlled
        else swap_register_ir((1,), (2,))
    )
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for control, left, right, spectator in product((0, 1), repeat=4):
        x, y = (
            controlled_swap_classical_values(control, left, right)[1:]
            if controlled
            else swap_classical_values(left, right)
        )
        before = (control << 3) | (left << 2) | (right << 1) | spectator
        after = (control << 3) | (x << 2) | (y << 1) | spectator
        expected[after] = initial[before]
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    assert all(g.name not in ("measure", "reset") for g in gates)


def test_relative_callback_permutation_does_not_erase_original_quantum_phases():
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.relative_controlled_swap import (
        relative_controlled_swap_ir,
    )
    from tests.unit.test_relative_controlled_swap import (
        assert_complex_stream,
        source_relative_swap,
    )

    gates = relative_controlled_swap_ir(0, (1,), (2,))
    inverse = relative_controlled_swap_ir(0, (1,), (2,), adjoint=True)
    source = source_relative_swap(0, (1,), (2,))
    assert gates == source
    assert_complex_stream(gates, source, inverse, 3)
    grid = torch.arange(1, 9, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    permutation = torch.zeros_like(initial)
    for c, x, y in product((0, 1), repeat=3):
        ctrl, left, right = controlled_swap_classical_values(c, x, y)
        permutation[(ctrl << 2) | (left << 1) | right] = initial[
            (c << 2) | (x << 1) | y
        ]
    actual = (
        Circuit(3, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(3, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual.abs(), permutation.abs(), rtol=0, atol=2e-12)
    assert not torch.allclose(actual, permutation, rtol=0, atol=2e-12)
