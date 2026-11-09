"""Distinct source greater-than target updates and activation policies."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_greater_comparison import (
    controlled_linear_greater_classical_values,
    linear_greater_classical_values,
)
from tests.unit.test_classical_comparison import PAIRS, TARGETS, compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("left,right", PAIRS)
@pytest.mark.parametrize("target", TARGETS)
def test_binary_assert_then_conditional_add_modulo_matches_original(
    signed, left, right, target
):
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    original = LinearDepthGreaterThan(3, signed=signed)
    compare_calls(
        lambda a, b, t: tuple(original.on_classical_vals(a, b, t).values()),
        linear_greater_classical_values,
        (left, right, target),
    )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", [0, 1, 3, -1, sympy.Integer(3), sympy.Symbol("n")])
def test_unused_source_width_and_signed_mode_do_not_normalize_values(width, signed):
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    original = LinearDepthGreaterThan(width, signed=signed)
    compare_calls(
        lambda a, b, t: tuple(original.on_classical_vals(a, b, t).values()),
        linear_greater_classical_values,
        (3, 0, 0),
    )


@pytest.mark.parametrize(
    "control,cv",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (-1, -1),
        (2, 1),
        (np.int8(1), 1),
        (np.array([1]), 1),
        (np.array([0, 1]), 1),
    ],
)
@pytest.mark.parametrize("target", TARGETS)
def test_controlled_exact_activation_and_no_extra_target_validation(
    control, cv, target
):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan

    original = CLinearDepthGreaterThan(QUInt(3), cv=cv)
    for left, right in PAIRS:
        compare_calls(
            lambda c, a, b, t: tuple(original.on_classical_vals(c, a, b, t).values()),
            lambda c, a, b, t: controlled_linear_greater_classical_values(
                c, a, b, t, control_value=cv
            ),
            (control, left, right, target),
        )


def test_inactive_branch_preserves_unchecked_objects_and_assertion_order():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        CLinearDepthGreaterThan,
        LinearDepthGreaterThan,
    )

    class Operand:
        def __gt__(self, other):
            raise RuntimeError("comparison must not run")

    left, right, target = Operand(), Operand(), object()
    actual = controlled_linear_greater_classical_values(0, left, right, target)
    original = tuple(
        CLinearDepthGreaterThan(QUInt(3))
        .on_classical_vals(0, left, right, target)
        .values()
    )
    for values in (actual, original):
        assert values[1] is left and values[2] is right and values[3] is target
    for function in (
        linear_greater_classical_values,
        lambda a, b, t: tuple(
            LinearDepthGreaterThan(3).on_classical_vals(a, b, t).values()
        ),
    ):
        with pytest.raises(AssertionError):
            function(left, right, 2)
        with pytest.raises(RuntimeError, match="must not run"):
            function(left, right, 1)


def test_boolean_target_add_modulo_type_differs_from_controlled_xor():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        CLinearDepthGreaterThan,
        LinearDepthGreaterThan,
    )

    target = np.bool_(True)
    direct = linear_greater_classical_values(2, 1, target)[-1]
    controlled = controlled_linear_greater_classical_values(1, 2, 1, target)[-1]
    assert type(direct) is type(
        LinearDepthGreaterThan(3).on_classical_vals(2, 1, target)["target"]
    )
    assert type(controlled) is type(
        CLinearDepthGreaterThan(QUInt(3)).on_classical_vals(1, 2, 1, target)["target"]
    )
    assert isinstance(direct, np.integer) and isinstance(controlled, np.bool_)
    assert direct == controlled == 0


@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_native_values_full_complex_states_all_histories_and_work(cv):
    import torch

    from flagquantum.compiler.registers.comparison.controlled_comparison import (
        controlled_greater_ir,
        controlled_greater_requirements,
    )
    from tests.unit.measurement_checks import forced_branch

    n = 1
    q, measurements = controlled_greater_requirements(n)
    work = tuple(range(4, 4 + q))
    gates = controlled_greater_ir(
        0, (1,), (2,), 3, work, tuple(range(40, 40 + measurements)), control_value=cv
    )
    data_bits = 4
    total = data_bits + q + 1
    grid = torch.arange(1, 33, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(16), (0, 1)):
        control, a, b, target = data >> 3, (data >> 2) & 1, (data >> 1) & 1, data & 1
        updated = controlled_linear_greater_classical_values(
            control, a, b, target, control_value=cv
        )[-1]
        output = (data & ~1) | updated
        initial[(data << (q + 1)) | spectator] = weights[2 * data + spectator]
        expected[(output << (q + 1)) | spectator] = weights[2 * data + spectator]
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (measurements / 2), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(actual.reshape(16, 1 << q, 2)[:, 1:, :]) == 0


def test_signed_root_numerical_gate_disagreement_is_preserved_not_repaired():
    import torch
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    from flagquantum.compiler.registers.comparison.linear_comparison import (
        linear_greater_ir,
    )
    from tests.unit.measurement_checks import forced_branch

    original = LinearDepthGreaterThan(2, signed=True)
    assert (
        original.on_classical_vals(3, 0, 0)["target"]
        == linear_greater_classical_values(3, 0, 0)[-1]
        == 1
    )
    gates = linear_greater_ir((0, 1), (2, 3), 4, (5,), (40,), signed=True)
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[3 << 4] = 1j
    for history in ((0,), (1,)):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, initial / 2**0.5, rtol=0, atol=2e-12)
