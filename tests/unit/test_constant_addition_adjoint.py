"""Literal AddK inverse and controlled inverse keep the original loaded constant."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic import (
    add_constant_adjoint_ir,
    add_constant_ir,
)
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
VALUES = [
    (n, k) for n in (1, 2, 3, 5, 8) for k in (0, 1, (1 << n) - 1, -1, -(1 << n) - 3)
]


def layout(n: int, controlled: bool) -> tuple[tuple[int, ...], ...]:
    p = int(controlled)
    return (
        tuple(range(p, p + n)),
        tuple(range(p + n, p + 2 * n)),
        tuple(range(p + 2 * n, p + 3 * n - 1)),
        tuple(range(40, 40 + n - 1)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,k", VALUES)
@pytest.mark.parametrize("cv", [None, 0, 1])
def test_literal_constant_inverse_gates_feedback_and_source_model(
    n: int, k: int, cv: int | None
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    target, constant, carry, bits = layout(n, cv is not None)
    source = AddK(QUInt(n), k)
    inputs = {"x": target}
    if cv is not None:
        source = source.controlled(CtrlSpec(cvs=cv))
        inputs["ctrl"] = (0,)
        gates = controlled_add_constant_adjoint_ir(
            0, target, k, constant, carry, bits, control_value=cv
        )
    else:
        gates = add_constant_adjoint_ir(target, k, constant, carry, bits)
    source = source.adjoint()
    expected = reference_arithmetic_gates(
        source, inputs, (*constant, *carry), dict(zip(carry, bits, strict=True))
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert counts["measure"] == model.measurement == n - 1
    assert counts["reset"] == n - 1


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [0, 1, -1])
@pytest.mark.parametrize("cv", [None, 0, 1])
def test_every_complex_control_target_spectator_and_branch(
    n: int, k: int, cv: int | None
) -> None:
    target, constant, carry, bits = layout(n, cv is not None)
    p, work = int(cv is not None), 2 * n - 1
    width = p + 3 * n
    data = torch.arange(1 << (p + n + 1), dtype=torch.int64)
    x, spectator = (data >> 1) & ((1 << n) - 1), data & 1
    active = (
        (data >> (n + 1)) == cv
        if cv is not None
        else torch.ones_like(x, dtype=torch.bool)
    )
    mapped = ((data >> 1) & ~((1 << n) - 1)) | ((x - active * k) % (1 << n))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(mapped << (work + 1)) | spectator] = amps
    if cv is None:
        inverse = add_constant_adjoint_ir(target, k, constant, carry, bits)
        forward = add_constant_ir(target, k, constant, carry, bits)
    else:
        inverse = controlled_add_constant_adjoint_ir(
            0, target, k, constant, carry, bits, control_value=cv
        )
        forward = controlled_add_constant_ir(
            0, target, k, constant, carry, bits, control_value=cv
        )
    for outcomes in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(state, inverse, outcomes),
            expected / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            forced_branch(expected, forward, outcomes),
            state / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


def test_constant_inverse_validation() -> None:
    target, constant, carry, bits = layout(2, True)
    for k in (True, 4):
        with pytest.raises(ValueError, match="Additive constant"):
            add_constant_adjoint_ir(target, k, constant, carry, bits)
    for cv in (True, 2):
        with pytest.raises(ValueError, match="zero or one"):
            controlled_add_constant_adjoint_ir(
                0, target, 1, constant, carry, bits, control_value=cv
            )
    with pytest.raises(ValueError, match="nonnegative"):
        controlled_add_constant_adjoint_ir(-1, target, 1, constant, carry, bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_add_constant_adjoint_ir(target[0], target, 1, constant, carry, bits)


def test_literal_inverse_is_not_negative_constant_relowering() -> None:
    target, constant, carry, bits = layout(3, False)
    literal = add_constant_adjoint_ir(target, 1, constant, carry, bits)
    equivalent = add_constant_ir(target, -1, constant, carry, bits)
    assert Counter(g.name for g in literal)["x"] == 2
    assert Counter(g.name for g in equivalent)["x"] == 6
    assert (
        literal != equivalent
    )  # Same permutation does not certify source gate parity.


def test_user_append_inverse_addition_example() -> None:
    from examples.algorithms.register_inverse_addition import main

    main()
