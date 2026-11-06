"""Original sequential parity/marker updates, reusable clean work and inverse."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.inverse_parity import inverse_parity_ir
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        2 * n,
        2 * n + 1,
        2 * n + 2,
        2 * n + 3,
        (2 * n + 4, 2 * n + 5),
        (40, 41),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_gates_feedback_outputs_and_costs(n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep2
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, b, a, m, f, w, bits = layout(n)
    source = _KaliskiIterationStep2(n)
    if adjoint:
        source = source.adjoint()
    gates = inverse_parity_ir(u, v, b, a, m, f, w, bits, adjoint=adjoint)
    # The shared slot is overwritten: bind measurement events in source order.
    expected = reference_arithmetic_gates(
        source,
        {"u": u, "v": v, "b": (b,), "a": (a,), "m": (m,), "f": (f,)},
        w,
        dict(zip(w, bits, strict=True)),
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12
    assert counts["measure"] == counts["reset"] == model.measurement == 3
    assert counts["cx"] == 22
    assert counts["x"] == 12
    assert get_cost_value(source, QubitCount()) == 2 * n + 6


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_every_input_every_history_and_inverse(n: int) -> None:
    u, v, b, a, m, f, w, bits = layout(n)
    outputs = []
    for raw in range(1 << (2 * n + 4)):
        x, y = raw >> (n + 4), raw >> 4 & ((1 << n) - 1)
        branch, parity, marker, active = (
            raw >> 3 & 1,
            raw >> 2 & 1,
            raw >> 1 & 1,
            raw & 1,
        )
        parity ^= int(x % 2 == 0) & active
        marker ^= int(y % 2 == 0) & (parity ^ 1) & active
        branch ^= parity ^ marker
        outputs.append(
            (x << (n + 4))
            | (y << 4)
            | (branch << 3)
            | (parity << 2)
            | (marker << 1)
            | active
        )
    check_flag_channel(
        inverse_parity_ir(u, v, b, a, m, f, w, bits),
        inverse_parity_ir(u, v, b, a, m, f, w, bits, adjoint=True),
        tuple(outputs),
        2 * n + 4,
        2,
        3,
        raw_branches=n <= 3,
    )


@pytest.mark.parametrize("bad", ["width", "adjoint", "alias", "work", "slots", "bits"])
def test_layout_rejections(bad: str) -> None:
    u, v, b, a, m, f, w, bits = layout(2)
    inverse = False
    if bad == "width":
        u = ()
    elif bad == "adjoint":
        inverse = 1
    elif bad == "alias":
        m = a
    elif bad == "work":
        w = w[:1]
    elif bad == "slots":
        bits = (40,)
    else:
        bits = (40, 40)
    with pytest.raises(ValueError):
        inverse_parity_ir(u, v, b, a, m, f, w, bits, adjoint=inverse)
