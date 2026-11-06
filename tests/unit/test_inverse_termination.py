"""Original termination detection and exact inverse, retaining logical flags."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.inverse_termination import inverse_termination_ir
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[tuple[int, ...], int, int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(range(n)),
        n,
        n + 1,
        n + 2,
        tuple(range(n + 3, 2 * n + 3)),
        tuple(range(40, 40 + n)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_gates_feedback_outputs_and_costs(n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep1
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    v, m, f, t, w, bits = layout(n)
    source = _KaliskiIterationStep1(n)
    if adjoint:
        source = source.adjoint()
    gates = inverse_termination_ir(v, m, f, t, w, bits, adjoint=adjoint)
    expected = reference_arithmetic_gates(
        source,
        {"v": v, "m": (m,), "f": (f,), "is_terminal": (t,)},
        w,
        dict(zip(w, bits, strict=True)),
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * n
    assert counts["measure"] == counts["reset"] == model.measurement == n
    assert counts["cx"] == 6 * n + 3
    assert counts["x"] == 4 * n
    assert get_cost_value(source, QubitCount()) == 2 * n + 3


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_every_input_every_history_and_inverse(n: int) -> None:
    v, m, f, t, w, bits = layout(n)
    outputs = []
    for raw in range(1 << (n + 3)):
        x, marker, active, terminal = raw >> 3, raw >> 2 & 1, raw >> 1 & 1, raw & 1
        marker ^= active & int(x == 0)
        active ^= marker
        terminal ^= marker
        outputs.append((x << 3) | (marker << 2) | (active << 1) | terminal)
    check_flag_channel(
        inverse_termination_ir(v, m, f, t, w, bits),
        inverse_termination_ir(v, m, f, t, w, bits, adjoint=True),
        tuple(outputs),
        n + 3,
        n,
        n,
        raw_branches=n <= 3,
    )


@pytest.mark.ftqc_reference
def test_original_classical_promise_and_one_bit_source_failure() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep1

    for n in (2, 3):
        source = _KaliskiIterationStep1(n)
        for x in range(1 << n):
            for marker in (0, 1):
                for active in (0, 1):
                    m = marker ^ (active & int(x == 0))
                    assert source.on_classical_vals(
                        v=x, m=marker, f=active, is_terminal=0
                    ) == {"v": x, "m": m, "f": active ^ m, "is_terminal": m}
        with pytest.raises(AssertionError):
            source.on_classical_vals(v=0, m=0, f=1, is_terminal=1)
    with pytest.raises(ValueError, match="at least 3"):
        _KaliskiIterationStep1(1).decompose_bloq()
    with pytest.raises(ValueError, match="at least two"):
        inverse_termination_ir((0,), 1, 2, 3, (4,), (40,))


@pytest.mark.parametrize("bad", ["adjoint", "alias", "work", "bits"])
def test_layout_rejections(bad: str) -> None:
    v, m, f, t, w, bits = layout(2)
    inverse = False
    if bad == "adjoint":
        inverse = 1
    elif bad == "alias":
        t = m
    elif bad == "work":
        w = w[:-1]
    else:
        bits = (40, 40)
    with pytest.raises(ValueError):
        inverse_termination_ir(v, m, f, t, w, bits, adjoint=inverse)
