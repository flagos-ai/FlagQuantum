"""Pinned signed comparator gates, exact branches and explicit source defects."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.linear_comparison import (
    linear_greater_ir,
    linear_greater_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...], tuple[int, ...]]:
    q, c = linear_greater_requirements(n, signed=True)
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        2 * n,
        tuple(range(2 * n + 1, 2 * n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


def source_predicate(n: int, a: int, b: int) -> int:
    # This is the executed source contract, including overflow and n=1.
    return int(a > b) if n == 1 else ((b - a) % (1 << n)) >> (n - 1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_gates_feedback_and_cost(n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, t, w, bits = layout(n)
    gates = linear_greater_ir(a, b, t, w, bits, signed=True)
    source = LinearDepthGreaterThan(n, signed=True)
    if adjoint:
        source = source.adjoint()
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    expected = reference_arithmetic_gates(
        source,
        {"a": a, "b": b, "target": (t,)},
        w,
        measured,
        preserve_components=True,
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    c = max(1, n - 1)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * c
    assert counts["measure"] == counts["reset"] == model.measurement == c
    assert get_cost_value(source, QubitCount()) == 2 * n + 1 + len(w)
    if n >= 2:
        assert counts["cx"] == 12 * n - 13
        assert counts["x"] == 2 * n + 2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_source_execution_all_columns_and_clean_work(n: int) -> None:
    a, b, t, w, bits = layout(n)
    width, q = 2 * n + 1 + len(w), len(w)
    # Exhaust all input patterns through five bits; eight bits uses edge cases.
    values = range(1 << n) if n <= 5 else (0, 1, 63, 127, 128, 129, 254, 255)
    before, after = [], []
    for x, y, z in product(values, values, (0, 1)):
        v = (((x << n) | y) << 1 | z) << q
        out = (((x << n) | y) << 1 | (z ^ source_predicate(n, x, y))) << q
        before.append(tuple((v >> (width - 1 - i)) & 1 for i in range(width)))
        after.append(tuple((out >> (width - 1 - i)) & 1 for i in range(width)))
    gates = linear_greater_ir(a, b, t, w, bits, signed=True)
    actual, events = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(after) and events == len(bits)
    restored, _ = certified_and_channel_map(gates, actual)
    assert restored == tuple(before)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.integration
def test_complex_amplitudes_spectator_all_raw_measurement_branches(n: int) -> None:
    a, b, t, w, bits = layout(n)
    q, events = len(w), len(bits)
    labels = list(product(range(1 << n), range(1 << n), (0, 1), (0, 1)))
    v = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (2 * n + q + 2), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (x, y, z, spectator), amp in zip(labels, amps, strict=True):
        before = (((x << n) | y) << 1) | z
        after = (((x << n) | y) << 1) | (z ^ source_predicate(n, x, y))
        state[(before << (q + 1)) | spectator] = amp
        expected[(after << (q + 1)) | spectator] = amp
    gates = linear_greater_ir(a, b, t, w, bits, signed=True)
    for branch in product((0, 1), repeat=events):
        result = forced_branch(state, gates, branch)
        torch.testing.assert_close(
            result, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            forced_branch(result, gates, branch),
            state / 2**events,
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "n,a,b,executed,mathematical",
    [(1, 1, 0, 1, 0), (2, 3, 0, 0, 0), (2, 2, 1, 1, 0), (3, 3, 4, 0, 1)],
)
def test_unmodified_source_classical_and_overflow_exceptions(
    n: int,
    a: int,
    b: int,
    executed: int,
    mathematical: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    source = LinearDepthGreaterThan(n, signed=True)
    result = source.decompose_bloq().call_classically(a=a, b=b, target=0)
    assert result == (a, b, executed)
    assert source.on_classical_vals(a=a, b=b, target=0)["target"] == int(a > b)
    signed_a = a - ((1 << n) if a >> (n - 1) else 0)
    signed_b = b - ((1 << n) if b >> (n - 1) else 0)
    assert int(signed_a > signed_b) == mathematical
    assert executed == source_predicate(n, a, b)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_mathematical_signed_comparison_on_no_overflow_domain(n: int) -> None:
    limit = 1 << (n - 1)
    for a, b in product(range(-limit, limit), repeat=2):
        if -limit <= b - a < limit:
            assert source_predicate(n, a % (1 << n), b % (1 << n)) == int(a > b)


def test_signed_validation_and_unsigned_budget_unchanged() -> None:
    with pytest.raises(ValueError, match="boolean"):
        linear_greater_requirements(3, signed=1)  # type: ignore[arg-type]
    a, b, t, w, bits = layout(3)
    with pytest.raises(ValueError, match="clean work"):
        linear_greater_ir(a, b, t, w[:-1], bits, signed=True)
    with pytest.raises(ValueError, match="classical"):
        linear_greater_ir(a, b, t, w, bits[:-1], signed=True)
    for n in (1, 2, 3, 8):
        assert linear_greater_requirements(n) == ((1, 1) if n == 1 else (n + 2, n))
