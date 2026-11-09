"""Original ModAdd inverse, source comparator palindrome and clean work reuse."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.linear_comparison import (
    linear_greater_ir,
)
from flagquantum.compiler.registers.modular.modular_addition import (
    _modular_workspace,
    modular_add_adjoint_ir,
    modular_add_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_addition import (
    CASES,
    layout,
    reference_binding,
    residue_states,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def inverse_binding(n: int, p: int) -> tuple[tuple[int, ...], dict[int, int]]:
    a, b, w, bits = layout(n)
    junk, sign, k, c, comp = _modular_workspace(a, b, p, w, bits)
    compare = comp if n == 1 else (comp[1], comp[0], *comp[2:])
    _, measured = reference_binding(n, p)
    return (sign, junk, *compare, *k[:n], *c[: n - 1], *k, *c, *c), measured


def basis_inputs(
    n: int, p: int, *, include_p: bool, subtract: bool
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    _, _, w, _ = layout(n)
    q, width = len(w), 2 * n + len(w)
    before, after = [], []
    for x in range(p + int(include_p)):
        for y in range(p):
            z = (y - x if subtract else y + x) % p
            a, b = ((x << n) | y) << q, ((x << n) | z) << q
            before.append(tuple((a >> (width - 1 - i)) & 1 for i in range(width)))
            after.append(tuple((b >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(before), tuple(after)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
def test_full_source_inverse_and_independent_models(n: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    source = ModAdd(n, p).adjoint()
    alloc, measured = inverse_binding(n, p)
    expected = reference_arithmetic_gates(
        source,
        {"x": a, "y": b},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = modular_add_adjoint_ir(a, b, p, w, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 16 * n - 4
    assert counts["measure"] == model.measurement == 4 * n - 1
    assert get_cost_value(source, QubitCount()) == 4 * n + 2
    assert counts == Counter(g.name for g in modular_add_ir(a, b, p, w, bits))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_original_unsigned_comparator_inverse_palindrome(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    a, b, w, bits = layout(n)
    _, sign, _, _, comp = _modular_workspace(a, b, 1, w, bits)
    gates = linear_greater_ir(a, b, sign, comp, bits)
    alloc = comp if n == 1 else (comp[1], comp[0], *comp[2:])
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    expected = reference_arithmetic_gates(
        LinearDepthGreaterThan(n, signed=False).adjoint(),
        {"a": a, "b": b, "target": (sign,)},
        alloc,
        measured,
        preserve_components=True,
    )
    assert_source(gates, expected)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_comparator_every_column_with_dirty_target(n: int) -> None:
    a, b, w, bits = layout(n)
    _, target, _, _, comp = _modular_workspace(a, b, 1, w, bits)
    # Compact remapping includes an arbitrary target, not an assumed zero sign.
    gates = linear_greater_ir(
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        2 * n,
        tuple(range(2 * n + 1, 2 * n + 1 + len(comp))),
        bits,
    )
    width = 2 * n + 1 + len(comp)
    before, after = [], []
    for x, y, t in product(range(1 << n), range(1 << n), (0, 1)):
        av = (((x << n) | y) << 1 | t) << len(comp)
        bv = (((x << n) | y) << 1 | (t ^ int(x > y))) << len(comp)
        before.append(tuple((av >> (width - 1 - i)) & 1 for i in range(width)))
        after.append(tuple((bv >> (width - 1 - i)) & 1 for i in range(width)))
    actual, m = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(after) and m == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_comparator_full_complex_dirty_target_spectator_all_raw_branches(
    n: int,
) -> None:
    _, _, w, bits = layout(n)
    a, b, _, _ = layout(n)
    _, _, _, _, comp = _modular_workspace(a, b, 1, w, bits)
    q = len(comp)
    target = 2 * n
    workspace = tuple(range(target + 1, target + 1 + q))
    gates = linear_greater_ir(
        tuple(range(n)), tuple(range(n, 2 * n)), target, workspace, bits
    )
    labels = [
        (x, y, t, s)
        for x, y, t, s in product(range(1 << n), range(1 << n), (0, 1), (0, 1))
    ]
    v = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (2 * n + q + 2), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (x, y, t, s), amp in zip(labels, amps, strict=True):
        before = (((x << n) | y) << 1) | t
        after = (((x << n) | y) << 1) | (t ^ int(x > y))
        state[(before << (q + 1)) | s] = amp
        expected[(after << (q + 1)) | s] = amp
    for branch in product((0, 1), repeat=n):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (n / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("n,p", CASES)
def test_every_valid_input_every_branch_and_restore(n: int, p: int) -> None:
    a, b, w, bits = layout(n)
    before, expected = basis_inputs(n, p, include_p=True, subtract=True)
    actual, m = certified_and_channel_map(
        modular_add_adjoint_ir(a, b, p, w, bits), before
    )
    assert actual == expected and m == 4 * n - 1
    restored, m2 = certified_and_channel_map(modular_add_ir(a, b, p, w, bits), actual)
    assert restored == before and m2 == m


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 5)])
def test_raw_complex_channels_spectator_and_work_zero(n: int, p: int) -> None:
    a, b, w, bits = layout(n)
    state, expected = residue_states(n, p, left_modulus=True, subtract=True)
    gates = modular_add_adjoint_ir(a, b, p, w, bits)
    m = 4 * n - 1
    branches = (
        product((0, 1), repeat=m)
        if n <= 2
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_shared_validation_and_distinct_modsub_stream() -> None:
    from flagquantum.compiler.registers.modular.modular_subtraction import (
        modular_subtract_ir,
    )

    a, b, w, bits = layout(3)
    exact = modular_add_adjoint_ir(a, b, 3, w, bits)
    other = modular_subtract_ir(a, b, 3, w, bits)
    assert sum(g.name in {"t", "tdg"} for g in exact) == 44
    assert sum(g.name in {"t", "tdg"} for g in other) == 60
    with pytest.raises(ValueError, match="classical"):
        modular_add_adjoint_ir(a, b, 3, w, ())
    with pytest.raises(ValueError, match="distinct"):
        modular_add_adjoint_ir(a, a, 3, w, bits)
    with pytest.raises(ValueError, match="Modulus"):
        modular_add_adjoint_ir(a, b, 8, w, bits)
