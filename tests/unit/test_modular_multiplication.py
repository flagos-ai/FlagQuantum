"""Pinned CModMulK scale/clear/exact-swap chain with clean coherent channels."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.modular_multiplication import (
    controlled_modular_multiply_constant_ir,
    modular_multiplication_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_constant_addition import (
    reference_binding as constant_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(2, 3, 1), (2, 3, 2), (3, 5, 2), (3, 6, 5), (5, 13, 7), (8, 251, 173)]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_multiplication_requirements(n)
    return (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, n + q + 1)),
        tuple(range(40, 40 + c)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p,k", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_original_source_components_feedback_and_resource_models(
    n: int, p: int, k: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, w, bits = layout(n)
    result, inner = w[:n], w[n:]
    first, m = constant_binding(result, inner[1:], bits, p)
    second, m2 = constant_binding(target, inner[1:], bits, p)
    assert m == m2
    m[inner[0]] = bits[0]
    alloc = (*result, *(inner[0],) * n, *(first * n), *(inner[0],) * n, *(second * n))
    source = CModMulK((QMontgomeryUInt if montgomery else QUInt)(n), k, p)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        m,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_multiply_constant_ir(0, target, k, p, w, bits)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == 40 * n * n + 23 * n
    )
    assert model.total_t_count() == 40 * n * n + 20 * n
    assert counts["measure"] == model.measurement == 2 * n * (5 * n + 2)
    assert get_cost_value(source, QubitCount()) == 1 + n + len(w)
    assert counts["cx"] == 2 * n * (58 * n + 7 + 2 * p.bit_count()) + 8 * n


def basis_inputs(
    n: int, p: int, k: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    q, _ = modular_multiplication_requirements(n)
    width = 1 + n + q
    before, after = [], []
    for c in (0, 1):
        for x in range(p):
            y = (x * k) % p if c else x
            a, b = ((c << n) | x) << q, ((c << n) | y) << q
            before.append(tuple((a >> (width - 1 - i)) & 1 for i in range(width)))
            after.append(tuple((b >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(before), tuple(after)


@pytest.mark.parametrize("n,p,k", CASES)
def test_every_residue_all_branches_zero_work_and_inverse_multiplier(
    n: int, p: int, k: int
) -> None:
    target, w, bits = layout(n)
    before, expected = basis_inputs(n, p, k)
    gates = controlled_modular_multiply_constant_ir(0, target, k, p, w, bits)
    actual, events = certified_and_channel_map(gates, before)
    assert actual == expected and events == 2 * n * (5 * n + 2)
    restored, again = certified_and_channel_map(
        controlled_modular_multiply_constant_ir(0, target, pow(k, -1, p), p, w, bits),
        actual,
    )
    assert restored == before and again == events


def test_raw_primitive_full_complex_channels_with_spectator() -> None:
    n, p, k = 2, 3, 2
    target, w, bits = layout(n)
    before, after = basis_inputs(n, p, k)
    state = torch.zeros(1 << (len(before[0]) + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    coeff = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        av = sum(bit << (len(a) - 1 - j) for j, bit in enumerate(a))
        bv = sum(bit << (len(b) - 1 - j) for j, bit in enumerate(b))
        for s in (0, 1):
            state[(av << 1) | s] = amps[2 * i + s]
            expected[(bv << 1) | s] = amps[2 * i + s]
    gates = controlled_modular_multiply_constant_ir(0, target, k, p, w, bits)
    m = 2 * n * (5 * n + 2)
    for branch in ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m))):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
def test_validation_and_source_outside_residue_promise() -> None:
    target, w, bits = layout(3)
    for k in (0, 6, True, 3):
        with pytest.raises(ValueError, match="constant"):
            controlled_modular_multiply_constant_ir(0, target, k, 6, w, bits)
    for p in (1, 8, True):
        with pytest.raises(ValueError, match="modulus"):
            controlled_modular_multiply_constant_ir(0, target, 1, p, w, bits)
    with pytest.raises(ValueError, match="positive"):
        modular_multiplication_requirements(0)
    with pytest.raises(ValueError, match="clean work"):
        controlled_modular_multiply_constant_ir(0, target, 1, 5, w[:-1], bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_multiply_constant_ir(target[0], target, 1, 5, w, bits)
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK

    source = CModMulK(QUInt(2), 2, 3)
    assert source.call_classically(ctrl=1, x=3) == (1, 3)
    assert source.decompose_bloq().call_classically(ctrl=1, x=3) == (1, 0)
    with pytest.raises((ValueError, AssertionError)):
        source.decompose_bloq().flatten().call_classically(ctrl=1, x=3)
    for k in (0, 3):
        with pytest.raises(AssertionError):
            CModMulK(QUInt(2), k, 3)


def test_user_append_example() -> None:
    from examples.algorithms.register_modular_multiplication import main

    main()
