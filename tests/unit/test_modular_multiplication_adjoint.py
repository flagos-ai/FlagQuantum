"""Pinned constant modular multiplication adjoint without inverse-k resynthesis."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.modular_multiplication import (
    controlled_modular_multiply_constant_adjoint_ir,
    controlled_modular_multiply_constant_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_constant_addition_adjoint import inverse_constant_binding
from tests.unit.test_modular_multiplication import CASES, basis_inputs, layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p,k", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_original_inverse_source_and_separate_model_weights(
    n: int, p: int, k: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, w, bits = layout(n)
    result, inner = w[:n], w[n:]
    first, measured = inverse_constant_binding(target, inner[1:], bits, p)
    second, measured2 = inverse_constant_binding(result, inner[1:], bits, p)
    assert measured == measured2
    measured[inner[0]] = bits[0]
    alloc = (*result, *(inner[0],) * n, *(first * n), *(inner[0],) * n, *(second * n))
    source = CModMulK((QMontgomeryUInt if montgomery else QUInt)(n), k, p).adjoint()
    assert CSwap(n).adjoint() == CSwap(n)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = controlled_modular_multiply_constant_adjoint_ir(0, target, k, p, w, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == 40 * n * n + 23 * n
    )
    assert model.total_t_count() == 40 * n * n + 20 * n
    assert counts["measure"] == model.measurement == 2 * n * (5 * n + 2)
    assert get_cost_value(source, QubitCount()) == 1 + n + len(w)
    assert counts == Counter(
        g.name
        for g in controlled_modular_multiply_constant_ir(0, target, k, p, w, bits)
    )


@pytest.mark.parametrize("n,p,k", CASES)
def test_every_valid_residue_all_branches_and_restore(n: int, p: int, k: int) -> None:
    target, w, bits = layout(n)
    before, expected = basis_inputs(n, p, pow(k, -1, p))
    actual, m = certified_and_channel_map(
        controlled_modular_multiply_constant_adjoint_ir(0, target, k, p, w, bits),
        before,
    )
    assert actual == expected and m == 2 * n * (5 * n + 2)
    restored, m2 = certified_and_channel_map(
        controlled_modular_multiply_constant_ir(0, target, k, p, w, bits), actual
    )
    assert restored == before and m2 == m


def test_raw_complex_channels_spectator_and_work_zero() -> None:
    n, p, k = 2, 3, 2
    target, w, bits = layout(n)
    before, after = basis_inputs(n, p, pow(k, -1, p))
    state = torch.zeros(1 << (len(before[0]) + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    v = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        av = sum(bit << (len(a) - 1 - j) for j, bit in enumerate(a))
        bv = sum(bit << (len(b) - 1 - j) for j, bit in enumerate(b))
        for spectator in (0, 1):
            state[(av << 1) | spectator] = amps[2 * i + spectator]
            expected[(bv << 1) | spectator] = amps[2 * i + spectator]
    gates = controlled_modular_multiply_constant_adjoint_ir(0, target, k, p, w, bits)
    m = 2 * n * (5 * n + 2)
    for branch in ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m))):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_not_forward_inverse_multiplier_and_shared_validation() -> None:
    target, w, bits = layout(3)
    exact = controlled_modular_multiply_constant_adjoint_ir(0, target, 2, 5, w, bits)
    other = controlled_modular_multiply_constant_ir(0, target, 3, 5, w, bits)
    assert exact != other
    from flagquantum.compiler.registers.controlled_swap import controlled_swap_ir

    swap = controlled_swap_ir(0, target, w[:3])
    assert exact[: len(swap)] == swap
    for k in (0, 6, True, 3):
        with pytest.raises(ValueError, match="constant"):
            controlled_modular_multiply_constant_adjoint_ir(0, target, k, 6, w, bits)
    with pytest.raises(ValueError, match="modulus"):
        controlled_modular_multiply_constant_adjoint_ir(0, target, 1, 8, w, bits)
    with pytest.raises(ValueError, match="clean work"):
        controlled_modular_multiply_constant_adjoint_ir(0, target, 1, 5, w[:-1], bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_multiply_constant_adjoint_ir(
            target[0], target, 1, 5, w, bits
        )


def test_append_example() -> None:
    from examples.algorithms.register_modular_multiplication import main

    main()
