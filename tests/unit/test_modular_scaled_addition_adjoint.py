"""Original reversed CtrlScaleModAdd traversal, channels, work and resources."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular.modular_scaled_addition import (
    controlled_modular_scaled_add_adjoint_ir,
    controlled_modular_scaled_add_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_constant_addition_adjoint import inverse_constant_binding
from tests.unit.test_modular_scaled_addition import CASES, basis_inputs, layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p,k", CASES)
def test_exact_reverse_source_traversal_and_models(n: int, p: int, k: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    alloc, measured = inverse_constant_binding(b, w[1:], bits, p)
    measured[w[0]] = bits[0]
    source = CtrlScaleModAdd(k=k, mod=p, bitsize=n).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        (*(w[0],) * n, *(alloc * n)),
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = controlled_modular_scaled_add_adjoint_ir(0, a, b, k, p, w, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == n * (20 * n + 8)
    assert counts["measure"] == model.measurement == n * (5 * n + 2)
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)
    forward = controlled_modular_scaled_add_ir(0, a, b, k, p, w, bits)
    assert counts == Counter(g.name for g in forward)


@pytest.mark.parametrize("n,p,k", CASES)
def test_every_valid_input_every_branch_and_restore(n: int, p: int, k: int) -> None:
    a, b, w, bits = layout(n)
    before, expected = basis_inputs(n, p, -k)
    actual, m = certified_and_channel_map(
        controlled_modular_scaled_add_adjoint_ir(0, a, b, k, p, w, bits), before
    )
    assert actual == expected and m == n * (5 * n + 2)
    restored, m2 = certified_and_channel_map(
        controlled_modular_scaled_add_ir(0, a, b, k, p, w, bits), actual
    )
    assert restored == before and m == m2


@pytest.mark.parametrize("n,p,k", [(1, 1, 0), (2, 3, -1)])
def test_raw_complex_channels_and_work_zero(n: int, p: int, k: int) -> None:
    a, b, w, bits = layout(n)
    before, after = basis_inputs(n, p, -k)
    width = len(before[0]) + 1
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    v = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xval = sum(bit << (len(x) - 1 - j) for j, bit in enumerate(x))
        yval = sum(bit << (len(y) - 1 - j) for j, bit in enumerate(y))
        for s in (0, 1):
            state[(xval << 1) | s] = amps[2 * i + s]
            expected[(yval << 1) | s] = amps[2 * i + s]
    gates = controlled_modular_scaled_add_adjoint_ir(0, a, b, k, p, w, bits)
    m = n * (5 * n + 2)
    branches = (
        product((0, 1), repeat=m)
        if n == 1
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_different_from_negated_scale_resynthesis_and_validation() -> None:
    a, b, w, bits = layout(3)
    exact = controlled_modular_scaled_add_adjoint_ir(0, a, b, 2, 5, w, bits)
    resynthesized = controlled_modular_scaled_add_ir(0, a, b, -2, 5, w, bits)
    assert exact != resynthesized
    # First outer And uses the low source bit in the original reverse traversal.
    assert exact[0].wires == (w[0],)  # H on And target precedes the first CX.
    assert exact[3].wires == (a[-1], w[0])
    for k in (True, 1.5):
        with pytest.raises(ValueError, match="concrete"):
            controlled_modular_scaled_add_adjoint_ir(0, a, b, k, 5, w, bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_scaled_add_adjoint_ir(a[0], a, b, 2, 5, w, bits)


def test_user_append_exact_adjoint_example() -> None:
    from examples.algorithms.register_modular_scaled_addition import main

    main()
