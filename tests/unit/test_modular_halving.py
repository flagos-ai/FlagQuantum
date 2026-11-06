"""Exact source ModDbl adjoint and inverse logical-port ownership."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_doubling import (
    modular_double_ir,
    modular_double_outputs,
)
from flagquantum.compiler.registers.modular_halving import (
    modular_half_ir,
    modular_half_outputs,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_doubling import CASES, basis, coherent, layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_source_adjoint_gates_ports_and_models(
    n: int, p: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    t, w, b = layout(n)
    high, sign = w[:2]
    constant, carry = w[2 : n + 4], w[n + 4 :]
    source = ModDbl((QMontgomeryUInt if montgomery else QUInt)(n), p).adjoint()
    out, clean = modular_half_outputs(t, w)
    ref = reference_arithmetic_gates(
        source,
        {"x": t},
        (sign, high, *constant[: n + 1], *carry[:n], *constant, *carry),
        dict(zip(carry, b, strict=True)),
        reuse_allocations=False,
        expected_outputs={"x": out},
    )
    gates = modular_half_ir(t, p, w, b)
    assert_source(gates, ref)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 8 * n + 4
    assert counts["measure"] == model.measurement == 2 * n + 1
    assert get_cost_value(source, QubitCount()) == n + len(w)
    assert len(clean) == len(w)


@pytest.mark.parametrize("n,p", CASES)
def test_all_inputs_all_branches_and_forward_restore(n: int, p: int) -> None:
    t, w, b = layout(n)
    before, expected = basis(n, p, half=True)
    actual, m = certified_and_channel_map(modular_half_ir(t, p, w, b), before)
    assert actual == expected and m == 2 * n + 1
    out, clean = modular_half_outputs(t, w)
    restored, m2 = certified_and_channel_map(
        modular_double_ir(out, p, clean, b), actual
    )
    assert (
        restored == before and m2 == m and modular_double_outputs(out, clean) == (t, w)
    )

    before, expected = basis(n, p)
    actual, m = certified_and_channel_map(modular_double_ir(t, p, w, b), before)
    assert actual == expected
    out, clean = modular_double_outputs(t, w)
    restored, m2 = certified_and_channel_map(modular_half_ir(out, p, clean, b), actual)
    assert restored == before and m2 == m and modular_half_outputs(out, clean) == (t, w)


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 5)])
def test_raw_inverse_complex_amplitudes_and_clean_returned_work(n: int, p: int) -> None:
    t, w, b = layout(n)
    g = modular_half_ir(t, p, w, b)
    state, expected = coherent(n, p, half=True)
    m = 2 * n + 1
    branches = (
        product((0, 1), repeat=m)
        if n <= 2
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(state, g, branch), expected / 2 ** (m / 2), rtol=0, atol=1e-12
        )


def test_append_example_and_returned_port_ownership() -> None:
    from examples.algorithms.register_modular_doubling import main

    main()
