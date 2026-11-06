"""Original ModSub inverse conjugation, full complex channels and shared pools."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_addition import (
    _modular_workspace,
    modular_add_ir,
)
from flagquantum.compiler.registers.modular_subtraction import (
    modular_subtract_adjoint_ir,
    modular_subtract_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_addition import layout, residue_states
from tests.unit.test_modular_addition_adjoint import basis_inputs, inverse_binding
from tests.unit.test_modular_subtraction import CASES
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_original_source_inverse_and_independent_models(
    n: int, p: int, montgomery: bool
) -> None:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModSub
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    _, _, k, c, _ = _modular_workspace(a, b, p, w, bits)
    alloc, measured = inverse_binding(n, p)
    outside = (*k[:n], *c[: n - 1])
    source = ModSub((QMontgomeryUInt if montgomery else QUInt)(n), p).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"x": a, "y": b},
        (*outside, *alloc, *outside),
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = modular_subtract_adjoint_ir(a, b, p, w, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 24 * n - 12
    assert counts["measure"] == model.measurement == 6 * n - 3
    assert get_cost_value(source, QubitCount()) == 4 * n + 2
    assert counts == Counter(g.name for g in modular_subtract_ir(a, b, p, w, bits))


@pytest.mark.parametrize("n,p", CASES)
def test_every_valid_input_every_branch_and_restore(n: int, p: int) -> None:
    a, b, w, bits = layout(n)
    before, expected = basis_inputs(n, p, include_p=False, subtract=False)
    actual, m = certified_and_channel_map(
        modular_subtract_adjoint_ir(a, b, p, w, bits), before
    )
    assert actual == expected and m == 6 * n - 3
    restored, m2 = certified_and_channel_map(
        modular_subtract_ir(a, b, p, w, bits), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("n,p", [(2, 2), (3, 3)])
def test_raw_complex_channels_spectator_and_work_zero(n: int, p: int) -> None:
    a, b, w, bits = layout(n)
    state, expected = residue_states(n, p)
    gates = modular_subtract_adjoint_ir(a, b, p, w, bits)
    m = 6 * n - 3
    branches = (
        product((0, 1), repeat=m)
        if n == 2
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_distinct_forward_modadd_and_boundary_validation() -> None:
    a, b, w, bits = layout(3)
    exact = modular_subtract_adjoint_ir(a, b, 3, w, bits)
    other = modular_add_ir(a, b, 3, w, bits)
    assert sum(g.name in {"t", "tdg"} for g in exact) == 60
    assert sum(g.name in {"t", "tdg"} for g in other) == 44
    with pytest.raises(ValueError, match="modulus\\+1"):
        modular_subtract_adjoint_ir(a, b, 7, w, bits)
    with pytest.raises(ValueError, match="clean work"):
        modular_subtract_adjoint_ir(a, b, 3, w[:-1], bits)
    with pytest.raises(ValueError, match="equal"):
        modular_subtract_adjoint_ir(a, b[:1], 3, w, bits)


def test_append_example() -> None:
    from examples.algorithms.register_modular_arithmetic import main

    main()
