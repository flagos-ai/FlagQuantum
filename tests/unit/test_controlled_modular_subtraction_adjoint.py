"""Literal CModSub adjoint, coherent channels and original conjugation costs."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    _controlled_modular_workspace,
    controlled_modular_add_ir,
)
from flagquantum.compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_adjoint_ir,
    controlled_modular_subtract_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_addition import (
    certificate_inputs,
    coherent_states,
    layout,
)
from tests.unit.test_controlled_modular_addition_adjoint import (
    adjoint_reference_binding,
)
from tests.unit.test_controlled_modular_subtraction import CASES
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_original_adjoint_gates_feedback_and_models(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    _, k, carry, _ = _controlled_modular_workspace(0, a, b, p, w, bits, cv)
    middle, measured = adjoint_reference_binding(n, p, cv)
    outside = (*k[:n], *carry[: n - 1])
    source = CModSub((QMontgomeryUInt if montgomery else QUInt)(n), p, cv).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        (*outside, *middle, *outside),
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = controlled_modular_subtract_adjoint_ir(
        0, a, b, p, w, bits, control_value=cv
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 28 * n - 4
    assert counts["measure"] == model.measurement == 7 * n - 1
    assert counts["cx"] == 82 * n - 29 + 2 * p.bit_count()
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_every_valid_basis_all_branches_and_restore(n: int, p: int, cv: int) -> None:
    a, b, w, bits = layout(n)
    before, _ = certificate_inputs(n, p, cv, subtract=True)
    # The forward subtraction is a permutation on precisely these residue pairs.
    after, events = certified_and_channel_map(
        controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv), before
    )
    restored, inverse_events = certified_and_channel_map(
        controlled_modular_subtract_adjoint_ir(0, a, b, p, w, bits, control_value=cv),
        after,
    )
    assert restored == before and events == inverse_events == 7 * n - 1
    inverse, _ = certified_and_channel_map(
        controlled_modular_subtract_adjoint_ir(0, a, b, p, w, bits, control_value=cv),
        before,
    )
    ideal, _ = certified_and_channel_map(
        controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv), before
    )
    assert inverse == ideal


@pytest.mark.parametrize("n,p", [(2, 2), (3, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_raw_complex_channel_and_work_zero(n: int, p: int, cv: int) -> None:
    a, b, w, bits = layout(n)
    # x=p is also a valid identity for this executable chain; the published
    # acceptance above conservatively retains the original residue-only domain.
    state, expected = coherent_states(n, p, cv)
    gates = controlled_modular_subtract_adjoint_ir(
        0, a, b, p, w, bits, control_value=cv
    )
    m = 7 * n - 1
    for branch in ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m))):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("cv", [0, 1])
def test_distinct_from_equivalent_forward_cmodadd(cv: int) -> None:
    a, b, w, bits = layout(3)
    inverse = controlled_modular_subtract_adjoint_ir(
        0, a, b, 3, w, bits, control_value=cv
    )
    forward = controlled_modular_add_ir(0, a, b, 3, w, bits, control_value=cv)
    assert sum(g.name in {"t", "tdg"} for g in inverse) == 80
    assert sum(g.name in {"t", "tdg"} for g in forward) == 64


def test_preserves_validation() -> None:
    a, b, w, bits = layout(2)
    with pytest.raises(ValueError, match="modulus\\+1"):
        controlled_modular_subtract_adjoint_ir(0, a, b, 3, w, bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_subtract_adjoint_ir(a[0], a, b, 2, w, bits)
    with pytest.raises(ValueError, match="control"):
        controlled_modular_subtract_adjoint_ir(0, a, b, 2, w, bits, control_value=2)
