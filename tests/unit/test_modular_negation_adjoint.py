"""Original ModNeg adjoint gate order despite its self-inverse logical map."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_negation import (
    modular_negate_adjoint_ir,
    modular_negate_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_negation import (
    CASES,
    basis_inputs,
    coherent_states,
    layout,
    reference_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_source_adjoint_gate_order_feedback_and_models(
    n: int, p: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    source = ModNeg((QMontgomeryUInt if montgomery else QUInt)(n), p).adjoint()
    alloc, measured = reference_binding(n, work, bits)
    expected = reference_arithmetic_gates(
        source,
        {"x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = modular_negate_adjoint_ir(target, p, work, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12 * (n - 1)
    assert counts["measure"] == model.measurement == 3 * (n - 1)
    assert get_cost_value(source, QubitCount()) == n + len(work)
    assert counts == Counter(g.name for g in modular_negate_ir(target, p, work, bits))


@pytest.mark.parametrize("n,p", CASES)
def test_every_residue_all_branches_and_restore(n: int, p: int) -> None:
    target, work, bits = layout(n)
    before, expected = basis_inputs(n, p)
    actual, m = certified_and_channel_map(
        modular_negate_adjoint_ir(target, p, work, bits), before
    )
    assert actual == expected and m == 3 * (n - 1)
    restored, m2 = certified_and_channel_map(
        modular_negate_ir(target, p, work, bits), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("n,p", [(2, 2), (3, 5)])
def test_every_small_raw_complex_branch_and_zero_work(n: int, p: int) -> None:
    target, work, bits = layout(n)
    inputs, expected = coherent_states(n, p)
    gates = modular_negate_adjoint_ir(target, p, work, bits)
    m = 3 * (n - 1)
    for branch in product((0, 1), repeat=m):
        torch.testing.assert_close(
            forced_branch(inputs, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_not_reusing_forward_stream_and_validation() -> None:
    target, work, bits = layout(3)
    assert modular_negate_adjoint_ir(target, 5, work, bits) != modular_negate_ir(
        target, 5, work, bits
    )
    with pytest.raises(ValueError, match="modulus\\+1"):
        modular_negate_adjoint_ir(target, 7, work, bits)
    with pytest.raises(ValueError, match="clean work"):
        modular_negate_adjoint_ir(target, 5, work[:-1], bits)
    with pytest.raises(ValueError, match="classical"):
        modular_negate_adjoint_ir(target, 5, work, ())
    with pytest.raises(ValueError, match="distinct"):
        modular_negate_adjoint_ir(target, 5, (target[0], *work[1:]), bits)
