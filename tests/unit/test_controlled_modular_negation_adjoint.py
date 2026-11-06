"""Pinned CModNeg adjoint, both polarities and original apply-And lifetime."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.controlled_modular_negation import (
    controlled_modular_negate_adjoint_ir,
    controlled_modular_negate_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_negation import layout
from tests.unit.test_modular_negation import (
    CASES,
    basis_inputs,
    coherent_states,
    reference_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_original_source_adjoint_both_polarities_and_models(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModNeg
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    source = CModNeg((QMontgomeryUInt if montgomery else QUInt)(n), p, cv).adjoint()
    alloc, measured = reference_binding(n, work, bits, controlled=True)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = controlled_modular_negate_adjoint_ir(
        0, target, p, work, bits, control_value=cv
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12 * (n - 1) + 4
    assert counts["measure"] == model.measurement == 3 * (n - 1) + 1
    assert get_cost_value(source, QubitCount()) == 1 + n + len(work)
    assert counts == Counter(
        g.name
        for g in controlled_modular_negate_ir(
            0, target, p, work, bits, control_value=cv
        )
    )


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_residues_all_branches_and_restore(n: int, p: int, cv: int) -> None:
    target, work, bits = layout(n)
    before, expected = basis_inputs(n, p, cv=cv)
    actual, m = certified_and_channel_map(
        controlled_modular_negate_adjoint_ir(
            0, target, p, work, bits, control_value=cv
        ),
        before,
    )
    assert actual == expected and m == 3 * (n - 1) + 1
    restored, m2 = certified_and_channel_map(
        controlled_modular_negate_ir(0, target, p, work, bits, control_value=cv), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("n,p", [(2, 2), (3, 5)])
@pytest.mark.parametrize("cv", [0, 1])
def test_every_small_raw_complex_branch_and_clean_work(n: int, p: int, cv: int) -> None:
    target, work, bits = layout(n)
    inputs, expected = coherent_states(n, p, cv=cv)
    gates = controlled_modular_negate_adjoint_ir(
        0, target, p, work, bits, control_value=cv
    )
    m = 3 * (n - 1) + 1
    for branch in product((0, 1), repeat=m):
        torch.testing.assert_close(
            forced_branch(inputs, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("cv", [0, 1])
def test_not_forward_stream_and_validation(cv: int) -> None:
    target, work, bits = layout(3)
    assert controlled_modular_negate_adjoint_ir(
        0, target, 5, work, bits, control_value=cv
    ) != controlled_modular_negate_ir(0, target, 5, work, bits, control_value=cv)
    with pytest.raises(ValueError, match="modulus\\+1"):
        controlled_modular_negate_adjoint_ir(0, target, 7, work, bits, control_value=cv)
    with pytest.raises(ValueError, match="zero or one"):
        controlled_modular_negate_adjoint_ir(
            0, target, 5, work, bits, control_value=True
        )
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_negate_adjoint_ir(
            target[0], target, 5, work, bits, control_value=cv
        )


def test_user_append_example() -> None:
    from examples.algorithms.register_modular_negation import main

    main()
