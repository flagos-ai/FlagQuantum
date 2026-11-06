"""Pinned CModNeg two-polarity gate dependencies and coherent clean channels."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.controlled_modular_negation import (
    controlled_modular_negate_ir,
    controlled_modular_negation_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_negation import (
    CASES,
    basis_inputs,
    coherent_states,
    reference_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = controlled_modular_negation_requirements(n)
    return (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_pinned_source_both_control_polarities_and_models(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModNeg
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    source = CModNeg((QMontgomeryUInt if montgomery else QUInt)(n), p, cv)
    alloc, measured = reference_binding(n, work, bits, controlled=True)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_negate_ir(0, target, p, work, bits, control_value=cv)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12 * (n - 1) + 4
    assert counts["measure"] == model.measurement == 3 * (n - 1) + 1
    assert get_cost_value(source, QubitCount()) == 1 + n + len(work)
    assert counts["cx"] == 26 * n - 20 + 2 * (p + 1).bit_count()
    assert counts["x"] == 8 * n + 2 + 4 * (cv == 0)


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_inputs_every_branch_and_involution(n: int, p: int, cv: int) -> None:
    target, work, bits = layout(n)
    gates = controlled_modular_negate_ir(0, target, p, work, bits, control_value=cv)
    before, expected = basis_inputs(n, p, cv=cv)
    actual, events = certified_and_channel_map(gates, before)
    assert actual == expected and events == 3 * (n - 1) + 1
    restored, again = certified_and_channel_map(gates, actual)
    assert restored == before and again == events


@pytest.mark.parametrize("n,p", [(2, 2), (3, 5)])
@pytest.mark.parametrize("cv", [0, 1])
def test_raw_complex_channels_spectator_and_every_small_outcome(
    n: int, p: int, cv: int
) -> None:
    target, work, bits = layout(n)
    gates = controlled_modular_negate_ir(0, target, p, work, bits, control_value=cv)
    inputs, expected = coherent_states(n, p, cv=cv)
    m = 3 * (n - 1) + 1
    branches = product((0, 1), repeat=m)
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(inputs, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("cv", [0, 1])
def test_source_all_ones_modulus_failure(n: int, cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModNeg

    target, work, bits = layout(n)
    p = (1 << n) - 1
    with pytest.raises(ValueError):
        CModNeg(QUInt(n), p, cv).decompose_bloq().flatten()
    with pytest.raises(ValueError, match="modulus\\+1"):
        controlled_modular_negate_ir(0, target, p, work, bits, control_value=cv)


def test_validation() -> None:
    target, work, bits = layout(2)
    for cv in (True, 2):
        with pytest.raises(ValueError, match="zero or one"):
            controlled_modular_negate_ir(0, target, 2, work, bits, control_value=cv)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_negate_ir(target[0], target, 2, work, bits)
    with pytest.raises(ValueError, match="positive"):
        controlled_modular_negation_requirements(0)


@pytest.mark.parametrize("cv", [0, 1])
def test_source_outside_domain_retains_dirty_flag_on_active_branch(cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModNeg

    target, work, bits = layout(2)
    source = CModNeg(QUInt(2), 2, cv)
    assert source.call_classically(ctrl=cv, x=2) == (cv, 2)
    with pytest.raises((ValueError, AssertionError)):
        source.decompose_bloq().call_classically(ctrl=cv, x=2)
    before = (cv, 1, 0, *(0,) * len(work))
    actual, events = certified_and_channel_map(
        controlled_modular_negate_ir(0, target, 2, work, bits, control_value=cv),
        (before,),
    )
    assert actual == ((cv, 0, 0, 1, 0, 0, 0, 0),) and events == 4


def test_user_append_example() -> None:
    from examples.algorithms.register_modular_negation import main

    main()
