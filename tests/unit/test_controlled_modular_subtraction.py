"""Controlled modular exact-adjoint composition and pinned source channels."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    _controlled_modular_workspace,
    controlled_modular_add_ir,
)
from flagquantum.compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_addition import (
    certificate_inputs,
    coherent_states,
    layout,
    reference_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(2, 1), (2, 2), (3, 3), (3, 5), (5, 13), (8, 251)]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_modsub_source_gates_feedback_and_resources(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    _, constant, carry, _ = _controlled_modular_workspace(0, a, b, p, w, bits, cv)
    middle, measured = reference_binding(n, p, cv)
    outside = (*constant[:n], *carry[: n - 1])
    source = CModSub((QMontgomeryUInt if montgomery else QUInt)(n), p, cv)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        (*outside, *middle, *outside),
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 28 * n - 4
    assert counts["measure"] == model.measurement == 7 * n - 1
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)
    assert counts["cx"] == 82 * n - 29 + 2 * p.bit_count()
    add = Counter(
        g.name for g in controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv)
    )
    assert counts["x"] == add["x"] + 2 * n + 4 * (p + 1).bit_count()


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_inputs_all_branches_and_add_sub_restore(n: int, p: int, cv: int) -> None:
    a, b, w, bits = layout(n)
    gates = controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv)
    inputs, expected = certificate_inputs(n, p, cv, subtract=True)
    actual, events = certified_and_channel_map(gates, inputs)
    assert actual == expected and events == 7 * n - 1
    restored, add_events = certified_and_channel_map(
        controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv), actual
    )
    assert restored == inputs and add_events == 5 * n + 1


@pytest.mark.parametrize("n,p", [(2, 2), (3, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_raw_primitive_complex_channels(n: int, p: int, cv: int) -> None:
    a, b, w, bits = layout(n)
    gates = controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv)
    state, expected = coherent_states(n, p, cv, subtract=True)
    events = 7 * n - 1
    for outcome in ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events))):
        torch.testing.assert_close(
            forced_branch(state, gates, outcome),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("cv", [0, 1])
def test_upstream_all_ones_modulus_boundary_is_explicit(n: int, cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub

    p = (1 << n) - 1
    a, b, w, bits = layout(n)
    with pytest.raises(ValueError):
        CModSub(QUInt(n), p, cv).decompose_bloq().flatten()
    with pytest.raises(ValueError, match="modulus\\+1"):
        controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv)


def test_user_append_example() -> None:
    from examples.algorithms.register_controlled_modular_arithmetic import main

    main()
