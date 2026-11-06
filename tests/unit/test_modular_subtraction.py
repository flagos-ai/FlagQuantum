"""Literal ModSub reuses exact AddK adjoints, including their original loads."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_addition import (
    _modular_workspace,
    modular_add_ir,
)
from flagquantum.compiler.registers.modular_subtraction import modular_subtract_ir
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_modular_addition import layout, reference_binding, residue_states
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(2, 1), (2, 2), (3, 3), (3, 5), (5, 13), (8, 251)]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,modulus", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_modular_subtract_source_chain_gates_and_model(
    n: int, modulus: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModSub
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    _, _, constant, carry, _ = _modular_workspace(a, b, modulus, w, bits)
    middle, measurements = reference_binding(n, modulus)
    stage = (*constant[:n], *carry[: n - 1])
    source = ModSub((QMontgomeryUInt if montgomery else QUInt)(n), modulus)
    gates = modular_subtract_ir(a, b, modulus, w, bits)
    expected = reference_arithmetic_gates(
        source,
        {"x": a, "y": b},
        (*stage, *middle, *stage),
        measurements,
        reuse_allocations=False,
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert get_cost_value(source, QubitCount()) == 2 * n + len(w) == 4 * n + 2
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 24 * n - 12
    assert counts["measure"] == model.measurement == 6 * n - 3
    assert counts["cx"] == 72 * n - 52 + 2 * modulus.bit_count()


@pytest.mark.parametrize("modulus", [1, 2])
def test_all_small_coherent_subtraction_events_and_add_restoration(
    modulus: int,
) -> None:
    n = 2
    a, b, w, bits = layout(n)
    inverse = modular_subtract_ir(a, b, modulus, w, bits)
    state, expected = residue_states(n, modulus, subtract=True)
    events = sum(g.name == "measure" for g in inverse)
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, inverse, outcomes),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )
    forward = modular_add_ir(a, b, modulus, w, bits)
    forward_events = sum(g.name == "measure" for g in forward)
    for outcomes in product((0, 1), repeat=forward_events):
        torch.testing.assert_close(
            forced_branch(expected, forward, outcomes),
            state / 2 ** (forward_events / 2),
            rtol=0,
            atol=1e-12,
        )


def test_nonbinary_modulus_complete_amplitudes_selected_branches() -> None:
    n, modulus = 3, 3
    a, b, w, bits = layout(n)
    inverse = modular_subtract_ir(a, b, modulus, w, bits)
    state, expected = residue_states(n, modulus, subtract=True)
    events = sum(g.name == "measure" for g in inverse)
    for outcomes in ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events))):
        torch.testing.assert_close(
            forced_branch(state, inverse, outcomes),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3])
def test_pinned_all_ones_modulus_cannot_load_modulus_plus_one(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModSub

    a, b, w, bits = layout(n)
    with pytest.raises(ValueError, match="modulus\\+1"):
        modular_subtract_ir(a, b, (1 << n) - 1, w, bits)
    with pytest.raises(ValueError):
        ModSub(QUInt(n), (1 << n) - 1).decompose_bloq().flatten()


def test_subtraction_uses_exact_adjoint_not_negative_constant() -> None:
    from flagquantum.compiler.registers.arithmetic import (
        add_constant_adjoint_ir,
        add_constant_ir,
    )
    from flagquantum.compiler.registers.bitwise import bitwise_not_ir

    a, b, w, bits = layout(3)
    _, _, constant, carry, _ = _modular_workspace(a, b, 3, w, bits)
    middle = modular_add_ir(a, b, 3, w, bits)
    before = add_constant_ir(a, 4, constant[:3], carry[:2], bits[:2])
    exact = add_constant_adjoint_ir(a, 4, constant[:3], carry[:2], bits[:2])
    gates = modular_subtract_ir(a, b, 3, w, bits)
    assert gates == (*bitwise_not_ir(a), *before, *middle, *exact, *bitwise_not_ir(a))
    assert exact != add_constant_ir(a, -4, constant[:3], carry[:2], bits[:2])


def test_user_append_modular_arithmetic_example() -> None:
    from examples.algorithms.register_modular_arithmetic import main

    main()
