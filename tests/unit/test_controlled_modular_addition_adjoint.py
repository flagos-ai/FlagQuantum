"""Exact CModAdd adjoint components, comparator palindrome and branch cleanup."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.controlled_comparison import controlled_greater_ir
from flagquantum.compiler.registers.controlled_modular_addition import (
    _controlled_modular_workspace,
    controlled_modular_add_ir,
)
from flagquantum.compiler.registers.controlled_modular_addition_adjoint import (
    controlled_modular_add_adjoint_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_addition import (
    CASES,
    certificate_inputs,
    coherent_states,
    layout,
    reference_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def adjoint_reference_binding(
    n: int, p: int, cv: int
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Bind original reversed component allocations to the shared physical pool."""
    a, b, w, bits = layout(n)
    sign, k, c, comp = _controlled_modular_workspace(0, a, b, p, w, bits, cv)
    _, measured = reference_binding(n, p, cv)
    alloc = (
        sign,
        comp[1],
        comp[0],
        *comp[2:-1][::-1],
        comp[-1],
        *k[:n],
        *c[: n - 1],
        *k,
        *c,
        *k[:n],
        *c,
    )
    return alloc, measured


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_full_source_adjoint_and_models(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    alloc, measured = adjoint_reference_binding(n, p, cv)
    source = CModAdd((QMontgomeryUInt if montgomery else QUInt)(n), p, cv).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_add_adjoint_ir(0, a, b, p, w, bits, control_value=cv)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 20 * n + 4
    assert counts["measure"] == model.measurement == 5 * n + 1
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 8])
@pytest.mark.parametrize("cv", [0, 1])
def test_original_comparator_adjoint_literal_palindrome(n: int, cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan

    a, b, w, bits = layout(n)
    sign, _, _, comp = _controlled_modular_workspace(0, a, b, 1, w, bits, cv)
    gates = controlled_greater_ir(0, a, b, sign, comp, bits[: n + 2], control_value=cv)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    expected = reference_arithmetic_gates(
        CLinearDepthGreaterThan(QUInt(n), cv).adjoint(),
        {"ctrl": (0,), "a": a, "b": b, "target": (sign,)},
        (comp[1], comp[0], *comp[2:-1][::-1], comp[-1]),
        measured,
    )
    assert_source(gates, expected)


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_valid_inputs_every_branch_and_forward_inverse(
    n: int, p: int, cv: int
) -> None:
    a, b, w, bits = layout(n)
    # Preserve x=p in the inverse domain, as promised by the source forward map.
    before, _ = certificate_inputs(n, p, cv)
    width = len(before[0])
    q = len(w)
    expected = []
    for word in before:
        ctrl = word[0]
        x = sum(bit << (n - 1 - i) for i, bit in enumerate(word[1 : n + 1]))
        y = sum(bit << (n - 1 - i) for i, bit in enumerate(word[n + 1 : 2 * n + 1]))
        z = (y - x) % p if ctrl == cv else y
        v = ((ctrl << (2 * n)) | (x << n) | z) << q
        expected.append(tuple((v >> (width - 1 - i)) & 1 for i in range(width)))
    actual, m = certified_and_channel_map(
        controlled_modular_add_adjoint_ir(0, a, b, p, w, bits, control_value=cv), before
    )
    assert actual == tuple(expected) and m == 5 * n + 1
    restored, m2 = certified_and_channel_map(
        controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_raw_complex_channels_and_clean_work(n: int, p: int, cv: int) -> None:
    a, b, w, bits = layout(n)
    state, _ = coherent_states(n, p, cv)
    # Bind the inverse ideal to the full forward-domain amplitudes, including x=p.
    before, _ = certificate_inputs(n, p, cv)
    v = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    expected = torch.zeros_like(state)
    for i, word in enumerate(before):
        ctrl = word[0]
        x = sum(bit << (n - 1 - j) for j, bit in enumerate(word[1 : n + 1]))
        y = sum(bit << (n - 1 - j) for j, bit in enumerate(word[n + 1 : 2 * n + 1]))
        z = (y - x) % p if ctrl == cv else y
        base = ((ctrl << (2 * n)) | (x << n) | z) << (len(w) + 1)
        for s in (0, 1):
            expected[base | s] = amps[2 * i + s]
    gates = controlled_modular_add_adjoint_ir(0, a, b, p, w, bits, control_value=cv)
    m = 5 * n + 1
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


def test_source_cmodsub_is_a_distinct_decomposition_with_separate_cost() -> None:
    from flagquantum.compiler.registers.controlled_modular_subtraction import (
        controlled_modular_subtract_ir,
    )

    n, p, cv = 3, 3, 1
    a, b, w, bits = layout(n)
    exact = controlled_modular_add_adjoint_ir(0, a, b, p, w, bits, control_value=cv)
    other = controlled_modular_subtract_ir(0, a, b, p, w, bits, control_value=cv)
    before, expected = certificate_inputs(n, p, cv, subtract=True)
    one, m1 = certified_and_channel_map(exact, before)
    two, m2 = certified_and_channel_map(other, before)
    assert one == two == expected and m1 == 16 and m2 == 20
    assert sum(g.name in {"t", "tdg"} for g in exact) == 64
    assert sum(g.name in {"t", "tdg"} for g in other) == 80


def test_append_example() -> None:
    from examples.algorithms.register_controlled_modular_adjoint import main

    main()
