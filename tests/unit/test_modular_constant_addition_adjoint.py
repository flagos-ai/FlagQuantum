"""Literal constant-modular adjoint with original k load and clean effect."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_constant_addition import (
    controlled_modular_add_constant_adjoint_ir,
    controlled_modular_add_constant_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_addition import layout as add_layout
from tests.unit.test_controlled_modular_addition_adjoint import (
    adjoint_reference_binding,
)
from tests.unit.test_modular_constant_addition import CASES, layout, states
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n,p,k", CASES)
def test_original_adjoint_source_and_independent_models(n: int, p: int, k: int) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    a, b, w, old_bits = add_layout(n)
    remap = dict(zip((*a, *b, *w), (*work[:n], *target, *work[n:]), strict=True))
    alloc, measured = adjoint_reference_binding(n, p, 1)
    bit_map = dict(zip(old_bits, bits, strict=True))
    source = CModAddK(k=k, mod=p, bitsize=n).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        (*work[:n], *(remap[x] for x in alloc)),
        {remap[x]: bit_map[v] for x, v in measured.items()},
        reuse_allocations=False,
        preserve_components=True,
    )
    gates = controlled_modular_add_constant_adjoint_ir(0, target, k, p, work, bits)
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 20 * n + 4
    assert counts["measure"] == model.measurement == 5 * n + 1
    assert get_cost_value(source, QubitCount()) == 1 + n + len(work)


@pytest.mark.parametrize("n,p,k", CASES)
def test_all_valid_inputs_all_branches_and_restore(n: int, p: int, k: int) -> None:
    target, work, bits = layout(n)
    width = 1 + n + len(work)
    inputs, ideal = [], []
    for c in (0, 1):
        for y in range(p):
            z = (y - k) % p if c else y
            xword = ((c << n) | y) << len(work)
            yword = ((c << n) | z) << len(work)
            inputs.append(tuple((xword >> (width - 1 - i)) & 1 for i in range(width)))
            ideal.append(tuple((yword >> (width - 1 - i)) & 1 for i in range(width)))
    actual, events = certified_and_channel_map(
        controlled_modular_add_constant_adjoint_ir(0, target, k, p, work, bits),
        tuple(inputs),
    )
    assert actual == tuple(ideal) and events == 5 * n + 1
    restored, events2 = certified_and_channel_map(
        controlled_modular_add_constant_ir(0, target, k, p, work, bits), actual
    )
    assert restored == tuple(inputs) and events == events2


@pytest.mark.parametrize("n,p,k", [(1, 1, 1), (2, 3, 1), (3, 5, 2)])
def test_raw_complex_channels_work_zero_and_spectator(n: int, p: int, k: int) -> None:
    target, work, bits = layout(n)
    # Reuse only the independent mathematical amplitude builder; keep original
    # positive k in the emitted inverse rather than normalizing its value.
    before, expected = states(n, p, -k)
    gates = controlled_modular_add_constant_adjoint_ir(0, target, k, p, work, bits)
    m = 5 * n + 1
    branches = (
        product((0, 1), repeat=m)
        if n == 1
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(before, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_same_k_not_resynthesis_and_validation() -> None:
    target, work, bits = layout(3)
    exact = controlled_modular_add_constant_adjoint_ir(0, target, 1, 5, work, bits)
    other = controlled_modular_add_constant_ir(0, target, 4, 5, work, bits)
    assert exact != other
    assert exact[0].name == "x" and exact[0].wires == (work[2],)
    assert other[0].name == "x" and other[0].wires == (work[0],)
    for k in (-1, 6, True):
        with pytest.raises(ValueError, match="constant"):
            controlled_modular_add_constant_adjoint_ir(0, target, k, 5, work, bits)
    with pytest.raises(ValueError, match="clean work"):
        controlled_modular_add_constant_adjoint_ir(0, target, 1, 5, work[:-1], bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_add_constant_adjoint_ir(target[0], target, 1, 5, work, bits)


def test_append_example() -> None:
    from examples.algorithms.register_modular_constant_adjoint import main

    main()
