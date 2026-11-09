"""Pinned CModAddK load/add/unload gates, channels and source model boundaries."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular.modular_constant_addition import (
    controlled_modular_add_constant_ir,
    modular_constant_requirements,
)
from flagquantum.simulation.classical_modular_constant_addition import (
    controlled_modular_add_constant_classical_values,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_controlled_modular_addition import (
    layout as add_layout,
)
from tests.unit.test_controlled_modular_addition import (
    reference_binding as add_binding,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 2, 0),
    (2, 3, 1),
    (2, 3, 3),
    (3, 3, 2),
    (3, 5, 5),
    (5, 13, 7),
    (8, 251, 0),
    (8, 251, 173),
]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_constant_requirements(n)
    return (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


def reference_binding(
    target: tuple[int, ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    p: int,
) -> tuple[tuple[int, ...], dict[int, int]]:
    n = len(target)
    a, b, w, old_bits = add_layout(n)
    remap = dict(zip((*a, *b, *w), (*work[:n], *target, *work[n:]), strict=True))
    alloc, measured = add_binding(n, p, 1)
    bit_map = dict(zip(old_bits, bits, strict=True))
    return (*work[:n], *(remap[x] for x in alloc)), {
        remap[x]: bit_map[v] for x, v in measured.items()
    }


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p,k", CASES)
def test_original_source_gates_feedback_and_independent_models(
    n: int, p: int, k: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    source = CModAddK(k=k, mod=p, bitsize=n)
    alloc, measured = reference_binding(target, work, bits, p)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_add_constant_ir(0, target, k, p, work, bits)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 20 * n + 4
    assert counts["measure"] == model.measurement == 5 * n + 1
    assert get_cost_value(source, QubitCount()) == 1 + n + len(work)
    assert counts["cx"] == 58 * n + 1 + 2 * p.bit_count() + 4 * (n == 1)
    assert (
        counts["x"]
        == 2 * k.bit_count() + 2 * ((-p) % (1 << (n + 1))).bit_count() + 4 * n + 7
    )


def states(n: int, p: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = modular_constant_requirements(n)
    width = 2 + n + q
    inputs = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(inputs)
    labels = [(c, y, s) for c in (0, 1) for y in range(p) for s in (0, 1)]
    x = torch.arange(len(labels), dtype=torch.float64)
    amplitudes = torch.complex(x + 1, 3 * x - 5)
    amplitudes /= amplitudes.norm()
    for (c, y, s), amp in zip(labels, amplitudes, strict=True):
        out = int(controlled_modular_add_constant_classical_values(c, y, k, p)[1])
        assert out == ((y + k) % p if c else y)
        inputs[(((c << n) | y) << (q + 1)) | s] = amp
        expected[(((c << n) | out) << (q + 1)) | s] = amp
    return inputs, expected


@pytest.mark.parametrize("n,p,k", CASES)
def test_every_valid_basis_input_all_measurement_branches(
    n: int, p: int, k: int
) -> None:
    target, work, bits = layout(n)
    gates = controlled_modular_add_constant_ir(0, target, k, p, work, bits)
    width = 1 + n + len(work)
    words, expected = [], []
    for ctrl in (0, 1):
        for value in range(p):
            out = int(
                controlled_modular_add_constant_classical_values(ctrl, value, k, p)[1]
            )
            assert out == ((value + k) % p if ctrl else value)
            before = ((ctrl << n) | value) << len(work)
            after = ((ctrl << n) | out) << len(work)
            words.append(tuple((before >> (width - 1 - i)) & 1 for i in range(width)))
            expected.append(tuple((after >> (width - 1 - i)) & 1 for i in range(width)))
    actual, events = certified_and_channel_map(gates, tuple(words))
    assert actual == tuple(expected) and events == 5 * n + 1


@pytest.mark.parametrize("n,p,k", [(1, 1, 1), (2, 3, 1), (3, 5, 2)])
def test_raw_complex_projection_with_spectator(n: int, p: int, k: int) -> None:
    target, work, bits = layout(n)
    gates = controlled_modular_add_constant_ir(0, target, k, p, work, bits)
    inputs, expected = states(n, p, k)
    m = 5 * n + 1
    branches = (
        product((0, 1), repeat=m)
        if n == 1
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(inputs, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
def test_validation_and_source_constant_overflow() -> None:
    target, work, bits = layout(2)
    for value in (-1, 4, True):
        with pytest.raises(ValueError, match="constant"):
            controlled_modular_add_constant_ir(0, target, value, 3, work, bits)
    with pytest.raises(ValueError, match="clean work"):
        controlled_modular_add_constant_ir(0, target, 1, 3, work[:-1], bits)
    with pytest.raises(ValueError, match="positive"):
        modular_constant_requirements(0)
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK

    with pytest.raises(ValueError):
        CModAddK(k=4, mod=3, bitsize=2).decompose_bloq().flatten()


@pytest.mark.ftqc_reference
def test_fitting_constant_outside_clean_source_domain_is_not_normalized() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK

    target, work, bits = layout(2)
    source = CModAddK(k=3, mod=2, bitsize=2)
    alloc, measured = reference_binding(target, work, bits, 2)
    gates = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    assert source.call_classically(ctrl=1, x=1) == (1, 0)
    # Unmodified executable source outputs y=2, not its classical model's y=0.
    # Reject the domain instead of normalizing k and claiming source parity.
    (actual,), events = certified_and_channel_map(
        gates, ((1, 0, 1, *(0,) * len(work)),)
    )
    assert actual == (1, 1, 0, *(0,) * len(work)) and events == 11
    with pytest.raises(ValueError, match="constant"):
        controlled_modular_add_constant_ir(0, target, 3, 2, work, bits)
