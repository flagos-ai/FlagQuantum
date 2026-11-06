"""Pinned CtrlScaleModAdd ordered And traversal, coherent channels and resources."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.modular_scaled_addition import (
    controlled_modular_scaled_add_ir,
    modular_scaled_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_modular_constant_addition import reference_binding
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (1, 1, 0),
    (2, 2, 0),
    (2, 3, 1),
    (2, 3, -1),
    (3, 3, 3),
    (3, 5, -7),
    (5, 13, 19),
    (8, 251, 173),
]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_scaled_requirements(n)
    return (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, 2 * n + 1)),
        tuple(range(2 * n + 1, 2 * n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


@pytest.mark.parametrize("n,p,k", CASES)
def test_original_source_gate_dependencies_and_model(n: int, p: int, k: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    source = CtrlScaleModAdd(k=k, mod=p, bitsize=n)
    alloc, measured = reference_binding(b, w[1:], bits, p)
    measured[w[0]] = bits[0]
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        (*(w[0],) * n, *(alloc * n)),
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = controlled_modular_scaled_add_ir(0, a, b, k, p, w, bits)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == n * (20 * n + 8)
    assert counts["measure"] == model.measurement == n * (5 * n + 2)
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)
    assert counts["cx"] == n * (58 * n + 7 + 2 * p.bit_count() + 4 * (n == 1))
    constants = [(k * (1 << (n - 1 - i))) % p for i in range(n)]
    assert counts["x"] == n * (
        2 * ((-p) % (1 << (n + 1))).bit_count() + 4 * n + 7
    ) + 2 * sum(x.bit_count() for x in constants)


def basis_inputs(
    n: int, p: int, k: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    q, _ = modular_scaled_requirements(n)
    width = 1 + 2 * n + q
    before, after = [], []
    for c in (0, 1):
        for a in range(1 << n):
            for b in range(p):
                y = (b + k * a) % p if c else b
                x = ((c << (2 * n)) | (a << n) | b) << q
                z = ((c << (2 * n)) | (a << n) | y) << q
                before.append(tuple((x >> (width - 1 - i)) & 1 for i in range(width)))
                after.append(tuple((z >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(before), tuple(after)


@pytest.mark.parametrize("n,p,k", CASES)
def test_all_valid_inputs_all_branches_and_inverse_scale(
    n: int, p: int, k: int
) -> None:
    a, b, w, bits = layout(n)
    before, expected = basis_inputs(n, p, k)
    actual, events = certified_and_channel_map(
        controlled_modular_scaled_add_ir(0, a, b, k, p, w, bits), before
    )
    assert actual == expected and events == n * (5 * n + 2)
    restored, inverse_events = certified_and_channel_map(
        controlled_modular_scaled_add_ir(0, a, b, -k, p, w, bits), actual
    )
    assert restored == before and inverse_events == events


@pytest.mark.parametrize("n,p,k", [(1, 1, 0), (2, 3, -1)])
def test_raw_primitive_complex_channels_with_spectator(n: int, p: int, k: int) -> None:
    a, b, w, bits = layout(n)
    before, after = basis_inputs(n, p, k)
    width = 2 + 2 * n + len(w)
    inputs = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(inputs)
    coeff = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xval = sum(bit << (len(x) - 1 - j) for j, bit in enumerate(x))
        yval = sum(bit << (len(y) - 1 - j) for j, bit in enumerate(y))
        for spectator in (0, 1):
            inputs[(xval << 1) | spectator] = amps[2 * i + spectator]
            expected[(yval << 1) | spectator] = amps[2 * i + spectator]
    gates = controlled_modular_scaled_add_ir(0, a, b, k, p, w, bits)
    m = n * (5 * n + 2)
    for outcome in ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m))):
        torch.testing.assert_close(
            forced_branch(inputs, gates, outcome),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_validation() -> None:
    a, b, w, bits = layout(2)
    with pytest.raises(ValueError, match="positive"):
        modular_scaled_requirements(0)
    for k in (True, 1.5):
        with pytest.raises(ValueError, match="concrete"):
            controlled_modular_scaled_add_ir(0, a, b, k, 3, w, bits)
    with pytest.raises(ValueError, match="distinct"):
        controlled_modular_scaled_add_ir(a[0], a, b, 1, 3, w, bits)
    with pytest.raises(ValueError, match="equal"):
        controlled_modular_scaled_add_ir(0, a, b[:1], 1, 3, w, bits)
    with pytest.raises(ValueError, match="Modulus"):
        controlled_modular_scaled_add_ir(0, a, b, 1, 4, w, bits)


def test_user_append_example() -> None:
    from examples.algorithms.register_modular_scaled_addition import main

    main()
