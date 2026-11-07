"""Pinned doubling gates, explicit source output aliases and clean channels."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_doubling import (
    modular_double_ir,
    modular_double_outputs,
    modular_double_requirements,
)
from flagquantum.simulation.classical_modular_multiplication import (
    modular_double_classical_value,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 1), (2, 3), (3, 5), (5, 13), (8, 251)]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_double_requirements(n)
    return tuple(range(n)), tuple(range(n, n + q)), tuple(range(40, 40 + c))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_source_gates_ports_and_models(n: int, p: int, montgomery: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    t, w, b = layout(n)
    low, sign = w[:2]
    constant, carry = w[2 : n + 4], w[n + 4 :]
    source = ModDbl((QMontgomeryUInt if montgomery else QUInt)(n), p)
    out, clean = modular_double_outputs(t, w)
    ref = reference_arithmetic_gates(
        source,
        {"x": t},
        (sign, low, *constant, *carry, *constant[: n + 1], *carry[:n]),
        dict(zip(carry, b, strict=True)),
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs={"x": out},
    )
    gates = modular_double_ir(t, p, w, b)
    assert_source(gates, ref)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 8 * n + 4
    assert counts["measure"] == model.measurement == 2 * n + 1
    assert get_cost_value(source, QubitCount()) == n + len(w)
    assert counts["cx"] == 24 * n + 7 + 2 * p.bit_count()
    assert len(clean) == len(w) and set(out + clean) == set(t + w)


def basis(
    n: int, p: int, *, half: bool = False
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    t, w, _ = layout(n)
    if half:
        from flagquantum.compiler.registers.modular_halving import modular_half_outputs

        out, clean = modular_half_outputs(t, w)
    else:
        out, clean = modular_double_outputs(t, w)
    width = n + len(w)
    a = []
    b = []
    for x in range(p):
        y = (
            (x + p * (x & 1)) // 2
            if half
            else int(modular_double_classical_value(x, p))
        )
        assert y == ((x + p * (x & 1)) // 2 if half else (2 * x) % p)
        row = [0] * width
        result = [0] * width
        for i, wire in enumerate(t):
            row[wire] = (x >> (n - 1 - i)) & 1
        for i, wire in enumerate(out):
            result[wire] = (y >> (n - 1 - i)) & 1
        assert not any(result[q] for q in clean)
        a.append(tuple(row))
        b.append(tuple(result))
    return tuple(a), tuple(b)


def coherent(
    n: int, p: int, *, half: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = basis(n, p, half=half)
    state = torch.zeros(1 << (len(a[0]) + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    coeff = torch.arange(2 * p, dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        xv = sum(bit << (len(x) - 1 - j) for j, bit in enumerate(x))
        yv = sum(bit << (len(y) - 1 - j) for j, bit in enumerate(y))
        for s in (0, 1):
            state[(xv << 1) | s] = amps[2 * i + s]
            expected[(yv << 1) | s] = amps[2 * i + s]
    return state, expected


@pytest.mark.parametrize("n,p", CASES)
def test_every_input_every_branch(n: int, p: int) -> None:
    t, w, b = layout(n)
    before, expected = basis(n, p)
    actual, m = certified_and_channel_map(modular_double_ir(t, p, w, b), before)
    assert actual == expected and m == 2 * n + 1


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 5)])
def test_raw_complex_all_small_branches_with_spectator(n: int, p: int) -> None:
    t, w, b = layout(n)
    g = modular_double_ir(t, p, w, b)
    state, expected = coherent(n, p)
    m = 2 * n + 1
    branches = (
        product((0, 1), repeat=m)
        if n <= 2
        else ((0,) * m, (1,) * m, tuple(i % 2 for i in range(m)))
    )
    for branch in branches:
        torch.testing.assert_close(
            forced_branch(state, g, branch), expected / 2 ** (m / 2), rtol=0, atol=1e-12
        )


def test_validation() -> None:
    t, w, b = layout(2)
    for p in (0, 2, 4, True):
        with pytest.raises(ValueError, match="odd"):
            modular_double_ir(t, p, w, b)
    with pytest.raises(ValueError, match="positive"):
        modular_double_requirements(0)
    with pytest.raises(ValueError, match="clean work"):
        modular_double_ir(t, 3, w[:-1], b)
    with pytest.raises(ValueError, match="classical"):
        modular_double_ir(t, 3, w, b[:-1])
    with pytest.raises(ValueError, match="distinct"):
        modular_double_outputs(t, (t[0], *w[1:]))


@pytest.mark.ftqc_reference
def test_upstream_out_of_residue_model_disagrees_without_dirty_work() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl

    t, w, b = layout(2)
    source = ModDbl(QUInt(2), 1)
    assert source.call_classically(x=2) == (2,)
    assert source.decompose_bloq().call_classically(x=2) == (3,)
    before = (1, 0, *(0,) * len(w))
    actual, m = certified_and_channel_map(modular_double_ir(t, 1, w, b), (before,))
    out, clean = modular_double_outputs(t, w)
    assert tuple(actual[0][q] for q in out) == (1, 1)
    assert not any(actual[0][q] for q in clean) and m == 5
