"""Pinned modular-add gates, coherent residue domain and every small branch."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_addition import (
    _modular_workspace,
    modular_add_ir,
    modular_add_requirements,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (3, 5), (5, 13), (8, 251)]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_add_requirements(n)
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        tuple(range(2 * n, 2 * n + q)),
        tuple(range(40, 40 + c)),
    )


def reference_binding(n: int, modulus: int) -> tuple[tuple[int, ...], dict[int, int]]:
    a, b, w, bits = layout(n)
    junk, sign, constant, carry, compare = _modular_workspace(a, b, modulus, w, bits)
    allocations = (
        junk,
        sign,
        *carry,
        *constant,
        *carry,
        *constant[:n],
        *carry[: n - 1],
        *compare,
    )
    measurements = dict(zip(carry, bits, strict=True))
    for wire, bit in zip(compare if n == 1 else compare[2:], bits, strict=True):
        assert wire not in measurements or measurements[wire] == bit
        measurements[wire] = bit
    return allocations, measurements


@pytest.mark.parametrize("n,modulus", CASES)
def test_modular_add_literal_source_stream_and_model(n: int, modulus: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    gates = modular_add_ir(a, b, modulus, w, bits)
    allocations, measurements = reference_binding(n, modulus)
    source = ModAdd(n, modulus)
    expected = reference_arithmetic_gates(
        source, {"x": a, "y": b}, allocations, measurements, reuse_allocations=False
    )
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    assert get_cost_value(source, QubitCount()) == 2 * n + len(w) == 4 * n + 2
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 16 * n - 4
    assert counts["measure"] == model.measurement == 4 * n - 1


def residue_states(
    n: int, modulus: int, *, left_modulus: bool = False, subtract: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = modular_add_requirements(n)
    width = 2 * n + q + 1
    pairs = [
        (a, b, s)
        for a in range(modulus + int(left_modulus))
        for b in range(modulus)
        for s in (0, 1)
    ]
    coeff = torch.arange(len(pairs), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (a, b, s), amp in zip(pairs, amps, strict=True):
        value = (b - a if subtract else b + a) % modulus
        state[(((a << n) | b) << (q + 1)) | s] = amp
        expected[(((a << n) | value) << (q + 1)) | s] = amp
    return state, expected


@pytest.mark.parametrize("n,modulus", [(1, 1), (2, 1), (2, 2), (2, 3)])
def test_every_residue_amplitude_left_modulus_and_all_measurement_events(
    n: int, modulus: int
) -> None:
    a, b, w, bits = layout(n)
    gates = modular_add_ir(a, b, modulus, w, bits)
    state, expected = residue_states(n, modulus, left_modulus=True)
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, gates, outcomes),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )


def test_target_at_modulus_is_not_a_clean_source_domain() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAdd

    source = ModAdd(2, 3)
    assert source.on_classical_vals(x=0, y=3) == {"x": 0, "y": 0}
    with pytest.raises((ValueError, AssertionError)):
        source.decompose_bloq().call_classically(x=0, y=3)
    a, b, w, bits = layout(2)
    gates = modular_add_ir(a, b, 3, w, bits)
    q, _ = modular_add_requirements(2)
    width = 4 + q
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[3 << q] = 1
    actual = forced_branch(state, gates, (0,) * 7)
    expected = torch.zeros_like(state)
    expected[1 << (width - 1 - w[1])] = 2 ** (-7 / 2)
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=1e-12
    )  # Source sign is dirty; do not patch it.


def test_modular_add_validation() -> None:
    for n in (0, -1, True):
        with pytest.raises(ValueError, match="positive"):
            modular_add_requirements(n)
    a, b, w, bits = layout(2)
    for p in (0, 4, True):
        with pytest.raises(ValueError, match="Modulus"):
            modular_add_ir(a, b, p, w, bits)
    with pytest.raises(ValueError, match="equal"):
        modular_add_ir(a, b[:1], 3, w, bits)
    with pytest.raises(ValueError, match="distinct"):
        modular_add_ir(a, a, 3, w, bits)
    with pytest.raises(ValueError, match="clean work"):
        modular_add_ir(a, b, 3, w[:-1], bits)
    for bad in ((40, 40), (-1, 40), (True, 40), ()):
        with pytest.raises(ValueError, match="classical"):
            modular_add_ir(a, b, 3, w, bad)
