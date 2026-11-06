"""Pinned ModNeg gates, complete residue channels and explicit source boundaries."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.modular_negation import (
    modular_negate_ir,
    modular_negation_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(2, 1), (2, 2), (3, 3), (3, 5), (5, 13), (8, 251)]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = modular_negation_requirements(n)
    return tuple(range(n)), tuple(range(n, n + q)), tuple(range(40, 40 + c))


def reference_binding(
    n: int,
    work: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    controlled: bool = False,
) -> tuple[tuple[int, ...], dict[int, int]]:
    start = 1 + int(controlled)
    constant, carry = work[start : start + n], work[start + n :]
    root = work[:start]
    return (*root, *carry, *constant, *carry, *carry), {
        **dict(zip(carry, bits[: n - 1], strict=True)),
        **({work[1]: bits[0]} if controlled else {}),
    }


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
def test_original_source_gates_and_resource_models(
    n: int, p: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, work, bits = layout(n)
    source = ModNeg((QMontgomeryUInt if montgomery else QUInt)(n), p)
    alloc, measured = reference_binding(n, work, bits)
    expected = reference_arithmetic_gates(
        source,
        {"x": target},
        alloc,
        measured,
        reuse_allocations=False,
        preserve_components=True,
    )
    actual = modular_negate_ir(target, p, work, bits)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12 * (n - 1)
    assert counts["measure"] == model.measurement == 3 * (n - 1)
    assert get_cost_value(source, QubitCount()) == n + len(work)
    assert counts["cx"] == 26 * n - 26 + 2 * (p + 1).bit_count()
    assert counts["x"] == 8 * n + 2


def basis_inputs(
    n: int, p: int, *, cv: int | None = None
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    q = 2 * n + int(cv is not None)
    width = n + q + int(cv is not None)
    inputs, outputs = [], []
    control_values = (0,) if cv is None else (0, 1)
    for ctrl in control_values:
        for x in range(p):
            y = (-x) % p if cv is None or ctrl == cv else x
            before = ((ctrl << n) | x) << q
            after = ((ctrl << n) | y) << q
            inputs.append(tuple((before >> (width - 1 - i)) & 1 for i in range(width)))
            outputs.append(tuple((after >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(inputs), tuple(outputs)


def coherent_states(
    n: int, p: int, *, cv: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    before, after = basis_inputs(n, p, cv=cv)
    inputs = torch.zeros(1 << (len(before[0]) + 1), dtype=torch.complex128)
    expected = torch.zeros_like(inputs)
    coeff = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        av = sum(bit << (len(a) - 1 - j) for j, bit in enumerate(a))
        bv = sum(bit << (len(b) - 1 - j) for j, bit in enumerate(b))
        for s in (0, 1):
            inputs[(av << 1) | s] = amps[2 * i + s]
            expected[(bv << 1) | s] = amps[2 * i + s]
    return inputs, expected


@pytest.mark.parametrize("n,p", CASES)
def test_all_residue_inputs_every_branch_and_involution(n: int, p: int) -> None:
    target, work, bits = layout(n)
    gates = modular_negate_ir(target, p, work, bits)
    before, expected = basis_inputs(n, p)
    actual, events = certified_and_channel_map(gates, before)
    assert actual == expected and events == 3 * (n - 1)
    restored, again = certified_and_channel_map(gates, actual)
    assert restored == before and again == events


@pytest.mark.parametrize("n,p", [(2, 2), (3, 5)])
def test_raw_complex_channels_spectator_and_work(n: int, p: int) -> None:
    target, work, bits = layout(n)
    gates = modular_negate_ir(target, p, work, bits)
    inputs, expected = coherent_states(n, p)
    m = 3 * (n - 1)
    outcomes = product((0, 1), repeat=m)
    for outcome in outcomes:
        torch.testing.assert_close(
            forced_branch(inputs, gates, outcome),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_source_all_ones_modulus_load_failure(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg

    target, work, bits = layout(n)
    p = (1 << n) - 1
    with pytest.raises(ValueError):
        ModNeg(QUInt(n), p).decompose_bloq().flatten()
    with pytest.raises(ValueError, match="modulus\\+1"):
        modular_negate_ir(target, p, work, bits)


def test_validation_and_source_outside_residue_domain() -> None:
    target, work, bits = layout(2)
    with pytest.raises(ValueError, match="positive"):
        modular_negation_requirements(0)
    with pytest.raises(ValueError, match="clean work"):
        modular_negate_ir(target, 2, work[:-1], bits)
    with pytest.raises(ValueError, match="classical"):
        modular_negate_ir(target, 2, work, ())
    with pytest.raises(ValueError, match="distinct"):
        modular_negate_ir(target, 2, (target[0], *work[1:]), bits)
    # The original classical model is identity outside residues, but the
    # decomposition retains a dirty flag on x=p; do not claim a total unitary.
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg

    source = ModNeg(QUInt(2), 2)
    assert source.call_classically(x=2) == (2,)
    with pytest.raises((ValueError, AssertionError)):
        source.decompose_bloq().call_classically(x=2)
    actual, events = certified_and_channel_map(
        modular_negate_ir(target, 2, work, bits), ((1, 0, 0, 0, 0, 0),)
    )
    assert actual == ((0, 0, 1, 0, 0, 0),) and events == 3
