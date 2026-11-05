"""Source half-comparison outputs, phases and all small cleanup branches."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.half_comparison import half_greater_ir
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        (2 * n,),
        tuple(range(2 * n + 1, 3 * n + 2)),
        (3 * n + 2, 3 * n + 3),
        tuple(range(40, 40 + n)),
    )


def assert_half_source(
    n: int,
    signed: bool,
    source: Any,
    compute: tuple[Instruction, ...],
    erase: tuple[Instruction, ...],
    *,
    flipped: bool = False,
    erased_outputs: dict[str, tuple[int, ...]] | None = None,
) -> None:
    a, b, target, difference, signs, bits = layout(n)
    inputs = {"a": a, "b": b}
    outputs = {**inputs, "c": difference, "target": target}
    measured = dict(zip(difference[::-1][1:], bits, strict=True))
    # Bloq traversal discovers sum outputs least-significant first. Bind names,
    # not an algorithm patch; expected_outputs checks physical output identities.
    expected_compute = reference_arithmetic_gates(
        source,
        inputs,
        (*signs, *difference[::-1], *target),
        {},
        expected_outputs=outputs,
        reuse_allocations=False,
    )
    assert_source(compute, expected_compute)
    expected_erase = reference_arithmetic_gates(
        source.adjoint(),
        outputs,
        signs,
        measured,
        expected_outputs=inputs if erased_outputs is None else erased_outputs,
        reuse_allocations=False,
    )
    assert_source(erase, expected_erase)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    forward_cost = get_cost_value(source, QECGatesCost())
    reverse_cost = get_cost_value(source.adjoint(), QECGatesCost())
    forward, reverse = Counter(g.name for g in compute), Counter(g.name for g in erase)
    assert forward["t"] + forward["tdg"] == forward_cost.total_t_count() == 4 * n
    assert reverse["t"] + reverse["tdg"] == reverse_cost.total_t_count() == 0
    assert forward["measure"] == forward_cost.measurement == 0
    assert reverse["measure"] == reverse_cost.measurement == n
    assert forward["cx"] == 11 * n + 3 + 4 * signed
    assert reverse["cx"] == 5 * n + 3 + 4 * signed
    assert forward["x"] == reverse["x"] == 3 * (n + 1) + flipped


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
def test_half_greater_source_outputs_gates_and_models(n: int, dtype: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import _HalfLinearDepthGreaterThan

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    a, b, target, difference, signs, bits = layout(n)
    signed = dtype == "signed"
    compute = half_greater_ir(a, b, difference, target[0], signs, signed=signed)
    erase = half_greater_ir(
        a,
        b,
        difference,
        target[0],
        signs,
        signed=signed,
        adjoint=True,
        classical_bits=bits,
    )
    assert_half_source(n, signed, _HalfLinearDepthGreaterThan(cls(n)), compute, erase)


def check_half_amplitudes(
    n: int,
    signed: bool,
    relation: str,
    compute: tuple[Instruction, ...],
    erase: tuple[Instruction, ...],
) -> None:
    a, b, target, difference, signs, bits = layout(n)
    width = 3 * n + 4
    scratch = n + 4  # retained flag/difference and two sign work bits
    data = torch.arange(1 << (2 * n), dtype=torch.int64)
    left, right = data >> n, data & ((1 << n) - 1)
    if signed:
        left = left - ((left >> (n - 1)) << n)
        right = right - ((right >> (n - 1)) << n)
    compared = {
        "gt": left > right,
        "ge": left >= right,
        "lt": left < right,
        "le": left <= right,
    }[relation]
    diff = ((right - left) if relation in ("gt", "le") else (left - right)) % (
        1 << (n + 1)
    )
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[data << scratch] = amps
    expected = torch.zeros_like(state)
    mapped = (data << scratch) | (compared.to(torch.int64) << (n + 3)) | (diff << 2)
    expected[mapped] = amps
    forward = (
        Circuit(width, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(width, compute))
        .state()
        .flatten()
    )
    torch.testing.assert_close(forward, expected, rtol=0, atol=1e-12)
    for outcomes in product((0, 1), repeat=n):
        actual = forced_branch(forward, erase, outcomes)
        torch.testing.assert_close(actual, state / 2 ** (n / 2), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
def test_half_greater_complex_difference_and_every_cleanup_branch(
    n: int, signed: bool
) -> None:
    a, b, target, difference, signs, bits = layout(n)
    compute = half_greater_ir(a, b, difference, target[0], signs, signed=signed)
    erase = half_greater_ir(
        a,
        b,
        difference,
        target[0],
        signs,
        signed=signed,
        adjoint=True,
        classical_bits=bits,
    )
    check_half_amplitudes(n, signed, "gt", compute, erase)


def test_half_greater_validation() -> None:
    a, b, target, difference, signs, bits = layout(2)
    for kwargs in ({"signed": 1}, {"adjoint": 1}):
        with pytest.raises(ValueError, match="boolean"):
            half_greater_ir(a, b, difference, target[0], signs, **kwargs)
    for left, right, out, work in (
        ((), (), (), signs),
        (a, b[:-1], difference, signs),
        (a, b, difference[:-1], signs),
        (a, b, difference, signs[:1]),
    ):
        with pytest.raises(ValueError, match="positive"):
            half_greater_ir(left, right, out, target[0], work)
    for work in ((0, 1), (-1, 20), (True, 20)):
        with pytest.raises(ValueError, match="distinct"):
            half_greater_ir(a, b, difference, target[0], work)
    for slots in ((), (40, 40), (-1, 0), (True, 0)):
        with pytest.raises(ValueError, match="classical"):
            half_greater_ir(
                a, b, difference, target[0], signs, adjoint=True, classical_bits=slots
            )
