"""Original conditional arithmetic program, coherent control and exact inverse."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic import (
    add_ir,
    controlled_add_or_subtract_ir,
    measurement_add_or_subtract_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
WIDTHS = [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (1, 2), (2, 3), (3, 5)]


def layout(m: int, n: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(1, m + 1)),
        tuple(range(m + 1, m + n + 1)),
        tuple(range(m + n + 1, m + 2 * n)),
        tuple(range(40, 40 + n - 1)),
    )


@pytest.mark.parametrize("m,n", WIDTHS)
@pytest.mark.parametrize("kind", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("add_on", [False, True])
def test_original_program_adjoint_feedback_and_models(
    m: int, n: int, kind: str, add_on: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[kind]
    a, b, w, bits = layout(m, n)
    source = ControlledAddOrSubtract(dtype(m), dtype(n), add_when_ctrl_is_on=add_on)
    assert source.adjoint().add_when_ctrl_is_on is not add_on
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        gates = measurement_add_or_subtract_ir(
            0, a, b, w, bits, add_when_ctrl_is_on=add_on ^ inverse
        )
        expected = reference_arithmetic_gates(
            original,
            {"ctrl": (0,), "a": a, "b": b},
            w,
            dict(zip(w, bits, strict=True)),
            preserve_components=True,
        )
        assert_source(gates, expected)
        counts = Counter(g.name for g in gates)
        model = get_cost_value(original, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
        assert counts["measure"] == counts["reset"] == model.measurement == n - 1
        assert get_cost_value(original, QubitCount()) == m + 2 * n
        assert (
            counts["cx"] == Counter(g.name for g in add_ir(a, b, w, bits))["cx"] + 2 * n
        )
        assert counts["x"] == 2 * int(add_on ^ inverse)
        if n == 1:
            assert model.clifford == counts["cx"] + counts["x"] - 4


@pytest.mark.parametrize(
    "m,n", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 5), (8, 8)]
)
@pytest.mark.parametrize("add_on", [False, True])
def test_every_small_input_all_histories_and_inverse(
    m: int, n: int, add_on: bool
) -> None:
    a, b, w, bits = layout(m, n)
    width, q = m + 2 * n, n - 1
    av = range(1 << m) if n <= 5 else (0, 1, 127, 128, 254, 255)
    bv = range(1 << n) if n <= 5 else (0, 1, 127, 128, 254, 255)
    before, after = [], []
    for c, x, y in product((0, 1), av, bv):
        z = (y + x if c == int(add_on) else y - x) % (1 << n)
        v = (((c << m) | x) << n | y) << q
        out = (((c << m) | x) << n | z) << q
        before.append(tuple((v >> (width - 1 - i)) & 1 for i in range(width)))
        after.append(tuple((out >> (width - 1 - i)) & 1 for i in range(width)))
    gates = measurement_add_or_subtract_ir(0, a, b, w, bits, add_when_ctrl_is_on=add_on)
    result, events = certified_and_channel_map(gates, tuple(before))
    assert result == tuple(after) and events == n - 1
    inverse = measurement_add_or_subtract_ir(
        0, a, b, w, bits, add_when_ctrl_is_on=not add_on
    )
    restored, _ = certified_and_channel_map(inverse, result)
    assert restored == tuple(before)


@pytest.mark.integration
@pytest.mark.parametrize("m,n", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("add_on", [False, True])
def test_complete_complex_control_and_spectator_all_raw_branches(
    m: int, n: int, add_on: bool
) -> None:
    a, b, w, bits = layout(m, n)
    q = n - 1
    labels = list(product((0, 1), range(1 << m), range(1 << n), (0, 1)))
    v = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (m + 2 * n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (c, x, y, spectator), amp in zip(labels, amps, strict=True):
        z = (y + x if c == int(add_on) else y - x) % (1 << n)
        before = (((c << m) | x) << n) | y
        after = (((c << m) | x) << n) | z
        state[(before << (q + 1)) | spectator] = amp
        expected[(after << (q + 1)) | spectator] = amp
    gates = measurement_add_or_subtract_ir(0, a, b, w, bits, add_when_ctrl_is_on=add_on)
    inverse = measurement_add_or_subtract_ir(
        0, a, b, w, bits, add_when_ctrl_is_on=not add_on
    )
    for branch in product((0, 1), repeat=q):
        result = forced_branch(state, gates, branch)
        torch.testing.assert_close(result, expected / 2 ** (q / 2), rtol=0, atol=1e-12)
        torch.testing.assert_close(
            forced_branch(result, inverse, branch), state / 2**q, rtol=0, atol=1e-12
        )


def test_shared_complement_is_replaceable_in_both_existing_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers import arithmetic

    calls = []

    def replacement(
        control: int, target: tuple[int, ...], value: int
    ) -> tuple[Instruction, ...]:
        calls.append((control, target, value))
        return (Instruction("z", (control,)),)

    monkeypatch.setattr(arithmetic, "controlled_xor_constant_ir", replacement)
    a, b, w, bits = layout(3, 3)
    measured = measurement_add_or_subtract_ir(0, a, b, w, bits)
    unitary = controlled_add_or_subtract_ir(0, a, b, w[0])
    assert calls == [(0, b, 7), (0, b, 7)]
    assert sum(g.name == "z" for g in measured) == 2
    assert sum(g.name == "z" for g in unitary) == 2


@pytest.mark.parametrize(
    "bad", ["mode", "control", "alias", "negative", "slots", "carry", "width"]
)
def test_invalid_layouts_fail_before_emission(bad: str) -> None:
    a, b, w, bits = layout(2, 3)
    c, add_on = 0, False
    if bad == "mode":
        add_on = 1
    elif bad == "control":
        c = True
    elif bad == "alias":
        w = (a[0], w[1])
    elif bad == "negative":
        bits = (-1, bits[1])
    elif bad == "slots":
        bits = (bits[0], bits[0])
    elif bad == "carry":
        w = w[:-1]
    else:
        a = (*a, 10, 11)
    with pytest.raises(ValueError):
        measurement_add_or_subtract_ir(c, a, b, w, bits, add_when_ctrl_is_on=add_on)


@pytest.mark.parametrize("add_on", [False, True])
def test_original_component_chain_is_retained(add_on: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.basic_gates import XGate

    source = ControlledAddOrSubtract(QUInt(3), QUInt(5), add_when_ctrl_is_on=add_on)
    counts = source.decompose_bloq().bloq_counts()
    expected = {Add(QUInt(3), QUInt(5)): 1, BitwiseNot(QUInt(5)).controlled(): 2}
    if add_on:
        expected[XGate()] = 2
    assert counts == expected


def test_narrow_signed_classical_model_and_source_gates_differ() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    source = ControlledAddOrSubtract(QInt(1), QInt(2))
    assert source.decompose_bloq().call_classically(ctrl=1, a=-1, b=0) == (1, -1, -1)
    a, b, w, bits = layout(1, 2)
    gates = measurement_add_or_subtract_ir(0, a, b, w, bits, add_when_ctrl_is_on=True)
    original = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "a": a, "b": b},
        w,
        dict(zip(w, bits, strict=True)),
        preserve_components=True,
    )
    assert_source(gates, original)
    # QInt(1) -1 is pattern 1. The original Add gate zero-extends that bit.
    result, _ = certified_and_channel_map(original, ((1, 1, 0, 0, 0),))
    assert result == ((1, 1, 0, 1, 0),)
    assert result[0][2:4] != (1, 1)  # QInt(2) -1 in the original classical model.
