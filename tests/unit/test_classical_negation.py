"""Typed source negation composition, scalar bounds and coherent clean work."""

from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.arithmetic import negate_ir
from flagquantum.simulation.classical_negation import negate_classical_value
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,width", tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 2, 3, 4)))
)
def test_negate_entire_source_classical_composite_domain(name, width):
    import qualtran
    from qualtran.bloqs.arithmetic.negate import Negate

    dt = getattr(qualtran, name)(width)
    source = Negate(dt)
    signed = name == "QInt"
    domain = (
        range(-(1 << (width - 1)), 1 << (width - 1)) if signed else range(1 << width)
    )
    for value in domain:
        assert (
            negate_classical_value(value, width, signed=signed),
        ) == source.call_classically(x=value)


@pytest.mark.parametrize(
    "signed,kind", tuple(product((False, True), ("int8", "uint8", "int64", "array")))
)
def test_typed_source_intermediate_failures_remain_visible(signed, kind):
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    value = np.array([1], dtype=np.int64) if kind == "array" else getattr(np, kind)(1)
    source = Negate(QInt(4) if signed else QUInt(4))
    try:
        expected = source.call_classically(x=value)[0]
    except (ValueError, TypeError, OverflowError) as exc:
        with pytest.raises(type(exc)):
            negate_classical_value(value, 4, signed=signed)
    else:
        result = negate_classical_value(value, 4, signed=signed)
        assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize("width", (1, 2, 3, 4))
@pytest.mark.parametrize("signed", (False, True))
def test_every_complex_amplitude_and_clean_negation_branch(width, signed):
    target, constant, carry, bits = layout(width)
    gates = negate_ir(target, constant, carry, bits)
    work = 2 * width - 1
    total = width + work + 1
    data = torch.arange(1 << (width + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amps
    expected = torch.zeros_like(state)
    for label in data.tolist():
        value = label >> 1
        if signed:
            value -= (1 << width) if value >= 1 << (width - 1) else 0
        mapped = negate_classical_value(value, width, signed=signed) % (1 << width)
        expected[(mapped << (work + 1)) | (label & 1)] = amps[label]
    for history in product((0, 1), repeat=width - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((width - 1) / 2),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            forced_branch(expected, gates, history),
            state / 2 ** ((width - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


def test_source_zero_symbolic_and_one_bit_signed_gate_boundaries_remain_distinct():
    import sympy
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate

    from tests.unit.test_register_constant_addition import source_gates

    for cls in (QUInt, QInt):
        with pytest.raises(TypeError):
            Negate(cls(0)).call_classically(x=0)
        with pytest.raises(TypeError):
            negate_classical_value(0, 0, signed=cls is QInt)
        with pytest.raises(ValueError):
            Negate(cls(sympy.Integer(3))).call_classically(x=0)
        with pytest.raises(ValueError):
            negate_classical_value(0, sympy.Integer(3), signed=cls is QInt)
    assert (
        negate_classical_value(-1, 1, signed=True)
        == Negate(QInt(1)).call_classically(x=-1)[0]
        == -1
    )
    with pytest.raises(ValueError):
        source_gates(1, 1, source=Negate(QInt(1)))


def test_negation_replaces_each_existing_stage_without_shortcut_arithmetic(monkeypatch):
    import flagquantum.simulation.classical_negation as module

    seen = []

    def validate(value, width, *, signed):
        seen.append(("validate", value, width, signed))

    def complement(value, width, *, unsigned):
        seen.append(("complement", value, width, unsigned))
        return 3

    def increment(value, constant, width, *, signed):
        seen.append(("add", value, constant, width, signed))
        return 2

    monkeypatch.setattr(module, "validate_integer_word_value", validate)
    monkeypatch.setattr(module, "bitwise_not_classical_value", complement)
    monkeypatch.setattr(module, "add_constant_classical_value", increment)
    assert module.negate_classical_value(1, 3, signed=True) == 2
    assert seen == [
        ("validate", 1, 3, True),
        ("complement", 1, 3, False),
        ("validate", 3, 3, True),
        ("add", 3, 1, 3, True),
        ("validate", 2, 3, True),
    ]
