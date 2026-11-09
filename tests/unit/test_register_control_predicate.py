"""Source register encodings and all small predicate cleanup histories."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.control.control_predicate import (
    ControlRegister,
    IntegerEncoding,
    control_predicate_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
ControlCase = tuple[
    tuple[str, int, tuple[int, ...], tuple[int, ...], IntegerEncoding], ...
]
CASES: tuple[ControlCase, ...] = (
    (("QBit", 1, (2,), (0, 1), "unsigned"),),
    (("QBit", 1, (2, 2), (1, 0, 0, 1), "unsigned"),),
    (("QAny", 2, (2,), (2, 1), "unsigned"),),
    (("QUInt", 3, (), (5,), "unsigned"), ("QInt", 2, (), (-2,), "twos_complement")),
    (("QIntOnesComp", 3, (2,), (-1, 2), "ones_complement"),),
    (("QMontgomeryUInt", 3, (), (3,), "unsigned"), ("BQUInt", 2, (), (1,), "unsigned")),
)


def control_fixture(
    case: ControlCase,
) -> tuple[Any, tuple[ControlRegister, ...], tuple[tuple[int, ...], ...]]:
    import qualtran

    np = pytest.importorskip("numpy")
    dtypes, values, regs, controls = [], [], [], []
    cursor = 0
    for name, width, shape, words, encoding in case:
        dtype = getattr(qualtran, name)
        dtypes.append(
            dtype()
            if name == "QBit"
            else (
                dtype(width, 3)
                if name == "BQUInt"
                else dtype(width, 5) if name == "QMontgomeryUInt" else dtype(width)
            )
        )
        values.append(np.asarray(words).reshape(shape))
        regs.append(ControlRegister(width, words, encoding))
        controls.append(tuple(range(cursor, cursor + width * len(words))))
        cursor += width * len(words)
    return qualtran.CtrlSpec(tuple(dtypes), tuple(values)), tuple(regs), tuple(controls)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", CASES)
def test_predicate_source_gates_branches_and_model(case: ControlCase) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    spec, regs, groups = control_fixture(case)
    source = CtrlSpecAnd(spec)
    n = int(source.n_ctrl_qubits)
    junk = tuple(range(n, 2 * n - 2))
    target = 2 * n - 2
    bits = tuple(range(30, 30 + n - 1))
    controls = {
        reg.name: group
        for reg, group in zip(source.control_registers, groups, strict=True)
    }
    outputs = {**controls, "target": (target,)}
    if junk:
        outputs["junk"] = junk
    encoded = tuple(bit for register in regs for bit in register.bits())
    assert encoded == source._flat_cvs
    assert get_cost_value(source, QubitCount()) == 2 * n - 1

    forward = control_predicate_ir(groups, regs, target, junk)
    erase = control_predicate_ir(
        groups, regs, target, junk, classical_bits=bits, uncompute=True
    )
    for gates, component, inputs, returned in (
        (forward, source, controls, outputs),
        (erase, source.adjoint(), outputs, controls),
    ):
        expected = reference_arithmetic_gates(
            component,
            inputs,
            (*junk, target),
            dict(zip((*junk, target), bits, strict=True)),
            expected_outputs=returned,
        )
        assert_source(gates, expected)
        counts = Counter(g.name for g in gates)
        model = get_cost_value(component, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == model.measurement
    assert sum(g.name == "measure" for g in erase) == n - 1

    initial = torch.zeros(1 << (2 * n - 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for value in range(1 << n):
        row = value << (n - 1)
        amplitude = complex(value + 1, value % 3 - 2) / (1 << n)
        initial[row] = amplitude
        prefix = 0
        matched = True
        for i, expected_bit in enumerate(encoded):
            matched = matched and ((value >> (n - i - 1)) & 1) == expected_bit
            if i:
                prefix = (prefix << 1) | matched
        expected[row | prefix] = amplitude
    torch.testing.assert_close(
        forced_branch(initial, forward, ()), expected, atol=2e-12, rtol=0
    )
    for outcomes in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(expected, erase, outcomes),
            initial * 2 ** (-(n - 1) / 2),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "register",
    [
        ControlRegister(0, (0,)),
        ControlRegister(1, (0,), "ones_complement"),
        ControlRegister(2, (-2,), "ones_complement"),
        ControlRegister(2, (2,), "twos_complement"),
    ],
)
def test_reject_unrepresentable_control_words(register: ControlRegister) -> None:
    with pytest.raises(ValueError, match="Control"):
        register.bits()
