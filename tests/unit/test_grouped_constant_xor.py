"""Unmodified grouped XorK source streams, resources and coherent erasure."""

from dataclasses import fields
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.bits.grouped_constant_xor import (
    grouped_controlled_xor_constant_ir,
    source_grouped_constant_xor_cost,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_grouped_controlled_constant_addition import EXTRA
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import CASES, control_fixture
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(m, n):
    target = tuple(range(m, m + n))
    work = tuple(range(m + n, 2 * m + n - 1))
    slots = tuple(range(30, 30 + max(0, m - 1)))
    return target, work, slots


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case,n,pattern", tuple(product(CASES + EXTRA, (1, 2, 5), (0, 1, 2)))
)
def test_original_grouped_source_stream_and_all_resource_fields(case, n, pattern):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    spec, regs, groups = control_fixture(case)
    m = sum(map(len, groups))
    target, work, slots = layout(m, n)
    value = (0, (1 << n) - 1, ((1 << n) - 1) // 3)[pattern]
    source = XorK(QUInt(n), value).controlled(ctrl_spec=spec)
    inputs = dict(zip(source.ctrl_reg_names, groups, strict=True)) | {"x": target}
    expected = reference_arithmetic_gates(
        source, inputs, work, dict(zip(work, slots, strict=True))
    )
    actual = grouped_controlled_xor_constant_ir(
        groups, regs, target, value, work, slots
    )
    assert_source(actual, expected)
    for legacy in (False, True):
        report = estimate_component_resources(
            source_grouped_constant_xor_cost(n, value, regs, legacy_shims=legacy)
        )
        reference = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(report.gates):
            assert getattr(report.gates, field.name) == getattr(reference, field.name)
        assert report.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "values,n,value",
    tuple(
        (v, n, k)
        for v in ((0,), (1,), (0, 1), (1, 0, 1))
        for n in (1, 2)
        for k in range(1 << n)
    ),
)
def test_all_complex_inputs_and_every_measurement_branch(values, n, value):
    m = len(values)
    groups = (tuple(range(m)),)
    regs = (ControlRegister(1, values),)
    target, work, slots = layout(m, n)
    gates = grouped_controlled_xor_constant_ir(groups, regs, target, value, work, slots)
    logical = m + n
    count = 2 * m + n  # Includes an entangled spectator after predicate storage.
    data = torch.arange(1 << (logical + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state[((data >> 1) << m) | (data & 1)] = amplitudes
    activation = sum(v << (m - i - 1) for i, v in enumerate(values))
    active = (data >> (n + 1)) == activation
    mapped = data ^ (active.to(torch.int64) * (value << 1))
    expected = torch.zeros_like(state)
    expected[((mapped >> 1) << m) | (mapped & 1)] = amplitudes
    for history in product((0, 1), repeat=m - 1):
        result = forced_branch(state, gates, history)
        torch.testing.assert_close(
            result, expected / 2 ** ((m - 1) / 2), rtol=0, atol=1e-12
        )
        restored = forced_branch(expected, gates, history)
        torch.testing.assert_close(
            restored, state / 2 ** ((m - 1) / 2), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("change", ("overlap", "work", "slots", "constant", "empty"))
def test_storage_and_pattern_failures(change):
    groups = ((0, 1),)
    regs = (ControlRegister(1, (0, 1)),)
    target = (2, 3)
    work = (4,)
    slots = (30,)
    value = 1
    if change == "overlap":
        target = (1, 3)
    if change == "work":
        work = ()
    if change == "slots":
        slots = (-1,)
    if change == "constant":
        value = 4
    if change == "empty":
        groups = ()
        regs = ()
    with pytest.raises(ValueError):
        grouped_controlled_xor_constant_ir(groups, regs, target, value, work, slots)


@pytest.mark.parametrize("width,value", ((True, 1), (0, 0), (2, True), (2, 4), (2, -1)))
def test_model_rejects_nonexecutable_dimensions(width, value):
    with pytest.raises(ValueError):
        source_grouped_constant_xor_cost(width, value, (ControlRegister(1, (1,)),))
