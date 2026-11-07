"""Original shaped integer controls around shared AddK and its measured inverse."""

from dataclasses import fields
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.multi_controlled_constant_addition import (
    grouped_controlled_add_constant_ir,
    source_grouped_controlled_constant_add_cost,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multi_controlled_constant_addition import layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import CASES, control_fixture
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
EXTRA = (
    (("QInt", 2, (), (-1,), "twos_complement"),),
    (("QIntOnesComp", 3, (), (-1,), "ones_complement"),),
    (("QBit", 1, (), (0,), "unsigned"),),
    (("QBit", 1, (), (1,), "unsigned"),),
)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case,n,adjoint", tuple(product(CASES + EXTRA, (1, 2), (False, True)))
)
def test_grouped_source_operands_feedback_and_resources(case, n, adjoint):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    spec, registers, groups = control_fixture(case)
    m = sum(len(group) for group in groups)
    _, target, pred, constant, carry, bits, pbits = layout(m, n)
    source = AddK(QUInt(n), -1).controlled(ctrl_spec=spec)
    if adjoint:
        source = source.adjoint()
    inputs = dict(zip(source.ctrl_reg_names, groups, strict=True)) | {"x": target}
    expected = reference_arithmetic_gates(
        source,
        inputs,
        (*pred, *constant, *carry),
        dict(zip((*carry, *pred), (*bits, *pbits), strict=True)),
    )
    actual = grouped_controlled_add_constant_ir(
        groups,
        registers,
        target,
        -1,
        pred,
        constant,
        carry,
        bits,
        pbits,
        adjoint=adjoint,
    )
    assert_source(actual, expected)
    for legacy in (False, True):
        report = estimate_component_resources(
            source_grouped_controlled_constant_add_cost(
                n, -1, registers, adjoint=adjoint, legacy_shims=legacy
            )
        )
        cost = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(report.gates):
            assert getattr(report.gates, field.name) == getattr(cost, field.name)
        assert report.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("case", (CASES[0], *EXTRA))
@pytest.mark.parametrize("adjoint", (False, True))
def test_every_small_coherent_grouped_input_history_and_work_cleanup(case, adjoint):
    _, registers, groups = control_fixture(case)
    m = sum(len(group) for group in groups)
    n = 1
    values = tuple(bit for reg in registers for bit in reg.bits())
    activation = sum(v << (m - i - 1) for i, v in enumerate(values))
    _, target, pred, constant, carry, bits, pbits = layout(m, n)
    gates = grouped_controlled_add_constant_ir(
        groups,
        registers,
        target,
        -1,
        pred,
        constant,
        carry,
        bits,
        pbits,
        adjoint=adjoint,
    )
    work = m + 2 * n - 2
    qubits = 2 * m + 3 * n - 1
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << qubits, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amplitudes
    active = (data >> (n + 1)) == activation
    x = (data >> 1) & ((1 << n) - 1)
    mapped = (x + active * (1 if adjoint else -1)) % (1 << n)
    logical = ((data >> (n + 1)) << n) | mapped
    expected = torch.zeros_like(state)
    expected[(logical << (work + 1)) | (data & 1)] = amplitudes
    for history in product((0, 1), repeat=n + m - 2):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((n + m - 2) / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("change", ("shape", "wire", "slots", "encoding", "empty"))
def test_grouped_port_and_ownership_failures_do_not_silently_flatten_bad_inputs(change):
    from flagquantum.compiler.registers.control_predicate import ControlRegister

    groups = ((0, 1),)
    registers = (ControlRegister(2, (-1,), "twos_complement"),)
    _, target, pred, constant, carry, bits, pbits = layout(2, 2)
    if change == "shape":
        groups = ((0,),)
    if change == "wire":
        target = (0, 3)
    if change == "slots":
        pbits = bits
    if change == "encoding":
        registers = (ControlRegister(2, (-3,), "twos_complement"),)
    if change == "empty":
        groups = ()
        registers = ()
    with pytest.raises(ValueError):
        grouped_controlled_add_constant_ir(
            groups, registers, target, -1, pred, constant, carry, bits, pbits
        )


def test_concrete_grouped_model_does_not_apply_executable_lifetimes_to_symbolic_constants():
    import sympy

    from flagquantum.compiler.registers.control_predicate import ControlRegister

    with pytest.raises(ValueError, match="Python integer constant"):
        source_grouped_controlled_constant_add_cost(
            3, sympy.Integer(5), (ControlRegister(2, (1,)),)
        )
