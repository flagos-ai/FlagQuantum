"""Literal multiple-controlled AddK source dependencies and coherent histories."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.multi_controlled_constant_addition import (
    multi_controlled_add_constant_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(m, n):
    controls = tuple(range(m))
    target = tuple(range(m, m + n))
    pred = tuple(range(m + n, 2 * m + n - 1))
    constant = tuple(range(2 * m + n - 1, 2 * m + 2 * n - 1))
    carry = tuple(range(2 * m + 2 * n - 1, 2 * m + 3 * n - 2))
    return (
        controls,
        target,
        pred,
        constant,
        carry,
        tuple(range(n - 1)),
        tuple(range(n - 1, n + m - 2)),
    )


CASES = [
    (m, n, values, adjoint)
    for m, n in ((2, 1), (2, 2), (3, 2), (4, 3))
    for values in ((1,) * m, (0,) * m, tuple(i % 2 for i in range(m)))
    for adjoint in (False, True)
]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n,values,adjoint", CASES)
def test_original_controlled_via_and_gate_dependencies(m, n, values, adjoint):
    import numpy as np
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.addition import AddK

    controls, target, pred, constant, carry, bits, pbits = layout(m, n)
    op = AddK(QUInt(n), -1).controlled(ctrl_spec=CtrlSpec(cvs=np.array(values)))
    if adjoint:
        op = op.adjoint()
    expected = reference_arithmetic_gates(
        op,
        {"ctrl": controls, "x": target},
        (*pred, *constant, *carry),
        dict(zip((*carry, *pred), (*bits, *pbits), strict=True)),
    )
    actual = multi_controlled_add_constant_ir(
        controls,
        values,
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


@pytest.mark.parametrize("m,n", ((2, 1), (2, 2), (3, 1)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_all_complex_input_and_measurement_branches_restore_every_work_wire(
    m, n, adjoint
):
    controls, target, pred, constant, carry, bits, pbits = layout(m, n)
    values = tuple(i % 2 for i in range(m))
    activation = sum(v << (m - i - 1) for i, v in enumerate(values))
    gates = multi_controlled_add_constant_ir(
        controls,
        values,
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
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << qubits, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amps
    active = (data >> (n + 1)) == activation
    x = (data >> 1) & ((1 << n) - 1)
    mapped = (x + active * (1 if adjoint else -1)) % (1 << n)
    logical = ((data >> (n + 1)) << n) | mapped
    expected = torch.zeros_like(state)
    expected[(logical << (work + 1)) | (data & 1)] = amps
    for history in product((0, 1), repeat=n + m - 2):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((n + m - 2) / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n,values,adjoint", CASES)
@pytest.mark.parametrize("legacy", (False, True))
def test_original_resource_categories_and_peak(m, n, values, adjoint, legacy):
    from dataclasses import fields

    import numpy as np
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.addition import AddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.multi_controlled_constant_addition import (
        source_multi_controlled_constant_add_cost,
    )

    source = AddK(QUInt(n), -1).controlled(ctrl_spec=CtrlSpec(cvs=np.array(values)))
    if adjoint:
        source = source.adjoint()
    actual = estimate_component_resources(
        source_multi_controlled_constant_add_cost(
            n, -1, values, adjoint=adjoint, legacy_shims=legacy
        )
    )
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for field in fields(actual.gates):
        assert getattr(actual.gates, field.name) == getattr(expected, field.name)
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("change", ("wire", "slot", "direction", "pattern"))
def test_ownership_and_control_validation_precedes_returned_instructions(change):
    controls, target, pred, constant, carry, bits, pbits = layout(2, 2)
    if change == "wire":
        pred = (controls[0],)
    if change == "slot":
        pbits = bits
    values = (1, 2) if change == "pattern" else (1, 0)
    with pytest.raises(ValueError):
        multi_controlled_add_constant_ir(
            controls,
            values,
            target,
            -1,
            pred,
            constant,
            carry,
            bits,
            pbits,
            adjoint=None if change == "direction" else False,
        )


def test_shared_component_implementations_can_be_replaced_without_consumer_changes(
    monkeypatch,
):
    import flagquantum.compiler.registers.multi_controlled_constant_addition as module
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
        estimate_component_resources,
    )

    compute = ComponentCost("predicate", 5, GateCost(t=2))
    erase = ComponentCost("erase", 7, GateCost(measurement=3))
    addition = ComponentCost("addition", 11, GateCost(cswap=5))
    monkeypatch.setattr(
        module,
        "control_predicate_cost",
        lambda *args, uncompute=False, **kwargs: erase if uncompute else compute,
    )
    monkeypatch.setattr(
        module, "source_controlled_constant_add_cost", lambda *args, **kwargs: addition
    )
    report = estimate_component_resources(
        module.source_multi_controlled_constant_add_cost(2, -1, (1, 0, 1))
    )
    assert (
        report.gates.t == 2
        and report.gates.cswap == 5
        and report.gates.measurement == 3
    )
    assert report.peak_logical_qubits == 15
