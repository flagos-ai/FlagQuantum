"""Compare inherited-control passthrough and measured child inversion."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.always import Always, always_component_cost
from flagquantum.compiler.registers.arithmetic import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_always_control_ports_adjoint_and_source_cost() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.bookkeeping import Always as SourceAlways
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    add = Add(QUInt(3), QUInt(3))
    source = SourceAlways(add)
    wrapped, adder = source.get_ctrl_system(CtrlSpec(cvs=(0, 1)))
    assert wrapped is source
    controls, ports = (6, 1), {"a": (3, 2, 0), "b": (5, 8, 4)}

    class Builder:
        def add_t(self, operation, **bindings):
            assert operation is source and bindings == ports
            return tuple(bindings.values())

    assert adder(Builder(), controls, ports) == (controls, tuple(ports.values()))
    assert source.adjoint() == SourceAlways(add.adjoint())

    native = Always(add_ir, add_adjoint_ir)
    returned, controlled = native.controlled(controls)
    assert returned is controls and controlled is native
    grouped = ((6, 1), (7,))
    assert native.controlled(grouped)[0] is grouped
    assert native.adjoint().adjoint() == native

    ccx = ComponentCost("And", 3, GateCost(and_bloq=1))
    erase = ComponentCost("And†", 3, GateCost(clifford=1, measurement=1))
    cnot = ComponentCost("CNOT", 2, GateCost(clifford=1))
    child = ComponentCost(
        "Add", 8, calls=(CostCall(ccx, 2), CostCall(erase, 2), CostCall(cnot, 9))
    )
    model = always_component_cost(child)
    counts = {And(): 2, And().adjoint(): 2, CNOT(): 9}
    assert source.call_graph()[1] == counts
    assert model.calls == child.calls
    estimate = estimate_component_resources(model)
    assert estimate == estimate_component_resources(source_add_cost(3, 3))
    assert estimate.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == get_cost_value(
        source, QECGatesCost()
    ).asdict()
    with pytest.raises(DecomposeNotImplementedError, match="does not declare"):
        get_cost_value(SourceAlways(XGate()), QECGatesCost())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("polarity", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_addk_always_body_and_all_complex_measurement_histories(
    polarity: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic import AddK

    width, value = 3, 5
    control, spectator = 0, 9
    target, constant, carry, bits = (1, 2, 3), (4, 5, 6), (7, 8), (10, 11)
    source = AddK(QUInt(width), value).controlled(ctrl_spec=CtrlSpec(cvs=polarity))
    if adjoint:
        source = source.adjoint()
    native = (
        controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    )
    gates = native(
        control, target, value, constant, carry, bits, control_value=polarity
    )
    reference = reference_arithmetic_gates(
        source,
        {"ctrl": (control,), "x": target},
        (*constant, *carry),
        dict(zip(carry, bits, strict=True)),
    )
    assert_source(gates, reference)
    body = (add_adjoint_ir if adjoint else add_ir)(constant, target, carry, bits)
    assert gates[2 + (polarity == 0) :][: len(body)] == body
    assert Counter(gate.name for gate in body)["measure"] == width - 1
    assert all(control not in gate.wires for gate in body)

    values = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(values, values.remainder(7) - 4)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (spectator + 1), dtype=amplitudes.dtype)
    expected = torch.zeros_like(initial)
    for index, amplitude in enumerate(amplitudes):
        enable, number, other = index >> 4, (index >> 1) & 7, index & 1
        result = (number + (-1 if adjoint else 1) * value * (enable == polarity)) % 8
        initial[(enable << 9) | (number << 6) | other] = amplitude
        expected[(enable << 9) | (result << 6) | other] = amplitude
    for history in product((0, 1), repeat=width - 1):
        result = forced_branch(initial, gates, history)
        torch.testing.assert_close(result, expected / 2, rtol=0, atol=2e-12)
