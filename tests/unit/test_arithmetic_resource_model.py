"""Source Add and abstract phase-add models with their original boundaries."""

from dataclasses import asdict

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.addition import add_ir
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
    source_phase_add_cost,
)

pytestmark = pytest.mark.unit


def assert_source_cost(model: ComponentCost, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    report = estimate_component_resources(model)
    gates = get_cost_value(source, QECGatesCost())
    assert {k: v for k, v in asdict(report.gates).items() if v} == gates.asdict()
    assert report.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("left,right", [(1, 1), (1, 2), (1, 5), (2, 4), (4, 4), (8, 8)])
def test_add_source_model_and_native_measurements(left: int, right: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    cost = source_add_cost(left, right)
    assert_source_cost(cost, Add(QUInt(left), QUInt(right)))
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    legacy = source_add_cost(left, right, legacy_shims=True)
    locked = get_cost_value(
        Add(QUInt(left), QUInt(right)), QECGatesCost(legacy_shims=True)
    )
    assert {
        key: value for key, value in asdict(legacy.gates).items() if value
    } == locked.asdict()
    gates = add_ir(
        tuple(range(left)),
        tuple(range(left, left + right)),
        tuple(range(left + right, cost.qubits)),
        tuple(range(right - 1)),
    )
    assert cost.gates.measurement == sum(gate.name == "measure" for gate in gates)
    assert cost.gates.total_t_count() == sum(
        gate.name in ("t", "tdg") for gate in gates
    )
    if right == 1:
        assert cost.gates.clifford == -3
        assert [gate.name for gate in gates] == ["cx"]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "widths,shift", [((1, 1), 0), ((2, 5), 2), ((5, 2), 3), ((8, 8), 0)]
)
@pytest.mark.parametrize("control", [None, 0, 1])
def test_phase_add_model(
    widths: tuple[int, int], shift: int, control: int | None
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    cost = source_phase_add_cost(*widths, controlled=control is not None)
    for sign in (1, -1):
        assert_source_cost(
            cost,
            AddIntoPhaseGrad(
                *widths, right_shift=shift, sign=sign, controlled_by=control
            ),
        )


@pytest.mark.parametrize("left,right", [(1, 1), (2, 3), (8, 8)])
@pytest.mark.parametrize("legacy", (False, True))
def test_add_cost_consumes_replaceable_native_recipe(monkeypatch, left, right, legacy):
    import flagquantum.compiler.registers.arithmetic.arithmetic_resource_model as module
    from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    seen = []

    def recipe(root):
        seen.append(root)
        return {FixedTCall("fredkin"): 9}

    monkeypatch.setattr(module, "arithmetic_callees", recipe)
    result = module.source_add_cost(left, right, legacy_shims=legacy)
    assert seen == [ArithmeticCall("add", (left, right))]
    assert result.gates.cswap == 9
    assert (
        result.gates.and_bloq == result.gates.clifford == result.gates.measurement == 0
    )
    assert result.qubits == left + 2 * right - 1
