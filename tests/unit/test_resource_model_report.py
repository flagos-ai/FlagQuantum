"""Reusable estimates retain source categories and never imply execution."""

import json

import pytest
from sympy import Symbol, sympify

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.arithmetic import add_ir
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.compiler.registers.symbolic_resource_model import (
    source_add_resource_formula,
)
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("left,right", [(1, 1), (1, 3), (3, 3), (8, 8)])
@pytest.mark.parametrize("legacy", [False, True])
def test_add_report_matches_locked_source(left, right, legacy):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Add(QUInt(left), QUInt(right))
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    report = resource_model_report(source_add_cost(left, right, legacy_shims=legacy))
    assert {k: v for k, v in report["logical_gates"].items() if v} == expected.asdict()
    assert report["peak_logical_qubits"] == get_cost_value(source, QubitCount())
    assert report["emitted_ir"] is report["execution_measurements"] is None
    assert json.loads(json.dumps(report)) == report


@pytest.mark.ftqc_reference
def test_symbolic_report_preserves_original_formula():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = Symbol("n", integer=True, positive=True)
    source = Add(QUInt(n), QUInt(n))
    expected = get_cost_value(source, QECGatesCost())
    report = resource_model_report(source_add_resource_formula(n, n))
    for category, value in expected.asdict().items():
        assert str(report["logical_gates"][category]) == str(value)
    assert report["peak_logical_qubits"] == str(get_cost_value(source, QubitCount()))
    assert sympify(report["t_equivalents"], locals={"n": n}).subs(n, 8) == 28
    assert report["success_probability"] is None
    assert json.loads(json.dumps(report)) == report


def test_recursive_calls_and_budget_are_independent_of_ir():
    child = ComponentCost(
        "child", 5, GateCost(t=2, rotation=1, and_bloq=3), success_probability=0.5
    )
    root = ComponentCost("root", 4, calls=(CostCall(child, 2, 3),))
    report = resource_model_report(root)
    assert report["logical_gates"]["and_bloq"] == 6
    assert report["peak_logical_qubits"] == 8  # Sequential calls, not 2 * width.
    assert report["success_probability"] == 0.25
    assert report["magic_states"] == {"n_t": 26, "n_ccz": 6}
    assert report["t_equivalents"] == 50
    assert report["conversion"]["t_per_rotation"] == 11
    assert report["conversion"]["t_per_ccz_equivalent"] == 4


def test_ir_column_counts_actual_decomposition_without_changing_model():
    model = source_add_cost(3, 3)
    before = resource_model_report(model)
    ir = CircuitIR(8, add_ir((0, 1, 2), (3, 4, 5), (6, 7), (0, 1)))
    report = resource_model_report(model, emitted_ir=ir)
    assert {k: v for k, v in report.items() if k != "emitted_ir"} == {
        k: v for k, v in before.items() if k != "emitted_ir"
    }
    emitted = report["emitted_ir"]
    assert emitted["basis"] == "emitted_instruction_sequence"
    assert emitted["t_count"] == report["t_equivalents"] == 8
    assert emitted["unconditional"]["measure"] == 2
    assert emitted["conditional"] == {"cz": 2}
    assert report["execution_measurements"] is None


def test_missing_model_is_refused():
    with pytest.raises(TypeError, match="requires a component model"):
        resource_model_report(object())
