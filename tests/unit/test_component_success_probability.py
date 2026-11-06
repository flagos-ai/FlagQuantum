"""Declared algorithm success, static overrides and actual measurement histories."""

from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_comparison import (
    compare_component_resources,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.and_compute import and_compute_ir, and_uncompute_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("override", [None, 0.0, 0.7, 1.0])
def test_success_multiplicity_and_override_match_locked_source(
    override: float | None,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import SuccessProb, get_cost_value

    leaf = ComponentCost("coin", 1, GateCost(measurement=1), success_probability=0.5)
    impossible = ComponentCost("failure", 1, success_probability=0.0)
    identity = ComponentCost("identity", 1)
    nested = ComponentCost(
        "nested",
        1,
        calls=(CostCall(leaf, 4), CostCall(impossible, 0), CostCall(identity, 2)),
        success_probability=override,
    )
    root = ComponentCost("root", 3, calls=(CostCall(nested, 2, 2), CostCall(leaf, 3)))
    result = estimate_component_resources(root)

    metric = SuccessProb()
    src_leaf = CostingBloq("coin", num_qubits=1, static_costs=[(metric, 0.5)])
    src_zero = CostingBloq("failure", num_qubits=1, static_costs=[(metric, 0.0)])
    src_identity = CostingBloq("identity", num_qubits=1)
    src_nested = CostingBloq(
        "nested",
        num_qubits=1,
        callees=[(src_leaf, 4), (src_zero, 0), (src_identity, 2)],
        static_costs=[] if override is None else [(metric, override)],
    )
    src_root = CostingBloq(
        "root", num_qubits=3, callees=[(src_nested, 2), (src_leaf, 3)]
    )
    assert result.success_probability == get_cost_value(src_root, metric)
    assert result.gates.measurement == 11
    assert result.peak_logical_qubits == 3


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_success_declaration(probability: float) -> None:
    with pytest.raises(ValueError, match="Success probability"):
        estimate_component_resources(
            ComponentCost("invalid", 1, success_probability=probability)
        )


def test_feedback_restores_all_histories_with_unit_declared_success() -> None:
    gates = (*and_compute_ir(0, 1, 2), *and_uncompute_ir(0, 1, 2, 0))
    model = ComponentCost("And round trip", 3, GateCost(and_bloq=1, measurement=1))
    ir = CircuitIR(3, gates)
    report = compare_component_resources(model, ir)
    assert report.to_dict()["component_model"]["success_probability"] == 1.0
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = torch.tensor((1, 1j, -1, -1j), dtype=torch.complex128) / 2
    for history in product((0, 1), repeat=1):
        branch = forced_branch(initial, gates, history)
        assert float(branch.norm() ** 2) == pytest.approx(0.5)
        torch.testing.assert_close(branch, initial / 2**0.5, rtol=0, atol=2e-12)
