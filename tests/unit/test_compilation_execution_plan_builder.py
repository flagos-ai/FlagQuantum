import pytest

from flagquantum.compilation.execution_plan_builder import (
    build_execution_plan,
    build_layer_plans,
)
from flagquantum.compilation.models import CircuitAnalysis
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _ir() -> CircuitIR:
    return CircuitIR(
        n_wires=3,
        instructions=(
            Instruction("h", (0,)),
            Instruction("rz", (2,), params={"theta": 0.25}),
            Instruction("cx", (0, 1)),
        ),
    )


def _analysis() -> CircuitAnalysis:
    return CircuitAnalysis(
        n_wires=3,
        n_instructions=3,
        depth=2,
        gate_counts={"h": 1, "rz": 1, "cx": 1},
        wire_usage=(2, 1, 1),
        max_gate_width=2,
        two_qubit_gates=1,
        multi_qubit_gates=0,
    )


def test_layer_plans_preserve_scheduler_order_and_sorted_wire_union() -> None:
    layers = build_layer_plans(_ir())

    assert tuple(layer.index for layer in layers) == (0, 1)
    assert tuple(instruction.name for instruction in layers[0].instructions) == (
        "h",
        "rz",
    )
    assert layers[0].wires == (0, 2)
    assert layers[1].wires == (0, 1)


def test_execution_plan_builder_assigns_distributed_product_contract() -> None:
    plan = build_execution_plan(
        ir=_ir(),
        analysis=_analysis(),
        state_bytes=64,
        recommended_mode="distributed_statevector",
        world_size=2,
        state_mode="statevector",
        runtime_config={"schema": "test"},
    )

    assert plan.world_size == 2
    assert plan.shardable_wires == (0, 1)
    assert plan.user_tier == "production_distributed"
    assert plan.usability_contract == "single_api_distributed_scale_out"
    assert plan.runtime_config == {"schema": "test"}


def test_execution_plan_calibrated_cost_preserves_plan_product_behavior() -> None:
    plan = build_execution_plan(
        ir=_ir(),
        analysis=_analysis(),
        state_bytes=64,
        recommended_mode="statevector",
        world_size=1,
        state_mode="statevector",
        runtime_config={"schema": "test"},
    )

    cost = plan.calibrated_cost(
        {
            "performance_gate": {"passed": True},
            "cost_model_calibration": {
                "seconds_per_work_unit": 0.25,
                "relative_uncertainty": 0.1,
                "source_benchmark": "cpu-golden-path",
            },
        }
    )

    assert cost.estimated_seconds == pytest.approx(0.75)
    assert cost.estimated_work_units == 3
    assert cost.source_benchmark == "cpu-golden-path"
