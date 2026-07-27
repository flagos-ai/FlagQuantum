"""Measured, fail-closed MPS production planner contracts."""

import pytest

import flagquantum as fq
from flagquantum.runtime.backends.mps.forward import NonlocalMPSCompilationError

pytestmark = pytest.mark.unit


def circuit(n_wires=8):
    value = fq.Circuit(n_wires)
    for wire in range(n_wires):
        value.h(wire)
    for wire in range(n_wires - 1):
        value.cx(wire, wire + 1)
    return value


def gates(**overrides):
    values = {
        "correctness_passed": True,
        "performance_passed": True,
        "capacity_passed": True,
        "correctness_artifact": "correctness.json",
        "performance_artifact": "speed.json",
        "capacity_artifact": "capacity.json",
    }
    values.update(overrides)
    return fq.MPSAcceptanceGates(**values)


def measurement(world_size, low, high, ci_low=1.1):
    return fq.MPSCrossoverMeasurement(
        world_size=world_size,
        workload_min_bytes=low,
        workload_max_bytes=high,
        speedup_mean=max(1.2, ci_low),
        speedup_ci_low=ci_low,
        speedup_ci_high=max(1.3, ci_low),
        sample_count=10,
    )


def test_incomplete_gates_keep_stable_planner_local_even_with_eight_gpus():
    plan = fq.plan_production_mps(
        circuit(),
        estimated_workload_bytes=100,
        single_gpu_capacity_bytes=1000,
        gates=gates(capacity_passed=False, capacity_artifact=None),
        crossover=(measurement(8, 0, 1000, 2.0),),
        available_gpu_count=8,
    )
    assert plan.execution_class == "single_gpu_fast_path"
    assert plan.production_promotion_allowed is False
    assert plan.summary()["gpu_availability_used_as_distribution_reason"] is False
    assert plan.summary()["acceptance_gates"]["correctness_passed"] is True
    assert plan.summary()["acceptance_gates"]["capacity_passed"] is False


def test_confidence_interval_selects_speed_path_and_exposes_rationale():
    plan = fq.plan_production_mps(
        circuit(),
        estimated_workload_bytes=500,
        single_gpu_capacity_bytes=1000,
        gates=gates(),
        crossover=(
            measurement(2, 0, 1000, 0.98),
            measurement(4, 0, 1000, 1.15),
        ),
        available_gpu_count=8,
    )
    assert plan.execution_class == "speed_oriented_distribution"
    assert plan.world_size == 4
    assert "confidence interval" in plan.rationale
    assert plan.distribution_semantics == "sharded_across_ranks"


def test_capacity_path_requires_matching_measured_artifact():
    with pytest.raises(fq.MPSProductionAcceptanceError, match="no matching"):
        fq.plan_production_mps(
            circuit(),
            estimated_workload_bytes=2000,
            single_gpu_capacity_bytes=1000,
            gates=gates(),
            crossover=(measurement(2, 0, 1000),),
        )
    plan = fq.plan_production_mps(
        circuit(),
        estimated_workload_bytes=2000,
        single_gpu_capacity_bytes=1000,
        gates=gates(),
        crossover=(measurement(4, 1500, 3000, 0.8),),
    )
    assert plan.execution_class == "capacity_oriented_distribution"
    assert plan.world_size == 4


def test_unsupported_topology_fails_during_planning_before_executor(monkeypatch):
    invalid = fq.Circuit(4).cx(0, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("executor allocation must not start")

    monkeypatch.setattr(
        "flagquantum.runtime.backends.mps.forward.execute_torch_distributed_mps_forward",
        forbidden,
    )
    with pytest.raises(NonlocalMPSCompilationError, match="requires MPS routing"):
        fq.plan_production_mps(
            invalid,
            estimated_workload_bytes=100,
            single_gpu_capacity_bytes=1000,
            gates=gates(),
            crossover=(measurement(2, 0, 1000),),
        )


def test_support_matrix_is_public_and_exact():
    support = fq.MPSProductionSupport().to_dict()
    assert support["two_site_topology"] == "adjacent_only"
    assert support["optimizers"] == ("sgd", "adam")
    assert support["precisions"] == ("complex64", "complex128")
    assert support["full_mps_materialization_allowed"] is False
    assert support["jax_required"] is False


def test_module_routes_incomplete_release_evidence_to_native_local_mps():
    def builder(parameters):
        return fq.Circuit(4).ry(0, parameters[0]).cx(0, 1).cx(1, 2)

    module = fq.Module(
        builder,
        1,
        policy=fq.RuntimePolicy(mode="mps", observable_wires=(0,)),
    )
    result = module.execute_production_mps(
        estimated_workload_bytes=100,
        single_gpu_capacity_bytes=1000,
        gates=gates(performance_passed=False, performance_artifact=None),
        crossover=(measurement(4, 0, 1000, 2.0),),
        available_gpu_count=8,
    )
    assert result.plan.execution_class == "single_gpu_fast_path"
    assert result.runtime["executor"] == "pytorch_native_local_mps"
    assert result.runtime["production_promotion_allowed"] is False
    assert result.runtime["full_mps_materialization"] is False


def test_release_artifact_requires_three_distinct_measured_gates():
    records = (
        {
            "gate": "correctness",
            "evidence_source": "measured_runtime",
            "artifact_path": "correctness.json",
            "artifact_sha256": "a" * 64,
        },
        {
            "gate": "performance",
            "evidence_source": "measured_runtime",
            "artifact_path": "speed.json",
            "artifact_sha256": "b" * 64,
            "speedup_ci_low": 1.05,
        },
        {
            "gate": "capacity",
            "evidence_source": "measured_runtime",
            "artifact_path": "capacity.json",
            "artifact_sha256": "c" * 64,
            "single_gpu_capacity_failure": True,
            "sharded_completion": True,
            "full_mps_materialization": False,
        },
    )
    artifact = fq.build_mps_release_artifact(gates=gates(), runtime_records=records)
    assert artifact["production_distributed_mps"] is True
    assert artifact["artifact_classification"] == "measured_production_release"
    with pytest.raises(fq.MPSProductionAcceptanceError, match="include correctness"):
        fq.build_mps_release_artifact(gates=gates(), runtime_records=records[:2])
    estimated = ({**records[0], "evidence_source": "estimated"},) + records[1:]
    with pytest.raises(fq.MPSProductionAcceptanceError, match="measured_runtime"):
        fq.build_mps_release_artifact(gates=gates(), runtime_records=estimated)
