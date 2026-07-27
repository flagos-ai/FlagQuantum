import json
from pathlib import Path

import pytest

import flagquantum as fq
from benchmarks.performance_catalog import BENCHMARK_CATALOG, OPTIMIZATION_BACKLOG
from flagquantum.compilation.performance_calibration import (
    select_calibrated_world_size,
)

ROOT = Path("benchmarks/development")
ARTIFACTS = {
    1: ROOT / "issue077_1gpu_local_kernel.json",
    2: ROOT / "issue077_2gpu_sharded.json",
    4: ROOT / "issue077_4gpu_topology.json",
    8: ROOT / "issue077_8gpu_topology.json",
}
JAX_ARTIFACT = ROOT / "issue077_1gpu_jax_kernel.json"
CROSSOVER_ARTIFACT = ROOT / "issue077_crossover.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.release_gate,
    pytest.mark.skipif(
        not all(
            path.exists()
            for path in (*ARTIFACTS.values(), JAX_ARTIFACT, CROSSOVER_ARTIFACT)
        ),
        reason="legacy ISSUE-077 evidence is not present",
    ),
]


def _load(path):
    return json.loads(path.read_text())


def test_catalog_covers_required_performance_layers_and_backlog():
    assert {item.layer for item in BENCHMARK_CATALOG} == {
        "micro",
        "kernel",
        "circuit",
        "training_step",
        "communication",
        "compile_time",
        "startup",
        "memory",
    }
    assert len(OPTIMIZATION_BACKLOG) >= 7


@pytest.mark.parametrize(("world_size", "path"), ARTIFACTS.items())
def test_a100_artifact_has_variance_memory_progress_and_correctness(world_size, path):
    payload = _load(path)
    assert payload["world_size"] == world_size
    assert payload["warmup"] >= 5
    assert payload["repetitions"] >= 20
    assert len(payload["samples_seconds"]) == payload["repetitions"]
    assert payload["correctness_passed"] is True
    assert payload["performance_gate"]["passed"] is True
    assert (
        payload["robust_coefficient_of_variation"]
        <= payload["regression_thresholds"]["max_coefficient_of_variation"]
    )
    assert (
        payload["allocated_memory_slope_bytes_per_step"]
        <= payload["regression_thresholds"]["max_memory_slope_bytes_per_step"]
    )
    assert (
        payload["reserved_memory_slope_bytes_per_step"]
        <= payload["regression_thresholds"]["max_memory_slope_bytes_per_step"]
    )
    assert payload["completed_work_units"] > 0
    assert (
        max(payload["heartbeat_gaps_seconds"])
        < payload["regression_thresholds"]["max_heartbeat_gap_seconds"]
    )
    assert 0 <= payload["communication_fraction"] <= 1
    assert payload["non_release_evidence"] is True
    assert payload["release_gate_allowed"] is False
    assert payload["scaling_classification"] in {
        "not_applicable_local",
        "weak_scaling_capacity",
    }
    assert payload["workload_dimensions"]
    assert payload["crossover"]["uncertainty_margin"] >= 0
    expected_seconds_per_unit = (
        sum(payload["samples_seconds"]) / payload["completed_work_units"]
    )
    assert payload["cost_model_calibration"]["seconds_per_work_unit"] == pytest.approx(
        expected_seconds_per_unit
    )


def test_local_and_sharded_claims_are_separate_and_rank_ownership_is_distinct():
    local = _load(ARTIFACTS[1])
    assert local["distribution_semantics"] == "single_device_fast_path"
    for world_size in (2, 4, 8):
        payload = _load(ARTIFACTS[world_size])
        assert payload["distribution_semantics"] == "sharded_across_ranks"
        ranges = [tuple(item["owned_range"]) for item in payload["rank_ownership"]]
        assert len(ranges) == world_size
        assert len(set(ranges)) == world_size
        assert payload["scaling_classification"] == "weak_scaling_capacity"
        assert payload["crossover"]["status"] == "not_measured"
        assert payload["crossover"]["distributed_selection_threshold"] is None
        assert payload["crossover"]["blockers"]
        assert payload["communication_timing_method"] in {
            "device_synchronized_wall_clock",
            "host_enqueue_unsynchronized_historical",
        }


def test_pytorch_and_jax_local_measurements_are_not_combined():
    pytorch = _load(ARTIFACTS[1])
    jax = _load(JAX_ARTIFACT)
    assert pytorch["backend"] == "pytorch"
    assert jax["backend"] == "jax"
    assert pytorch["benchmark"] != jax["benchmark"]
    assert jax["performance_gate"]["passed"] is True
    assert jax["correctness_passed"] is True
    calibrated = fq.Circuit(2).h(0).cx(0, 1).plan().calibrated_cost(pytorch)
    assert calibrated.estimated_seconds > 0
    assert calibrated.relative_uncertainty == pytorch["robust_coefficient_of_variation"]


def test_matched_crossover_calibrates_planner_world_size_with_uncertainty():
    payload = _load(CROSSOVER_ARTIFACT)
    assert payload["dimensions"] == ["global_elements", "world_size"]
    assert {row["world_size"] for row in payload["measurements"]} == {1, 2, 4, 8}
    local = select_calibrated_world_size(1 << 20, payload)
    distributed = select_calibrated_world_size(1 << 24, payload)
    assert local.world_size == 1
    assert distributed.world_size in {2, 4, 8}
    assert distributed.relative_uncertainty > 0
    for source in payload["source_artifacts"]:
        measured = _load(Path(source))
        assert measured["scaling_classification"] == "strong_scaling_matched_workload"
        assert measured["communication_timing_method"] in {
            "not_applicable",
            "device_synchronized_wall_clock",
        }
        assert measured["correctness_passed"] is True
        assert measured["performance_gate"]["passed"] is True
