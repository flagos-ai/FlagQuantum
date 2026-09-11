import pytest

from flagquantum.runtime.observability.performance import (
    PerformanceMonitor,
    PerformanceRecord,
    calibrate_cost_model,
    calibrate_crossover,
    classify_no_progress,
    evaluate_performance,
    memory_slope,
)

pytestmark = pytest.mark.unit


def _record(**changes):
    values = {
        "benchmark": "unit",
        "layer": "training_step",
        "backend": "pytorch",
        "device": "cpu",
        "distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "warmup": 2,
        "repetitions": 4,
        "samples_seconds": (1.0, 1.0, 1.0, 1.0),
        "initialization_seconds": 0.1,
        "teardown_seconds": 0.1,
        "peak_memory_allocated_bytes": 100,
        "peak_memory_reserved_bytes": 120,
        "allocated_memory_by_step": (100, 100, 100, 100),
        "reserved_memory_by_step": (120, 120, 120, 120),
        "useful_work_seconds": 4.0,
        "communication_seconds": 0.0,
        "idle_seconds": 0.0,
        "completed_work_units": 4,
        "heartbeat_gaps_seconds": (1.0, 1.0, 1.0, 1.0),
        "correctness_passed": True,
    }
    values.update(changes)
    return PerformanceRecord(**values)


def test_record_reports_variance_utilization_and_memory_trend():
    summary = _record().summary()
    assert summary["coefficient_of_variation"] == 0.0
    assert summary["useful_work_fraction"] == 1.0
    assert summary["allocated_memory_slope_bytes_per_step"] == 0.0


def test_memory_leak_slope_fails_closed():
    record = _record(allocated_memory_by_step=(0, 2048, 4096, 6144))
    result = evaluate_performance(record)
    assert not result.passed
    assert "unbounded_memory_growth_suspected" in result.errors
    assert memory_slope(record.allocated_memory_by_step) == 2048.0


def test_latency_and_memory_regressions_compare_to_baseline():
    baseline = _record()
    current = _record(
        samples_seconds=(1.2, 1.2, 1.2, 1.2),
        peak_memory_reserved_bytes=140,
    )
    result = evaluate_performance(current, baseline=baseline)
    assert {"latency_regression", "memory_regression"}.issubset(result.errors)


def test_variance_and_heartbeat_are_hardware_gate_inputs():
    record = _record(
        samples_seconds=(0.5, 1.5, 0.5, 1.5),
        heartbeat_gaps_seconds=(1.0, 1.0, 31.0, 1.0),
    )
    result = evaluate_performance(record)
    assert "latency_variance_exceeds_threshold" in result.errors
    assert "no_progress_heartbeat_timeout" in result.errors


def test_low_activity_requires_explanation_or_fails():
    stalled = _record(useful_work_seconds=0.01, idle_seconds=4.0)
    assert (
        "persistently_low_useful_work_fraction" in evaluate_performance(stalled).errors
    )
    explained = _record(
        useful_work_seconds=0.01,
        idle_seconds=4.0,
        explanation="expected compile interval",
    )
    assert evaluate_performance(explained).passed


def test_no_progress_distinguishes_expected_phase_and_stall():
    record = _record(completed_work_units=0)
    assert classify_no_progress(record, phase="jit") == "expected_jit_interval"
    assert classify_no_progress(record, phase="execute") == "stalled"


def test_cost_calibration_reports_uncertainty():
    calibration = calibrate_cost_model(_record())
    assert calibration.seconds_per_work_unit == 1.0
    assert calibration.relative_uncertainty == 0.0
    assert calibration.sample_count == 4


def test_cost_calibration_uses_total_sample_time_for_total_work_units():
    record = _record(
        repetitions=3,
        samples_seconds=(2.0, 2.0, 2.0),
        completed_work_units=6,
    )
    assert calibrate_cost_model(record).seconds_per_work_unit == 1.0


def test_crossover_requires_a_win_beyond_measurement_uncertainty():
    rows = (
        {
            "world_size": 1,
            "global_elements": 1024,
            "median_seconds": 1.0,
            "relative_uncertainty": 0.05,
        },
        {
            "world_size": 2,
            "global_elements": 1024,
            "median_seconds": 1.02,
            "relative_uncertainty": 0.04,
        },
        {
            "world_size": 1,
            "global_elements": 4096,
            "median_seconds": 2.0,
            "relative_uncertainty": 0.03,
        },
        {
            "world_size": 2,
            "global_elements": 4096,
            "median_seconds": 1.5,
            "relative_uncertainty": 0.04,
        },
    )
    selection = calibrate_crossover(rows, distributed_world_size=2)
    assert selection.minimum_global_elements == 4096
    assert selection.measured_world_sizes == (1, 2)


def test_monitor_captures_activity_memory_compile_and_progress_time_series():
    monitor = PerformanceMonitor()
    monitor.sample(
        gpu_utilization_percent=75.0,
        kernel_active=True,
        memory_allocated_bytes=100,
        memory_reserved_bytes=120,
        communication_progress_bytes=64,
        cpu_wait_seconds=0.01,
        compilation_state="steady_state",
        completed_work_units=1,
    )
    assert len(monitor.samples) == 1
    assert monitor.samples[0].kernel_active
    assert monitor.samples[0].communication_progress_bytes == 64
