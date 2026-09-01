import json

import pytest

import flagquantum as fq
import flagquantum.compilation.planner as fqxp
from flagquantum.compilation import (
    build_tn_working_set_calibration,
    load_tn_working_set_calibration,
)

pytestmark = pytest.mark.unit


def _grid(rows: int, columns: int, cycles: int) -> fq.Circuit:
    circuit = fq.Circuit(rows * columns)
    for cycle in range(cycles):
        for wire in range(rows * columns):
            circuit.ry(wire, theta=0.01 * (cycle + 1))
        for row in range(rows):
            for column in range(columns - 1):
                left = row * columns + column
                circuit.cx(left, left + 1)
        for row in range(rows - 1):
            for column in range(columns):
                top = row * columns + column
                circuit.cx(top, top + columns)
    return circuit


def test_30q_grid_expectation_gradient_prefers_statevector_when_it_fits() -> None:
    circuit = _grid(5, 6, 7)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="expectation",
        require_gradients=True,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    assert decision.selected_backend == "statevector"
    tn = next(item for item in decision.candidates if item.backend == "tensor_network")
    assert "dense_statevector_is_lower_risk_for_this_capacity" in tn.blockers


def test_shallow_50q_chain_uses_mps_after_dense_capacity_failure() -> None:
    circuit = fq.Circuit(50)
    for wire in range(49):
        circuit.cx(wire, wire + 1)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    assert decision.selected_backend == "mps"
    assert decision.interaction_width_proxy == 1
    assert decision.estimated_mps_bond == 2


def test_full_state_capacity_failure_does_not_hide_behind_mps() -> None:
    circuit = fq.Circuit(50)
    for wire in range(49):
        circuit.cx(wire, wire + 1)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="full_state",
        memory_limit_bytes=80 << 30,
    )

    assert decision.selected_backend == "statevector"
    mps = next(item for item in decision.candidates if item.backend == "mps")
    assert "mps_cannot_satisfy_full_state_without_dense_materialization" in (
        mps.blockers
    )


def test_nonlocal_low_treewidth_single_amplitude_uses_tn() -> None:
    circuit = fq.Circuit(50)
    for child in range(1, 50):
        circuit.cx((child - 1) // 2, child)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    assert decision.selected_backend == "tensor_network"
    assert decision.interaction_width_proxy == 1
    mps = next(item for item in decision.candidates if item.backend == "mps")
    assert "unbounded_bond_growth_on_nonlocal_topology" in mps.blockers


def test_tn_reserved_memory_calibration_scopes_and_inflates_estimate() -> None:
    circuit = fq.Circuit(50)
    for child in range(1, 50):
        circuit.cx((child - 1) // 2, child)
    measurements = (
        {
            "predicted_working_set_bytes": 1 << 20,
            "cuda_peak_allocated_bytes": 20 << 20,
            "cuda_peak_reserved_bytes": 24 << 20,
        },
        {
            "predicted_working_set_bytes": 8 << 20,
            "cuda_peak_allocated_bytes": 40 << 20,
            "cuda_peak_reserved_bytes": 48 << 20,
        },
        {
            "predicted_working_set_bytes": 32 << 20,
            "cuda_peak_allocated_bytes": 96 << 20,
            "cuda_peak_reserved_bytes": 128 << 20,
        },
    )
    calibration = build_tn_working_set_calibration(
        measurements,
        accelerator_name="NVIDIA-A800-SXM4-80GB",
        complex_bytes=8,
        world_size=1,
        topology_class="single_gpu",
    )
    baseline = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        memory_limit_bytes=80 << 30,
    )
    calibrated = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        memory_limit_bytes=80 << 30,
        accelerator_name="NVIDIA-A800-SXM4-80GB",
        tn_memory_calibration=calibration,
    )
    baseline_tn = next(
        item for item in baseline.candidates if item.backend == "tensor_network"
    )
    calibrated_tn = next(
        item for item in calibrated.candidates if item.backend == "tensor_network"
    )

    assert calibration.passed is True
    assert calibration.fixed_reserved_overhead_bytes == 24 << 20
    assert calibrated.tn_calibration_identity == calibration.identity
    assert calibrated.tn_working_set_safety_factor > 1.0
    assert calibrated_tn.estimated_memory_bytes > baseline_tn.estimated_memory_bytes
    assert "reserved_memory_calibration_applied" in calibrated_tn.reasons

    mismatched = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        memory_limit_bytes=80 << 30,
        accelerator_name="different-accelerator",
        tn_memory_calibration=calibration,
    )
    mismatched_tn = next(
        item for item in mismatched.candidates if item.backend == "tensor_network"
    )
    assert "tn_memory_calibration_scope_mismatch" in mismatched_tn.blockers


def test_tn_calibration_requirement_fails_closed_when_missing() -> None:
    circuit = fq.Circuit(50)
    for child in range(1, 50):
        circuit.cx((child - 1) // 2, child)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="single_amplitude",
        memory_limit_bytes=80 << 30,
        require_tn_memory_calibration=True,
        accelerator_name="NVIDIA-A800-SXM4-80GB",
    )
    tn = next(item for item in decision.candidates if item.backend == "tensor_network")

    assert tn.available is False
    assert "tn_memory_calibration_required" in tn.blockers


def test_tn_calibration_round_trip_and_runtime_plan_integration(tmp_path) -> None:
    measurements = tuple(
        {
            "predicted_working_set_bytes": scale << 20,
            "cuda_peak_allocated_bytes": (2 * scale + 8) << 20,
            "cuda_peak_reserved_bytes": (3 * scale + 16) << 20,
        }
        for scale in (1, 8, 32)
    )
    calibration = build_tn_working_set_calibration(
        measurements,
        accelerator_name="NVIDIA-A800-SXM4-80GB",
        complex_bytes=8,
        world_size=1,
        topology_class="single_gpu",
    )
    path = tmp_path / "tn-calibration.json"
    path.write_text(
        json.dumps(calibration.summary(), sort_keys=True),
        encoding="utf-8",
    )
    restored = load_tn_working_set_calibration(path)

    assert restored == calibration
    circuit = fq.Circuit(50)
    for child in range(1, 50):
        circuit.cx((child - 1) // 2, child)
    plan = fqxp.plan_runtime_selection(
        circuit,
        target="single_amplitude",
        require_gradients=False,
        memory_limit_bytes=80 << 30,
        accelerator_name="NVIDIA-A800-SXM4-80GB",
        tn_memory_calibration=restored,
    )
    evidence = plan.recommended_candidate.summary()["backend_cost_selection"]
    assert evidence["tn_calibration_identity"] == calibration.identity

    payload = calibration.summary()
    payload["recommended_safety_factor"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_tn_working_set_calibration(path)


def test_bounded_local_chain_prefers_mps() -> None:
    circuit = fq.Circuit(40)
    for wire in range(39):
        circuit.cx(wire, wire + 1)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="samples",
        max_bond=32,
        memory_limit_bytes=1 << 30,
    )

    assert decision.selected_backend == "mps"
    assert decision.nearest_neighbour_fraction == 1.0


def test_mps_gradient_estimate_accounts_for_reverse_working_set() -> None:
    circuit = fq.Circuit(40)
    for wire in range(39):
        circuit.cx(wire, wire + 1)

    forward = fqxp.select_backend_by_cost(
        circuit,
        target="expectation",
        memory_limit_bytes=80 << 30,
    )
    reverse = fqxp.select_backend_by_cost(
        circuit,
        target="expectation",
        require_gradients=True,
        memory_limit_bytes=80 << 30,
    )
    forward_mps = next(item for item in forward.candidates if item.backend == "mps")
    reverse_mps = next(item for item in reverse.candidates if item.backend == "mps")

    assert reverse_mps.estimated_memory_bytes == 3 * forward_mps.estimated_memory_bytes


def test_forced_tn_is_preserved_but_reports_predicted_degradation() -> None:
    decision = fqxp.select_backend_by_cost(
        _grid(5, 6, 2),
        target="expectation",
        require_gradients=True,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
        requested_backend="tensor_network",
    )

    assert decision.selected_backend == "tensor_network"
    assert decision.warnings == (
        "forced_backend_has_predicted_degradation_or_blockers",
    )


def test_runtime_plan_uses_and_exposes_the_same_cost_decision() -> None:
    circuit = _grid(5, 6, 2)

    plan = fqxp.plan_runtime_selection(
        circuit,
        target="expectation",
        require_gradients=True,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    assert plan.recommended_candidate.state_mode == "statevector"
    evidence = plan.recommended_candidate.summary()["backend_cost_selection"]
    assert evidence["selected_backend"] == "statevector"
    assert evidence["target"] == "expectation"


def test_few_amplitudes_select_tn_but_large_output_batch_does_not() -> None:
    circuit = fq.Circuit(50)
    for child in range(1, 50):
        circuit.cx((child - 1) // 2, child)

    sparse = fqxp.select_backend_by_cost(
        circuit,
        target="few_amplitudes",
        target_count=16,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )
    large = fqxp.select_backend_by_cost(
        circuit,
        target="few_amplitudes",
        target_count=4096,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    assert sparse.selected_backend == "tensor_network"
    tn = next(item for item in large.candidates if item.backend == "tensor_network")
    assert "amplitude_target_batch_too_large_for_sparse_tn_path" in tn.blockers


def test_large_observable_batch_is_not_treated_as_sparse_output() -> None:
    circuit = fq.Circuit(50)
    for wire in range(49):
        circuit.cx(wire, wire + 1)

    decision = fqxp.select_backend_by_cost(
        circuit,
        target="local_observables",
        target_count=512,
        complex_bytes=16,
        memory_limit_bytes=80 << 30,
    )

    tn = next(item for item in decision.candidates if item.backend == "tensor_network")
    assert "observable_batch_too_large_for_shared_tn_path" in tn.blockers
