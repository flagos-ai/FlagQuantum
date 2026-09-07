"""Tests for native FlagQuantum noise and density-matrix execution."""

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.noise as fqn
import flagquantum.noise as noise
import flagquantum.runtime.planner as fqxp
from flagquantum.compiler import lower_noise_model
from flagquantum.noise import (
    CorrelatedReadoutError,
    DeviceNoiseProfile,
    GateDuration,
    KrausChannel,
    NoiseRule,
    QubitNoiseCalibration,
    ReadoutError,
    amplitude_damping_channel,
    bit_flip_channel,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    reset_error_channel,
    thermal_relaxation_channel,
)
from flagquantum.runtime.backends.statevector import (
    merge_noisy_statevector_results,
    run_noisy_statevector,
)
from flagquantum.runtime.execution import run_advanced
from flagquantum.runtime.planner import (
    NOISE_SELECTOR_CALIBRATION_SCHEMA,
    estimate_density_bytes,
    plan_noise_execution_selection,
    select_execution_mode,
)
from flagquantum.simulation.density_matrix import (
    apply_kraus_density,
    density_matrix_from_ir,
    expectation_z_density,
)


def test_batched_statevector_trajectory_is_batch_size_invariant():
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    model = fqn.NoiseModel().add("cx", depolarizing_channel(0.2))

    serial = run_noisy_statevector(
        circuit,
        model,
        trajectories=37,
        trajectory_batch_size=1,
        seed=19,
        retain_trajectories=True,
    )
    batched = run_noisy_statevector(
        circuit,
        model,
        trajectories=37,
        trajectory_batch_size=11,
        seed=19,
        retain_trajectories=True,
    )

    assert torch.equal(serial.retained_states, batched.retained_states)
    assert torch.equal(serial.expectation_z, batched.expectation_z)
    assert serial.trajectory_seeds == batched.trajectory_seeds
    assert batched.pauli_fast_path_events == 37 * circuit.bsz * 2


def test_batched_statevector_generic_kraus_matches_density_matrix():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", amplitude_damping_channel(0.3))

    sampled = run_noisy_statevector(
        circuit,
        model,
        trajectories=4000,
        trajectory_batch_size=128,
        seed=23,
    )
    exact = expectation_z_density(fqn.noisy_density_matrix(circuit, model))

    assert torch.allclose(sampled.expectation_z, exact, atol=0.04)
    assert sampled.amplitude_damping_fast_path_events == 4000
    assert sampled.generic_kraus_events == 0
    assert sampled.noise_model_identity == model.identity


def test_batched_statevector_applies_readout_and_preserves_circuit_batch():
    inputs = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
    circuit = fq.Circuit(1, bsz=2, inputs=inputs)
    model = fqn.NoiseModel().add_readout(0, ReadoutError(((0.75, 0.25), (0.1, 0.9))))

    result = run_noisy_statevector(
        circuit, model, trajectories=3, trajectory_batch_size=2, seed=5
    )

    assert result.expectation_z.shape == (2, 1)
    assert torch.allclose(result.expectation_z, torch.tensor([[0.5], [-0.8]]))


def test_batched_statevector_two_wire_kraus_matches_density_and_batching():
    probability = 0.3
    identity = torch.eye(4, dtype=torch.complex64)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    channel = KrausChannel(
        "correlated_flip",
        (
            (1 - probability) ** 0.5 * identity,
            probability**0.5 * torch.kron(x, x),
        ),
    )
    model = fqn.NoiseModel().add("cx", channel)
    circuit = fq.Circuit(2).x(0).cx(0, 1)

    serial = run_noisy_statevector(
        circuit,
        model,
        trajectories=4000,
        trajectory_batch_size=1,
        seed=17,
        retain_trajectories=True,
    )
    batched = run_noisy_statevector(
        circuit,
        model,
        trajectories=4000,
        trajectory_batch_size=127,
        seed=17,
        retain_trajectories=True,
    )
    exact = expectation_z_density(fqn.noisy_density_matrix(circuit, model))

    assert torch.equal(serial.retained_states, batched.retained_states)
    assert torch.allclose(batched.expectation_z, exact, atol=0.04)
    assert batched.generic_kraus_events == 4000


def test_public_run_reports_noisy_statevector_runtime_mode():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.25))

    result = run_advanced(circuit, noise_model=model, trajectories=4, seed=3)

    assert result.runtime["mode"] == "noisy_statevector"
    assert result.plan.noisy_execution_plan.noise_model_identity == model.identity


def test_rank_local_batched_statevector_results_merge_by_global_id():
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    model = fqn.NoiseModel().add("cx", depolarizing_channel(0.2))
    single = run_noisy_statevector(
        circuit,
        model,
        trajectories=31,
        trajectory_batch_size=7,
        seed=101,
        retain_trajectories=True,
    )
    local = tuple(
        run_noisy_statevector(
            circuit,
            model,
            trajectories=31,
            trajectory_batch_size=4,
            seed=101,
            retain_trajectories=True,
            rank=rank,
            world_size=4,
        )
        for rank in range(4)
    )

    merged = merge_noisy_statevector_results(local)

    assert tuple(sorted(i for item in local for i in item.trajectory_ids)) == tuple(
        range(31)
    )
    assert merged.trajectory_seeds == single.trajectory_seeds
    assert torch.equal(merged.retained_states, single.retained_states)
    assert torch.allclose(merged.expectation_z, single.expectation_z, atol=1e-7)
    assert merged.statistics.count == 31


def test_rank_local_batched_statevector_merge_rejects_incomplete_set():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.5))
    rank_zero = run_noisy_statevector(
        circuit, model, trajectories=4, seed=9, rank=0, world_size=2
    )

    with pytest.raises(ValueError, match="incomplete"):
        merge_noisy_statevector_results((rank_zero,))


def test_batched_statevector_adaptive_stops_on_batch_boundary():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    result = run_noisy_statevector(
        circuit,
        model,
        trajectories=100,
        trajectory_batch_size=4,
        min_trajectories=8,
        target_standard_error=1e-5,
        seed=31,
    )

    assert result.statistics.count == 8
    assert result.converged is True
    assert result.stopped_early is True
    assert result.target_standard_error == 1e-5


def test_batched_statevector_adaptive_reports_unmet_target():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.5))

    result = run_noisy_statevector(
        circuit,
        model,
        trajectories=12,
        trajectory_batch_size=4,
        min_trajectories=4,
        target_standard_error=1e-12,
        seed=37,
    )

    assert result.statistics.count == 12
    assert result.converged is False
    assert result.stopped_early is False


def test_public_batched_statevector_plan_records_adaptive_policy():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    result = run_advanced(
        circuit,
        noise_model=model,
        trajectories=20,
        trajectory_batch_size=4,
        min_trajectories=8,
        target_standard_error=1e-5,
        seed=41,
    )

    trajectory_plan = result.plan.noisy_execution_plan.trajectory
    assert trajectory_plan.count == 20
    assert trajectory_plan.min_count == 8
    assert trajectory_plan.target_standard_error == 1e-5
    assert result.native().statistics.count == 8


def test_batched_statevector_checkpoint_resume_matches_continuous(tmp_path):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    model = fqn.NoiseModel().add("cx", depolarizing_channel(0.2))
    checkpoint_path = tmp_path / "statevector.pt"
    continuous = run_noisy_statevector(
        circuit, model, trajectories=24, trajectory_batch_size=4, seed=43
    )

    partial = run_noisy_statevector(
        circuit,
        model,
        trajectories=24,
        trajectory_batch_size=4,
        seed=43,
        checkpoint_path=checkpoint_path,
        max_batches_per_run=2,
    )
    resumed = run_noisy_statevector(
        circuit,
        model,
        trajectories=24,
        trajectory_batch_size=4,
        seed=43,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert partial.statistics.count == 8
    assert resumed.statistics.count == 24
    assert resumed.trajectory_ids == continuous.trajectory_ids
    assert torch.equal(resumed.expectation_z, continuous.expectation_z)
    assert torch.equal(resumed.variance, continuous.variance)


def test_batched_statevector_checkpoint_rejects_changed_identity(tmp_path):
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.2))
    checkpoint_path = tmp_path / "statevector.pt"
    run_noisy_statevector(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
        seed=47,
        checkpoint_path=checkpoint_path,
        max_batches_per_run=1,
    )

    changed_noise = fqn.NoiseModel().add("x", bit_flip_channel(0.3))
    with pytest.raises(ValueError, match="noise model identity"):
        run_noisy_statevector(
            circuit,
            changed_noise,
            trajectories=8,
            trajectory_batch_size=4,
            seed=47,
            checkpoint_path=checkpoint_path,
            resume=True,
        )
    with pytest.raises(ValueError, match="circuit digest"):
        run_noisy_statevector(
            fq.Circuit(1).h(0),
            model,
            trajectories=8,
            trajectory_batch_size=4,
            seed=47,
            checkpoint_path=checkpoint_path,
            resume=True,
        )


def test_batched_statevector_isolates_failure_and_retries_from_checkpoint(
    tmp_path, monkeypatch
):
    from flagquantum.runtime.backends.statevector import noisy

    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.25))
    checkpoint_path = tmp_path / "failure.pt"
    original = noisy._execute_trajectory_batch

    def injected_failure(initial, ids, **kwargs):
        if len(ids) > 1:
            raise RuntimeError("injected batch failure")
        if ids == (3,):
            raise RuntimeError("injected trajectory failure")
        return original(initial, ids, **kwargs)

    monkeypatch.setattr(noisy, "_execute_trajectory_batch", injected_failure)
    partial = run_noisy_statevector(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
        seed=53,
        continue_on_error=True,
        checkpoint_path=checkpoint_path,
    )

    assert partial.statistics.count == 7
    assert partial.trajectory_ids == (0, 1, 2, 4, 5, 6, 7)
    assert len(partial.failures) == 1
    assert partial.failures[0].trajectory_id == 3
    assert partial.failures[0].error_type == "RuntimeError"

    monkeypatch.setattr(noisy, "_execute_trajectory_batch", original)
    resumed = run_noisy_statevector(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
        seed=53,
        checkpoint_path=checkpoint_path,
        resume=True,
        retry_failed=True,
    )

    assert resumed.statistics.count == 8
    assert resumed.trajectory_ids == tuple(range(8))
    assert resumed.failures == ()


def test_kraus_channel_rejects_non_trace_preserving_operators():
    with pytest.raises(ValueError, match="trace-preserving"):
        KrausChannel(
            "invalid",
            (torch.eye(2, dtype=torch.complex64) * 0.5,),
        )


def test_noise_model_round_trip_preserves_stable_identity():
    model = (
        fqn.NoiseModel()
        .add(("x", "sx"), phase_damping_channel(0.2), wires=1)
        .add("cx", depolarizing_channel(0.03))
        .add_readout(0, ReadoutError(((0.98, 0.02), (0.07, 0.93))))
    )

    restored = fqn.NoiseModel.from_dict(model.to_dict())

    assert restored.to_dict() == model.to_dict()
    assert restored.identity == model.identity
    assert len(model.identity) == 64


def test_readout_confusion_is_classical_and_wire_local():
    model = fqn.NoiseModel().add_readout(
        1,
        ReadoutError(((0.9, 0.1), (0.2, 0.8))),
    )
    ideal = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # true |01>

    observed = model.apply_readout_probabilities(ideal, n_wires=2)

    assert torch.allclose(observed, torch.tensor([[0.2, 0.8, 0.0, 0.0]]))
    assert torch.allclose(observed.sum(dim=-1), torch.ones(1))
    observed_z = model.apply_readout_expectation_z(torch.tensor([[1.0, -1.0]]))
    assert torch.allclose(observed_z, torch.tensor([[1.0, -0.6]]))


def test_noisy_mps_aggregates_post_readout_z_expectation():
    model = fqn.NoiseModel().add_readout(
        0,
        ReadoutError(((0.75, 0.25), (0.1, 0.9))),
    )

    result = fqb.run_noisy_mps(
        fq.Circuit(1),
        model,
        trajectories=3,
        seed=5,
        retain_trajectories=False,
    )

    assert torch.allclose(result.expectation_z_mean, torch.tensor([[0.5]]))


def test_readout_confusion_rejects_invalid_rows_and_duplicate_wires():
    with pytest.raises(ValueError, match="sum to 1"):
        ReadoutError(((0.8, 0.3), (0.0, 1.0)))
    model = fqn.NoiseModel().add_readout(0, ReadoutError(((1.0, 0.0), (0.0, 1.0))))
    with pytest.raises(ValueError, match="only one"):
        model.add_readout(0, ReadoutError(((0.9, 0.1), (0.1, 0.9))))


def test_correlated_readout_confusion_preserves_joint_assignment_errors():
    matrix = (
        (0.90, 0.04, 0.05, 0.01),
        (0.03, 0.91, 0.02, 0.04),
        (0.06, 0.02, 0.89, 0.03),
        (0.01, 0.07, 0.08, 0.84),
    )
    model = fqn.NoiseModel().add_correlated_readout(
        (0, 1), CorrelatedReadoutError(matrix)
    )

    observed = model.apply_readout_probabilities(
        torch.tensor([0.0, 0.0, 1.0, 0.0]), n_wires=2
    )

    assert observed.tolist() == pytest.approx(matrix[2])
    restored = fqn.NoiseModel.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()


def test_phase_damping_matches_analytic_coherence():
    circuit = fq.Circuit(1).h(0)
    model = fqn.NoiseModel().add("h", phase_damping_channel(0.75))

    rho = fqn.noisy_density_matrix(circuit, model)

    assert torch.allclose(rho[0, 0, 1], torch.tensor(0.25 + 0j), atol=1e-6)
    assert torch.allclose(torch.diagonal(rho[0]), torch.full((2,), 0.5 + 0j), atol=1e-6)


def test_reset_error_and_coherent_overrotation_have_expected_limits():
    reset = fqn.NoiseModel().add("x", reset_error_channel(1.0))
    reset_rho = fqn.noisy_density_matrix(fq.Circuit(1).x(0), reset)
    assert torch.allclose(
        expectation_z_density(reset_rho, 0), torch.ones(1, 1), atol=1e-6
    )

    angle = 0.37
    rotation = fqn.NoiseModel().add("x", coherent_overrotation_channel(angle, axis="x"))
    rotation_rho = fqn.noisy_density_matrix(fq.Circuit(1).x(0), rotation)
    assert torch.allclose(
        expectation_z_density(rotation_rho, 0),
        torch.tensor([[-math.cos(angle)]]),
        atol=1e-6,
    )


def test_thermal_relaxation_matches_t1_population_decay():
    t1, duration = 10.0, 2.5
    model = fqn.NoiseModel().add("x", thermal_relaxation_channel(t1, 2 * t1, duration))

    rho = fqn.noisy_density_matrix(fq.Circuit(1).x(0), model)

    expected_excited = torch.exp(torch.tensor(-duration / t1))
    assert torch.allclose(torch.real(rho[0, 1, 1]), expected_excited, atol=1e-6)


def _device_profile(*, x0_duration=20.0, x1_duration=10.0):
    return DeviceNoiseProfile(
        qubits=(
            QubitNoiseCalibration(
                0,
                t1=100.0,
                t2=150.0,
                readout_error=ReadoutError(((0.98, 0.02), (0.05, 0.95))),
            ),
            QubitNoiseCalibration(1, t1=110.0, t2=170.0),
        ),
        gate_durations=(
            GateDuration("x", x0_duration, wires=(0,)),
            GateDuration("x", x1_duration, wires=(1,)),
            GateDuration("cx", 30.0),
        ),
        source="test-calibration",
        captured_at="2026-08-06T12:00:00+08:00",
        time_unit="ns",
    )


def test_device_noise_profile_round_trip_and_model_identity():
    profile = _device_profile()
    restored = DeviceNoiseProfile.from_dict(profile.to_dict())
    model = fqn.NoiseModel.from_device_profile(restored)
    model_round_trip = fqn.NoiseModel.from_dict(model.to_dict())

    assert restored == profile
    assert restored.identity == profile.identity
    assert model_round_trip.identity == model.identity
    assert model.readout_rules[0].wires == (0,)


def test_device_profile_lowers_gate_and_idle_relaxation_with_provenance():
    profile = _device_profile()
    model = fqn.NoiseModel.from_device_profile(profile)
    circuit = fq.Circuit(2).x(0).x(1).cx(0, 1)

    lowered = lower_noise_model(circuit, model)
    channels = [item for item in lowered if item.metadata.get("is_channel")]
    placements = [item.metadata["placement"] for item in channels]

    assert placements.count("during_gate_approximation") == 4
    assert placements.count("idle_before_gate") == 1
    idle = next(
        item for item in channels if item.metadata["placement"] == "idle_before_gate"
    )
    assert idle.wires == (1,)
    assert idle.metadata["duration"] == 10.0
    assert idle.metadata["source_gate_name"] == "cx"
    assert idle.metadata["device_profile_identity"] == profile.identity


def test_device_profile_relaxation_executes_on_density_backend():
    t1 = 10.0
    duration = math.log(2) * t1
    profile = DeviceNoiseProfile(
        qubits=(QubitNoiseCalibration(0, t1=t1, t2=2 * t1),),
        gate_durations=(GateDuration("x", duration),),
        source="analytic",
        captured_at="2026-08-06T12:00:00+08:00",
    )

    rho = fqn.noisy_density_matrix(
        fq.Circuit(1).x(0), fqn.NoiseModel.from_device_profile(profile)
    )

    assert torch.allclose(expectation_z_density(rho, 0), torch.zeros(1, 1), atol=1e-6)

    model = fqn.NoiseModel.from_device_profile(profile)
    _, plan = fqb.run_native(
        fq.Circuit(1).x(0),
        noise_model=model,
        mode="density_matrix",
        return_plan=True,
    )
    assert plan.noisy_execution_plan.noise_model_identity == model.identity


def test_device_profile_missing_gate_duration_fails_closed():
    profile = DeviceNoiseProfile(
        qubits=(QubitNoiseCalibration(0, t1=10.0, t2=15.0),),
        gate_durations=(GateDuration("x", 1.0),),
        source="incomplete",
        captured_at="2026-08-06T12:00:00+08:00",
    )
    with pytest.raises(ValueError, match="no duration for gate 'h'"):
        lower_noise_model(
            fq.Circuit(1).h(0), fqn.NoiseModel.from_device_profile(profile)
        )


def test_device_profile_requires_timestamped_provenance():
    with pytest.raises(ValueError, match="timezone"):
        DeviceNoiseProfile(
            qubits=(QubitNoiseCalibration(0, t1=10.0, t2=15.0),),
            gate_durations=(GateDuration("x", 1.0),),
            source="device",
            captured_at="2026-08-06T12:00:00",
        )


def test_two_qubit_mps_trajectory_samples_kraus_branches():
    probability = 0.5
    identity = torch.eye(4, dtype=torch.complex64)
    xx = torch.kron(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    )
    channel = KrausChannel(
        "correlated_flip",
        (
            (1 - probability) ** 0.5 * identity,
            probability**0.5 * xx,
        ),
    )
    circuit = fq.Circuit(2).x(0).cx(0, 1)
    model = fqn.NoiseModel().add("cx", channel)

    exact = expectation_z_density(fqn.noisy_density_matrix(circuit, model))
    sampled = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=512,
        seed=811,
        retain_trajectories=False,
    )

    assert torch.allclose(sampled.expectation_z_mean, exact, atol=0.12)


def test_noise_semantics_have_one_canonical_public_identity():
    import flagquantum.compiler as noise_compiler
    import flagquantum.simulation.density_matrix as density_backend

    assert noise.NoiseModel is fqn.NoiseModel
    assert noise.NoiseRule is NoiseRule
    assert noise.KrausChannel is KrausChannel
    assert lower_noise_model is noise_compiler.lower_noise_model
    assert density_matrix_from_ir is density_backend.density_matrix_from_ir


def test_noisy_density_runtime_delegates_numerics_to_simulation(monkeypatch):
    import flagquantum.simulation.density_matrix as density_backend

    expected = torch.eye(2, dtype=torch.complex64).reshape(1, 2, 2)
    calls = []

    def replacement(ir, **options):
        calls.append((ir, options))
        return expected

    monkeypatch.setattr(density_backend, "density_matrix_from_ir", replacement)

    assert fqn.noisy_density_matrix(fq.Circuit(1)) is expected
    assert len(calls) == 1
    assert isinstance(calls[0][0], fq.CircuitIR)
    assert calls[0][1] == {"bsz": 1, "device": "cpu", "dtype": None}


def test_structured_noisy_execution_plan_separates_evolution_semantics():
    from flagquantum.runtime.planner import (
        build_noisy_execution_plan,
    )

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    execution_plan = fqxp.plan_advanced(circuit, state_mode="density_matrix")
    plan = build_noisy_execution_plan(
        execution_plan,
        representation="density_matrix",
        evolution="exact_channel",
        memory_limit_bytes=execution_plan.state_bytes,
    )

    assert plan.representation == "density_matrix"
    assert plan.evolution == "exact_channel"
    assert plan.trajectory is None
    assert plan.memory.fits is True
    assert plan.summary()["sampling_error_enabled"] is False


def test_quantum_trajectory_plan_requires_sampling_controls():
    from flagquantum.runtime.planner import (
        build_noisy_execution_plan,
    )

    execution_plan = fqxp.plan_advanced(fq.Circuit(1), state_mode="mps")
    with pytest.raises(
        ValueError, match="quantum trajectory evolution requires a trajectory plan"
    ):
        build_noisy_execution_plan(
            execution_plan,
            representation="mps",
            evolution="quantum_trajectory",
        )

    plan = build_noisy_execution_plan(
        execution_plan,
        representation="mps",
        evolution="quantum_trajectory",
        trajectories=128,
        seed=7,
    )
    assert plan.trajectory is not None
    assert plan.trajectory.count == 128
    assert plan.trajectory.seed == 7
    assert plan.error_budget.sampling_error_enabled is True


def test_density_noise_executor_is_resolved_through_registry(monkeypatch):
    from flagquantum.compiler import lower_noise_model
    from flagquantum.runtime.noise_registry import execute_noisy_plan
    from flagquantum.runtime.planner import (
        build_noisy_execution_plan,
    )

    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))
    lowered = lower_noise_model(circuit, model)
    execution_plan = fqxp.plan_advanced(circuit, noise_model=model)
    noisy_plan = build_noisy_execution_plan(
        execution_plan,
        representation="density_matrix",
        evolution="exact_channel",
    )

    def reject_lowering(*args, **kwargs):
        raise AssertionError("planned density execution must not lower noise again")

    monkeypatch.setattr("flagquantum.compiler.lower_noise_model", reject_lowering)

    rho = execute_noisy_plan(lowered, noisy_plan)

    assert torch.allclose(expectation_z_density(rho, 0), torch.ones(1, 1))


def test_density_matrix_from_native_circuit():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    rho = circuit.density_matrix()

    assert rho.shape == (1, 4, 4)
    assert torch.allclose(
        torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)), torch.ones(1)
    )
    assert torch.allclose(expectation_z_density(rho), torch.zeros(1, 2), atol=1e-6)


def test_bit_flip_channel_on_density_matrix():
    circuit = fq.Circuit(1)
    rho = apply_kraus_density(
        circuit.density_matrix(),
        bit_flip_channel(1.0),
        (0,),
        circuit.n_wires,
    )

    assert torch.allclose(expectation_z_density(rho, 0), -torch.ones(1, 1), atol=1e-6)


def test_noisy_density_matrix_gate_noise_model():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    rho = circuit.noisy_density_matrix(model)

    assert torch.allclose(expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_amplitude_damping_channel():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", amplitude_damping_channel(1.0))

    rho = fqn.noisy_density_matrix(circuit, model)

    assert torch.allclose(expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_density_matrix_matches_statevector_expectation():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1).rz(1, theta=0.3)

    rho = fqn.noisy_density_matrix(circuit)

    assert torch.allclose(
        expectation_z_density(rho),
        circuit.expectation_z(),
        atol=1e-6,
    )


def test_noise_model_lowers_to_unified_ir_and_planner():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    lowered = lower_noise_model(circuit, model)
    plan = fqxp.plan_advanced(circuit, noise_model=model)

    assert [inst.name for inst in lowered] == ["x", "bit_flip"]
    assert lowered.instructions[1].metadata["is_channel"] is True
    assert plan.analysis.channel_count == 1
    assert plan.analysis.has_noise is True
    assert plan.state_mode == "density_matrix"
    assert plan.recommended_mode == "density"
    assert plan.state_bytes == estimate_density_bytes(1)


def test_circuit_run_uses_density_matrix_for_noise():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    result = circuit.run(noise_model=model)
    rho = result.state
    plan = result.plan
    rho_from_ir = density_matrix_from_ir(lower_noise_model(circuit, model))

    assert torch.allclose(rho, rho_from_ir)
    assert plan.state_mode == "density_matrix"
    assert plan.noisy_execution_plan is not None
    assert plan.noisy_execution_plan.representation == "density_matrix"
    assert plan.noisy_execution_plan.evolution == "exact_channel"
    assert plan.noisy_execution_plan.noise_model_identity == model.identity
    assert torch.allclose(expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_run_native_statevector_path():
    circuit = fq.Circuit(1)
    circuit.h(0)

    state, plan = fqb.run_native(circuit, return_plan=True)

    assert state.shape == (1, 2)
    assert plan.state_mode == "statevector"
    assert torch.allclose(state.abs() ** 2, torch.full((1, 2), 0.5), atol=1e-6)


def test_auto_mode_selects_density_matrix_for_noise():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    rho, plan = fqb.run_native(circuit, noise_model=model, return_plan=True)

    assert plan.state_mode == "density_matrix"
    assert select_execution_mode(circuit, noise_model=model) == "density_matrix"
    assert torch.allclose(expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_auto_mode_selects_batched_statevector_when_trajectory_controls_fit():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    result, plan = fqb.run_native(
        circuit,
        noise_model=model,
        trajectories=4,
        return_plan=True,
    )

    assert (
        select_execution_mode(circuit, noise_model=model, trajectories=4)
        == "noisy_statevector"
    )
    assert plan.state_mode == "statevector"
    assert plan.noisy_execution_plan.representation == "statevector"
    assert plan.noisy_execution_plan.memory.estimated_bytes == 4 * 2 * 8 * 4
    assert result.statistics.count == 4
    assert torch.allclose(result.expectation_z, torch.ones(1, 1), atol=1e-6)


def test_planned_noisy_statevector_consumes_lowered_ir_without_recompiling(
    monkeypatch: pytest.MonkeyPatch,
):
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))
    expected, plan = fqb.run_native(
        circuit,
        noise_model=model,
        mode="noisy_statevector",
        trajectories=4,
        seed=7,
        return_plan=True,
    )
    lowered = lower_noise_model(circuit, model)

    def forbidden(*args, **kwargs):
        raise AssertionError("planned noisy execution must not lower noise again")

    monkeypatch.setattr(
        "flagquantum.compiler.lower_noise_model",
        forbidden,
    )
    actual, returned_plan = fqb.run_native(
        lowered,
        noise_model=model,
        mode="noisy_statevector",
        trajectories=4,
        seed=7,
        return_plan=True,
        _execution_plan=plan,
    )

    assert returned_plan is plan
    assert torch.equal(actual.expectation_z, expected.expectation_z)
    assert actual.trajectory_seeds == expected.trajectory_seeds


def test_auto_mode_fails_when_every_noisy_candidate_exceeds_memory():
    circuit = fq.Circuit(2)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    with pytest.raises(ValueError, match="no noisy execution candidate"):
        fqb.run_native(
            circuit,
            noise_model=model,
            memory_limit_bytes=1,
            trajectories=2,
            return_plan=True,
        )


def test_noise_selector_reports_candidates_and_rejection_reasons():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
    )
    candidates = {item.mode: item for item in selection.candidates}

    assert selection.selected_mode == "noisy_statevector"
    assert selection.noise_event_count == 1
    assert selection.maximum_kraus_rank == 2
    assert candidates["density_matrix"].rejection_reasons == (
        "trajectory_execution_explicitly_requested",
    )
    assert candidates["noisy_statevector"].sampling_error is True
    assert candidates["noisy_statevector"].truncation_error is False
    assert candidates["noisy_mps"].truncation_error is True


def test_distributed_selector_rejects_public_noisy_mps_candidate():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        world_size=2,
    )
    candidates = {item.mode: item for item in selection.candidates}

    assert selection.selected_mode == "noisy_statevector"
    assert candidates["noisy_mps"].eligible is False
    assert (
        "distributed_collective_not_implemented"
        in candidates["noisy_mps"].rejection_reasons
    )
    assert (
        candidates["noisy_mps"].metadata["distributed_public_execution_supported"]
        is False
    )


def test_explicit_distributed_noisy_mps_request_fails_closed():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(ValueError, match="requested noisy execution candidate"):
        plan_noise_execution_selection(
            circuit,
            model,
            trajectories=8,
            world_size=2,
            max_bond=2,
        )

    with pytest.raises(NotImplementedError, match="distributed statistics reduction"):
        fqb.run_native(
            circuit,
            noise_model=model,
            mode="noisy_mps",
            trajectories=8,
            world_size=2,
        )


def test_noise_selector_respects_exact_and_explicit_mps_policy():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    exact = plan_noise_execution_selection(circuit, model, exact_noise=True)
    compressed = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        max_bond=2,
        cutoff=1e-8,
    )

    assert exact.selected_mode == "density_matrix"
    assert exact.selected_candidate.exact_quantum_channel is True
    assert compressed.selected_mode == "noisy_mps"
    assert "explicit_mps_controls" in compressed.selected_candidate.reasons


def _selector_calibration(
    circuit, model, *, statevector_seconds, mps_seconds, density_seconds=2.0
):
    common = {
        "n_wires": circuit.n_wires,
        "depth": 1,
        "channel_count": 1,
        "circuit_digest": circuit.to_ir().content_hash,
        "noise_model_identity": model.identity,
        "noise_kind": "test",
        "max_cuda_peak_allocated_bytes": 1024,
        "executed_trajectories": 8,
    }
    return {
        "schema": NOISE_SELECTOR_CALIBRATION_SCHEMA,
        "device_name": "test accelerator",
        "torch_version": torch.__version__,
        "trajectory_batch_size": 4,
        "requested_trajectories": 8,
        "world_size": 1,
        "records": [
            {
                **common,
                "mode": "density_matrix",
                "median_seconds": density_seconds,
                "executed_trajectories": None,
            },
            {
                **common,
                "mode": "noisy_statevector",
                "median_seconds": statevector_seconds,
            },
            {**common, "mode": "noisy_mps", "median_seconds": mps_seconds},
        ],
    }


def test_noise_selector_uses_only_exact_device_calibration_matches():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))
    calibration = _selector_calibration(
        circuit, model, statevector_seconds=10.0, mps_seconds=1.0
    )

    calibrated = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
        calibration=calibration,
    )
    fallback = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=2,
        calibration=calibration,
    )
    distributed_fallback = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=8,
        trajectory_batch_size=4,
        world_size=2,
        calibration=calibration,
    )

    assert calibrated.selected_mode == "noisy_mps"
    assert calibrated.selection_basis == "exact_device_calibration"
    assert calibrated.calibration_device == "test accelerator"
    assert fallback.selected_mode == "noisy_statevector"
    assert fallback.selection_basis == "analytic_policy"
    assert distributed_fallback.selection_basis == "analytic_policy"


def test_run_native_consumes_selector_calibration_option():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))
    calibration = _selector_calibration(
        circuit, model, statevector_seconds=1.0, mps_seconds=10.0
    )

    result, plan = fqb.run_native(
        circuit,
        noise_model=model,
        trajectories=8,
        trajectory_batch_size=4,
        noise_performance_calibration=calibration,
        return_plan=True,
    )

    assert plan.state_mode == "statevector"
    assert result.statistics.count == 8


def test_noise_selector_rejects_unknown_calibration_schema():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(ValueError, match="unsupported noise selector calibration"):
        plan_noise_execution_selection(
            circuit,
            model,
            trajectories=8,
            calibration={"schema": "future", "records": []},
        )


def test_noise_selector_reports_time_to_target_trajectory_evidence():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=128,
        min_trajectories=16,
        target_standard_error=0.1,
    )
    insufficient = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=32,
        target_standard_error=0.1,
    )

    assert selection.estimated_trajectories_to_target == 100
    assert selection.target_feasible_within_cap is True
    assert insufficient.target_feasible_within_cap is False
    assert (
        selection.selected_candidate.metadata["trajectory_estimate_basis"]
        == "bounded_pauli_variance_le_one"
    )


def test_noise_selector_target_error_requires_trajectory_ceiling():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(ValueError, match="explicit trajectory ceiling"):
        plan_noise_execution_selection(
            circuit,
            model,
            target_standard_error=0.1,
        )


def test_noise_selector_uses_reproducible_pilot_variance_evidence():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=128,
        min_trajectories=16,
        target_standard_error=0.1,
        pilot_variance=0.25,
        pilot_trajectories=32,
    )

    assert selection.estimated_trajectories_to_target == 25
    assert selection.trajectory_variance_estimate == 0.25
    assert selection.pilot_trajectories == 32
    assert selection.selected_candidate.metadata["trajectory_estimate_basis"] == (
        "pilot_variance"
    )


def test_noise_selector_uses_simultaneous_pilot_variance_upper_bound():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=4096,
        min_trajectories=32,
        target_standard_error=0.02,
        pilot_variance=0.0481456369,
        pilot_trajectories=32,
        pilot_confidence_level=0.95,
        pilot_observable_count=8,
    )

    assert selection.pilot_sample_variance == pytest.approx(0.0481456369)
    assert selection.trajectory_variance_estimate == 1.0
    assert selection.estimated_trajectories_to_target == 2500
    assert selection.pilot_confidence_level == 0.95
    assert selection.pilot_observable_count == 8


def test_target_error_cost_can_select_faster_exact_density_backend():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))
    calibration = _selector_calibration(
        circuit,
        model,
        statevector_seconds=10.0,
        mps_seconds=8.0,
        density_seconds=2.0,
    )

    selection = plan_noise_execution_selection(
        circuit,
        model,
        trajectories=128,
        trajectory_batch_size=4,
        target_standard_error=0.1,
        calibration=calibration,
    )

    assert selection.selected_mode == "density_matrix"
    assert selection.selection_basis == "exact_device_calibration"
    assert selection.selected_candidate.sampling_error is False


@pytest.mark.parametrize(
    ("pilot_variance", "pilot_trajectories", "message"),
    ((-0.1, 32, "finite and non-negative"), (0.2, 1, "at least two")),
)
def test_noise_selector_validates_pilot_variance_evidence(
    pilot_variance, pilot_trajectories, message
):
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(ValueError, match=message):
        plan_noise_execution_selection(
            circuit,
            model,
            trajectories=128,
            target_standard_error=0.1,
            pilot_variance=pilot_variance,
            pilot_trajectories=pilot_trajectories,
        )


def test_noise_selector_validates_pilot_confidence_evidence():
    circuit = fq.Circuit(2).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(ValueError, match="between zero and one"):
        plan_noise_execution_selection(
            circuit,
            model,
            trajectories=128,
            target_standard_error=0.1,
            pilot_variance=0.2,
            pilot_trajectories=32,
            pilot_confidence_level=1.0,
        )
