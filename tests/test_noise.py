"""Tests for native FlagQuantum noise and density-matrix execution."""

import torch

import flagquantum as fq


def test_density_matrix_from_native_circuit():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    rho = circuit.density_matrix()

    assert rho.shape == (1, 4, 4)
    assert torch.allclose(
        torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)), torch.ones(1)
    )
    assert torch.allclose(fq.expectation_z_density(rho), torch.zeros(1, 2), atol=1e-6)


def test_bit_flip_channel_on_density_matrix():
    circuit = fq.Circuit(1)
    rho = fq.apply_kraus_density(
        circuit.density_matrix(),
        fq.bit_flip_channel(1.0),
        (0,),
        circuit.n_wires,
    )

    assert torch.allclose(
        fq.expectation_z_density(rho, 0), -torch.ones(1, 1), atol=1e-6
    )


def test_noisy_density_matrix_gate_noise_model():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    rho = circuit.noisy_density_matrix(model)

    assert torch.allclose(fq.expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_amplitude_damping_channel():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.amplitude_damping_channel(1.0))

    rho = fq.noisy_density_matrix(circuit, model)

    assert torch.allclose(fq.expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_density_matrix_matches_statevector_expectation():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1).rz(1, theta=0.3)

    rho = fq.noisy_density_matrix(circuit)

    assert torch.allclose(
        fq.expectation_z_density(rho),
        circuit.expectation_z(),
        atol=1e-6,
    )


def test_noise_model_lowers_to_unified_ir_and_planner():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    lowered = fq.lower_noise_model(circuit, model)
    plan = fq.plan(circuit, noise_model=model)

    assert [inst.name for inst in lowered] == ["x", "bit_flip"]
    assert lowered.instructions[1].metadata["is_channel"] is True
    assert plan.analysis.channel_count == 1
    assert plan.analysis.has_noise is True
    assert plan.state_mode == "density_matrix"
    assert plan.recommended_mode == "density"
    assert plan.state_bytes == fq.estimate_density_bytes(1)


def test_circuit_run_uses_density_matrix_for_noise():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    rho, plan = circuit.run(noise_model=model, return_plan=True)
    rho_from_ir = fq.density_matrix_from_ir(fq.lower_noise_model(circuit, model))

    assert torch.allclose(rho, rho_from_ir)
    assert plan.state_mode == "density_matrix"
    assert torch.allclose(fq.expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_run_native_statevector_path():
    circuit = fq.Circuit(1)
    circuit.h(0)

    state, plan = fq.run_native(circuit, return_plan=True)

    assert state.shape == (1, 2)
    assert plan.state_mode == "statevector"
    assert torch.allclose(state.abs() ** 2, torch.full((1, 2), 0.5), atol=1e-6)


def test_auto_mode_selects_density_matrix_for_noise():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    rho, plan = fq.run_native(circuit, noise_model=model, return_plan=True)

    assert plan.state_mode == "density_matrix"
    assert fq.select_execution_mode(circuit, noise_model=model) == "density_matrix"
    assert torch.allclose(fq.expectation_z_density(rho, 0), torch.ones(1, 1), atol=1e-6)


def test_auto_mode_selects_noisy_mps_when_trajectory_controls_are_set():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    result, plan = fq.run_native(
        circuit,
        noise_model=model,
        trajectories=4,
        return_plan=True,
    )

    assert (
        fq.select_execution_mode(circuit, noise_model=model, trajectories=4)
        == "noisy_mps"
    )
    assert plan.state_mode == "mps"
    assert plan.recommended_mode == "noisy_mps"
    assert result.n_trajectories == 4
    assert torch.allclose(result.expectation_z_mean, torch.ones(1, 1), atol=1e-6)


def test_auto_mode_selects_noisy_mps_when_density_memory_is_limited():
    circuit = fq.Circuit(2)
    circuit.x(0)
    model = fq.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    result, plan = fq.run_native(
        circuit,
        noise_model=model,
        memory_limit_bytes=1,
        trajectories=2,
        return_plan=True,
    )

    assert (
        fq.select_execution_mode(
            circuit,
            noise_model=model,
            memory_limit_bytes=1,
        )
        == "noisy_mps"
    )
    assert plan.state_mode == "mps"
    assert result.n_trajectories == 2
