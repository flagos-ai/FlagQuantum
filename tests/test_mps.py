"""Tests for native FlagQuantum MPS execution."""

import torch

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.noise as fqn
import flagquantum.simulation.mps.models as mps_models
from flagquantum.noise import amplitude_damping_channel, bit_flip_channel
from flagquantum.runtime.executors.mps.compiled_training import (
    MPSTrainingStep,
    compile_mps_training_step,
)
from flagquantum.runtime.planner import estimate_mps_bytes
from flagquantum.simulation.density_matrix import expectation_z_density
from flagquantum.simulation.mps.entrypoints import (
    run_mps_adaptive,
    run_noisy_mps_trajectory,
)
from flagquantum.simulation.mps.models import (
    MPSAdaptiveBondPlan,
    MPSAdaptiveRunResult,
    MPSBondProfile,
    MPSLocalRefinementPlan,
    MPSTruncationRecord,
)


def test_run_mps_wrapper_delegates_local_numerics(monkeypatch):
    import flagquantum.simulation.mps.entrypoints as mps_entrypoints

    expected = object()
    calls = []

    def replacement(ir, state, **options):
        calls.append((ir, state, options))
        return expected

    monkeypatch.delenv("FQ_MPS_SPATIAL_BUCKET", raising=False)
    monkeypatch.setattr(mps_entrypoints, "run_local_mps", replacement)

    assert fqb.run_mps(fq.Circuit(2).h(0)) is expected
    assert len(calls) == 1
    assert isinstance(calls[0][0], fq.CircuitIR)
    assert calls[0][1].n_wires == 2
    assert calls[0][2]["fuse_single_qubit"] is True
    assert calls[0][2]["spatial_bucket"] is True


def test_mps_bell_state_matches_statevector():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    mps, plan = fqb.run_native(circuit, mode="mps", return_plan=True)

    assert plan.state_mode == "mps"
    assert plan.recommended_mode == "mps"
    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)
    assert torch.allclose(mps.expectation_z(), circuit.expectation_z(), atol=1e-6)
    assert plan.state_bytes == estimate_mps_bytes(2, max_bond=2)
    assert mps.summary()["state_mode"] == "mps"
    assert mps.max_bond == 2
    assert mps.left_canonical_residual() < 1e-6
    assert isinstance(mps.bond_profile(), MPSBondProfile)
    assert mps.summary()["parameter_count"] == mps.parameter_count
    assert abs(mps.summary()["state_norm_min"] - 1.0) < 1e-6


def test_mps_instruction_schedule_is_reused_for_dynamic_parameters():
    mps_models._MPS_INSTRUCTION_SCHEDULE_CACHE.clear()
    theta = torch.tensor(0.2, requires_grad=True)
    first = fqb.run_mps(fq.Circuit(2).ry(0, theta).rz(0, theta).cx(0, 1))
    first.expectation_z(0).sum().backward()
    cache_size = len(mps_models._MPS_INSTRUCTION_SCHEDULE_CACHE)

    phi = torch.tensor(-0.3, requires_grad=True)
    second = fqb.run_mps(fq.Circuit(2).ry(0, phi).rz(0, phi).cx(0, 1))
    second.expectation_z(0).sum().backward()

    assert cache_size == 1
    assert len(mps_models._MPS_INSTRUCTION_SCHEDULE_CACHE) == cache_size
    assert theta.grad is not None and phi.grad is not None


def test_mps_program_is_shape_and_truncation_specialized():
    circuit = fq.Circuit(3).ry(0, 0.2).cx(0, 1).cx(0, 2)

    fqb.run_mps(circuit, max_bond=2, cutoff=0.0)
    first_program = next(
        value for key, value in circuit._backend_programs.items() if key[0] == "mps"
    )
    fqb.run_mps(circuit, max_bond=4, cutoff=1e-6)
    programs = [
        value for key, value in circuit._backend_programs.items() if key[0] == "mps"
    ]

    assert isinstance(first_program, mps_models.CompiledMPSProgram)
    assert len(programs) == 2
    assert programs[0].signature != programs[1].signature
    assert {operation.kind for operation in first_program.operations} >= {
        "one",
        "adjacent_two",
        "remote_two",
    }


def test_mps_parameterized_adjacent_two_qubit_gate():
    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.31).rxx(0, 1, theta=0.42).rz(1, theta=-0.2)

    mps = fqb.run_mps(circuit)

    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)


def test_mps_spatial_two_site_bucket_is_default_and_matches_sequential(monkeypatch):
    circuit = fq.Circuit(4).h(0).h(2).rxx(0, 1, theta=0.31).rzz(2, 3, theta=-0.27)

    monkeypatch.delenv("FQ_MPS_SPATIAL_BUCKET", raising=False)
    bucketed = fqb.run_mps(circuit)
    monkeypatch.setenv("FQ_MPS_SPATIAL_BUCKET", "0")
    sequential = fqb.run_mps(circuit)

    torch.testing.assert_close(
        bucketed.to_statevector(), sequential.to_statevector(), atol=1e-6, rtol=1e-6
    )
    assert bucketed.summary()["spatial_two_site_bucket_count"] == 1
    assert bucketed.summary()["spatial_two_site_bucketed_gate_count"] == 2
    assert bucketed.summary()["triton_mps_two_site_enabled"] is False
    assert sequential.summary()["spatial_two_site_bucket_count"] == 0


def test_mps_spatial_two_site_bucket_preserves_parameter_gradients(monkeypatch):
    theta = torch.tensor(0.31, requires_grad=True)
    phi = torch.tensor(-0.27, requires_grad=True)
    reference_theta = theta.detach().clone().requires_grad_(True)
    reference_phi = phi.detach().clone().requires_grad_(True)
    circuit = fq.Circuit(4).rxx(0, 1, theta).rzz(2, 3, phi)
    reference = fq.Circuit(4).rxx(0, 1, reference_theta).rzz(2, 3, reference_phi)

    monkeypatch.delenv("FQ_MPS_SPATIAL_BUCKET", raising=False)
    loss = fqb.run_mps(circuit).expectation_z_sum().sum()
    reference_loss = reference.expectation_z().sum()
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(loss, reference_loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(theta.grad, reference_theta.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(phi.grad, reference_phi.grad, atol=2e-6, rtol=2e-6)


def test_mps_complex128_two_site_gate_preserves_dtype_and_gradient():
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    reference_theta = theta.detach().clone().requires_grad_(True)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.ry(0, theta).rxx(0, 1, theta).rzz(1, 2, theta)
    reference = fq.Circuit(3, dtype=torch.complex128)
    reference.ry(0, reference_theta).rxx(0, 1, reference_theta).rzz(
        1, 2, reference_theta
    )

    mps_loss = fqb.run_mps(circuit).expectation_ps(z=(0, 2)).mean()
    dense_loss = reference.expectation_ps(z=(0, 2)).mean()
    mps_loss.backward()
    dense_loss.backward()

    assert fqb.run_mps(circuit).dtype == torch.complex128
    torch.testing.assert_close(mps_loss, dense_loss, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(theta.grad, reference_theta.grad, atol=1e-11, rtol=1e-11)


def test_mps_complex_two_site_autograd_matches_statevector():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=True)
    ref_params = params.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.h(0)
        circuit.rx(1, theta=values[0])
        circuit.ry(2, theta=values[1])
        circuit.rz(3, theta=values[2])
        circuit.cx(0, 3)
        circuit.rzz(1, 2, theta=values[3])
        return circuit

    mps_loss = fqb.run_mps(build(params), max_bond=8).expectation_z((0, 2)).sum()
    ref_loss = build(ref_params).expectation_z((0, 2)).sum()
    mps_loss.backward()
    ref_loss.backward()

    assert params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(mps_loss.detach(), ref_loss.detach(), atol=1e-6)
    assert torch.allclose(params.grad, ref_params.grad, atol=1e-5)


def test_mps_expectation_z_sum_matches_vectorized_z_gradient():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19, 0.11, -0.07], requires_grad=True)
    ref_params = params.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(4)
        for wire in range(4):
            circuit.rx(wire, theta=values[wire % values.numel()])
            circuit.ry(wire, theta=values[(wire + 1) % values.numel()])
        circuit.cx(0, 1).cx(1, 2).cx(2, 3)
        return circuit

    loss = fqb.run_mps(build(params), max_bond=8).expectation_z_sum().sum()
    ref_loss = fqb.run_mps(build(ref_params), max_bond=8).expectation_z().sum()
    loss.backward()
    ref_loss.backward()

    assert params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-6)
    assert torch.allclose(params.grad, ref_params.grad, atol=1e-5)


def test_mps_expectation_z_sum_preserves_wire_multiplicity_and_empty_inputs():
    circuit = fq.Circuit(3)
    circuit.h(0).ry(1, theta=0.2).cx(0, 2)
    mps = fqb.run_mps(circuit)

    repeated = mps.expectation_z_sum((0, 0, 2))
    vectorized = mps.expectation_z((0, 0, 2)).sum()
    empty = mps.expectation_z_sum(())

    assert torch.allclose(repeated, vectorized, atol=1e-6)
    assert torch.allclose(empty, torch.zeros_like(empty))


def test_compile_mps_training_step_allows_parameter_updates_and_matches_mps():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(3)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 1).cx(1, 2)
        circuit.ry(0, theta=values[3])
        return circuit

    step = compile_mps_training_step(
        build,
        params,
        compile=False,
        max_bond=8,
        dense_observable_wires=8,
    )
    loss, grad = step(params)

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = (
        fqb.run_mps(build(ref_params), max_bond=8, dense_observable_wires=8)
        .expectation_z_sum()
        .sum()
    )
    ref_loss.backward()

    shifted = params + 0.07
    shifted_loss, shifted_grad = step(shifted)

    assert isinstance(step, MPSTrainingStep)
    assert torch.allclose(loss, ref_loss.detach(), atol=1e-6)
    assert ref_params.grad is not None
    assert torch.allclose(grad, ref_params.grad, atol=1e-5)
    assert not torch.allclose(shifted_loss, loss)
    assert shifted_grad.shape == params.shape


def test_compile_mps_training_step_compile_failure_falls_back(monkeypatch):
    params = torch.tensor([0.2, -0.1])

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0]).ry(1, theta=values[1]).cx(0, 1)
        return circuit

    def failing_compile(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", failing_compile, raising=False)
    step = compile_mps_training_step(build, params, compile=True, max_bond=4)
    loss, grad = step(params)

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(grad))
    assert step.summary()["status"] == "compile_failed_fallback"
    assert "compiler unavailable" in str(step.summary()["compile_error"])


def test_mps_non_adjacent_gate_uses_local_swaps():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).rz(1, theta=0.5)

    mps = fqb.run_native(circuit, mode="mps")

    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)
    assert mps.summary()["local_swap_count"] == 2


def test_mps_non_adjacent_reverse_wire_order():
    circuit = fq.Circuit(3)
    circuit.h(2).cx(2, 0).ry(1, theta=0.2)

    mps = fqb.run_mps(circuit)

    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)
    assert mps.summary()["local_swap_count"] == 2


def test_mps_parameterized_remote_two_qubit_gate():
    circuit = fq.Circuit(4)
    circuit.h(0).ry(3, theta=0.4).rxx(0, 3, theta=-0.35).rz(2, theta=0.2)

    mps = fqb.run_mps(circuit)

    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)
    assert mps.summary()["local_swap_count"] == 4


def test_mps_from_initial_state():
    initial = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.complex64)
    circuit = fq.Circuit(2, inputs=initial)
    circuit.cx(0, 1)

    mps = fqb.run_mps(circuit)

    assert torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)


def test_mps_pauli_string_expectation_matches_statevector():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    mps = fqb.run_mps(circuit)

    assert torch.allclose(
        mps.expectation_ps(z=[0, 1]), circuit.expectation_ps(z=[0, 1])
    )
    assert torch.allclose(
        mps.expectation_ps(x=[0, 1]), circuit.expectation_ps(x=[0, 1])
    )
    assert torch.allclose(
        mps.expectation_ps(y=[0, 1]), circuit.expectation_ps(y=[0, 1])
    )


def test_mps_expectations_use_transfer_contraction():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    mps = fqb.run_mps(circuit)

    def blocked_statevector():
        raise AssertionError("expectations should not materialize statevector")

    mps.to_statevector = blocked_statevector

    assert torch.allclose(mps.expectation_z(), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(mps.expectation_ps(x=[0, 1]), torch.ones(1), atol=1e-6)


def test_mps_sampling_and_counts_match_basis_state():
    generator = torch.Generator().manual_seed(1234)
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)

    mps = fqb.run_mps(circuit)
    samples = mps.sample(8, generator=generator)
    counts = mps.counts(8, generator=torch.Generator().manual_seed(1234))

    assert samples.shape == (1, 8, 3)
    assert torch.all(samples[..., 0] == 1)
    assert torch.all(samples[..., 1] == 0)
    assert torch.all(samples[..., 2] == 1)
    assert counts == [{"101": 8}]


def test_mps_sampling_uses_conditional_contraction():
    circuit = fq.Circuit(3)
    circuit.x(0).x(2)
    mps = fqb.run_mps(circuit)

    def blocked_dense():
        raise AssertionError("sampling should not materialize dense probabilities")

    mps.to_statevector = blocked_dense
    mps.probabilities = blocked_dense

    samples = mps.sample(4, generator=torch.Generator().manual_seed(7))
    counts = mps.counts(4, generator=torch.Generator().manual_seed(7))

    assert torch.all(samples == torch.tensor([[[1, 0, 1]] * 4]))
    assert counts == [{"101": 4}]


def test_mps_counts_int_format():
    circuit = fq.Circuit(2)
    circuit.x(1)

    mps = fqb.run_native(circuit, mode="mps")

    assert mps.counts(4, generator=torch.Generator().manual_seed(1), format="int") == [
        {1: 4}
    ]


def test_mps_truncation_error_is_reported():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    mps = fqb.run_mps(circuit, max_bond=1)
    summary = mps.summary()

    assert mps.max_bond == 1
    assert summary["truncation_steps"] >= 1
    assert summary["truncation_error"] > 0
    assert summary["truncation_by_bond"]
    assert not torch.allclose(mps.to_statevector(), circuit.state(), atol=1e-6)


def test_mps_truncation_records_are_reported_per_bond():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    mps = fqb.run_mps(circuit, max_bond=1)
    profile = mps.bond_profile()

    assert profile.truncation_records
    assert all(
        isinstance(record, MPSTruncationRecord) for record in profile.truncation_records
    )
    assert all(record.kept_rank <= 1 for record in profile.truncation_records)
    assert profile.truncation_by_bond == tuple(
        sorted(mps.truncation_error_by_bond().items())
    )
    assert (
        sum(value for _, value in profile.truncation_by_bond)
        == profile.truncation_error
    )
    assert profile.summary()["truncation_records"][0]["source"] in {
        "from_statevector",
        "two_site",
    }


def test_mps_adaptive_bond_plan_uses_truncation_hotspots():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    mps = fqb.run_mps(circuit, max_bond=1)
    plan = mps.adaptive_bond_plan(global_error_budget=0.0, growth_factor=2.0)

    assert isinstance(plan, MPSAdaptiveBondPlan)
    assert not plan.budget_satisfied
    assert plan.observed_error == mps.summary()["truncation_error"]
    assert plan.hot_bonds
    assert plan.suggested_max_bond > plan.current_max_bond
    assert all(
        bond in dict(mps.summary()["truncation_by_bond"]) for bond in plan.hot_bonds
    )
    assert mps.summary()["adaptive_suggested_max_bond"] >= plan.current_max_bond
    assert not mps.truncation_error_within_budget(0.0)


def test_mps_local_refinement_plan_maps_hot_bonds_to_windows():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    mps = fqb.run_mps(circuit, max_bond=1)
    plan = mps.local_refinement_plan(global_error_budget=0.0, window_radius=0)

    assert isinstance(plan, MPSLocalRefinementPlan)
    assert plan.hot_bonds
    assert plan.windows
    assert all(
        any(left <= bond < right for left, right in plan.windows)
        for bond in plan.hot_bonds
    )
    assert mps.summary()["local_refinement_windows"] == plan.windows


def test_run_mps_adaptive_reruns_with_suggested_bond():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    result = run_mps_adaptive(
        circuit,
        initial_max_bond=1,
        global_error_budget=0.0,
        max_bond_cap=4,
    )

    assert isinstance(result, MPSAdaptiveRunResult)
    assert result.rerun
    assert result.initial_plan.hot_bonds
    assert result.final_plan.current_max_bond > result.initial_plan.current_max_bond
    assert result.summary()["state_mode"] == "adaptive_mps"
    assert result.summary()["rerun"] is True
    assert result.summary()["refinement_plan"]["windows"]
    assert torch.allclose(
        result.to_statevector(),
        fqb.run_native(circuit, mode="mps", max_bond=4).to_statevector(),
        atol=1e-6,
    )


def test_run_mps_adaptive_skips_rerun_when_budget_is_satisfied():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = run_mps_adaptive(
        circuit,
        initial_max_bond=2,
        global_error_budget=0.0,
    )

    assert not result.rerun
    assert result.initial_state is result.state
    assert result.summary()["final_plan"]["budget_satisfied"]
    assert torch.allclose(result.to_statevector(), circuit.state(), atol=1e-6)


def test_circuit_run_accepts_adaptive_mps_mode():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    result, plan = fqb.run_native(
        circuit,
        mode="adaptive_mps",
        initial_max_bond=1,
        global_error_budget=0.0,
        max_bond_cap=4,
        return_plan=True,
    )

    assert isinstance(result, MPSAdaptiveRunResult)
    assert result.summary()["state_mode"] == "adaptive_mps"
    assert plan.state_mode == "mps"
    assert torch.allclose(
        result.to_statevector(),
        fqb.run_mps(circuit, max_bond=4).to_statevector(),
        atol=1e-6,
    )


def test_mps_canonicalize_rebuilds_left_canonical_form():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.25)

    mps = fqb.run_mps(circuit)
    before = mps.to_statevector()
    mps.canonicalize()

    assert torch.allclose(mps.to_statevector(), before, atol=1e-6)
    assert mps.left_canonical_residual() < 1e-6


def test_mps_orthogonalize_left_preserves_state_and_profiles():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.25).rzz(2, 3, theta=-0.31)

    mps = fqb.run_mps(circuit)
    mps.apply_one(torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=mps.dtype), 0)
    before = mps.to_statevector()

    mps.orthogonalize_left()
    profile = mps.bond_profile()

    assert isinstance(profile, MPSBondProfile)
    assert torch.allclose(mps.to_statevector(), before, atol=1e-6)
    assert profile.left_canonical_residual < 1e-6
    assert profile.parameter_count == mps.parameter_count
    assert profile.state_norm_min > 0.999
    assert profile.state_norm_max < 1.001


def test_mps_orthogonalize_right_preserves_state_and_profiles():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.25).rzz(2, 3, theta=-0.31)

    mps = fqb.run_mps(circuit)
    before = mps.to_statevector()

    mps.orthogonalize_right()
    profile = mps.bond_profile()

    assert torch.allclose(mps.to_statevector(), before, atol=1e-6)
    assert mps.orthogonality_center == 0
    assert profile.right_canonical_residual < 1e-6
    assert profile.mixed_canonical_residual < 1e-6


def test_mps_move_orthogonality_center_to_middle():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.25).rzz(2, 3, theta=-0.31).cx(1, 3)

    mps = fqb.run_mps(circuit)
    before = mps.to_statevector()

    mps.move_orthogonality_center(2)
    profile = mps.bond_profile()

    assert torch.allclose(mps.to_statevector(), before, atol=1e-6)
    assert mps.orthogonality_center == 2
    assert profile.orthogonality_center == 2
    assert profile.mixed_canonical_residual < 1e-6
    assert mps.summary()["orthogonality_center"] == 2


def test_noisy_mps_trajectory_bit_flip_matches_density_path():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    mps = run_noisy_mps_trajectory(circuit, model)
    rho = fqn.noisy_density_matrix(circuit, model)

    assert torch.allclose(
        mps.expectation_z(0), expectation_z_density(rho, 0), atol=1e-6
    )
    assert torch.allclose(mps.expectation_z(0), torch.ones(1, 1), atol=1e-6)


def test_noisy_mps_wrapper_passes_lowered_ir_and_explicit_rng(monkeypatch):
    import flagquantum.simulation.mps.entrypoints as mps_entrypoints

    expected = object()
    calls = []

    def replacement(lowered_ir, state, *, generator):
        calls.append((lowered_ir, state, generator))
        return expected

    monkeypatch.setattr(mps_entrypoints, "run_local_noisy_mps_trajectory", replacement)
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(0.25))
    generator = torch.Generator().manual_seed(11)

    assert run_noisy_mps_trajectory(circuit, model, generator=generator) is expected
    assert isinstance(calls[0][0], fq.CircuitIR)
    assert any(item.metadata.get("is_channel") for item in calls[0][0])
    assert calls[0][1].n_wires == 1
    assert calls[0][2] is generator


def test_noisy_mps_trajectory_amplitude_damping():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", amplitude_damping_channel(1.0))

    mps, plan = fqb.run_native(
        circuit,
        noise_model=model,
        mode="mps_trajectory",
        return_plan=True,
    )

    assert plan.state_mode == "mps"
    assert plan.noisy_execution_plan is not None
    assert plan.noisy_execution_plan.representation == "mps"
    assert plan.noisy_execution_plan.evolution == "quantum_trajectory"
    assert plan.noisy_execution_plan.trajectory.count == 1
    assert torch.allclose(mps.expectation_z(0), torch.ones(1, 1), atol=1e-6)


def test_noisy_mps_monte_carlo_aggregates_deterministic_channel():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", bit_flip_channel(1.0))

    result = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=4,
        generator=torch.Generator().manual_seed(3),
    )

    assert result.n_trajectories == 4
    assert len(result.trajectories) == 4
    assert torch.allclose(result.expectation_z_mean, torch.ones(1, 1), atol=1e-6)
    assert torch.allclose(result.expectation_z_variance, torch.zeros(1, 1), atol=1e-6)
    assert result.summary()["state_mode"] == "mps_trajectory"


def test_run_native_noisy_mps_mode():
    circuit = fq.Circuit(1)
    circuit.x(0)
    model = fqn.NoiseModel().add("x", amplitude_damping_channel(1.0))

    result, plan = fqb.run_native(
        circuit,
        noise_model=model,
        mode="noisy_mps",
        trajectories=3,
        return_plan=True,
    )

    assert plan.state_mode == "mps"
    assert plan.noisy_execution_plan is not None
    assert plan.noisy_execution_plan.trajectory.count == 3
    assert result.n_trajectories == 3
    assert torch.allclose(result.expectation_z_mean, torch.ones(1, 1), atol=1e-6)
