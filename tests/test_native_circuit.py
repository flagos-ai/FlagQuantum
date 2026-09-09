"""Tests for the native FlagQuantum circuit core."""

import os
import socket
from collections import Counter
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.compiler as compiler
import flagquantum.runtime as fqr
import flagquantum.runtime.planner as fqxp
import flagquantum.simulation.mps as fqmps
from flagquantum.compiler import CouplingMap
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.gradients import (
    batched_parameter_shift_gradient,
    parameter_shift_gradient,
)
from flagquantum.runtime.audit import audit_distributed_scalability
from flagquantum.runtime.backend_registry import get_backend_capabilities
from flagquantum.runtime.configuration import get_backend, set_backend
from flagquantum.runtime.distributed import (
    destroy_torch_distributed,
)
from flagquantum.runtime.executors.mps.distributed_state import (
    DistributedBoundaryProtocol,
    DistributedBoundarySync,
    DistributedMPSState,
    DistributedShardPlan,
    ShardedMPSState,
)
from flagquantum.runtime.executors.tensor_network.sliced_tasks import (
    DistributedTNSliceTask,
)
from flagquantum.runtime.executors.tensor_network.state import (
    DistributedTensorNetworkState,
)
from flagquantum.runtime.planner import estimate_state_bytes, select_execution_mode
from flagquantum.simulation.matrices import GATE_MAT_DICT
from flagquantum.simulation.mps.state import MPSState
from flagquantum.simulation.statevector.operations import (
    _apply_matrix,
    _compose_gate_matrices,
    _gate_matrix,
)
from flagquantum.simulation.tensor_network.entrypoints import build_tensor_network

pytestmark = pytest.mark.integration


def _free_tcp_init_method():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return f"tcp://127.0.0.1:{port}"


def _skip_inside_outer_torchrun():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        pytest.skip("single-rank process-group test must not run inside outer torchrun")


def test_native_bell_state():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    state = circuit.state()
    expected = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=state.dtype)

    assert torch.allclose(state, expected, atol=1e-6)
    assert torch.allclose(circuit.expectation_z(), torch.zeros(1, 2), atol=1e-6)


def test_simulation_gate_composition_follows_execution_order():
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    torch.testing.assert_close(_compose_gate_matrices((x, z)), z @ x)


def test_circuit_prefers_qubit_count_and_preserves_wire_aliases():
    circuit = fq.Circuit(n_qubits=3)

    assert circuit.n_qubits == 3
    assert circuit.num_qubits == 3
    assert circuit.n_wires == 3
    assert fq.Circuit(n_wires=3).n_qubits == 3
    assert fq.Circuit(nqubits=3).n_qubits == 3
    assert fq.Circuit(n_qubits=3, n_wires=3, nqubits=3).n_qubits == 3


def test_circuit_rejects_conflicting_qubit_count_aliases():
    with pytest.raises(ValueError, match="conflicting qubit counts"):
        fq.Circuit(n_qubits=2, n_wires=3)


def test_native_parameter_gradient():
    theta = torch.tensor(0.3, requires_grad=True)

    circuit = fq.Circuit(1)
    circuit.rx(0, theta=theta)
    loss = circuit.expectation_z(0).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


def test_diagonal_statevector_gates_avoid_dense_bmm_and_preserve_gradients(monkeypatch):
    import flagquantum.simulation.statevector.local as statevector_runtime

    diagonal_gates = statevector_runtime._DIAGONAL_STATEVECTOR_GATES
    reference_theta = torch.tensor(0.31, requires_grad=True)
    theta = torch.tensor(0.31, requires_grad=True)
    initial = torch.tensor([[0.5, 0.5j, -0.5j, 0.5]], dtype=torch.complex64)
    reference = fq.Circuit(2, inputs=initial)
    reference.rz(0, reference_theta).cphase(0, 1, reference_theta * 0.5).rzz(
        1, 0, reference_theta * 0.25
    )
    circuit = fq.Circuit(2, inputs=initial)
    circuit.rz(0, theta).cphase(0, 1, theta * 0.5).rzz(1, 0, theta * 0.25)

    monkeypatch.setattr(statevector_runtime, "_DIAGONAL_STATEVECTOR_GATES", frozenset())
    reference_state = reference.state()
    reference_loss = reference_state.real.sum()
    reference_loss.backward()
    monkeypatch.setattr(
        statevector_runtime, "_DIAGONAL_STATEVECTOR_GATES", diagonal_gates
    )

    def unexpected_bmm(*args, **kwargs):
        raise AssertionError("known diagonal gates must not launch torch.bmm")

    monkeypatch.setattr(statevector_runtime.torch, "bmm", unexpected_bmm)
    state = circuit.state()
    loss = state.real.sum()
    loss.backward()

    assert torch.allclose(state, reference_state, atol=1e-6)
    assert theta.grad is not None
    assert reference_theta.grad is not None
    assert torch.allclose(theta.grad, reference_theta.grad, atol=1e-6)
    assert circuit._last_statevector_runtime["diagonal_elementwise_gates"] == 3


@pytest.mark.parametrize("wires", [(0, 3), (3, 0), (1, 2), (2, 1)])
def test_cx_statevector_uses_permutation_without_dense_bmm(monkeypatch, wires):
    import flagquantum.circuit as circuit_runtime

    generator = torch.Generator().manual_seed(903)
    initial = torch.randn(2, 16, dtype=torch.complex64, generator=generator)
    initial = initial / torch.linalg.vector_norm(initial, dim=-1, keepdim=True)
    reference = _apply_matrix(
        initial,
        GATE_MAT_DICT["cx"],
        wires,
        4,
    )
    circuit = fq.Circuit(4, inputs=initial).cx(*wires)

    def unexpected_bmm(*args, **kwargs):
        raise AssertionError("CNOT must use the permutation path")

    monkeypatch.setattr(circuit_runtime.torch, "bmm", unexpected_bmm)
    actual = circuit.state()

    assert torch.equal(actual, reference)
    assert circuit._last_statevector_runtime["permutation_gates"] == 1


@pytest.mark.parametrize(
    ("name", "wires"),
    [("x", (0,)), ("x", (3,)), ("swap", (0, 3)), ("swap", (2, 1))],
)
def test_fixed_basis_permutations_avoid_dense_bmm(monkeypatch, name, wires):
    import flagquantum.circuit as circuit_runtime

    generator = torch.Generator().manual_seed(904)
    initial = torch.randn(2, 16, dtype=torch.complex64, generator=generator)
    reference = _apply_matrix(initial, GATE_MAT_DICT[name], wires, 4)
    circuit = fq.Circuit(4, inputs=initial)
    getattr(circuit, name)(*wires)

    def unexpected_bmm(*args, **kwargs):
        raise AssertionError(f"{name} must use the permutation path")

    monkeypatch.setattr(circuit_runtime.torch, "bmm", unexpected_bmm)
    actual = circuit.state()

    assert torch.equal(actual, reference)
    assert circuit._last_statevector_runtime["permutation_gates"] == 1


@pytest.mark.parametrize("wire", [0, 1, 2, 3])
def test_fixed_y_specialization_avoids_dense_bmm(monkeypatch, wire):
    import flagquantum.circuit as circuit_runtime

    name = "y"
    generator = torch.Generator().manual_seed(905)
    initial = torch.randn(2, 16, dtype=torch.complex64, generator=generator)
    reference = _apply_matrix(initial, GATE_MAT_DICT[name], (wire,), 4)
    circuit = fq.Circuit(4, inputs=initial)
    getattr(circuit, name)(wire)

    def unexpected_bmm(*args, **kwargs):
        raise AssertionError("Y must use the fixed specialization path")

    monkeypatch.setattr(circuit_runtime.torch, "bmm", unexpected_bmm)
    actual = circuit.state()

    assert torch.allclose(actual, reference, atol=1e-7, rtol=1e-7)
    assert (
        circuit._last_statevector_runtime["fixed_single_qubit_specialized_gates"] == 1
    )


def test_same_wire_gate_fusion_preserves_state_and_gradient():
    initial = torch.tensor([[0.5, 0.5j, -0.5j, 0.5]], dtype=torch.complex64)
    reference_theta = torch.tensor(0.23, requires_grad=True)
    theta = reference_theta.detach().clone().requires_grad_(True)
    reference_circuit = fq.Circuit(2, inputs=initial)
    reference_circuit.rx(1, reference_theta).ry(1, reference_theta * 0.7).rz(
        1, reference_theta - 0.11
    )
    reference_state = initial
    for instruction in reference_circuit.to_ir().instructions:
        matrix = _gate_matrix(
            instruction,
            bsz=initial.shape[0],
            device=initial.device,
            dtype=initial.dtype,
        )
        reference_state = _apply_matrix(reference_state, matrix, instruction.wires, 2)
    reference_loss = reference_state.real.sum()
    reference_loss.backward()

    circuit = fq.Circuit(2, inputs=initial)
    circuit.rx(1, theta).ry(1, theta * 0.7).rz(1, theta - 0.11)
    actual_state = circuit.state()
    actual_loss = actual_state.real.sum()
    actual_loss.backward()

    assert torch.allclose(actual_state, reference_state, atol=1e-6, rtol=1e-6)
    assert reference_theta.grad is not None
    assert theta.grad is not None
    assert torch.allclose(theta.grad, reference_theta.grad, atol=1e-6, rtol=1e-6)
    assert circuit._last_statevector_runtime["fused_gate_regions"] == 1
    assert circuit._last_statevector_runtime["fused_gate_count"] == 3
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 1


def test_parameter_shift_gradient_matches_autograd_for_training_loss():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], dtype=torch.float32)
    autograd_params = params.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.h(0)
        circuit.rx(1, theta=values[0])
        circuit.ry(2, theta=values[1])
        circuit.rz(3, theta=values[2])
        circuit.cx(0, 3)
        circuit.rzz(1, 2, theta=values[3])
        return circuit

    def loss_fn(circuit):
        return circuit.expectation_z((0, 2)).sum()

    autograd_loss = loss_fn(build(autograd_params))
    autograd_loss.backward()
    shift_grad = parameter_shift_gradient(build, params, loss_fn)

    assert autograd_params.grad is not None
    assert torch.allclose(shift_grad, autograd_params.grad.detach(), atol=1e-5)


def test_batched_parameter_shift_uses_one_evaluation_and_matches_autograd():
    params = torch.tensor([0.17, -0.31], dtype=torch.float64)
    autograd_params = params.detach().clone().requires_grad_(True)
    batch_sizes = []

    def build(values):
        return (
            fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .rx(0, theta=values[0])
            .ry(1, theta=values[1])
            .cx(0, 1)
        )

    def loss(circuit):
        return circuit.expectation_z((0, 1)).sum()

    def evaluate_batch(circuits):
        batch_sizes.append(len(circuits))
        return tuple(loss(circuit) for circuit in circuits)

    reference = loss(build(autograd_params))
    reference.backward()
    gradient = batched_parameter_shift_gradient(build, params, evaluate_batch)

    assert batch_sizes == [4]
    assert autograd_params.grad is not None
    torch.testing.assert_close(gradient, autograd_params.grad)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda values: fq.Circuit(1).rx(0, theta=values[0]).ry(0, theta=values[0]),
            "exactly one gate occurrence",
        ),
        (
            lambda values: fq.Circuit(1).rx(0, theta=2 * values[0]),
            "enter its gate angle directly",
        ),
        (
            lambda values: fq.Circuit(2).rzz(0, 1, theta=values[0]),
            "supports only H, X, RX, RY, RZ, and CX",
        ),
    ],
)
def test_batched_parameter_shift_rejects_unsupported_parameter_use(build, message):
    def unexpected_batch(_circuits):
        pytest.fail("invalid shifted circuits must fail before batch execution")

    with pytest.raises(ValueError, match=message):
        batched_parameter_shift_gradient(
            build,
            torch.tensor([0.2]),
            unexpected_batch,
        )


def test_native_named_parameters_bind_before_execution_and_export():
    theta = fq.Parameter("theta")
    phi = fq.Parameter("phi")
    circuit = fq.Circuit(1)
    circuit.rx(0, theta=theta).rz(0, theta=phi + 0.1)

    assert circuit.is_parameterized()
    assert circuit.parameter_names == ("phi", "theta")
    with pytest.raises(ValueError, match="bind_parameters"):
        circuit.state()
    with pytest.raises(ValueError, match="bind_parameters"):
        emit_openqasm(circuit)

    bound = circuit.bind_parameters({"theta": 0.3, phi: torch.tensor(0.2)})

    assert not bound.is_parameterized()
    assert bound.parameter_names == ()
    assert torch.allclose(
        bound.expectation_z(0), torch.cos(torch.tensor([[0.3]])), atol=1e-6
    )
    assert "rx(0.3)" in emit_openqasm(bound, version=2.0)
    assert "RZ Q0 0.3" in emit_qcis(bound)


def test_native_ir_and_compiler():
    circuit = fq.Circuit(1)
    circuit.x(0).x(0).h(0)

    ir = circuit.to_ir()
    compiled = compiler.optimize(ir)

    assert len(ir) == 3
    assert len(compiled) == 1
    assert compiled.instructions[0].name == "h"


def test_native_compiler_rotation_merge_and_layers():
    circuit = fq.Circuit(3)
    circuit.rx(0, theta=0.1)
    circuit.rx(0, theta=0.2)
    circuit.rz(1, theta=0.0)
    circuit.x(2).x(2)
    circuit.h(0).h(2)

    compiled = circuit.compile()
    layers = compiled.layers()

    assert len(compiled) == 3
    assert compiled.to_ir().instructions[0].name == "rx"
    theta = torch.as_tensor(compiled.to_ir().instructions[0].params["theta"])
    assert torch.allclose(theta, torch.tensor(0.3))
    assert {inst.name for inst in layers[0]} == {"rx", "h"}
    assert torch.allclose(circuit.state(), compiled.state(), atol=1e-6)


def test_native_compiler_routes_to_line_topology():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(2, theta=0.2).rzz(3, 1, theta=0.4)
    coupling = CouplingMap.line(4)

    compiled = circuit.compile(coupling_map=coupling)
    instructions = compiled.to_ir().instructions

    assert len(compiled) > len(circuit)
    assert any(inst.name == "swap" for inst in instructions)
    for instruction in instructions:
        if len(instruction.wires) == 2:
            assert coupling.has_edge(*instruction.wires)
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)


def test_native_compiler_topology_helpers():
    line = CouplingMap.line(4)
    ring = CouplingMap.ring(4)
    grid = CouplingMap.grid(2, 2)

    assert line.shortest_path(0, 3) == (0, 1, 2, 3)
    assert ring.has_edge(0, 3)
    assert grid.has_edge(0, 1)
    assert grid.has_edge(0, 2)


def test_top_level_uses_native_runtime():
    assert get_backend() == "pytorch"
    assert set_backend("torch") == "pytorch"


def test_native_multi_parameter_and_ising_gates():
    circuit = fq.Circuit(2)
    circuit.u3(0, theta=0.1, phi=0.2, lbd=0.3)
    circuit.rxx(0, 1, theta=0.4)
    circuit.ryy(0, 1, theta=0.5)
    circuit.rzz(0, 1, theta=0.6)

    state = circuit.state()
    probs = circuit.probabilities()

    assert state.shape == (1, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_native_copy_and_pauli_string_expectation():
    circuit = fq.Circuit(2)
    circuit.x(0)

    copied = circuit.copy()

    assert copied is not circuit
    assert len(copied) == len(circuit)
    assert torch.allclose(copied.expectation_ps(z=[0]), torch.tensor([-1.0]))


def test_native_qasm_export():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1).rx(1, theta=0.25)

    qasm = emit_openqasm(circuit, version=3.0)

    assert "OPENQASM 3.0;" in qasm
    assert "qubit[2] q;" in qasm
    assert "h q[0];" in qasm
    assert "cx q[0], q[1];" in qasm
    assert "rx(0.25) q[1];" in qasm


def test_native_distributed_bridge_cpu():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    measured = circuit.run_distributed(device="cpu", world_sz=1, measure=True)

    assert torch.allclose(measured, torch.zeros(1, 2), atol=1e-5)


def test_native_distributed_bridge_uses_compiled_plan():
    circuit = fq.Circuit(1)
    circuit.x(0).x(0).rx(0, theta=0.1).rx(0, theta=0.2)

    result, plan = circuit.run_distributed(
        device="cpu",
        world_size=1,
        return_plan=True,
    )

    assert result.plan.gate_plans[0].name == "rx"
    assert plan.analysis.n_instructions == 1
    assert result.plan.world_size == 1
    assert torch.allclose(result.state, circuit.state(), atol=1e-5)


def test_native_distributed_bridge_preserves_complex128_end_to_end():
    theta = torch.tensor(0.371 + 2**-30, dtype=torch.float64)
    custom = torch.tensor([[1.0, 2**-30j], [2**-30j, 1.0]], dtype=torch.complex128)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=theta)
    circuit.any(0, unitary=custom)

    result, plan = circuit.run_distributed(
        device="cpu",
        world_size=1,
        return_plan=True,
    )

    assert result.state.dtype == torch.complex128
    assert plan.state_bytes == 2 * torch.complex128.itemsize
    torch.testing.assert_close(result.state, circuit.state(), atol=1e-15, rtol=1e-15)


def test_native_pauli_string_entangled_correlation():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    assert torch.allclose(circuit.expectation_ps(z=[0, 1]), torch.ones(1), atol=1e-6)
    assert torch.allclose(circuit.expectation_ps(x=[0, 1]), torch.ones(1), atol=1e-6)
    assert torch.allclose(circuit.expectation_ps(y=[0, 1]), -torch.ones(1), atol=1e-6)


def test_native_sampling_and_counts():
    generator = torch.Generator().manual_seed(1234)
    circuit = fq.Circuit(2)
    circuit.x(0)

    samples = circuit.sample(8, generator=generator)
    counts = circuit.counts(8, generator=torch.Generator().manual_seed(1234))

    assert samples.shape == (1, 8, 2)
    assert torch.all(samples[..., 0] == 1)
    assert torch.all(samples[..., 1] == 0)
    assert counts == [{"10": 8}]


def test_native_planner_analysis_and_execution_plan():
    circuit = fq.Circuit(3, bsz=2)
    circuit.h(0).cx(0, 1).rz(2, theta=0.2).cx(1, 2)

    analysis = circuit.analysis()
    plan = fqxp.plan_advanced(circuit, bsz=2, world_size=2, memory_limit_bytes=1)

    assert analysis.n_wires == 3
    assert analysis.n_instructions == 4
    assert analysis.gate_counts == {"h": 1, "cx": 2, "rz": 1}
    assert analysis.two_qubit_gates == 2
    assert analysis.max_gate_width == 2
    assert plan.recommended_mode == "distributed_statevector"
    assert plan.world_size == 2
    assert plan.user_tier == "production_distributed"
    assert plan.usability_contract == "single_api_distributed_scale_out"
    assert plan.state_bytes == estimate_state_bytes(3, bsz=2)
    assert len(plan.layers) == analysis.depth
    assert plan.layers[0].wires == (0, 2)


def test_native_planner_accepts_coupling_map():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)

    plan = fqxp.plan_advanced(circuit, coupling_map=CouplingMap.line(3))

    assert plan.analysis.gate_counts["swap"] == 2
    assert plan.analysis.gate_counts["cx"] == 1
    for layer in plan.layers:
        for instruction in layer.instructions:
            if len(instruction.wires) == 2:
                assert abs(instruction.wires[0] - instruction.wires[1]) == 1


def test_auto_mode_selects_statevector_by_default():
    circuit = fq.Circuit(1)
    circuit.h(0)

    result, plan = fqr.run_native(circuit, return_plan=True)

    assert plan.state_mode == "statevector"
    assert plan.user_tier == "single_device"
    assert plan.usability_contract == "single_api_fast_path"
    assert plan.summary()["distribution_semantics"] == "single_device_fast_path"
    assert plan.summary()["scalability_claim_allowed"] is False
    assert audit_distributed_scalability(plan.summary()).valid
    assert torch.allclose(result, circuit.state(), atol=1e-6)
    assert select_execution_mode(circuit) == "statevector"


def test_run_native_uses_one_resolved_precision_for_execution_and_plan():
    circuit = fq.Circuit(1, dtype=torch.complex64).h(0)

    promoted, promoted_plan = fqr.run_native(
        circuit,
        dtype=torch.complex128,
        return_plan=True,
    )
    standalone_ir = replace(
        fq.Circuit(1, dtype=torch.complex128).h(0).to_ir(),
        metadata={},
    )
    inherited, inherited_plan = fqr.run_native(standalone_ir, return_plan=True)

    assert promoted.dtype == torch.complex128
    assert promoted_plan.runtime_config["complex_dtype"] == "complex128"
    assert promoted_plan.state_bytes == 2 * torch.complex128.itemsize
    assert inherited.dtype == torch.complex128
    assert inherited_plan.runtime_config["complex_dtype"] == "complex128"


def test_backend_native_execution_rejects_program_precision_demotion():
    circuit = fq.Circuit(1, dtype=torch.complex128).h(0)

    with pytest.raises(ValueError, match="would demote a complex128 program"):
        fqr.run_native(circuit, dtype=torch.complex64)
    with pytest.raises(ValueError, match="would demote a complex128 program"):
        circuit.run_distributed(
            device="cpu",
            world_size=1,
            precision=torch.complex64,
        )


def test_auto_mode_selects_mps_for_bond_control_and_preserves_full_state_contract():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)

    result, plan = fqr.run_native(circuit, max_bond=2, return_plan=True)
    memory_result, memory_plan = fqr.run_native(
        circuit,
        memory_limit_bytes=1,
        return_plan=True,
    )

    assert plan.state_mode == "mps"
    assert result.summary()["state_mode"] == "mps"
    assert memory_plan.state_mode == "statevector"
    assert torch.allclose(memory_result, circuit.state(), atol=1e-6)
    assert select_execution_mode(circuit, max_bond=2) == "mps"


def test_runtime_selection_prefers_local_jax_mps_training_fast_path():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).ry(2, theta=0.2).cx(2, 3)

    selection = circuit.runtime_plan(
        max_bond=4,
        prefer_jax=True,
        require_gradients=True,
    )
    summary = selection.summary()

    assert selection.recommended_mode == "jax_kernel_mps"
    assert (
        summary["recommended_candidate"]["distribution_semantics"]
        == "single_device_fast_path"
    )
    assert summary["recommended_candidate"]["gradient_support"] == "jax_value_and_grad"
    assert summary["recommended_candidate"]["scalability_claim_allowed"] is False
    assert "execution_plan" not in summary["recommended_candidate"]
    assert "rank_shards" not in summary["recommended_candidate"]
    assert "mps_runtime_summary" not in summary["recommended_candidate"]
    assert "mps_runtime_blockers" not in summary["recommended_candidate"]
    assert "runtime_readiness_blockers" not in summary["recommended_candidate"]
    assert summary["usability_contract"] == "single_api_fast_path"
    assert (
        fqxp.plan_runtime_selection(
            circuit, max_bond=4, prefer_jax=True
        ).recommended_mode
        == "jax_kernel_mps"
    )


def test_runtime_selection_marks_rank_local_jax_as_not_capacity_scaling():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).cx(2, 3)

    selection = circuit.runtime_plan(
        world_size=2,
        max_bond=4,
        prefer_jax=True,
        require_gradients=True,
    )
    candidates = {
        candidate.mode: candidate.summary() for candidate in selection.candidates
    }

    assert (
        candidates["jax_kernel_mps"]["distribution_semantics"]
        == "rank_local_replicated_kernel"
    )
    assert candidates["jax_kernel_mps"]["scalability_claim_allowed"] is False
    assert (
        "rank_local_jax_kernel_is_not_capacity_scaling"
        in candidates["jax_kernel_mps"]["warnings"]
    )
    assert selection.summary()["user_tier"] == "production_distributed"


def test_runtime_selection_includes_memory_communication_gradient_deployment_plans():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 1).ry(2, theta=0.1).cx(3, 4)

    selection = circuit.runtime_plan(
        world_size=4,
        local_world_size=2,
        node_count=2,
        prefer_jax=True,
        require_gradients=True,
        memory_limit_bytes=estimate_state_bytes(5, complex_bytes=8) // 2,
    )
    summary = selection.summary()
    candidates = {
        candidate.mode: candidate.summary() for candidate in selection.candidates
    }
    distributed_sv = candidates["distributed_statevector"]
    rank_local_jax = candidates["jax_kernel_statevector"]

    assert summary["world_size"] == 4
    assert summary["local_world_size"] == 2
    assert summary["node_count"] == 2
    assert summary["memory_plan"] == summary["recommended_candidate"]["memory_plan"]
    assert (
        summary["communication_plan"]
        == summary["recommended_candidate"]["communication_plan"]
    )
    assert summary["gradient_plan"] == summary["recommended_candidate"]["gradient_plan"]
    assert (
        summary["deployment_plan"]
        == summary["recommended_candidate"]["deployment_plan"]
    )
    assert distributed_sv["memory_plan"]["single_device_expected_oom"] is True
    assert (
        distributed_sv["communication_plan"]["communication_semantics"]
        == "amplitude_shard_transport"
    )
    assert distributed_sv["communication_plan"]["node_count"] == 2
    assert len(distributed_sv["communication_plan"]["rank_ownership"]) == 4
    assert distributed_sv["gradient_plan"]["parameter_gradient_ready"] is False
    assert distributed_sv["gradient_plan"]["fail_closed"] is True
    assert (
        rank_local_jax["communication_plan"]["communication_semantics"]
        == "rank_local_replicated_kernel"
    )
    assert rank_local_jax["communication_plan"]["estimated_communication_bytes"] == 0
    assert rank_local_jax["scalability_claim_allowed"] is False


def test_runtime_selection_uses_jax_statevector_training_readiness_plan():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    selection = circuit.runtime_plan(
        world_size=4,
        local_world_size=2,
        prefer_jax=True,
        require_gradients=True,
        distributed_profile="production",
        jax_backward_backend="pmap",
        assume_jax_devices_ready=True,
    )
    candidates = {
        candidate.mode: candidate.summary() for candidate in selection.candidates
    }
    jax_statevector = candidates["jax_sharded_statevector"]

    assert jax_statevector["scalability_claim_allowed"] is False
    assert selection.recommended_mode != "jax_sharded_statevector"
    assert jax_statevector["available"] is False
    assert "statevector_training_not_claimable" in jax_statevector["blockers"]
    assert "optimizer_update_semantics_not_measured" in jax_statevector["blockers"]
    assert jax_statevector["sharding_plan_available"] is True
    assert jax_statevector["gradient_plan"]["parameter_gradient_ready"] is True
    assert (
        jax_statevector["gradient_plan"]["details"]["planner"]
        == "jax_sharded_statevector_training"
    )
    assert (
        jax_statevector["gradient_plan"]["details"]["backward_execution"]
        == "jax_pmap_backward"
    )
    assert jax_statevector["gradient_plan"]["details"]["blockers"] == ()
    assert jax_statevector["execution_plan"]["runtime_tier"] == "planned_pmap_execution"
    assert jax_statevector["execution_plan"]["claimable_production_training"] is False
    assert (
        jax_statevector["statevector_training_claimability_status"] == "preflight_only"
    )
    assert jax_statevector["claimable_production_training"] is False
    assert len(jax_statevector["rank_shards"]) == 4
    assert len(jax_statevector["rank_ownership"]) == 4
    assert (
        jax_statevector["memory_plan"]["state_partition"]
        == "amplitude_or_qubit_address_shards"
    )
    assert len(jax_statevector["memory_plan"]["per_rank_shard_bytes"]) == 4
    assert "all_to_all" in jax_statevector["communication_plan"]["transport_patterns"]
    assert (
        jax_statevector["communication_plan"]["statevector_route_source"]
        == "distributed_statevector_plan"
    )
    assert (
        jax_statevector["communication_plan"]["inter_node_route"]
        == "topology_dependent"
    )
    assert (
        jax_statevector["communication_plan"]["collective_route_resolution"]
        == "topology_dependent"
    )
    assert (
        jax_statevector["gradient_plan"]["gradient_distribution_semantics"]
        == "sharded_across_ranks"
    )
    assert (
        jax_statevector["gradient_plan"]["optimizer_update_semantics"] == "not_measured"
    )
    assert jax_statevector["gradient_plan"]["fail_closed"] is True
    assert (
        "optimizer_update_semantics_not_measured"
        in jax_statevector["runtime_readiness_blockers"]
    )
    assert (
        "statevector_training_not_claimable"
        in jax_statevector["deployment_plan"]["readiness_blockers"]
    )


def test_runtime_selection_projects_jax_statevector_shard_map_blockers():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    selection = circuit.runtime_plan(
        world_size=4,
        local_world_size=4,
        prefer_jax=True,
        require_gradients=True,
        distributed_profile="production",
        jax_backward_backend="shard_map",
        assume_jax_devices_ready=True,
    )
    candidates = {
        candidate.mode: candidate.summary() for candidate in selection.candidates
    }
    jax_statevector = candidates["jax_sharded_statevector"]

    assert jax_statevector["available"] is False
    assert jax_statevector["scalability_claim_allowed"] is False
    assert selection.recommended_mode != "jax_sharded_statevector"
    assert (
        jax_statevector["execution_plan"]["runtime_tier"]
        == "planned_shard_map_execution"
    )
    assert (
        jax_statevector["execution_plan"]["planned_backward_execution"]
        == "jax_shard_map_backward"
    )
    assert (
        jax_statevector["execution_plan"]["production_training_preflight_ready"]
        is False
    )
    assert (
        jax_statevector["statevector_training_claimability_status"] == "preflight_only"
    )
    assert jax_statevector["claimable_production_training"] is False
    assert (
        "shard_map_statevector_multi_sharded_wire_transport_pending:all_to_all"
        in jax_statevector["blockers"]
    )
    assert "statevector_training_not_claimable" in jax_statevector["blockers"]
    assert "optimizer_update_semantics_not_measured" in jax_statevector["blockers"]
    assert jax_statevector["gradient_plan"]["parameter_gradient_ready"] is False
    assert (
        jax_statevector["gradient_plan"]["gradient_distribution_semantics"]
        == "incomplete"
    )
    assert jax_statevector["gradient_plan"]["fail_closed"] is True
    assert (
        jax_statevector["communication_plan"]["communication_execution"]
        == "jax_shard_map_backward"
    )
    assert "all_to_all" in jax_statevector["communication_plan"]["transport_patterns"]
    assert "shard_map_statevector_multi_sharded_wire_transport_pending:all_to_all" in (
        jax_statevector["communication_plan"]["collective_blockers"]
    )
    assert (
        jax_statevector["communication_plan"]["statevector_route_source"]
        == "distributed_statevector_plan"
    )


def test_runtime_selection_exposes_distributed_mps_tn_training_blockers():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 1).cx(2, 3).cx(4, 5)

    selection = circuit.runtime_plan(
        world_size=4,
        state_mode="mps",
        max_bond=4,
        require_gradients=True,
    )
    candidates = {
        candidate.mode: candidate.summary() for candidate in selection.candidates
    }

    assert (
        candidates["distributed_mps"]["distribution_semantics"]
        == "requires_runtime_summary"
    )
    assert (
        candidates["distributed_mps"]["intended_distribution_semantics"]
        == "sharded_across_ranks"
    )
    assert candidates["distributed_mps"]["scalability_claim_allowed"] is False
    assert (
        "mps_boundary_adjoint_exchange_pending"
        in candidates["distributed_mps"]["blockers"]
    )
    assert (
        candidates["distributed_mps"]["gradient_plan"]["parameter_gradient_ready"]
        is False
    )
    assert (
        candidates["distributed_mps"]["communication_plan"]["communication_semantics"]
        == "mps_site_or_bond_boundary_exchange"
    )
    assert candidates["jax_sharded_mps"]["scalability_claim_allowed"] is False
    assert (
        "mps_boundary_adjoint_exchange_pending"
        in candidates["jax_sharded_mps"]["blockers"]
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"]["planner"]
        == "jax_sharded_mps_training"
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"][
            "local_development_gradient_ready"
        ]
        is True
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"][
            "rank_local_jax_kernel_allowed_for_capacity_claim"
        ]
        is False
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"][
            "full_mps_reconstruction_allowed_for_claimable_backward"
        ]
        is False
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"]["parameter_flow"][
            "planner"
        ]
        == "jax_sharded_mps_parameter_flow"
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["details"]["parameter_flow"][
            "parameter_gate_count"
        ]
        == 0
    )
    assert candidates["jax_sharded_mps"]["available"] is False
    assert candidates["jax_sharded_mps"]["execution_plan"]["runtime_tier"] == "blocked"
    assert candidates["jax_sharded_mps"]["memory_plan"]["evidence_status"] == "planned"
    assert (
        candidates["jax_sharded_mps"]["communication_plan"]["evidence_status"]
        == "planned"
    )
    assert (
        candidates["jax_sharded_mps"]["gradient_plan"]["mps_backward_readiness_status"]
        == "blocked"
    )
    assert (
        "mps_parameter_gradient_ownership_pending"
        in candidates["jax_sharded_mps"]["runtime_readiness_blockers"]
    )
    assert (
        "mps_optimizer_update_ownership_pending"
        in candidates["jax_sharded_mps"]["runtime_readiness_blockers"]
    )
    assert candidates["jax_sharded_mps"]["claimable_production_training"] is False
    assert candidates["jax_sharded_mps"]["mps_backward_readiness_status"] == "blocked"
    assert (
        "mps_optimizer_update_semantics_not_measured"
        in candidates["jax_sharded_mps"]["mps_backward_readiness_blockers"]
    )
    assert (
        candidates["jax_sharded_tensor_network"]["gradient_support"]
        == "tensor_node_gradients_only"
    )
    assert (
        candidates["jax_sharded_tensor_network"]["communication_plan"][
            "communication_semantics"
        ]
        == "tensor_network_slice_reduce"
    )
    assert (
        "jax_sliced_tensor_network_parameter_gate_pullback_pending"
        in candidates["jax_sharded_tensor_network"]["blockers"]
    )


@pytest.mark.parametrize(
    "readiness_status",
    (
        "control_plane_ready",
        "backward_preflight_ready",
        "production_backward_evidence",
    ),
)
def test_runtime_selection_projects_mps_evidence_status(
    monkeypatch,
    readiness_status,
):
    from flagquantum.runtime.executors.jax.mps import planning as mps_planning

    class _TrainingPlan:
        def summary(self):
            blockers = ("synthetic_phase5_release_evidence_pending",)
            evidence_status = {
                "boundary_adjoint_exchange": "planned",
                "parameter_gradient_ownership": "planned",
                "backward_memory": "planned",
                "backward_communication": "planned",
                "optimizer_update_ownership": "pending",
            }
            return {
                "claim_evidence_type": "plan_preflight",
                "distribution_semantics": "requires_runtime_summary",
                "intended_distribution_semantics": "sharded_across_ranks",
                "sharding_plan_available": True,
                "world_size": 2,
                "local_world_size": 2,
                "node_count": 1,
                "backward_execution": "jax_pmap_backward",
                "mps_forward_distribution_semantics": "sharded_across_ranks",
                "mps_backward_distribution_semantics": "sharded_across_ranks",
                "site_shard_ownership": ({"rank": 0}, {"rank": 1}),
                "bond_shard_ownership": ({"left_rank": 0, "right_rank": 1},),
                "parameter_gradient_ownership": ({"rank": 0}, {"rank": 1}),
                "boundary_gradient_ownership": ({"left_rank": 0, "right_rank": 1},),
                "boundary_adjoint_exchange": {"status": "planned_not_executed"},
                "boundary_gradient_routes": ({"left_rank": 0, "right_rank": 1},),
                "mps_backward_memory_plan": {"status": "complete_estimate"},
                "mps_backward_communication_plan": {"status": "planned_not_executed"},
                "optimizer_update_semantics": "not_measured",
                "optimizer_update_ownership": (),
                "blockers": blockers,
                "mps_backward_readiness_gate": {
                    "status": readiness_status,
                    "production_training_claimable": False,
                },
                "mps_backward_readiness_status": readiness_status,
                "mps_backward_readiness_blockers": blockers,
                "mps_runtime_summary": {
                    "status": readiness_status,
                    "production_training_claimable": False,
                    "fail_closed": True,
                    "evidence_status": evidence_status,
                    "runtime_blockers": blockers,
                },
                "mps_runtime_blockers": blockers,
                "claimable_production_training": False,
                "scalability_claim_allowed": False,
            }

    monkeypatch.setattr(
        mps_planning,
        "plan_jax_sharded_mps_training",
        lambda *args, **kwargs: _TrainingPlan(),
    )
    selection = fqxp.plan_runtime_selection(
        fq.Circuit(4),
        world_size=2,
        state_mode="mps",
        require_gradients=True,
    )
    candidate = next(
        item.summary()
        for item in selection.candidates
        if item.mode == "jax_sharded_mps"
    )

    assert candidate["execution_plan"]["runtime_tier"] == readiness_status
    assert candidate["memory_plan"]["evidence_status"] == "planned"
    assert candidate["memory_plan"]["world_size"] == 2
    assert candidate["memory_plan"]["rank_ownership"] == ({"rank": 0}, {"rank": 1})
    assert candidate["communication_plan"]["evidence_status"] == "planned"
    assert (
        candidate["communication_plan"]["communication_semantics"]
        == "mps_boundary_adjoint_and_gradient_exchange"
    )
    assert candidate["deployment_plan"]["runtime_readiness"] == readiness_status
    assert (
        "synthetic_phase5_release_evidence_pending"
        in candidate["deployment_plan"]["blockers"]
    )
    assert (
        candidate["gradient_plan"]["mps_backward_readiness_status"] == readiness_status
    )
    assert candidate["gradient_plan"]["evidence_status"] == {
        "boundary_adjoint_exchange": "planned",
        "parameter_gradient_ownership": "planned",
        "backward_memory": "planned",
        "backward_communication": "planned",
        "optimizer_update_ownership": "pending",
    }
    assert candidate["mps_runtime_summary"]["status"] == readiness_status
    assert candidate["runtime_readiness_blockers"] == (
        "synthetic_phase5_release_evidence_pending",
    )
    assert candidate["available"] is False
    assert candidate["claimable_production_training"] is False
    assert candidate["scalability_claim_allowed"] is False


def test_run_native_executes_routed_statevector():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).rz(1, theta=0.2)

    result, plan = fqr.run_native(
        circuit,
        coupling_map=CouplingMap.line(3),
        return_plan=True,
    )

    assert plan.analysis.gate_counts["swap"] == 2
    expected = circuit.state()
    assert result.device == expected.device
    assert torch.allclose(result, expected, atol=1e-6)


def test_auto_mode_can_dispatch_to_distributed_cpu():
    circuit = fq.Circuit(1)
    circuit.x(0)

    result, plan = fqr.run_native(
        circuit,
        world_size=1,
        mode="distributed_statevector",
        device="cpu",
        return_plan=True,
    )

    assert plan.recommended_mode == "local"
    assert result.plan.world_size == 1
    assert torch.allclose(result.state, circuit.state(), atol=1e-6)
    assert select_execution_mode(circuit, world_size=2) == "distributed_statevector"


def test_distributed_mode_alias_and_plan_name_are_explicit_statevector():
    circuit = fq.Circuit(1)
    circuit.h(0)

    qdev_alias, alias_plan = fqr.run_native(
        circuit,
        mode="distributed",
        device="cpu",
        world_size=1,
        return_plan=True,
    )
    qdev_explicit, explicit_plan = fqr.run_native(
        circuit,
        mode="distributed_statevector",
        device="cpu",
        world_size=1,
        return_plan=True,
    )
    planned = fqxp.plan_advanced(circuit, world_size=2)

    assert alias_plan.state_mode == "statevector"
    assert explicit_plan.state_mode == "statevector"
    assert planned.recommended_mode == "distributed_statevector"
    assert planned.summary()["user_tier"] == "production_distributed"
    assert planned.summary()["usability_contract"] == "single_api_distributed_scale_out"
    assert planned.summary()["distribution_semantics"] == "sharded_across_ranks"
    assert planned.summary()["scalability_claim_allowed"] is False
    assert planned.summary()["sharding_plan_available"] is True
    assert planned.state_mode == "statevector"
    assert torch.allclose(qdev_alias.state, qdev_explicit.state, atol=1e-6)


def test_distributed_mps_mode_exposes_rank_shards_and_matches_mps():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2)

    result, plan = fqr.run_native(
        circuit, mode="distributed_mps", world_size=2, max_bond=4, return_plan=True
    )

    assert isinstance(result, DistributedMPSState)
    assert plan.state_mode == "mps"
    assert plan.recommended_mode == "distributed_mps"
    assert plan.summary()["distribution_semantics"] == "requires_runtime_summary"
    assert plan.summary()["scalability_claim_allowed"] is False
    assert (
        "runtime_summary_required_for_scalability_claim"
        in plan.summary()["scalability_blockers"]
    )
    assert result.summary()["state_mode"] == "distributed_mps"
    assert result.summary()["world_size"] == 2
    assert tuple(shard.wires for shard in result.shards) == ((0, 1), (2, 3))
    assert torch.allclose(
        result.to_statevector(),
        fqr.run_native(circuit, mode="mps").to_statevector(),
        atol=1e-6,
    )
    assert select_execution_mode(circuit, world_size=2, max_bond=4) == "distributed_mps"


def test_distributed_mps_accepts_adaptive_bond_policy():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).h(2).cx(2, 3).cx(1, 2)

    result, plan = fqr.run_native(
        circuit,
        mode="distributed_mps",
        world_size=2,
        adaptive=True,
        initial_max_bond=1,
        global_error_budget=0.0,
        max_bond_cap=4,
        return_plan=True,
    )
    summary = result.summary()

    assert isinstance(result, DistributedMPSState)
    assert plan.state_mode == "mps"
    assert summary["adaptive"]
    assert summary["adaptive_rerun"]
    assert summary["adaptive_initial_plan"]["hot_bonds"]
    assert (
        summary["adaptive_final_plan"]["current_max_bond"]
        > summary["adaptive_initial_plan"]["current_max_bond"]
    )
    assert summary["adaptive_refinement_plan"]["windows"]
    assert torch.allclose(
        result.to_statevector(),
        fqmps.run_mps(circuit, max_bond=4).to_statevector(),
        atol=1e-6,
    )


def test_distributed_mps_local_tensor_strict_chain_shards_without_full_sync():
    circuit = fq.Circuit(4)
    circuit.h(0).ry(1, theta=0.2).cx(1, 2).rz(3, theta=-0.1)

    result = fqr.run_native(
        circuit,
        mode="distributed_mps",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
        max_bond=4,
        strict_sharded=True,
    )
    summary = result.summary()

    assert summary["executor"] == "local_tensor_development_simulator"
    assert summary["mps_execution"] == "local_tensor_site_sharded_simulator"
    assert (
        summary["distribution_semantics"]
        == "hybrid_sharded_forward_with_replicated_state"
    )
    assert summary["scalability_claim_allowed"] is False
    assert "single_process_development_simulator" in summary["scalability_blockers"]
    assert "mps_full_local_replay_not_claimable" in summary["scalability_blockers"]
    assert "full_mps_sync_fallback" not in summary["scalability_blockers"]
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert summary["mps_backward_readiness_gate"]["fail_closed"] is True
    assert not any(key.startswith("phase5_mps_") for key in summary)
    runtime = summary["mps_runtime_summary"]
    assert runtime["status"] == "blocked"
    assert runtime["evidence_status"]["boundary_adjoint_exchange"] == "pending"
    assert runtime["evidence_status"]["parameter_gradient_ownership"] == "pending"
    assert runtime["evidence_status"]["backward_memory"] == "pending"
    assert runtime["evidence_status"]["backward_communication"] == "pending"
    assert "mps_full_local_replay_not_claimable" in runtime["runtime_blockers"]
    assert "mps_parameter_gradient_ownership_pending" in runtime["runtime_blockers"]
    assert "mps_backward_memory_evidence_pending" in runtime["runtime_blockers"]
    assert "mps_backward_communication_evidence_pending" in runtime["runtime_blockers"]
    assert (
        "mps_full_local_state_view_not_claimable"
        in summary["mps_backward_readiness_blockers"]
    )
    assert summary["boundary_sync_count"] == 1
    assert summary["full_sync_count"] == 0
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["unsupported_instruction_count"] == 0
    assert summary["strict_sharded"] is True
    assert summary["local_tensor_wires_by_rank"] == {0: (0, 1), 1: (2, 3)}
    assert len(summary["local_memory_bytes_by_rank"]) == 2
    assert torch.allclose(
        result.to_statevector(),
        fqmps.run_mps(circuit, max_bond=4).to_statevector(),
        atol=1e-6,
    )


def test_distributed_mps_strict_sharded_rejects_nonlocal_gate():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3)

    with pytest.raises(RuntimeError, match="Strict distributed MPS only supports"):
        fqr.run_native(
            circuit,
            mode="distributed_mps",
            world_size=2,
            distributed_profile="development",
            torch_backend="local_tensor",
            max_bond=4,
            strict_sharded=True,
        )


def test_distributed_tensor_network_mode_exposes_slice_tasks_and_matches_tn():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    local_plan = build_tensor_network(circuit)
    target_peak = local_plan.contraction_cost()["peak_size"]
    label_counts = Counter(label for node in local_plan.nodes for label in node.labels)
    sliced_label = next(
        label
        for label, count in label_counts.items()
        if count > 1 and label not in local_plan.output_labels
    )

    result, plan = fqr.run_native(
        circuit,
        mode="distributed_tensor_network",
        world_size=2,
        max_intermediate_size=target_peak,
        sliced_labels=(sliced_label,),
        return_plan=True,
    )

    assert isinstance(result, DistributedTensorNetworkState)
    assert DistributedTensorNetworkState.__module__.endswith("tensor_network.state")
    assert plan.state_mode == "tensor_network"
    assert plan.recommended_mode == "distributed_tensor_network"
    assert result.summary()["state_mode"] == "distributed_tensor_network"
    assert "jax_distributed_plan" not in result.summary()
    assert result.summary()["world_size"] == 2
    assert result.summary()["slice_tasks"] == len(result.tasks)
    assert set(result.summary()["tasks_by_rank"]) == {0, 1}
    assert all(isinstance(task, DistributedTNSliceTask) for task in result.tasks)
    assert torch.allclose(
        result.state(),
        fqr.run_native(circuit, mode="tensor_network").state(),
        atol=1e-6,
    )


def test_distributed_tensor_network_local_tensor_simulates_slice_parallel_state():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    local_plan = build_tensor_network(circuit)
    target_peak = local_plan.contraction_cost()["peak_size"]
    label_counts = Counter(label for node in local_plan.nodes for label in node.labels)
    sliced_label = next(
        label
        for label, count in label_counts.items()
        if count > 1 and label not in local_plan.output_labels
    )

    result = fqr.run_native(
        circuit,
        mode="distributed_tensor_network",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
        max_intermediate_size=target_peak,
        sliced_labels=(sliced_label,),
    )
    summary = result.summary()

    assert summary["executor"] == "local_tensor_development_simulator"
    assert summary["distribution_semantics"] == "slice_parallel_state_with_local_facade"
    assert summary["scalability_claim_allowed"] is False
    assert "single_process_development_simulator" in summary["scalability_blockers"]
    assert "full_local_tensor_network_facade" in summary["scalability_blockers"]
    assert summary["communication_tiers"]["model"] == "local_simulated_all_reduce"
    assert set(summary["tasks_by_rank"]) == {0, 1}
    assert set(summary["rank_partial_bytes_by_rank"]) == {0, 1}
    assert len(summary["local_memory_bytes_by_rank"]) == 2
    assert torch.allclose(
        result.state(),
        fqr.run_native(circuit, mode="tensor_network").state(),
        atol=1e-6,
    )


def test_distributed_tensor_network_alias_and_backend_capability():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = fqr.run_native(
        circuit, mode="distributed_tn", world_size=2, max_intermediate_size=4
    )

    assert isinstance(result, DistributedTensorNetworkState)
    assert get_backend_capabilities().supports_mode("distributed_mps")
    assert get_backend_capabilities().supports_mode("distributed_tensor_network")


def test_distributed_tensor_network_torch_executor_single_rank():
    _skip_inside_outer_torchrun()
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    init_uri = _free_tcp_init_method()

    try:
        result = fqr.run_native(
            circuit,
            mode="distributed_tensor_network",
            world_size=1,
            distributed_executor="torch",
            init_method=init_uri,
            backend="gloo",
            device="cpu",
            max_intermediate_size=8,
        )

        assert result.summary()["executor"] == "torch_distributed"
        assert (
            result.summary()["distribution_semantics"]
            == "slice_parallel_state_with_local_facade"
        )
        assert result.summary()["scalability_claim_allowed"] is False
        assert (
            "full_local_tensor_network_facade"
            in result.summary()["scalability_blockers"]
        )
        assert result.summary()["rank"] == 0
        assert result.summary()["rank_placement"]["node_count"] == 1
        assert (
            result.summary()["communication_tiers"]["model"]
            == "collective_topology_dependent"
        )
        assert torch.allclose(
            result.state(),
            fqr.run_native(circuit, mode="tensor_network").state(),
            atol=1e-6,
        )
    finally:
        destroy_torch_distributed()


def test_distributed_mps_torch_executor_single_rank():
    _skip_inside_outer_torchrun()
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=theta)
    init_uri = _free_tcp_init_method()

    try:
        result = fqr.run_native(
            circuit,
            mode="distributed_mps",
            world_size=1,
            distributed_executor="torch",
            init_method=init_uri,
            backend="gloo",
            device="cpu",
            max_bond=4,
            boundary_transport="auto",
        )

        assert result.summary()["executor"] == "torch_distributed"
        assert (
            result.summary()["distribution_semantics"]
            == "hybrid_sharded_forward_with_replicated_state"
        )
        assert result.summary()["scalability_claim_allowed"] is False
        assert (
            "mps_full_local_replay_not_claimable"
            in result.summary()["scalability_blockers"]
        )
        assert "replicated_mps_autograd" in result.summary()["scalability_blockers"]
        assert result.summary()["mps_execution"] == "site_sharded_sync"
        assert result.summary()["gradient_execution"] == "replicated_mps_autograd"
        assert result.summary()["mps_backward_readiness_status"] == "blocked"
        assert (
            "mps_replicated_autograd_not_claimable"
            in result.summary()["mps_backward_readiness_blockers"]
        )
        assert result.summary()["rank"] == 0
        assert result.summary()["rank_placement"]["node_count"] == 1
        assert result.summary()["communication_tiers"]["node_count"] == 1
        assert result.summary()["owned_instruction_count"] == len(circuit)
        assert result.summary()["sharded_kernel_count"] == 2
        assert result.summary()["tensor_sync_count"] == 2
        assert result.summary()["full_sync_count"] == 1
        assert result.summary()["sync_bytes"] == 0
        assert result.summary()["boundary_transfer_bytes"] == 0
        assert result.summary()["storage"] == "sharded"
        assert set(result.local_shard_tensors) == {0, 1, 2}
        assert isinstance(result.sharded_state, ShardedMPSState)
        assert result.sharded_state.summary()["local_tensor_wires"] == (0, 1, 2)
        assert torch.allclose(
            result.to_statevector(),
            fqr.run_native(circuit, mode="mps").to_statevector(),
            atol=1e-6,
        )
        assert torch.allclose(
            result.sharded_state.to_statevector(),
            fqr.run_native(circuit, mode="mps").to_statevector(),
            atol=1e-6,
        )
        loss = result.expectation_z(1).sum()
        loss.backward()
        assert theta.grad is not None
    finally:
        destroy_torch_distributed()


def test_distributed_mps_site_local_two_qubit_gate_uses_tensor_sync():
    _skip_inside_outer_torchrun()
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    init_uri = _free_tcp_init_method()

    try:
        result = fqr.run_native(
            circuit,
            mode="distributed_mps",
            world_size=1,
            distributed_executor="torch",
            init_method=init_uri,
            backend="gloo",
            device="cpu",
            max_bond=4,
        )

        assert result.summary()["mps_execution"] == "site_sharded_sync"
        assert result.summary()["sharded_kernel_count"] == 2
        assert result.summary()["tensor_sync_count"] == 2
        assert result.summary()["full_sync_count"] == 0
        assert torch.allclose(
            result.to_statevector(),
            fqr.run_native(circuit, mode="mps").to_statevector(),
            atol=1e-6,
        )
    finally:
        destroy_torch_distributed()


def test_sharded_mps_apply_one_local_matches_mps_kernel():
    mps = MPSState.zero(2)
    sharded = ShardedMPSState(
        n_wires=2,
        bsz=1,
        config=mps.config,
        local_tensors={0: mps.tensors[0]},
        shards=(
            DistributedShardPlan(
                rank=0, world_size=1, wires=(0,), left_boundary=None, right_boundary=0
            ),
        ),
    )
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=mps.dtype)

    sharded.apply_one_local(matrix, 0)
    mps.apply_one(matrix, 0)

    assert torch.allclose(sharded.local_tensors[0], mps.tensors[0], atol=1e-6)


def test_sharded_mps_apply_two_local_matches_mps_kernel():
    mps = MPSState.zero(2)
    sharded = ShardedMPSState(
        n_wires=2,
        bsz=1,
        config=mps.config,
        local_tensors={0: mps.tensors[0], 1: mps.tensors[1]},
        shards=(
            DistributedShardPlan(
                rank=0, world_size=1, wires=(0, 1), left_boundary=None, right_boundary=1
            ),
        ),
    )
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=mps.dtype,
    )

    sharded.apply_two_local(matrix, 0)
    mps.apply_two(matrix, 0)

    assert torch.allclose(sharded.local_tensors[0], mps.tensors[0], atol=1e-6)
    assert torch.allclose(sharded.local_tensors[1], mps.tensors[1], atol=1e-6)


def test_distributed_mps_identifies_cross_shard_boundary_gate():
    from flagquantum.runtime.executors.mps import execution as dist_runtime

    circuit = fq.Circuit(4)
    circuit.cx(1, 2)
    instruction = circuit.to_ir().instructions[0]
    shards = dist_runtime._mps_shards(4, 2)

    assert tuple(shard.wires for shard in shards) == ((0, 1), (2, 3))
    assert dist_runtime._instruction_is_boundary_local(instruction, shards)
    assert not dist_runtime._instruction_is_site_local(instruction, shards)
    assert dist_runtime._boundary_touched_wires(instruction) == (1, 2)
    record = dist_runtime._boundary_sync_record(instruction, shards)
    assert isinstance(record, DistributedBoundarySync)
    assert record.left_wire == 1
    assert record.right_wire == 2
    assert record.left_rank == 0
    assert record.right_rank == 1
    assert record.owner_rank == 0
    mps = MPSState.zero(4)
    protocol = dist_runtime._boundary_protocol_record(mps, record)
    assert isinstance(protocol, DistributedBoundaryProtocol)
    assert protocol.stages == (
        "isend_boundary_to_owner",
        "apply_two_site_update",
        "isend_updated_boundary_shards",
    )
    assert protocol.recipient_ranks == (0, 1)
    assert protocol.transport == "async_p2p"
    assert protocol.estimated_transfer_bytes == protocol.tensor_bytes * 2
    assert protocol.point_to_point_messages == 2
    assert protocol.collective_messages == 0
    assert protocol.tensor_bytes > 0
    assert protocol.sync is record

    p2p_protocol = dist_runtime._boundary_protocol_record(mps, record, transport="p2p")
    assert p2p_protocol.stages == (
        "send_boundary_to_owner",
        "apply_two_site_update",
        "send_updated_boundary_shards",
    )
    assert p2p_protocol.transport == "p2p"
    assert p2p_protocol.point_to_point_messages == 2
    assert p2p_protocol.collective_messages == 0

    broadcast_protocol = dist_runtime._boundary_protocol_record(
        mps, record, transport="broadcast"
    )
    assert broadcast_protocol.stages == (
        "gather_boundary",
        "apply_two_site_update",
        "scatter_boundary_shards",
    )
    assert broadcast_protocol.transport == "broadcast"
    assert broadcast_protocol.recipient_ranks == (0, 1)
    assert (
        broadcast_protocol.estimated_transfer_bytes
        == broadcast_protocol.tensor_bytes * 2
    )
    assert broadcast_protocol.point_to_point_messages == 0
    assert broadcast_protocol.collective_messages == 2
