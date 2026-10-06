"""Single-rank reverse-mode and custom-autograd contracts."""

import math
from types import SimpleNamespace

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.statevector import plan_distributed_statevector
from flagquantum.runtime.executors.statevector.gradient_dispatch import (
    _triton_local_adjoint_vjp_decision,
    _triton_local_reversible_vjp_decision,
    _triton_local_reversible_vjp_tensor_decision,
    _triton_sharded_adjoint_vjp_decision,
    _triton_sharded_adjoint_vjp_tensor_decision,
)
from flagquantum.runtime.executors.statevector.gradient_reduction import (
    AsyncGradientReducer,
)
from flagquantum.runtime.executors.statevector.local_execution import (
    _rank_global_indices,
    use_compact_global_indices,
)
from flagquantum.runtime.executors.statevector.reverse import (
    BackwardExecutionEvidence,
    StatevectorCheckpointPolicy,
    _parameter_layout,
    execute_torch_distributed_statevector_reverse,
    resolve_checkpoint_policy,
)
from flagquantum.runtime.executors.statevector.reverse_adjoint_kernels import (
    _cpu_direct_adjoint_gate_enabled,
)
from flagquantum.runtime.executors.statevector.reverse_support import (
    _compact_reverse_global_indices,
)
from flagquantum.runtime.executors.statevector.reverse_template import (
    _clear_adjoint_ir_template_cache,
    prepare_adjoint_ir_template,
)
from flagquantum.simulation.native_cpu import native_cpu_cx_adjoint_inplace_available

pytestmark = pytest.mark.unit


def test_local_adjoint_vjp_decision_binds_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    decision = _triton_local_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )

    assert decision.accelerated
    assert decision.semantic_id == "gradient.vjp.adjoint_1q.local"
    assert decision.implementation_id == "FQKI-TRITON-GR-001-A"
    assert decision.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_local_adjoint_vjp_decision_reports_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")

    decision = _triton_local_adjoint_vjp_decision(
        device_type=device_type,
        dtype=dtype,
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.implementation_id is None
    assert decision.catalog_mismatches == (mismatch,)


def test_local_adjoint_vjp_decision_preserves_runtime_contract(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    decision = _triton_local_adjoint_vjp_decision(
        runtime_supported=False,
        device_type="cuda",
        dtype="complex64",
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.catalog_mismatches == ()


def test_local_adjoint_vjp_decision_honors_policy_and_availability(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", raising=False)
    disabled = _triton_local_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert disabled.reason == "disabled_by_policy"

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: False,
    )
    unavailable = _triton_local_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert unavailable.reason == "triton_unavailable"


def test_sharded_adjoint_vjp_decision_binds_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch."
        "triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    decision = _triton_sharded_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )

    assert decision.accelerated
    assert decision.semantic_id == "gradient.vjp.adjoint_1q.sharded"
    assert decision.implementation_id == "FQKI-TRITON-GR-003-A"
    assert decision.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_sharded_adjoint_vjp_decision_reports_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")

    decision = _triton_sharded_adjoint_vjp_decision(
        device_type=device_type,
        dtype=dtype,
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.implementation_id is None
    assert decision.catalog_mismatches == (mismatch,)


def test_sharded_adjoint_vjp_tensor_decision_preserves_runtime_contract(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def capture_decision(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.gradient_dispatch."
        "_triton_sharded_adjoint_vjp_decision",
        capture_decision,
    )
    before = torch.zeros((1, 4), dtype=torch.complex64)
    adjoint = torch.zeros((1, 5), dtype=torch.complex64)

    decision = _triton_sharded_adjoint_vjp_tensor_decision(before, adjoint)

    assert decision is sentinel
    assert captured == {
        "runtime_supported": False,
        "device_type": "cpu",
        "dtype": "complex64",
    }


def test_sharded_adjoint_vjp_decision_honors_policy_and_availability(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", raising=False)
    disabled = _triton_sharded_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert disabled.reason == "disabled_by_policy"

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: False,
    )
    unavailable = _triton_sharded_adjoint_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert unavailable.reason == "triton_unavailable"


def test_local_reversible_vjp_decision_binds_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    decision = _triton_local_reversible_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )

    assert decision.accelerated
    assert decision.semantic_id == "gradient.vjp.reversible_1q.local"
    assert decision.implementation_id == "FQKI-TRITON-GR-002-A"
    assert decision.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_local_reversible_vjp_decision_reports_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")

    decision = _triton_local_reversible_vjp_decision(
        device_type=device_type,
        dtype=dtype,
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.implementation_id is None
    assert decision.catalog_mismatches == (mismatch,)


def test_local_reversible_vjp_decision_rejects_aliasing_states(monkeypatch):
    captured = {}
    sentinel = object()

    def capture_decision(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.gradient_dispatch."
        "_triton_local_reversible_vjp_decision",
        capture_decision,
    )
    state = torch.zeros((1, 4), dtype=torch.complex64)

    decision = _triton_local_reversible_vjp_tensor_decision(state, state)

    assert decision is sentinel
    assert captured == {
        "runtime_supported": False,
        "device_type": "cpu",
        "dtype": "complex64",
    }


def test_local_reversible_vjp_decision_honors_policy_and_availability(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", raising=False)
    disabled = _triton_local_reversible_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert disabled.reason == "disabled_by_policy"

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: False,
    )
    unavailable = _triton_local_reversible_vjp_decision(
        device_type="cuda",
        dtype="complex64",
    )
    assert unavailable.reason == "triton_unavailable"


def test_adjoint_ir_template_cache_is_parameter_free_and_has_rollback(monkeypatch):
    _clear_adjoint_ir_template_cache()
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    ir = fq.Circuit(2, dtype=torch.complex128).ry(0, theta).cx(0, 1).to_ir()
    _, slots, _ = _parameter_layout(ir)

    first = prepare_adjoint_ir_template(ir, slots)
    repeated = prepare_adjoint_ir_template(ir, slots)

    assert repeated is first
    assert first.instructions[0].params["theta"] == 0.0
    assert all(
        not isinstance(value, torch.Tensor)
        for instruction in first.instructions
        for value in instruction.params.values()
    )

    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_TEMPLATE_CACHE", "0")
    assert prepare_adjoint_ir_template(ir, slots) is not first


def test_adjoint_ir_template_cache_bypasses_non_parameter_tensors():
    _clear_adjoint_ir_template_cache()
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, theta)
    circuit.any(0, unitary=torch.eye(2, dtype=torch.complex128))
    ir = circuit.to_ir()
    _, slots, _ = _parameter_layout(ir)

    first = prepare_adjoint_ir_template(ir, slots)
    repeated = prepare_adjoint_ir_template(ir, slots)

    assert repeated is not first


def test_repeated_adjoint_execution_observes_updated_parameter_value():
    _clear_adjoint_ir_template_cache()
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)

    first = execute_torch_distributed_statevector_reverse(circuit)
    first.backward()
    torch.testing.assert_close(theta.grad, -torch.sin(theta.detach()))

    theta.grad = None
    with torch.no_grad():
        theta.fill_(0.71)
    second = execute_torch_distributed_statevector_reverse(circuit)
    second.backward()

    torch.testing.assert_close(theta.grad, -torch.sin(theta.detach()))


def test_parameter_layout_deduplicates_exact_views_of_one_leaf_element():
    parameters = torch.tensor((0.2, 0.4), dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit.rx(0, parameters[0]).rx(1, parameters[0]).rz(1, parameters[1])

    values, slots, occurrences = _parameter_layout(circuit.to_ir())

    assert len(values) == 2
    assert tuple(slot[2] for slot in slots) == (0, 0, 1)
    assert occurrences == (2, 1)


def test_parameter_layout_keeps_independent_aliasing_leaves_distinct():
    first = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    second = first.detach().requires_grad_()
    circuit = fq.Circuit(2, dtype=torch.complex128).rx(0, first).rx(1, second)

    values, slots, occurrences = _parameter_layout(circuit.to_ir())

    assert values[0] is first
    assert values[1] is second
    assert tuple(slot[2] for slot in slots) == (0, 1)
    assert occurrences == (1, 1)


def test_cached_adjoint_template_revalidates_updated_parameter_value():
    _clear_adjoint_ir_template_cache()
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)
    execute_torch_distributed_statevector_reverse(circuit)

    with torch.no_grad():
        theta.fill_(float("nan"))

    with pytest.raises(ValueError, match="finite"):
        execute_torch_distributed_statevector_reverse(circuit)


def test_adjoint_backward_skips_revalidating_saved_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.runtime.executors.statevector import reverse_support

    calls = 0
    original = reverse_support._normalize_angle

    def counting_normalize_angle(value, *, opcode, parameter):
        nonlocal calls
        calls += 1
        return original(value, opcode=opcode, parameter=parameter)

    monkeypatch.setattr(reverse_support, "_normalize_angle", counting_normalize_angle)
    theta = torch.tensor((0.23, -0.41), dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, theta[0]).rz(1, theta[1])

    result = execute_torch_distributed_statevector_reverse(circuit)
    assert calls == 2
    result.backward()

    assert calls == 2


def test_adjoint_saved_parameter_revalidation_has_a_complete_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.runtime.executors.statevector import reverse_support

    calls = 0
    original = reverse_support._normalize_angle

    def counting_normalize_angle(value, *, opcode, parameter):
        nonlocal calls
        calls += 1
        return original(value, opcode=opcode, parameter=parameter)

    monkeypatch.setattr(reverse_support, "_normalize_angle", counting_normalize_angle)
    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_REVALIDATE_SAVED_PARAMETERS", "1")
    theta = torch.tensor((0.23, -0.41), dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, theta[0]).rz(1, theta[1])

    result = execute_torch_distributed_statevector_reverse(circuit)
    result.backward()

    assert calls == 4


def test_adjoint_saved_parameter_version_check_precedes_backward_binding() -> None:
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)
    result = execute_torch_distributed_statevector_reverse(circuit)

    with torch.no_grad():
        theta.add_(0.1)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.backward()


def test_cpu_direct_adjoint_gate_has_explicit_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _cpu_direct_adjoint_gate_enabled() is True
    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_CPU_DIRECT", "0")
    assert _cpu_direct_adjoint_gate_enabled() is False


def test_generic_matrix_jvp_matches_rotation_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.runtime.executors.statevector import reverse_adjoint_sweep

    monkeypatch.setattr(
        reverse_adjoint_sweep, "_analytic_rotation_derivative", lambda *args: None
    )
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)
    result = execute_torch_distributed_statevector_reverse(circuit)
    result.backward()

    torch.testing.assert_close(theta.grad, -torch.sin(theta.detach()))


@pytest.mark.parametrize("jvp_result", [None, (torch.tensor(0.0), None)])
def test_generic_matrix_jvp_rejects_invalid_derivative(
    monkeypatch: pytest.MonkeyPatch, jvp_result: object
) -> None:
    from flagquantum.runtime.executors.statevector import reverse_adjoint_sweep

    monkeypatch.setattr(
        reverse_adjoint_sweep, "_analytic_rotation_derivative", lambda *args: None
    )
    monkeypatch.setattr(
        torch.autograd.functional, "jvp", lambda *args, **kwargs: jvp_result
    )
    theta = torch.tensor(0.23, requires_grad=True)
    result = execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta))

    with pytest.raises(TypeError, match="Matrix JVP must return a tensor derivative"):
        result.backward()
    assert not result.summary()["parameter_gradient_ready"]


def test_address_sharded_reverse_always_uses_compact_global_indices():
    small_shard = SimpleNamespace(local_amplitudes=1 << 23)
    address_sharded = SimpleNamespace(
        distribution="qubit_address_sharded", shards=(small_shard,)
    )
    range_partitioned = SimpleNamespace(
        distribution="range_partitioned", shards=(small_shard,)
    )

    assert _compact_reverse_global_indices(address_sharded, 0) is True
    assert _compact_reverse_global_indices(range_partitioned, 0) is False


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_address_sharded_indices_are_built_from_local_ordinal(world_size):
    plan = plan_distributed_statevector(fq.Circuit(8), world_size=world_size)
    rank_bits = len(plan.sharded_qubits)

    for rank in range(world_size):
        indices = _rank_global_indices(plan, rank, device=torch.device("cpu"))
        expected = (
            torch.arange(plan.shards[rank].local_amplitudes, dtype=torch.long)
            << rank_bits
        ) | rank
        torch.testing.assert_close(indices, expected)
        assert indices.numel() == plan.shards[rank].local_amplitudes


def test_compact_index_policy_is_shared_by_forward_and_reverse():
    plan = plan_distributed_statevector(fq.Circuit(8), world_size=4)

    for rank in range(plan.world_size):
        assert use_compact_global_indices(plan, rank)
        assert _compact_reverse_global_indices(plan, rank)


def test_parameter_gradients_are_all_reduced_in_dtype_packed_buckets(monkeypatch):
    calls = []

    class CompletedWork:
        def wait(self):
            return True

    def fake_all_reduce(tensor, **kwargs):
        calls.append(tensor.clone())
        tensor.add_(10)
        assert kwargs["async_op"] is True
        return CompletedWork()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0, dtype=torch.float64),
    ]
    evidence = BackwardExecutionEvidence()

    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=len(gradients),
        max_bytes=sum(
            gradient.numel() * gradient.element_size() for gradient in gradients
        ),
    )
    for index in range(len(gradients)):
        reducer.mark_ready(index, overlap_opportunity=False)
    reducer.finish()

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(calls[1], torch.tensor([3.0], dtype=torch.float64))
    assert [gradient.item() for gradient in gradients] == [11.0, 12.0, 13.0]
    assert evidence.communication_count == 2
    assert evidence.async_gradient_collective_count == 2
    assert evidence.communication_bytes == 16
    assert evidence.gradient_collective_count == 2
    assert evidence.gradient_collective_bytes == 16


def test_ready_gradient_bucket_launches_before_adjoint_completion(monkeypatch):
    waits = []

    class PendingWork:
        def wait(self):
            waits.append(True)

    def fake_all_reduce(tensor, **kwargs):
        tensor.mul_(2)
        return PendingWork()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    evidence = BackwardExecutionEvidence()
    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=2,
        max_bytes=1 << 20,
    )

    reducer.mark_ready(0)
    assert evidence.gradient_collective_count == 0
    reducer.mark_ready(1)
    assert evidence.gradient_collective_count == 1
    assert evidence.overlapped_gradient_collective_count == 1
    assert waits == []

    reducer.mark_ready(2)
    reducer.finish()

    assert waits == [True, True]
    assert [gradient.item() for gradient in gradients] == [2.0, 4.0, 6.0]
    assert evidence.async_gradient_collective_count == 2
    assert evidence.communication_bytes == 12
    assert evidence.gradient_collective_count == 2
    assert evidence.gradient_collective_bytes == 12


def test_synchronous_gradient_reduction_has_no_overlap_evidence(monkeypatch):
    calls = []

    def fake_all_reduce(tensor, **kwargs):
        calls.append(kwargs["async_op"])
        tensor.mul_(2)
        return

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [torch.tensor(1.0), torch.tensor(2.0)]
    evidence = BackwardExecutionEvidence()
    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=2,
        max_bytes=1 << 20,
        async_op=False,
    )
    reducer.mark_ready(0)
    reducer.mark_ready(1)
    reducer.finish()

    assert calls == [False]
    assert [gradient.item() for gradient in gradients] == [2.0, 4.0]
    assert evidence.async_gradient_collective_count == 0
    assert evidence.overlapped_gradient_collective_count == 0


def test_multi_layer_multi_parameter_gradients_match_dense_autograd():
    theta = torch.tensor(0.23, requires_grad=True)
    phi = torch.tensor(-0.37, requires_grad=True)
    circuit = fq.Circuit(3).h(0).ry(2, theta).rxx(0, 2, phi).rz(1, theta)
    dense = circuit.expectation_z(2)
    dense_grad = torch.autograd.grad(dense, (theta, phi), retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(circuit, observable_qubit=2)
    pending = result.summary()
    assert pending["parameter_gradient_ready"] is False
    assert pending["backward_status"] == "pending"
    assert pending["gradient_reduction"] == "pending"
    assert pending["backward_distribution_semantics"] == "pending"
    assert pending["local_world_size"] == 1
    assert pending["node_count"] == 1
    assert pending["layout_optimization_target"] == "training_step"
    assert len(pending["logical_to_physical_wires"]) == 3
    assert pending["local_state_bytes"] > 0
    assert pending["backward_communication_count"] == 0
    assert pending["backward_communication_bytes"] == 0
    assert pending["peak_backward_scratch_bytes"] == 0
    assert all(item["owner_rank"] is None for item in pending["parameter_ownership"])
    assert all(
        not item["participating_ranks"] for item in pending["parameter_ownership"]
    )
    result.backward()
    assert float(theta.grad) == pytest.approx(float(dense_grad[0]), abs=2e-5)
    assert float(phi.grad) == pytest.approx(float(dense_grad[1]), abs=2e-5)
    assert result.ownership[0].occurrence_count == 2
    completed = result.summary()
    assert completed["backward_uses_full_state_replay"] is False
    assert completed["saved_forward_state_reused"] is True
    assert completed["checkpoint_policy"]["strategy"] == "reversible_adjoint"
    assert (
        completed["checkpoint_policy"]["selection_reason"]
        == "forward_state_reuse_fits_budget"
    )
    assert completed["parameter_gradient_ready"] is True
    assert completed["backward_status"] == "completed"
    assert completed["gradient_distribution"] == "local"
    assert completed["backward_distribution_semantics"] == "single_device_fast_path"
    assert completed["peak_backward_scratch_bytes"] > 0
    assert completed["parameter_ownership"][0]["participating_ranks"] == (0,)
    dispatch = completed["kernel_dispatch"]
    assert dispatch["triton_execution_count"] == 0
    assert dispatch["pytorch_fallback_count"] == 3
    assert {
        (record["feature"], record["reason"]) for record in dispatch["decisions"]
    } == {
        ("local_reversible_vjp", "disabled_by_policy"),
        ("vjp_adjoint", "disabled_by_policy"),
    }


def test_parameter_shift_and_finite_difference_match_native_vjp():
    theta = torch.tensor(0.41, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, theta)
    result = execute_torch_distributed_statevector_reverse(circuit)
    result.backward()
    parameter_shift = (math.cos(0.41 + math.pi / 2) - math.cos(0.41 - math.pi / 2)) / 2
    epsilon = 1e-4
    finite_difference = (math.cos(0.41 + epsilon) - math.cos(0.41 - epsilon)) / (
        2 * epsilon
    )
    assert float(theta.grad) == pytest.approx(parameter_shift, abs=1e-5)
    assert float(theta.grad) == pytest.approx(finite_difference, abs=2e-4)


def test_multi_z_sum_uses_one_forward_and_one_combined_adjoint(monkeypatch):
    import flagquantum.runtime.executors.statevector.reverse as reverse

    calls = 0
    execute_forward = reverse.execute_torch_distributed_statevector

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return execute_forward(*args, **kwargs)

    monkeypatch.setattr(
        reverse,
        "execute_torch_distributed_statevector",
        counted_forward,
    )
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128).rx(0, theta).ry(1, phi).cx(0, 1)
    dense = circuit.expectation_z(0) + circuit.expectation_z(1)
    expected = torch.autograd.grad(dense, (theta, phi), retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubits=(0, 1),
    )
    result.backward()

    assert calls == 1
    torch.testing.assert_close(result.value, dense.squeeze(), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(theta.grad, expected[0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(phi.grad, expected[1], atol=1e-10, rtol=1e-10)
    summary = result.summary()
    assert summary["gradient_method"] == "statevector_adjoint"
    assert summary["observable_profile"] == "sum_of_single_wire_pauli_z_terms"
    assert summary["observable_wires"] == (0, 1)
    assert summary["backward_distribution_semantics"] == "single_device_fast_path"


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_weighted_z_zz_hamiltonian_matches_dense_autograd(real_dtype):
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    theta = torch.tensor(0.23, dtype=real_dtype, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=real_dtype, requires_grad=True)
    circuit = (
        fq.Circuit(3, dtype=complex_dtype)
        .rx(0, theta)
        .ry(1, phi)
        .cx(0, 2)
        .rxx(1, 2, theta)
    )
    dense = (
        0.7 * circuit.expectation_ps(z=(0, 2))
        + 0.2 * circuit.expectation_z(1)
        - 0.3 * circuit.expectation_ps(z=(1, 2))
    ).squeeze()
    expected = torch.autograd.grad(dense, (theta, phi), retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_terms=((0.7, (0, 2)), (0.2, (1,)), (-0.3, (1, 2))),
    )
    result.backward()

    tolerance = 2e-5 if real_dtype == torch.float32 else 1e-10
    torch.testing.assert_close(result.value, dense, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(theta.grad, expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(phi.grad, expected[1], atol=tolerance, rtol=tolerance)
    summary = result.summary()
    assert summary["observable_profile"] == "weighted_z_zz_hamiltonian"
    assert summary["observable_wires"] == (0, 1, 2)
    assert summary["observable_terms"] == (
        (0.7, (0, 2)),
        (0.2, (1,)),
        (-0.3, (1, 2)),
    )


@pytest.mark.parametrize(
    ("terms", "error", "match"),
    [
        ((), ValueError, "at least one"),
        (((1.0, (0, 0)),), ValueError, "repeat"),
        (((1.0, (0, 1, 2)),), ValueError, "only Z and ZZ"),
        (((float("inf"), (0,)),), ValueError, "finite"),
        (((1.0, (0.5,)),), TypeError, "integers"),
    ],
)
def test_weighted_observable_validation(terms, error, match):
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = fq.Circuit(3).rx(0, theta)

    with pytest.raises(error, match=match):
        execute_torch_distributed_statevector_reverse(circuit, observable_terms=terms)


def test_multi_z_sum_rejects_empty_or_invalid_wire_sets():
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = fq.Circuit(1).rx(0, theta)

    with pytest.raises(ValueError, match="at least one"):
        execute_torch_distributed_statevector_reverse(circuit, observable_qubits=())
    with pytest.raises(TypeError, match="integers"):
        execute_torch_distributed_statevector_reverse(
            circuit,
            observable_qubits=(0.5,),
        )
    with pytest.raises(ValueError, match="outside"):
        execute_torch_distributed_statevector_reverse(circuit, observable_qubits=(1,))


def test_checkpoint_policy_is_versioned_and_fail_closed():
    assert StatevectorCheckpointPolicy().strategy == "auto"
    assert StatevectorCheckpointPolicy().version == "statevector_checkpoint_v5"
    assert StatevectorCheckpointPolicy(strategy="interval", interval=8).interval == 8
    with pytest.raises(ValueError, match="interval > 0"):
        StatevectorCheckpointPolicy(strategy="interval", interval=0)


@pytest.mark.parametrize("strategy", ("full_rematerialization", "interval"))
def test_rematerialization_skips_reversible_cx_optimization(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    monkeypatch.setenv("FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "1")
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.gate("ry", 0, theta=theta)
    circuit.gate("cx", (0, 1))
    circuit.gate("cx", (1, 2))
    expected = circuit.expectation_z(2)
    (expected_gradient,) = torch.autograd.grad(expected, theta)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=2,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy=strategy, interval=1),
    )
    result.backward()

    torch.testing.assert_close(result.value, expected.squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    assert result.summary()["checkpoint_policy"]["strategy"] == strategy


def test_auto_checkpoint_reuses_forward_state_when_working_set_fits():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4096),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "reversible_adjoint"
    assert policy.selection_reason == "forward_state_reuse_fits_budget"
    assert policy.estimated_required_bytes == 4096


def test_auto_checkpoint_fails_over_to_rematerialization_when_budget_is_tight():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4095),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "full_rematerialization"
    assert policy.selection_reason == "checkpoint_working_set_exceeds_budget"
    assert policy.estimated_required_bytes == 3072
    assert policy.estimated_reversible_bytes == 4096


def test_auto_checkpoint_selects_budgeted_blocks_for_cpu_cx_workload():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4096),
        local_state_bytes=1024,
        device=torch.device("cpu"),
        instruction_count=10,
        contains_local_cx=True,
    )

    assert policy.strategy == "interval"
    assert policy.interval == 5
    assert policy.selection_reason == "budgeted_block_checkpoints"
    assert policy.estimated_checkpoint_count == 2
    assert policy.estimated_required_bytes == 4096
    assert policy.estimated_reversible_bytes == 5120


def test_auto_checkpoint_selects_low_memory_reversible_cpu_cx_workload():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4096),
        local_state_bytes=1024,
        device=torch.device("cpu"),
        instruction_count=10,
        contains_local_cx=True,
        low_memory_cpu_cx_available=True,
        compact_cpu_cx_auxiliary_bytes=512,
    )

    assert policy.strategy == "reversible_adjoint"
    assert policy.selection_reason == "low_memory_cpu_cx_reversible_fits_budget"
    assert policy.estimated_required_bytes == 4096
    assert policy.estimated_reversible_bytes == 4096
    assert policy.estimated_fused_reversible_bytes == 5120
    assert policy.estimated_compact_reversible_bytes == 4608
    assert policy.low_memory_cpu_cx
    assert not policy.compact_cpu_cx_cycles


def test_auto_checkpoint_selects_budgeted_compact_cpu_cx_cycles():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4608),
        local_state_bytes=1024,
        device=torch.device("cpu"),
        instruction_count=10,
        contains_local_cx=True,
        low_memory_cpu_cx_available=True,
        compact_cpu_cx_auxiliary_bytes=512,
    )

    assert policy.strategy == "reversible_adjoint"
    assert policy.selection_reason == "compact_cpu_cx_cycles_reversible_fits_budget"
    assert policy.estimated_required_bytes == 4608
    assert policy.estimated_reversible_bytes == 4608
    assert policy.estimated_compact_reversible_bytes == 4608
    assert policy.estimated_fused_reversible_bytes == 5120
    assert policy.low_memory_cpu_cx
    assert policy.compact_cpu_cx_cycles


@pytest.mark.parametrize(
    ("budget", "expected_strategy", "expected_checkpoints"),
    [
        (3 * 1024, "full_rematerialization", 1),
        (4 * 1024, "interval", 2),
        (5 * 1024, "reversible_adjoint", 0),
    ],
)
def test_auto_checkpoint_respects_cpu_cx_budget_boundaries(
    budget, expected_strategy, expected_checkpoints
):
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=budget),
        local_state_bytes=1024,
        device=torch.device("cpu"),
        instruction_count=100,
        contains_local_cx=True,
    )

    assert policy.strategy == expected_strategy
    assert policy.estimated_checkpoint_count == expected_checkpoints
    assert policy.estimated_required_bytes <= budget


def test_budgeted_block_checkpoints_reduce_replay_and_preserve_gradient(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_ADJOINT_INPLACE", "0")
    theta = torch.tensor([0.17, -0.23, 0.31], dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.ry(0, theta[0]).rx(1, theta[1]).rz(2, theta[2])
    circuit.cx(0, 1).cx(1, 2)
    expected = circuit.expectation_z(2)
    (expected_gradient,) = torch.autograd.grad(expected, theta)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=2,
        checkpoint_policy=StatevectorCheckpointPolicy(memory_budget_bytes=512),
    )
    result.backward()
    summary = result.summary()

    torch.testing.assert_close(result.value, expected.squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    assert summary["checkpoint_policy"]["strategy"] == "interval"
    assert summary["checkpoint_policy"]["interval"] == 3
    assert summary["backward_checkpoint_count"] == 2
    assert summary["backward_rematerialized_gate_count"] == 9


def test_low_memory_cpu_cx_reversible_path_preserves_gradient():
    if not native_cpu_cx_adjoint_inplace_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor([0.17, -0.23, 0.31], dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.ry(0, theta[0]).rx(1, theta[1]).rz(2, theta[2])
    circuit.cx(0, 1).cx(1, 2)
    expected = circuit.expectation_z(2)
    (expected_gradient,) = torch.autograd.grad(expected, theta)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=2,
        checkpoint_policy=StatevectorCheckpointPolicy(memory_budget_bytes=512),
    )
    result.backward()
    summary = result.summary()

    torch.testing.assert_close(result.value, expected.squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    assert summary["checkpoint_policy"]["strategy"] == "reversible_adjoint"
    assert summary["checkpoint_policy"]["low_memory_cpu_cx"]
    assert not summary["checkpoint_policy"]["compact_cpu_cx_cycles"]
    assert summary["backward_rematerialized_gate_count"] == 0


def test_budgeted_compact_cpu_cx_cycles_preserve_gradient():
    if not native_cpu_cx_adjoint_inplace_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor([0.17, -0.23, 0.31], dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.ry(0, theta[0]).rx(1, theta[1]).rz(3, theta[2])
    for control, target in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 2),
        (2, 1),
        (1, 3),
        (3, 2),
    ):
        circuit.cx(control, target)
    expected = circuit.expectation_z(3)
    (expected_gradient,) = torch.autograd.grad(expected, theta)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=3,
        checkpoint_policy=StatevectorCheckpointPolicy(memory_budget_bytes=1168),
    )
    result.backward()
    summary = result.summary()

    torch.testing.assert_close(result.value, expected.squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    assert summary["checkpoint_policy"]["strategy"] == "reversible_adjoint"
    assert summary["checkpoint_policy"]["compact_cpu_cx_cycles"]
    assert summary["checkpoint_policy"]["estimated_required_bytes"] == 1168
    assert summary["peak_backward_scratch_bytes"] == 144
    assert summary["backward_rematerialized_gate_count"] == 0


def test_auto_checkpoint_uses_platform_memory_snapshot(monkeypatch):
    platform = SimpleNamespace(
        is_available=lambda: True,
        memory_snapshot=lambda device: SimpleNamespace(free_bytes=10_000),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.checkpointing.get_platform_runtime",
        lambda device_type: platform,
    )

    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(),
        local_state_bytes=2_000,
        device=torch.device("cuda:0"),
    )

    assert policy.memory_budget_bytes == 7_000
    assert policy.strategy == "full_rematerialization"


def test_auto_checkpoint_uses_host_memory_snapshot_at_24_qubits(monkeypatch):
    available = 2 << 30
    platform = SimpleNamespace(
        is_available=lambda: True,
        memory_snapshot=lambda device: SimpleNamespace(free_bytes=available),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.checkpointing.get_platform_runtime",
        lambda device_type: platform,
    )

    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(),
        local_state_bytes=256 << 20,
        device=torch.device("cpu"),
        instruction_count=95,
        contains_local_cx=True,
    )

    assert policy.memory_budget_bytes == int(available * 0.7)
    assert policy.strategy == "reversible_adjoint"
    assert policy.estimated_required_bytes == 5 * (256 << 20)


def test_auto_checkpoint_keeps_24_qubit_host_headroom(monkeypatch):
    available = 1 << 30
    platform = SimpleNamespace(
        is_available=lambda: True,
        memory_snapshot=lambda device: SimpleNamespace(free_bytes=available),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.checkpointing.get_platform_runtime",
        lambda device_type: platform,
    )

    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(),
        local_state_bytes=256 << 20,
        device=torch.device("cpu"),
        instruction_count=95,
        contains_local_cx=True,
    )

    assert policy.memory_budget_bytes == int(available * 0.7)
    assert policy.strategy == "full_rematerialization"


def test_auto_checkpoint_handles_exhausted_platform_memory(monkeypatch):
    platform = SimpleNamespace(
        is_available=lambda: True,
        memory_snapshot=lambda device: SimpleNamespace(free_bytes=0),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.checkpointing.get_platform_runtime",
        lambda device_type: platform,
    )

    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.memory_budget_bytes == 1
    assert policy.strategy == "full_rematerialization"


def test_explicit_checkpoint_strategy_has_priority_over_auto_budget():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(
            strategy="full_rematerialization", memory_budget_bytes=1
        ),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "full_rematerialization"
    assert policy.selection_reason == "explicit_strategy"


def test_reverse_exchange_chunk_is_configurable_and_audited(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES", str(1 << 20))
    theta = torch.tensor(0.19, requires_grad=True)
    result = execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta))
    result.backward()
    summary = result.summary()
    assert summary["backward_exchange_chunk_amplitudes"] == 1 << 20
    assert summary["backward_exchange_chunk_bytes"] == 8 << 20

    monkeypatch.setenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES", "1000")
    with pytest.raises(ValueError, match="positive power of two"):
        execute_torch_distributed_statevector_reverse(
            fq.Circuit(1).ry(0, torch.tensor(0.2, requires_grad=True))
        )


def test_reversible_adjoint_matches_dense_autograd_on_all_active_wires():
    parameters = tuple(
        torch.tensor(0.11 + 0.07 * wire, requires_grad=True) for wire in range(4)
    )
    circuit = fq.Circuit(4)
    for wire, parameter in enumerate(parameters):
        circuit.ry(wire, parameter)
    for wire in range(3):
        circuit.cx(wire, wire + 1)
    circuit.cx(3, 0)
    dense = circuit.expectation_z(2)
    expected = torch.autograd.grad(dense, parameters, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=2,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"),
    )
    result.backward()

    torch.testing.assert_close(result.value, dense.squeeze(), atol=2e-5, rtol=2e-5)
    for parameter, reference in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, reference, atol=3e-5, rtol=3e-5)
    assert result.summary()["saved_forward_state_reused"] is True


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reverse_local_cnot_uses_cataloged_triton_route(monkeypatch):
    import flagquantum.runtime.executors.statevector.reverse_adjoint_kernels as kernels

    calls = 0
    original = kernels._vectorized_local_cx_gate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels, "_vectorized_local_cx_gate", counted)
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_CX", "1")
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(2).ry(0, theta).cx(0, 1)
    dense = circuit.expectation_z(1)
    (expected_gradient,) = torch.autograd.grad(dense, theta, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=1,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"),
        device="cuda",
    )
    result.backward()

    assert calls > 0
    torch.testing.assert_close(result.value.cpu(), dense.detach().squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reverse_local_adjoint_vjp_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setenv("FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "0")
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, theta)
    dense = circuit.expectation_z(0)
    (expected_gradient,) = torch.autograd.grad(dense, theta, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=0,
        device="cuda",
    )
    result.backward()

    torch.testing.assert_close(result.value.cpu(), dense.detach().squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    record = next(
        record
        for record in result.summary()["kernel_dispatch"]["decisions"]
        if record["feature"] == "local_adjoint_vjp"
    )
    route = record["kernel_route"]
    assert record["selected"] == "triton"
    assert record["count"] == 1
    assert route["semantic_id"] == "gradient.vjp.adjoint_1q.local"
    assert route["implementation_id"] == "FQKI-TRITON-GR-001-A"
    assert route["catalog_mismatches"] == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reverse_local_reversible_vjp_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "1")
    monkeypatch.setenv("FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "1")
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, theta)
    dense = circuit.expectation_z(0)
    (expected_gradient,) = torch.autograd.grad(dense, theta, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=0,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"),
        device="cuda",
    )
    result.backward()

    torch.testing.assert_close(result.value.cpu(), dense.detach().squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    record = next(
        record
        for record in result.summary()["kernel_dispatch"]["decisions"]
        if record["feature"] == "local_reversible_vjp"
    )
    route = record["kernel_route"]
    assert record["selected"] == "triton"
    assert record["count"] == 1
    assert route["semantic_id"] == "gradient.vjp.reversible_1q.local"
    assert route["implementation_id"] == "FQKI-TRITON-GR-002-A"
    assert route["catalog_mismatches"] == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reverse_local_cnot_sequence_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(3).ry(0, theta).cx(0, 1).cx(1, 2)
    dense = circuit.expectation_z(2)
    (expected_gradient,) = torch.autograd.grad(dense, theta, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_qubit=2,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"),
        device="cuda",
    )
    result.backward()

    torch.testing.assert_close(result.value.cpu(), dense.detach().squeeze())
    torch.testing.assert_close(theta.grad, expected_gradient)
    record = next(
        record
        for record in result.summary()["kernel_dispatch"]["decisions"]
        if record["feature"] == "local_cx_segment"
    )
    route = record["kernel_route"]
    assert record["count"] == 2
    assert route["semantic_id"] == "statevector.apply.cnot_sequence.local"
    assert route["implementation_id"] == "FQKI-TRITON-SV-003-A"


def test_reverse_executor_import_does_not_initialize_jax():
    import subprocess
    import sys

    code = (
        "import sys; import flagquantum.runtime.executors.statevector.reverse; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_custom_autograd_boundary_passes_double_precision_gradcheck():
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)

    def expectation(value):
        circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, value)
        return execute_torch_distributed_statevector_reverse(circuit).value

    assert torch.autograd.gradcheck(
        expectation, (theta,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_reverse_memory_debug_uses_active_platform(monkeypatch, capsys):
    monkeypatch.setenv("FQ_STATEVECTOR_DEBUG_MEMORY", "1")
    theta = torch.tensor(0.19, requires_grad=True)
    execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta)).backward()
    assert "reversible_adjoint_initialized" in capsys.readouterr().out


def test_backward_failure_is_reported_and_never_marks_gradient_ready(monkeypatch):
    import flagquantum.runtime.executors.statevector.reverse_adjoint as reverse_adjoint

    theta = torch.tensor(0.19, requires_grad=True)
    result = execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta))

    def fail(*args, **kwargs):
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(reverse_adjoint, "_explicit_sharded_adjoint", fail)
    with pytest.raises(RuntimeError, match="injected backward failure"):
        result.backward()
    summary = result.summary()
    assert summary["backward_status"] == "failed"
    assert summary["backward_distribution_semantics"] == "failed"
    assert summary["parameter_gradient_ready"] is False
    assert "injected backward failure" in summary["backward_error"]
    assert result.ownership[0].distribution == "failed"


class _SimulatedWorld:
    """A whole process group simulated inside one test process.

    The real collective is NCCL's; what these tests can check is the packing the
    collective is handed and what is done with the block it returns. So every
    rank's own packed input is built here, the block that ``rank`` would receive
    is computed by summing those inputs, and the reducer is then given that
    block exactly as a real reduce-scatter would deliver it.
    """

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        self.segments: list[int] = []
        self.inputs: list[torch.Tensor] = []
        self.calls = 0

    def install(self, monkeypatch) -> None:
        def get_world_size(group=None):
            return self.world_size

        def get_rank(group=None):
            return self.rank

        def reduce_scatter_tensor(output, tensor, **kwargs):
            self.calls += 1
            segment = tensor.numel() // self.world_size
            self.segments.append(segment)
            self.inputs.append(tensor.detach().clone())
            output.copy_(
                tensor.detach()[self.rank * segment : (self.rank + 1) * segment]
            )

        monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
        monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
        monkeypatch.setattr(
            torch.distributed, "reduce_scatter_tensor", reduce_scatter_tensor
        )


def _sharded_reducer(gradients, owners, *, evidence=None, rank=0, world_size=2):
    return AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence if evidence is not None else BackwardExecutionEvidence(),
        owners=owners,
        async_op=True,
        max_parameters=len(gradients),
        max_bytes=1 << 20,
    )


def test_owner_sharded_reduction_packs_each_owner_block_contiguously(monkeypatch):
    world = _SimulatedWorld(world_size=2, rank=0)
    world.install(monkeypatch)
    gradients = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0]),
        torch.tensor([4.0, 5.0]),
    ]
    evidence = BackwardExecutionEvidence(
        ownership=[
            SimpleNamespace(
                owner_rank=None, reduction="pending", distribution="pending"
            )
            for _ in gradients
        ]
    )
    reducer = _sharded_reducer(gradients, [0, 1, 1], evidence=evidence)

    for index in range(len(gradients)):
        reducer.mark_ready(index)
    reducer.finish()

    # Owner 0 holds two amplitudes, owner 1 holds three, so the widest block
    # sets the segment and owner 0's lane is padded with zeros.
    assert world.calls == 1
    assert world.segments == [3]
    assert world.inputs[0].tolist() == [1.0, 2.0, 0.0, 3.0, 4.0, 5.0]
    # Rank 0 owns only parameter 0, so only its block survives here.
    assert [gradient.tolist() for gradient in gradients] == [
        [1.0, 2.0],
        [0.0],
        [0.0, 0.0],
    ]
    assert evidence.gradient_reduction_scheme == "owner_sharded_reduce_scatter"
    assert evidence.gradient_collective_count == 1
    assert evidence.gradient_collective_bytes == 6 * 4


def test_owner_sharded_reduction_delivers_the_other_ranks_block(monkeypatch):
    world = _SimulatedWorld(world_size=2, rank=1)
    world.install(monkeypatch)
    gradients = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0]),
        torch.tensor([4.0, 5.0]),
    ]
    reducer = _sharded_reducer(gradients, [0, 1, 1])

    for index in range(len(gradients)):
        reducer.mark_ready(index)
    reducer.finish()

    assert [gradient.tolist() for gradient in gradients] == [
        [0.0, 0.0],
        [3.0],
        [4.0, 5.0],
    ]


def test_owner_sharded_reduction_groups_separate_dtypes_into_separate_collectives(
    monkeypatch,
):
    world = _SimulatedWorld(world_size=2, rank=0)
    world.install(monkeypatch)
    gradients = [
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float64),
    ]
    reducer = _sharded_reducer(gradients, [0, 1])

    for index in range(len(gradients)):
        reducer.mark_ready(index)
    reducer.finish()

    assert world.calls == 2
    assert [item.dtype for item in world.inputs] == [torch.float32, torch.float64]


def test_owner_sharded_reduction_rejects_a_mismatched_owner_count():
    gradients = [torch.tensor(1.0), torch.tensor(2.0)]

    with pytest.raises(ValueError, match="one owner per gradient"):
        AsyncGradientReducer(
            gradients,
            process_group=None,
            evidence=BackwardExecutionEvidence(),
            owners=[0],
        )


def test_reducer_without_owners_keeps_the_replicated_all_reduce(monkeypatch):
    calls = []

    def fake_all_reduce(tensor, **kwargs):
        calls.append(tensor.detach().clone())
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [torch.tensor([1.0]), torch.tensor([2.0])]
    evidence = BackwardExecutionEvidence()
    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=2,
        max_bytes=1 << 20,
        async_op=False,
    )

    for index in range(len(gradients)):
        reducer.mark_ready(index)
    reducer.finish()

    assert len(calls) == 1
    assert [gradient.tolist() for gradient in gradients] == [[2.0], [4.0]]
    assert evidence.gradient_reduction_scheme == "replicated_all_reduce"
