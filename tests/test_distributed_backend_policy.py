import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb
from flagquantum.errors import ExecutionError
from flagquantum.runtime.distributed import (
    LocalTensor,
    resolve_distributed_backend_policy,
)
from flagquantum.runtime.execution import run_advanced
from flagquantum.runtime.executors.statevector import (
    simulate_distributed_statevector_local,
)
from flagquantum.runtime.executors.statevector.models import (
    LocalDistributedStatevectorResult,
)
from flagquantum.runtime.parity import (
    local_distributed_development_preflight,
    require_development_production_parity,
    validate_development_production_parity,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.distributed,
    pytest.mark.distributed_cpu,
]


def test_distributed_backend_policy_defaults_to_development_without_torchrun(
    monkeypatch,
):
    monkeypatch.delenv("FQ_DISTRIBUTED_PROFILE", raising=False)
    monkeypatch.delenv("FQ_JAX_DISTRIBUTED_BACKEND", raising=False)
    monkeypatch.delenv("FQ_TORCH_DISTRIBUTED_BACKEND", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    policy = resolve_distributed_backend_policy()

    assert policy.profile == "development"
    assert policy.jax_backend == "pmap_local_cpu"
    assert policy.torch_backend == "local_tensor"
    assert policy.requires_torchrun is False
    assert policy.requires_gpu is False


def test_distributed_backend_policy_uses_env_for_production(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "production")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "8")

    policy = resolve_distributed_backend_policy()

    assert policy.profile == "production"
    assert policy.jax_backend == "pmap"
    assert policy.torch_backend == "torch_distributed"
    assert policy.local_world_size == 8
    assert policy.requires_torchrun is True
    assert policy.requires_gpu is True


def test_distributed_backend_policy_allows_explicit_backend_overrides(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_JAX_DISTRIBUTED_BACKEND", "pmap_local_cpu")
    monkeypatch.setenv("FQ_TORCH_DISTRIBUTED_BACKEND", "local_tensor")

    policy = resolve_distributed_backend_policy()

    assert policy.summary()["source"]["FQ_DISTRIBUTED_PROFILE"] == "development"
    assert policy.jax_backend == "pmap_local_cpu"
    assert policy.torch_backend == "local_tensor"


def test_local_tensor_simulates_rank_sharded_pytorch_tensor():
    tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    local = LocalTensor.from_tensor(tensor, world_size=3, dim=0)
    doubled = local.map_shards(lambda shard, rank: shard + rank)

    expected = torch.cat(
        [
            tensor[0:2] + 0,
            tensor[2:4] + 1,
            tensor[4:6] + 2,
        ],
        dim=0,
    )
    assert local.world_size == 3
    assert torch.equal(local.full_tensor(), tensor)
    assert torch.equal(doubled.full_tensor(), expected)
    assert local.summary()["backend"] == "local_tensor"


def test_local_distributed_statevector_summary_uses_development_policy(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(3)
    circuit.h(0).x(2)

    result = simulate_distributed_statevector_local(circuit, world_size=2)
    summary = result.summary()

    assert summary["distributed_backend_policy"]["profile"] == "development"
    assert summary["distributed_backend_policy"]["jax_backend"] == "pmap_local_cpu"
    assert summary["distributed_backend_policy"]["torch_backend"] == "local_tensor"


def test_circuit_run_development_distributed_statevector_is_transparent(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(3)
    circuit.h(0).x(2).cx(0, 2)

    result = run_advanced(
        circuit,
        mode="distributed_statevector",
        world_size=2,
        device="cpu",
    )
    plan = result.plan

    assert isinstance(result, fq.ExecutionResult)
    assert result.compatibility["source_type"] == "LocalDistributedStatevectorResult"
    assert result.summary()["distributed_backend_policy"]["profile"] == "development"
    assert result.plan.world_size == 2
    assert plan.recommended_mode == "distributed_statevector"
    assert torch.allclose(result.state, circuit.state(), atol=1e-6)


def test_distributed_statevector_uses_one_explicit_complex128_precision(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(2, dtype=torch.complex64).ry(0, theta=0.371).cx(0, 1)

    result = run_advanced(
        circuit,
        mode="distributed_statevector",
        world_size=2,
        device="cpu",
        dtype=torch.complex128,
    )

    assert result.state is not None
    assert result.state.dtype == torch.complex128
    assert result.plan.runtime_config["complex_dtype"] == "complex128"
    assert result.plan.state_bytes == 4 * torch.complex128.itemsize


def test_distributed_statevector_uses_env_local_world_size_without_code_change(
    monkeypatch,
):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "3")
    circuit = fq.Circuit(3)
    circuit.h(0).x(2)

    result = run_advanced(
        circuit,
        mode="distributed_statevector",
        device="cpu",
    )
    plan = result.plan

    assert isinstance(result, fq.ExecutionResult)
    assert result.compatibility["source_type"] == "LocalDistributedStatevectorResult"
    assert result.plan.world_size == 3
    assert plan.world_size == 3
    assert result.summary()["world_size"] == 3
    assert torch.allclose(result.state, circuit.state(), atol=1e-6)


def test_circuit_run_development_distributed_statevector_measure(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    measured, plan = fqb.run_native(
        circuit,
        mode="distributed_statevector",
        world_size=2,
        device="cpu",
        measure=True,
        return_plan=True,
    )

    assert plan.recommended_mode == "distributed_statevector"
    assert measured.shape == (1, 2)
    assert torch.allclose(measured, circuit.expectation_z(), atol=1e-6)


def test_circuit_run_single_rank_uses_statevector_backend_result(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = run_advanced(
        circuit,
        mode="distributed_statevector",
        world_size=1,
        device="cpu",
    )
    qdev = result.native()
    plan = result.plan

    assert isinstance(qdev, LocalDistributedStatevectorResult)
    assert qdev.backend_policy.profile == "development"
    assert torch.allclose(qdev.state, circuit.state(), atol=1e-6)
    assert plan.recommended_mode == "local"


def test_distributed_statevector_backend_override_is_consumed_before_device(
    monkeypatch,
):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = run_advanced(
        circuit,
        mode="distributed_statevector",
        world_size=1,
        device="cpu",
        torch_backend="local_tensor",
        jax_backend="pmap_local_cpu",
    )
    qdev = result.native()
    plan = result.plan

    assert qdev.backend_policy.torch_backend == "local_tensor"
    assert qdev.backend_policy.jax_backend == "pmap_local_cpu"
    assert qdev.plan.world_size == 1
    assert plan.world_size == 1


def test_production_statevector_uses_torch_distributed_executor(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "production")
    circuit = fq.Circuit(1).x(0)
    sentinel = object()
    captured = {}

    def execute(program, **options):
        captured["program"] = program
        captured["options"] = options
        return sentinel

    monkeypatch.setattr(
        "flagquantum.runtime.execution.execute_torch_distributed_statevector",
        execute,
    )

    result, plan = fqb.run_native(
        circuit,
        mode="distributed_statevector",
        world_size=1,
        device="cpu",
        return_plan=True,
    )

    assert result is sentinel
    assert captured["program"].n_wires == 1
    assert captured["options"]["device"] == "cpu"
    assert captured["options"]["dtype"] == torch.complex64
    assert plan.world_size == 1


def test_production_statevector_requires_initialized_process_group(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "production")

    with pytest.raises(ExecutionError, match="initialized torch.distributed"):
        fqb.run_native(
            fq.Circuit(2).h(0),
            mode="distributed_statevector",
            world_size=2,
            device="cpu",
        )


def test_distributed_mps_run_uses_backend_policy_transparently(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rx(2, theta=0.2)

    result = run_advanced(
        circuit,
        mode="distributed_mps",
        world_size=2,
        max_bond=4,
    )
    plan = result.plan
    summary = result.summary()

    assert plan.recommended_mode == "distributed_mps"
    assert summary["distributed_backend_policy"]["profile"] == "development"
    assert summary["distributed_backend_policy"]["torch_backend"] == "local_tensor"
    assert summary["executor"] == "local_tensor_development_simulator"
    assert summary["mps_execution"] == "local_tensor_site_sharded_simulator"
    assert summary["scalability_claim_allowed"] is False


def test_distributed_mps_uses_env_local_world_size_without_code_change(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "3")
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 1).rx(2, theta=0.2).cx(3, 4).rz(5, theta=-0.1)

    result = run_advanced(
        circuit,
        mode="distributed_mps",
        max_bond=4,
    )
    plan = result.plan
    summary = result.summary()

    assert plan.world_size == 3
    assert summary["world_size"] == 3
    assert summary["executor"] == "local_tensor_development_simulator"
    assert summary["local_tensor_wires_by_rank"] == {0: (0, 1), 1: (2, 3), 2: (4, 5)}
    assert torch.allclose(
        result.to_statevector(),
        fqb.run_mps(circuit, max_bond=4).to_statevector(),
        atol=1e-6,
    )


def test_distributed_tensor_network_run_uses_backend_policy_transparently(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rz(2, theta=0.4)

    result = run_advanced(
        circuit,
        mode="distributed_tensor_network",
        world_size=2,
        max_intermediate_size=8,
    )
    plan = result.plan
    summary = result.summary()

    assert plan.recommended_mode == "distributed_tensor_network"
    assert summary["distributed_backend_policy"]["profile"] == "development"
    assert summary["distributed_backend_policy"]["torch_backend"] == "local_tensor"
    assert summary["executor"] == "local_tensor_development_simulator"
    assert summary["communication_tiers"]["model"] == "local_simulated_all_reduce"
    assert summary["scalability_claim_allowed"] is False


def test_distributed_tensor_network_uses_env_local_world_size_without_code_change(
    monkeypatch,
):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "3")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rz(2, theta=0.4)

    result = run_advanced(
        circuit,
        mode="distributed_tensor_network",
        max_intermediate_size=8,
    )
    plan = result.plan
    summary = result.summary()

    assert plan.world_size == 3
    assert summary["world_size"] == 3
    assert set(summary["tasks_by_rank"]) == {0, 1, 2}
    assert summary["executor"] == "local_tensor_development_simulator"
    assert torch.allclose(
        result.to_statevector(),
        run_advanced(circuit, mode="tensor_network").to_statevector(),
        atol=1e-6,
    )


def test_explicit_local_modes_ignore_distributed_env_world_size(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "4")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rz(2, theta=0.4)

    state_plan = run_advanced(circuit, mode="statevector").plan
    mps_plan = run_advanced(circuit, mode="mps", max_bond=4).plan
    tn_plan = run_advanced(circuit, mode="tensor_network").plan

    assert state_plan.world_size == 1
    assert mps_plan.world_size == 1
    assert tn_plan.world_size == 1


def test_development_production_statevector_parity_contract():
    circuit = fq.Circuit(3)
    circuit.h(0).x(2).cx(0, 2).rz(1, theta=0.3)

    report = require_development_production_parity(
        circuit,
        mode="distributed_statevector",
        world_size=2,
        device="cpu",
        atol=1e-6,
    )
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["development_signature"] == summary["production_signature"]
    assert summary["numerical_reference_error"] <= 1e-6
    assert (
        summary["development_signature"]["distribution_semantics"]
        == "sharded_across_ranks"
    )


def test_development_production_mps_parity_contract():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).rx(2, theta=0.2).cx(2, 3)

    report = validate_development_production_parity(
        circuit,
        mode="distributed_mps",
        world_size=2,
        max_bond=8,
        atol=1e-6,
    )
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["development_signature"] == summary["production_signature"]
    assert summary["numerical_reference_error"] <= 1e-6


def test_development_production_tensor_network_parity_contract():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rz(2, theta=0.4)

    report = validate_development_production_parity(
        circuit,
        mode="distributed_tensor_network",
        world_size=2,
        max_intermediate_size=8,
        atol=1e-6,
    )
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["development_signature"] == summary["production_signature"]
    assert summary["numerical_reference_error"] <= 1e-6


def test_local_distributed_development_preflight_default_program():
    report = local_distributed_development_preflight(world_size=2)
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["development_policy"]["profile"] == "development"
    assert summary["production_policy"]["profile"] == "production"
    assert set(summary["parity_reports"]) == {
        "distributed_statevector",
        "distributed_mps",
        "distributed_tensor_network",
    }
    assert all(item["passed"] for item in summary["parity_reports"].values())


def test_local_distributed_development_preflight_accepts_custom_program():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).rz(2, theta=0.5)

    report = local_distributed_development_preflight(
        circuit,
        world_size=2,
        modes=("distributed_statevector",),
    )

    assert report.passed is True
    assert report.modes == ("distributed_statevector",)
    assert report.parity_reports["distributed_statevector"].passed is True
