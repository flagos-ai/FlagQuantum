import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb

pytestmark = pytest.mark.unit


def test_run_target_uses_statevector_when_dense_path_fits() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)

    result = fqb.run_target(
        circuit,
        target="local_observables",
        observables=({"z": (0, 5)}, {"x": (2,)}),
        memory_limit_bytes=1 << 30,
    )

    assert result.backend == "statevector"
    assert torch.allclose(
        result.values,
        torch.stack(
            [
                circuit.expectation_ps(z=(0, 5)),
                circuit.expectation_ps(x=(2,)),
            ],
            dim=-1,
        ),
    )


def test_run_target_forced_tn_executes_sparse_kernel_not_full_state() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)
    targets = ("000000", "100001", "010100")
    indices = torch.tensor([int(bits, 2) for bits in targets])

    result = fqb.run_target(
        circuit,
        target="few_amplitudes",
        bitstrings=targets,
        mode="tensor_network",
    )

    assert result.backend == "tensor_network"
    assert torch.allclose(result.values, circuit.state()[:, indices], atol=1e-6)
    assert result.summary()["full_state_materialized"] is False


def test_run_target_distributed_tn_preserves_sparse_execution_evidence() -> None:
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2)

    result = fqb.run_target(
        circuit,
        target="local_observables",
        observables=({"z": (0, 5)}, {"x": (2,)}),
        mode="tensor_network",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )

    assert result.backend == "tensor_network"
    assert result.execution_summary["full_state_materialized"] is False
    assert result.execution_summary["shared_contraction"] is True


def test_run_target_routes_shallow_chain_to_sparse_mps_execution() -> None:
    circuit = fq.Circuit(20)
    for wire in range(19):
        circuit.cx(wire, wire + 1)
    targets = ("0" * 20, "1" * 20)

    result = fqb.run_target(
        circuit,
        target="few_amplitudes",
        bitstrings=targets,
        memory_limit_bytes=1 << 20,
    )

    assert result.backend == "mps"
    assert result.execution_summary["full_state_materialized"] is False
    assert result.execution_summary["max_bond_observed"] <= 2
    assert torch.allclose(
        result.values,
        circuit.state()[:, torch.tensor([0, 2**20 - 1])],
    )
