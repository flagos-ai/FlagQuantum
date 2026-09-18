import os

import pytest
import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.execution import run_advanced

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_cpu,
    pytest.mark.distributed_launch,
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="requires torchrun with distributed RANK/WORLD_SIZE environment",
    ),
]


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _precision_assert(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6
) -> float:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    max_error = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
    assert torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=atol)
    assert max_error <= atol
    return float(max_error)


def _reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(4)
    circuit.h(0).cx(1, 2).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    return circuit


def _parameterized_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
    circuit = fq.Circuit(4, device=device)
    circuit.h(0)
    circuit.rx(1, theta=params[0])
    circuit.ry(2, theta=params[1])
    circuit.rz(3, theta=params[2])
    circuit.cx(0, 3)
    circuit.rzz(1, 2, theta=params[3])
    return circuit


def _z_loss_from_state(
    state: torch.Tensor, wires: tuple[int, ...] = (0, 2)
) -> torch.Tensor:
    flat = state.reshape(state.shape[0], -1)
    n_wires = int(torch.log2(torch.tensor(flat.shape[-1], dtype=torch.float32)).item())
    probs = torch.abs(flat) ** 2
    values = []
    for wire in wires:
        bit = (
            torch.arange(flat.shape[-1], device=flat.device) >> (n_wires - 1 - wire)
        ) & 1
        signs = (1 - 2 * bit).to(dtype=probs.dtype)
        values.append((probs * signs).sum(dim=-1))
    return torch.stack(values, dim=-1).sum()


def _training_loss(result: object) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return _z_loss_from_state(result)
    if hasattr(result, "expectation_z"):
        return result.expectation_z((0, 2)).sum()
    if hasattr(result, "to_statevector"):
        return _z_loss_from_state(result.to_statevector())
    return _z_loss_from_state(result.state())


def _loss_and_grad(
    mode: str,
    *,
    device: str,
    world_size: int,
    distributed_executor: str | None = None,
    aggregate_grad: bool = False,
    **options,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = torch.tensor(
        [0.17, -0.31, 0.23, -0.19],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    circuit = _parameterized_circuit(params, device=device)
    run_options = dict(options)
    if distributed_executor is not None:
        run_options["distributed_executor"] = distributed_executor
    if mode.startswith("distributed"):
        run_options["world_size"] = world_size
    result = run_advanced(circuit, mode=mode, device=device, **run_options)
    if mode == "distributed_tensor_network":
        loss = _z_loss_from_state(result.to_statevector())
    else:
        loss = _training_loss(result)
    if loss.requires_grad:
        loss.backward()
    if params.grad is None:
        params.grad = torch.zeros_like(params)
    if aggregate_grad and dist.is_available() and dist.is_initialized():
        dist.all_reduce(params.grad, op=dist.ReduceOp.SUM)
    return loss.detach().cpu(), params.grad.detach().cpu()


def _gradient_precision_assert(
    actual_loss: torch.Tensor,
    actual_grad: torch.Tensor,
    expected_loss: torch.Tensor,
    expected_grad: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> tuple[float, float]:
    loss_error = torch.max(torch.abs(actual_loss - expected_loss)).item()
    grad_error = torch.max(torch.abs(actual_grad - expected_grad)).item()
    assert torch.allclose(actual_loss, expected_loss, atol=atol, rtol=atol)
    assert torch.allclose(actual_grad, expected_grad, atol=atol, rtol=atol)
    assert loss_error <= atol
    assert grad_error <= atol
    return float(loss_error), float(grad_error)


def test_distributed_tensor_network_torchrun_precision_alignment():
    circuit = _reference_circuit()
    world_size = _world_size()
    device = _device()

    distributed = run_advanced(
        circuit,
        mode="distributed_tensor_network",
        world_size=world_size,
        distributed_executor="torch",
        device=device,
        # The complete four-qubit output itself is 16 elements, so a budget
        # of 8 cannot be satisfied by slicing the contraction at all. The
        # script this file mirrors has carried 16 since it was written.
        max_intermediate_size=16,
    )
    local = run_advanced(
        circuit,
        mode="tensor_network",
        device=device,
        # The complete four-qubit output itself is 16 elements, so a budget
        # of 8 cannot be satisfied by slicing the contraction at all. The
        # script this file mirrors has carried 16 since it was written.
        max_intermediate_size=16,
    )

    max_error = _precision_assert(
        distributed.native().state(), local.native().state(), atol=1e-6
    )
    summary = distributed.summary()
    assert summary["executor"] == "torch_distributed"
    assert summary["state_mode"] == "distributed_tensor_network"
    assert summary["world_size"] == world_size
    assert summary["rank"] == _rank()
    # The slice list lives on the native result; the wrapper's summary reports
    # a count of it, and comparing the two is what makes the summary
    # trustworthy.
    assert summary["slice_tasks"] == len(distributed.native().tasks)
    assert set(summary["tasks_by_rank"]) == set(range(world_size))
    assert max_error <= 1e-6


def test_distributed_statevector_torchrun_partitions_the_state_and_reports_its_plan():
    """Check the sharded statevector path where this runtime lets it be checked.

    This test used to compare a distributed gradient against a local one. It
    cannot: `full_state` on the production executor raises
    `FullStateMaterializationError` by design, because a distributed result is
    rank-local, and the shard-to-global index mapping is not published either —
    `global_indices` comes back empty and `amplitude_start` reports a contiguous
    range that the amplitudes do not follow. Comparing amplitudes would mean
    pinning an unpublished layout into a test.

    What is checkable is what the sibling script `statevector_correctness.py`
    checks, plus one thing it does not: that the ranks partition the state.
    """

    world_size = _world_size()
    device = _device()

    distributed = run_advanced(
        _reference_circuit(),
        mode="distributed_statevector",
        world_size=world_size,
        device=device,
    )
    summary = distributed.summary()
    native = distributed.native()

    assert summary["executor"] == "pytorch_native_distributed_statevector_v1"
    assert summary["distribution_semantics"] == "sharded_across_ranks"

    plan = native.plan
    report = plan.summary()
    assert report["world_size"] == world_size
    # Both ranks are on one host, so the plan must not claim the sharding spans
    # nodes and must not let that be read as a scalability claim.
    assert report["node_count"] == 1
    assert report["rank_address_bits"] > 0
    assert report["sharded_wires"] == (3,)
    assert report["release_gate_allowed"] is False
    assert report["scalability_claim_allowed"] is False
    assert len(report["rank_shards"]) == world_size
    # Every rank owns a non-empty shard, and the shards tile the whole state.
    assert min(shard["local_amplitudes"] for shard in report["rank_shards"]) > 0
    assert sum(shard["local_amplitudes"] for shard in report["rank_shards"]) == (
        plan.total_amplitudes
    )

    # The plan's dry run covers every event it traced and processes work on
    # every rank.
    dry_run = plan.execute_dry_run()
    assert dry_run.valid, dry_run.errors
    dry_run_report = dry_run.summary()
    assert dry_run_report["trace_event_count"] > 0
    assert dry_run_report["rank_count"] == world_size
    assert dry_run_report["total_local_bytes_processed"] > 0

    # The ranks together carry a normalized state, to the precision a rank-local
    # check can reach without materializing the whole thing on one rank.
    local_norm_squared = float(
        torch.linalg.vector_norm(native.shard_state.amplitudes.to(torch.complex128))
        .square()
        .item()
    )
    total = torch.tensor(local_norm_squared, dtype=torch.float64)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    reference_norm_squared = float(
        torch.linalg.vector_norm(
            _reference_circuit().state().reshape(-1).to(torch.complex128)
        )
        .square()
        .item()
    )
    assert float(total) == pytest.approx(reference_norm_squared, abs=1e-5)
    # No rank may hold the entire state, which is the property the executor
    # refuses to give up and the reason a local comparison is unavailable.
    assert native.shard_state.amplitudes.numel() < _reference_circuit().state().numel()


def test_distributed_tensor_network_torchrun_gradient_precision_alignment():
    world_size = _world_size()
    device = _device()

    distributed_loss, distributed_grad = _loss_and_grad(
        "distributed_tensor_network",
        device=device,
        world_size=world_size,
        distributed_executor="torch",
        aggregate_grad=True,
        # The complete four-qubit output itself is 16 elements, so a budget
        # of 8 cannot be satisfied by slicing the contraction at all. The
        # script this file mirrors has carried 16 since it was written.
        max_intermediate_size=16,
    )
    reference_loss, reference_grad = _loss_and_grad(
        "tensor_network",
        device=device,
        world_size=world_size,
        # The complete four-qubit output itself is 16 elements, so a budget
        # of 8 cannot be satisfied by slicing the contraction at all. The
        # script this file mirrors has carried 16 since it was written.
        max_intermediate_size=16,
    )

    _gradient_precision_assert(
        distributed_loss,
        distributed_grad,
        reference_loss,
        reference_grad,
        atol=1e-5,
    )


def test_distributed_mps_torchrun_precision_alignment():
    circuit = _reference_circuit()
    world_size = _world_size()
    device = _device()

    distributed = run_advanced(
        circuit,
        mode="distributed_mps",
        world_size=world_size,
        distributed_executor="torch",
        device=device,
        max_bond=8,
        boundary_transport="auto",
    )
    local = run_advanced(circuit, mode="mps", device=device, max_bond=8)

    # `sharded_state`, `local_shard_tensors`, and the MPS state are properties of
    # the native distributed result. `ExecutionResult` carries neither, and its
    # `to_statevector` cannot unwrap this one, so reaching for them directly on
    # the wrapper fails in two different ways.
    native = distributed.native()
    max_error = _precision_assert(
        native.sharded_state.to_statevector(), local.to_statevector(), atol=1e-6
    )
    summary = distributed.summary()
    assert summary["executor"] == "torch_distributed"
    assert summary["state_mode"] == "distributed_mps"
    assert summary["world_size"] == world_size
    assert summary["rank"] == _rank()
    assert summary["storage"] == "sharded"
    assert summary["mps_execution"] == "site_sharded_sync"
    assert native.local_shard_tensors
    assert native.sharded_state is not None
    assert max_error <= 1e-6


def test_distributed_mps_torchrun_gradient_precision_alignment():
    world_size = _world_size()
    device = _device()

    distributed_loss, distributed_grad = _loss_and_grad(
        "distributed_mps",
        device=device,
        world_size=world_size,
        distributed_executor="torch",
        max_bond=8,
        boundary_transport="auto",
    )
    reference_loss, reference_grad = _loss_and_grad(
        "mps",
        device=device,
        world_size=world_size,
        max_bond=8,
    )

    _gradient_precision_assert(
        distributed_loss,
        distributed_grad,
        reference_loss,
        reference_grad,
        atol=1e-5,
    )
