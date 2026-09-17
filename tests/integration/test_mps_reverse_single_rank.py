"""Single-rank rank-owned MPS reverse: exactness and the reported contract.

The multi-rank rank-ownership semantics belong to
`tests/distributed/test_mps_reverse.py`, which spawns torchrun at 2-4 ranks. This
module runs the same executor in one process against a real world-size-1 gloo
group, so the reverse pipeline is exercised by the default local lane and the
coverage gate, and it pins the exact gradient against dense statevector autograd
and against the local MPS runner. A single rank is a correctness statement only:
it is not distributed scalability evidence.
"""

from __future__ import annotations

import socket

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)
from flagquantum.simulation.mps.entrypoints import run_mps

pytestmark = pytest.mark.integration

OBSERVABLE_WIRES = (1, 2)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def single_rank_group() -> None:
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("a gloo process group is unavailable")
    if dist.is_initialized():
        pytest.skip("another process group owns this interpreter")
    dist.init_process_group(
        backend="gloo",
        world_size=1,
        rank=0,
        init_method=f"tcp://127.0.0.1:{_free_port()}",
    )
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _workload(theta: torch.Tensor, n_wires: int = 4) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=theta.device)
    for wire in range(n_wires):
        circuit.ry(wire, theta)
    for wire in range(n_wires - 1):
        circuit.rxx(wire, wire + 1, theta)
    circuit.ry(n_wires // 2, theta)
    for wire in range(n_wires - 2, -1, -1):
        circuit.rzz(wire, wire + 1, theta)
    return circuit


def _dense_z(circuit: fq.Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    state = circuit.state(refresh=True)
    indices = torch.arange(state.shape[-1], device=state.device)
    sign = torch.ones(state.shape[-1], device=state.device)
    for wire in wires:
        bit = (indices >> (circuit.n_wires - wire - 1)) & 1
        sign *= 1 - 2 * bit
    return torch.sum(state.abs().square() * sign, dim=-1).mean()


def _observable() -> dict[int, str]:
    return {wire: "z" for wire in OBSERVABLE_WIRES}


def test_reverse_gradient_matches_dense_statevector_autograd(
    single_rank_group: None,
) -> None:
    theta = torch.tensor(0.23, requires_grad=True)

    result = execute_torch_distributed_mps_reverse(
        _workload(theta), observable=_observable(), max_bond=32
    )
    result.backward()

    dense_theta = torch.tensor(0.23, requires_grad=True)
    dense_loss = _dense_z(_workload(dense_theta), OBSERVABLE_WIRES)
    dense_loss.backward()

    local_theta = torch.tensor(0.23, requires_grad=True)
    local_loss = (
        run_mps(_workload(local_theta), max_bond=32)
        .expectation_ps(z=OBSERVABLE_WIRES)
        .mean()
    )
    local_loss.backward()

    torch.testing.assert_close(result.value, dense_loss, atol=5e-5, rtol=5e-5)
    assert theta.grad is not None
    torch.testing.assert_close(theta.grad, dense_theta.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(theta.grad, local_theta.grad, atol=2e-4, rtol=2e-4)


def test_reverse_reports_a_deterministic_single_rank_tape(
    single_rank_group: None,
) -> None:
    def run() -> tuple[str, list[tuple[str, int, int]]]:
        theta = torch.tensor(0.23, requires_grad=True)
        result = execute_torch_distributed_mps_reverse(
            _workload(theta), observable=_observable(), max_bond=32
        )
        records = [
            (record.operation_id, record.forward_sequence, record.reverse_sequence)
            for record in result.tape.records
        ]
        return result.tape.identity, records

    first_identity, first_records = run()
    second_identity, second_records = run()

    assert first_records and first_records == second_records
    assert first_identity == second_identity


def test_reverse_summary_claims_only_what_single_rank_earned(
    single_rank_group: None,
) -> None:
    theta = torch.tensor(0.23, requires_grad=True)
    result = execute_torch_distributed_mps_reverse(
        _workload(theta), observable=_observable(), max_bond=32
    )
    result.backward()
    summary = result.summary()

    assert summary["mps_backward_execution"] == "completed"
    assert summary["gradient_accuracy"] == "exact"
    assert summary["gradient_policy"] == "exact"
    assert summary["replicated_autograd"] is False
    assert summary["full_mps_reconstruction"] is False
    assert summary["statevector_fallback"] is False
    assert summary["jax_required"] is False
    assert summary["rank"] == 0
    assert summary["world_size"] == 1
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["reverse_tape_identity"] == result.tape.identity
    assert summary["reverse_tape_records"] == len(result.tape.records)
    assert summary["discarded_weight"] == 0.0
    assert summary["svd_factorization_count"] == 0
    assert summary["qr_factorization_count"] > 0
    assert summary["watchdog_diagnostics"]["tape_cursor"] == len(result.tape.records)
    assert summary["watchdog_diagnostics"]["rank"] == 0
    assert summary["blockers"] == ()
    assert summary["backward_error"] is None


def test_reverse_gradient_can_drive_an_optimizer_step(single_rank_group: None) -> None:
    theta = torch.tensor(0.23, requires_grad=True)
    before = theta.detach().clone()

    result = execute_torch_distributed_mps_reverse(
        _workload(theta), observable=_observable(), max_bond=32
    )
    result.backward()
    torch.optim.SGD([theta], lr=0.01).step()

    assert not torch.equal(theta.detach(), before)


def test_reverse_approximate_policy_is_reported_as_approximate(
    single_rank_group: None,
) -> None:
    theta = torch.tensor(0.41, requires_grad=True)

    result = execute_torch_distributed_mps_reverse(
        _workload(theta),
        observable=_observable(),
        max_bond=1,
        gradient_policy="approximate",
        gradient_tolerance=10.0,
    )
    result.backward()
    summary = result.summary()

    assert summary["mps_backward_execution"] == "completed"
    assert summary["gradient_accuracy"] == "approximate"
    assert summary["gradient_policy"] == "approximate"
    assert summary["svd_factorization_count"] > 0
    assert summary["discarded_weight"] > 0.0
    assert torch.isfinite(result.value)
    assert theta.grad is not None and torch.isfinite(theta.grad)


def test_reverse_accepts_hamiltonian_objectives(single_rank_group: None) -> None:
    theta = torch.tensor(0.17, requires_grad=True)
    terms = [({wire: "z"}, 1.0) for wire in range(4)] + [
        ({wire: "x", wire + 1: "x"}, 0.5) for wire in range(3)
    ]

    result = execute_torch_distributed_mps_reverse(
        _workload(theta), hamiltonian_terms=terms, max_bond=32
    )
    result.backward()

    assert result.summary()["mps_backward_execution"] == "completed"
    assert torch.isfinite(result.value)
    assert theta.grad is not None and torch.isfinite(theta.grad)


def test_reverse_rejects_conflicting_objective_inputs(single_rank_group: None) -> None:
    theta = torch.tensor(0.23)

    with pytest.raises(ValueError, match="mutually exclusive"):
        execute_torch_distributed_mps_reverse(
            _workload(theta),
            observable=_observable(),
            observable_terms=[({0: "z"}, 1.0)],
        )
