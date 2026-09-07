from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)
from flagquantum.simulation.mps.entrypoints import run_mps


def workload(world: int, theta: torch.Tensor) -> Circuit:
    n_wires = max(6, 2 * world)
    circuit = Circuit(n_wires, device=theta.device)
    for wire in range(n_wires):
        circuit.ry(wire, theta)
    for wire in range(n_wires - 1):
        circuit.rxx(wire, wire + 1, theta)
    circuit.ry(n_wires // 2, theta)
    for wire in range(n_wires - 2, -1, -1):
        circuit.rzz(wire, wire + 1, theta)
    return circuit


def dense_z(circuit: Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    state = circuit.state(refresh=True)
    indices = torch.arange(state.shape[-1], device=state.device)
    sign = torch.ones(state.shape[-1], device=state.device)
    for wire in wires:
        bit = (indices >> (circuit.n_wires - wire - 1)) & 1
        sign *= 1 - 2 * bit
    return torch.sum(state.abs().square() * sign, dim=-1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args()
    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        theta = torch.tensor(0.23, device=device, requires_grad=True)
        circuit = workload(world, theta)
        wires = (1, circuit.n_wires - 2)
        result = execute_torch_distributed_mps_reverse(
            circuit,
            observable={wire: "z" for wire in wires},
            max_bond=32,
            device=device,
            reverse_delay_seconds=0.01,
        )
        result.backward()
        summary = result.summary()
        assert summary["mps_backward_execution"] == "completed"
        assert summary["gradient_accuracy"] == "exact"
        assert summary["replicated_autograd"] is False
        assert summary["full_mps_reconstruction"] is False
        assert summary["watchdog_diagnostics"]["tape_cursor"] == len(
            result.tape.records
        )
        identities: list[object] = [None] * world
        dist.all_gather_object(identities, result.tape.identity)
        assert len(set(identities)) == 1
        assert theta.grad is not None and torch.isfinite(theta.grad)
        before_update = theta.detach().clone()
        if rank == 0:
            dense_theta = torch.tensor(0.23, device=device, requires_grad=True)
            dense_loss = dense_z(workload(world, dense_theta), wires)
            dense_loss.backward()
            local_theta = torch.tensor(0.23, device=device, requires_grad=True)
            local = run_mps(workload(world, local_theta), max_bond=32)
            local_loss = local.expectation_ps(z=wires).mean()
            local_loss.backward()
            torch.testing.assert_close(result.value, dense_loss, atol=5e-5, rtol=5e-5)
            torch.testing.assert_close(
                theta.grad, dense_theta.grad, atol=2e-4, rtol=2e-4
            )
            torch.testing.assert_close(
                theta.grad, local_theta.grad, atol=2e-4, rtol=2e-4
            )
        torch.optim.SGD([theta], lr=0.01).step()
        assert not torch.equal(theta.detach(), before_update)
        updated: list[object] = [None] * world
        dist.all_gather_object(updated, float(theta.detach()))
        assert len(set(updated)) == 1
        dist.barrier()

        approximate_theta = torch.tensor(0.41, device=device, requires_grad=True)
        approximate_circuit = workload(world, approximate_theta)
        approximate = execute_torch_distributed_mps_reverse(
            approximate_circuit,
            observable={wire: "z" for wire in wires},
            max_bond=1,
            device=device,
            gradient_policy="approximate",
            gradient_tolerance=10.0,
        )
        approximate.backward()
        approximate_summary = approximate.summary()
        assert approximate_summary["gradient_accuracy"] == "approximate"
        assert approximate_summary["discarded_weight"] > 0
        if rank == 0:
            local_theta = torch.tensor(0.41, device=device, requires_grad=True)
            local = run_mps(workload(world, local_theta), max_bond=1)
            local_loss = local.expectation_ps(z=wires).mean()
            local_loss.backward()
            torch.testing.assert_close(
                approximate.value, local_loss, atol=8e-5, rtol=8e-5
            )
            torch.testing.assert_close(
                approximate_theta.grad, local_theta.grad, atol=5e-4, rtol=5e-4
            )
            print(
                json.dumps(
                    {
                        "world_size": world,
                        "status": "passed",
                        "exact": summary,
                        "approximate": approximate_summary,
                    },
                    default=str,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
