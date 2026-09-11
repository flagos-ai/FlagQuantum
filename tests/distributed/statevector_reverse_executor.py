"""Torchrun gradient differential for sharded statevector reverse mode."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.statevector.reverse import (
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--persistent-layout", action="store_true")
    args = parser.parse_args()
    if args.persistent_layout:
        os.environ["FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT"] = "1"
        os.environ["FQ_STATEVECTOR_TRITON_VJP_ADJOINT"] = "1"
        os.environ["FQ_STATEVECTOR_TRITON_LOCAL_CX"] = "1"
        os.environ["FQ_STATEVECTOR_TRITON_CX_SEGMENT"] = "1"
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group(args.backend)
    try:
        n_wires = max(4, world.bit_length() + 1)
        theta = torch.tensor(0.23, device=device, requires_grad=True)
        phi = torch.tensor(-0.37, device=device, requires_grad=True)
        circuit = fq.Circuit(n_wires, device=device)
        circuit.h(0).ry(n_wires - 1, theta).rxx(0, n_wires - 1, phi)
        circuit.rz(1, theta).crx(0, n_wires - 1, phi)
        if args.persistent_layout:
            for wire in range(n_wires):
                circuit.ry(wire, theta)
                circuit.rx(wire, phi)
            circuit.cx(1, n_wires - 1)
            circuit.cx(0, 1)
            circuit.cx(1, 2)
        if world >= 4:
            circuit.ry(1, theta).rzz(0, 1, phi)
        if world >= 8:
            circuit.crz(2, n_wires - 1, theta).rxx(1, 2, phi)

        dense_value = circuit.expectation_z(n_wires - 1).sum()
        dense_gradients = torch.autograd.grad(
            dense_value, (theta, phi), retain_graph=True
        )
        result = execute_torch_distributed_statevector_reverse(
            circuit,
            observable_wire=n_wires - 1,
            checkpoint_policy=StatevectorCheckpointPolicy(
                strategy=(
                    "reversible_adjoint" if args.persistent_layout else "interval"
                ),
                interval=0 if args.persistent_layout else 3,
            ),
            device=device,
        )
        pending = result.summary()
        assert pending["parameter_gradient_ready"] is False
        assert pending["backward_status"] == "pending"
        assert pending["backward_distribution_semantics"] == "pending"
        assert all(
            item["owner_rank"] is None and not item["participating_ranks"]
            for item in pending["parameter_ownership"]
        )
        result.backward()
        torch.testing.assert_close(result.value, dense_value, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(theta.grad, dense_gradients[0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(phi.grad, dense_gradients[1], atol=3e-5, rtol=3e-5)
        summary = result.summary()
        assert summary["backward_uses_full_state_replay"] is False
        assert summary["backward_status"] == "completed"
        assert summary["backward_distribution_semantics"] == "sharded_across_ranks"
        assert summary["local_world_size"] == local_world_size
        assert summary["node_count"] == max(
            1, (world + local_world_size - 1) // local_world_size
        )
        assert summary["local_state_bytes"] > 0
        assert summary["peak_backward_scratch_bytes"] > 0
        assert summary["backward_communication_count"] > 0
        assert summary["backward_communication_bytes"] > 0
        assert summary["gradient_distribution"] == "replicated_after_all_reduce"
        if args.persistent_layout:
            assert summary["analytic_rotation_derivative_count"] > 0
            assert summary["fused_parameter_adjoint_count"] > 0
        assert all(
            item["owner_rank"] is None for item in summary["parameter_ownership"]
        )
        assert all(
            item.owner_rank is None
            and item.distribution == "replicated_after_all_reduce"
            and item.participating_ranks == tuple(range(world))
            for item in result.ownership
        )
        assert all(
            item["participating_ranks"] == tuple(range(world))
            for item in summary["parameter_ownership"]
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        print(
            json.dumps(
                {
                    **summary,
                    "value": float(result.value.detach().cpu()),
                    "gradients": [float(theta.grad.cpu()), float(phi.grad.cpu())],
                    "cleanup_verified": not dist.is_initialized(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
