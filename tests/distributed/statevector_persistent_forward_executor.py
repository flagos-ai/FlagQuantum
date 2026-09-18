"""Torchrun differential for persistent-layout forward execution.

The circuit shapes below are the ones that make the persistent-layout planner
produce swaps, so that the executor's swap replay has something to replay.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.core.ir import ensure_circuit_ir
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.layout import (
    plan_persistent_statevector_layout,
    schedule_statevector_dependency_dag,
)


def _canonical_state(result) -> torch.Tensor:
    local = result.shard_state.amplitudes[0]
    gathered = [torch.empty_like(local) for _ in range(result.plan.world_size)]
    dist.all_gather(gathered, local)
    internal = torch.empty(
        result.plan.total_amplitudes, dtype=local.dtype, device=local.device
    )
    for rank, shard in enumerate(gathered):
        internal[rank :: result.plan.world_size] = shard
    mapping = result.logical_to_physical_wires
    canonical = torch.empty_like(internal)
    for logical_basis in range(internal.numel()):
        physical_basis = 0
        for logical_wire, physical_wire in enumerate(mapping):
            bit = (logical_basis >> (len(mapping) - logical_wire - 1)) & 1
            physical_basis |= bit << (len(mapping) - physical_wire - 1)
        canonical[logical_basis] = internal[physical_basis]
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # The three accelerated dispatch paths in the forward sweep are guarded
        # on a CUDA device as well as on these flags, so they are requested only
        # where a device can honour them. The swap replay is device-agnostic and
        # is reached on both backends.
        os.environ["FQ_STATEVECTOR_LOCAL_BLOCK_FUSION"] = "1"
        os.environ["FQ_STATEVECTOR_TRITON_LOCAL_CX"] = "1"
        os.environ["FQ_STATEVECTOR_TRITON_CX_SEGMENT"] = "1"
    if world > 1:
        dist.init_process_group(args.backend)
    try:
        circuit = (
            fq.Circuit(5)
            .h(0)
            .ry(4, 0.31)
            .cx(4, 1)
            .rx(3, -0.27)
            .cx(0, 4)
            .rz(4, 0.19)
            .cx(3, 2)
        )
        # The executor schedules before it plans, so the plan it replayed is the
        # one over the scheduled IR, not over the circuit as written.
        scheduled = schedule_statevector_dependency_dag(
            ensure_circuit_ir(circuit), world_size=world
        )
        plan = plan_persistent_statevector_layout(scheduled, world_size=world)
        assert plan.swaps, "the persistent plan must swap, or nothing is replayed"

        canonical = execute_torch_distributed_statevector(
            circuit, device=device, dtype=torch.complex64
        )
        persistent = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=torch.complex64,
            persistent_wire_layout=True,
        )
        # Comparing states is not enough to prove the swaps were replayed: a run
        # that skips every swap and also skips the wire remap executes the
        # canonical schedule and returns the canonical state, while the plan's
        # communication saving is silently lost. The mapping is what the replay
        # produces, so it is checked against the plan's own swap sequence.
        expected_mapping = list(range(scheduled.n_wires))
        for swap in plan.swaps:
            local, sharded = swap.local_logical_wire, swap.sharded_logical_wire
            expected_mapping[local], expected_mapping[sharded] = (
                expected_mapping[sharded],
                expected_mapping[local],
            )
        assert persistent.logical_to_physical_wires == tuple(
            expected_mapping
        ), "the replayed swaps do not reproduce the plan's wire permutation"
        torch.testing.assert_close(
            _canonical_state(persistent),
            _canonical_state(canonical),
            atol=2e-6,
            rtol=2e-6,
        )
        assert persistent.wire_layout == "persistent"

        repeated = fq.Circuit(5).rx(4, 0.1).ry(4, 0.2).rx(4, -0.3)
        repeated_canonical = execute_torch_distributed_statevector(
            repeated, device=device, dtype=torch.complex64
        )
        repeated_persistent = execute_torch_distributed_statevector(
            repeated,
            device=device,
            dtype=torch.complex64,
            persistent_wire_layout=True,
        )
        torch.testing.assert_close(
            _canonical_state(repeated_persistent),
            _canonical_state(repeated_canonical),
            atol=2e-6,
            rtol=2e-6,
        )
        assert (
            repeated_persistent.communication_bytes
            < repeated_canonical.communication_bytes
        ), "persistent layout must not cost more communication than the baseline"

        cx_chain = fq.Circuit(5).h(0).h(2).cx(0, 1).cx(1, 2).cx(2, 3)
        cx_chain_canonical = execute_torch_distributed_statevector(
            cx_chain, device=device, dtype=torch.complex64
        )
        cx_chain_compiled = execute_torch_distributed_statevector(
            cx_chain,
            device=device,
            dtype=torch.complex64,
            persistent_wire_layout=True,
        )
        torch.testing.assert_close(
            _canonical_state(cx_chain_compiled),
            _canonical_state(cx_chain_canonical),
            atol=2e-6,
            rtol=2e-6,
        )
        evidence = persistent.kernel_dispatch_evidence.summary()
        if dist.is_initialized():
            dist.destroy_process_group()
        print(
            json.dumps(
                {
                    "persistent_layout_enabled": True,
                    "wire_layout": persistent.wire_layout,
                    "persistent_swap_count": len(plan.swaps),
                    "logical_to_physical_wires": list(
                        persistent.logical_to_physical_wires
                    ),
                    "canonical_communication_bytes": canonical.communication_bytes,
                    "persistent_communication_bytes": persistent.communication_bytes,
                    "local_gate_count": persistent.local_gate_count,
                    "distributed_gate_count": persistent.distributed_gate_count,
                    "triton_execution_count": evidence["triton_execution_count"],
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
