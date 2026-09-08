"""NCCL correctness check for persistent-layout forward execution."""

import os

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.layout import (
    plan_persistent_statevector_layout,
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
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    os.environ["FQ_STATEVECTOR_LOCAL_BLOCK_FUSION"] = "1"
    os.environ["FQ_STATEVECTOR_TRITON_LOCAL_CX"] = "1"
    os.environ["FQ_STATEVECTOR_TRITON_CX_SEGMENT"] = "1"
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
        canonical = execute_torch_distributed_statevector(
            circuit, device=device, dtype=torch.complex64
        )
        persistent = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=torch.complex64,
            persistent_wire_layout=True,
        )
        canonical_state = _canonical_state(canonical)
        persistent_state = _canonical_state(persistent)
        assert torch.allclose(persistent_state, canonical_state, atol=2e-6, rtol=2e-6)
        assert persistent.wire_layout == "persistent"
        assert plan_persistent_statevector_layout(circuit, world_size=2).swaps

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
        assert torch.allclose(
            _canonical_state(repeated_persistent),
            _canonical_state(repeated_canonical),
            atol=2e-6,
            rtol=2e-6,
        )
        assert (
            repeated_persistent.communication_bytes
            < repeated_canonical.communication_bytes
        )
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
        assert torch.allclose(
            _canonical_state(cx_chain_compiled),
            _canonical_state(cx_chain_canonical),
            atol=2e-6,
            rtol=2e-6,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            print("persistent-layout NCCL forward: PASS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
