from __future__ import annotations

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist

from flagquantum.circuit import Circuit
from flagquantum.runtime.executors.mps.reverse import (
    MPSReverseCheckpointPolicy,
    MPSReverseContractError,
    _recv,
    _send,
    execute_torch_distributed_mps_reverse,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("mismatch", "memory", "sequence", "timeout"), required=True
    )
    args = parser.parse_args()
    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=2 if args.mode == "timeout" else 30)
    )
    rank = dist.get_rank()
    try:
        theta = torch.tensor(0.2, requires_grad=True)
        circuit = Circuit(4).ry(0, theta).rxx(0, 1, theta).rxx(1, 2, theta).ry(3, theta)
        if args.mode == "mismatch":
            try:
                execute_torch_distributed_mps_reverse(
                    circuit, tape_identity_override="altered" if rank == 0 else None
                )
            except MPSReverseContractError as error:
                assert "tape mismatch" in str(error)
            else:
                raise AssertionError("tape mismatch did not fail closed")
        elif args.mode == "memory":
            try:
                execute_torch_distributed_mps_reverse(
                    circuit,
                    checkpoint_policy=MPSReverseCheckpointPolicy(max_saved_bytes=1),
                )
            except MPSReverseContractError as error:
                assert "limit is 1" in str(error)
            else:
                raise AssertionError("oversized checkpoint did not fail closed")
        elif args.mode == "sequence":
            reference = torch.zeros(1, 1, 2, 1, dtype=torch.complex64)
            if rank == 0:
                _send(reference, destination=1, sequence=41)
            else:
                try:
                    _recv(reference, source=0, sequence=42)
                except MPSReverseContractError as error:
                    assert "peer 0" in str(error)
                    assert "expected 42" in str(error)
                    assert "received 41" in str(error)
                else:
                    raise AssertionError("P2P sequence mismatch did not fail closed")
            dist.barrier()
        else:
            result = execute_torch_distributed_mps_reverse(
                circuit, reverse_delay_seconds=3.0
            )
            result.backward()
            raise AssertionError("collective timeout was not triggered")
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except RuntimeError:
                pass


if __name__ == "__main__":
    main()
