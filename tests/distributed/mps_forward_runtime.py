from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import torch
import torch.distributed as dist

import flagquantum.runtime.backends.mps.forward as mps_forward
from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.forward import (
    MPSFullMaterializationError,
    NonlocalMPSCompilationError,
    execute_torch_distributed_mps_forward,
    gather_mps_for_validation,
)
from flagquantum.runtime.backends.mps.site_kernels import (
    configure_site_kernel_cache,
    reset_site_kernel_stats,
    site_kernel_cache_policy,
)
from flagquantum.simulation.mps import run_mps


def circuit_for_world(world: int, *, device: torch.device) -> Circuit:
    n_wires = max(6, world * 2)
    circuit = Circuit(n_wires, device=device)
    for wire in range(n_wires):
        circuit.ry(wire, 0.07 * (wire + 1))
    for left in range(n_wires - 1):
        circuit.rxx(left, left + 1, 0.11 + 0.01 * left)
    for left in range(n_wires - 2, -1, -1):
        circuit.ry(left, 0.03).rxx(left, left + 1, -0.08 - 0.01 * left)
    circuit.rxx(0, 1, 0.19).rxx(0, 1, -0.13)
    return circuit


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
    rank = dist.get_rank()
    world = dist.get_world_size()
    try:
        circuit = circuit_for_world(world, device=device)
        result = execute_torch_distributed_mps_forward(
            circuit,
            device=device,
            max_bond=8,
            rebalance_threshold=1.01,
        )
        owned: list[object] = [None] * world
        dist.all_gather_object(owned, tuple(result.shard_state.local_tensors))
        flat = [wire for wires in owned for wire in wires]
        assert sorted(flat) == list(range(circuit.n_wires))
        assert len(flat) == len(set(flat))
        assert all(
            wire in result.shard_state.ownership[rank]
            for wire in result.shard_state.local_tensors
        )
        try:
            result.full_state()
        except MPSFullMaterializationError:
            pass
        else:
            raise AssertionError("production result unexpectedly materialized full MPS")

        gathered = gather_mps_for_validation(result)
        if rank == 0:
            assert gathered is not None
            expected = run_mps(circuit, max_bond=8)
            torch.testing.assert_close(
                gathered.to_statevector(),
                expected.to_statevector(),
                atol=3e-5,
                rtol=3e-5,
            )
            torch.testing.assert_close(
                gathered.to_statevector(),
                circuit.state(),
                atol=3e-5,
                rtol=3e-5,
            )
        dist.barrier()

        invalid = Circuit(circuit.n_wires).h(0).cx(0, circuit.n_wires - 1)
        original_apply = mps_forward._apply_one_mps_tensor

        def forbidden_apply(*args, **kwargs):
            raise AssertionError("prevalidation must precede partial execution")

        mps_forward._apply_one_mps_tensor = forbidden_apply
        try:
            try:
                execute_torch_distributed_mps_forward(invalid)
            except NonlocalMPSCompilationError as error:
                assert "route it before MPS execution" in str(error)
            else:
                raise AssertionError("nonlocal gate did not fail compilation")
        finally:
            mps_forward._apply_one_mps_tensor = original_apply
        assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
        assert result.summary()["jax_required"] is False
        assert result.summary()["jax_imported"] is False
        assert result.summary()["backend"] == args.backend
        assert result.summary()["claim_evidence_type"] == (
            "accelerator_semantics"
            if args.backend == "nccl"
            else "development_semantics"
        )
        assert result.summary()["communication_messages"] >= result.boundary_messages
        assert result.summary()["communication_bytes"] >= result.boundary_bytes

        original_policy = site_kernel_cache_policy()
        lifecycle_records = []
        try:
            configure_site_kernel_cache(
                replace(
                    original_policy,
                    backend="eager",
                    mode=None,
                    policy_id="issue107_distributed_lifetime",
                )
            )
            reset_site_kernel_stats(clear_cache=True)
            lifecycle = execute_torch_distributed_mps_forward(
                circuit,
                device=device,
                max_bond=8,
                rebalance_threshold=float("inf"),
                compile_site_kernels=True,
                layer_lifecycle_callback=lifecycle_records.append,
            )
        finally:
            configure_site_kernel_cache(original_policy)
            reset_site_kernel_stats(clear_cache=True)
        lifetime = lifecycle.summary()["forward_tensor_lifetime"]
        assert lifetime["policy"] == "layer_local_destructive_consumption_v1"
        assert lifetime["gate_matrix_materialization"] == "per_instruction_last_use"
        assert lifetime["layer_cache_empty_at_return"] is True
        assert lifetime["layer_drain_count"] == len(lifecycle_records) > 0
        assert all(
            record["remaining_precomputed_entries"] == 0 for record in lifecycle_records
        )

        def contains_tensor(value):
            if isinstance(value, torch.Tensor):
                return True
            if isinstance(value, dict):
                return any(contains_tensor(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(contains_tensor(item) for item in value)
            return False

        assert not contains_tensor(lifecycle.summary())
        sentinel = object()
        sys.modules["jax"] = sentinel  # prove the summary is measured, not hard-coded
        try:
            assert result.summary()["jax_imported"] is True
        finally:
            del sys.modules["jax"]
        if rank == 0:
            print(
                json.dumps(
                    {
                        "world_size": world,
                        "backend": args.backend,
                        "n_wires": circuit.n_wires,
                        "status": "passed",
                        "rebalance_count": result.rebalance_count,
                        "partition_history": result.partition_history,
                        "bond_dimensions": result.bond_dimensions,
                        "hostname": os.uname().nodename,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
