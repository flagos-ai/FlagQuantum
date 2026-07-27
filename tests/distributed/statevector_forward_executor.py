"""Torchrun numerical differential for the ISSUE-041 forward executor."""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.statevector.forward import (
    execute_torch_distributed_statevector,
)


def circuit_for_world(world_size: int) -> fq.Circuit:
    n_wires = max(4, world_size.bit_length() + 1)
    last = n_wires - 1
    circuit = fq.Circuit(n_wires)
    circuit.h(0).x(last).rz(last, -0.2).ry(last, 0.31).cx(last - 1, last)
    circuit.h(0).rx(last, 0.13)
    if world_size >= 4:
        circuit.rxx(0, 1, 0.23).swap(1, last)
    if world_size >= 8:
        circuit.ccx(0, 1, last).rzz(1, 2, -0.17)
    return circuit


def seeded_random_circuit(world_size: int) -> fq.Circuit:
    rng = random.Random(4100 + world_size)
    n_wires = max(5, world_size.bit_length() + 2)
    circuit = fq.Circuit(n_wires)
    one_site = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "phase")
    two_site = ("cx", "cy", "cz", "swap", "rxx", "ryy", "rzz")
    parameterized = {"rx", "ry", "rz", "phase", "rxx", "ryy", "rzz"}
    for _ in range(24):
        if rng.random() < 0.55:
            name = rng.choice(one_site)
            wires = (rng.randrange(n_wires),)
        else:
            name = rng.choice(two_site)
            wires = tuple(rng.sample(range(n_wires), 2))
        params = {"theta": rng.uniform(-0.7, 0.7)} if name in parameterized else {}
        circuit.gate(name, wires, **params)
    # Guarantee communication independent of the random draw.
    circuit.h(n_wires - 1).rxx(0, n_wires - 1, 0.19)
    return circuit


def assert_matches_dense(circuit: fq.Circuit, device: torch.device):
    dense = circuit.state().to(device)
    result = execute_torch_distributed_statevector(circuit, device=device)
    indices = result.shard_state.global_indices
    if not indices.numel():
        local = torch.arange(
            result.shard_state.amplitudes.shape[1],
            device=device,
            dtype=torch.long,
        )
        indices = (local << len(result.plan.sharded_wires)) | result.shard_state.rank
    expected = dense[:, indices]
    torch.testing.assert_close(
        result.shard_state.amplitudes, expected, atol=1e-5, rtol=1e-5
    )
    return dense, result


def assert_fusion_reduces_communication(circuit: fq.Circuit, device: torch.device):
    fused = execute_torch_distributed_statevector(
        circuit, device=device, fuse_cross_shard_gates=True
    )
    baseline = execute_torch_distributed_statevector(
        circuit, device=device, fuse_cross_shard_gates=False
    )
    torch.testing.assert_close(
        fused.shard_state.amplitudes,
        baseline.shard_state.amplitudes,
        atol=1e-5,
        rtol=1e-5,
    )
    assert fused.communication_count < baseline.communication_count
    assert fused.communication_bytes < baseline.communication_bytes
    assert (
        fused.executed_distributed_segment_count
        < baseline.executed_distributed_segment_count
    )
    return fused, baseline


def assert_pair_pipeline_matches_synchronous(circuit: fq.Circuit, device: torch.device):
    pipelined = execute_torch_distributed_statevector(
        circuit,
        device=device,
        exchange_buffer_bytes=64,
        pipeline_pair_exchange=True,
    )
    synchronous = execute_torch_distributed_statevector(
        circuit,
        device=device,
        exchange_buffer_bytes=64,
        pipeline_pair_exchange=False,
    )
    torch.testing.assert_close(
        pipelined.shard_state.amplitudes,
        synchronous.shard_state.amplitudes,
        atol=1e-5,
        rtol=1e-5,
    )
    assert pipelined.exchange_pipeline_prefetch_count > 0
    assert pipelined.exchange_pipelined_gate_count > 0
    assert synchronous.exchange_pipeline_prefetch_count == 0
    assert synchronous.exchange_pipelined_gate_count == 0
    return pipelined, synchronous


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
    if world > 1:
        dist.init_process_group(args.backend)
    try:
        dense, result = assert_matches_dense(circuit_for_world(world), device)
        if world > 1:
            fused, baseline = assert_fusion_reduces_communication(
                circuit_for_world(world), device
            )
            pipelined, synchronous = assert_pair_pipeline_matches_synchronous(
                circuit_for_world(world), device
            )
            assert fused.cross_shard_fusion_enabled is True
            assert baseline.cross_shard_fusion_enabled is False
        _, random_result = assert_matches_dense(seeded_random_circuit(world), device)
        assert (
            random_result.local_gate_count + random_result.distributed_gate_count == 26
        )
        if world > 1:
            assert result.distributed_gate_count > 0
            assert result.communication_count > 0
            # Live amplitude scratch includes the rank-local output buffer in
            # addition to exchange chunks, gathered inputs, and matrix output.
            assert result.peak_scratch_bytes >= 6 * result.plan.per_rank_state_bytes
            assert result.summary()["scratch_accounting"] == (
                "peak_live_amplitude_tensors_including_output_buffer"
            )
        summary = result.summary()
        assert summary["exchange_pipeline_enabled"] is True
        if world > 1:
            assert summary["exchange_workspace_allocation_count"] >= 1
            assert summary["exchange_workspace_reuse_count"] >= 1
            assert summary["exchange_workspace_reserved_bytes"] > 0
        if world > 1:
            summary["fusion_ab"] = {
                "fused_communication_count": fused.communication_count,
                "baseline_communication_count": baseline.communication_count,
                "fused_communication_bytes": fused.communication_bytes,
                "baseline_communication_bytes": baseline.communication_bytes,
                "fused_executed_segments": fused.executed_distributed_segment_count,
                "baseline_executed_segments": baseline.executed_distributed_segment_count,
            }
            summary["pipeline_ab"] = {
                "prefetch_count": pipelined.exchange_pipeline_prefetch_count,
                "pipelined_gate_count": pipelined.exchange_pipelined_gate_count,
                "subgroup_pipelined_gate_count": pipelined.exchange_subgroup_pipelined_gate_count,
                "synchronous_prefetch_count": synchronous.exchange_pipeline_prefetch_count,
            }
        assert summary["local_diagonal_gate_count"] >= 1
        if world > 1:
            assert summary["fused_cross_shard_regions"] >= 1
            assert summary["fused_cross_shard_gate_count"] >= 3
            assert (
                summary["executed_distributed_segment_count"]
                < summary["distributed_gate_count"]
            )
        assert summary["full_state_materialization"] is False
        assert (
            summary["local_state_bytes"]
            == dense.numel() * dense.element_size() // world
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
