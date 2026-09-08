from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.runtime.executors.mps.reverse import (
    _fused_z_zz_mse_and_adjoints,
    _parse_z_zz_terms,
    site_sharded_z_zz_objective_pipeline,
)
from flagquantum.runtime.executors.mps.state import (
    RankOwnedMPSState,
    initial_mps_ownership,
)
from flagquantum.simulation.mps.models import MPSConfig


def _states(slots: int, *, device: torch.device):
    rank, world, wires, batch = dist.get_rank(), dist.get_world_size(), 8, 3
    ownership = initial_mps_ownership(wires, world)
    output = []
    for slot in range(slots):
        local = {}
        for wire in ownership[rank]:
            generator = torch.Generator(device="cpu").manual_seed(
                1000 + slot * wires + wire
            )
            value = torch.randn(
                batch, 1, 2, 1, generator=generator, dtype=torch.float64
            )
            value = value / torch.linalg.vector_norm(value, dim=2, keepdim=True)
            local[wire] = value.to(device=device, dtype=torch.complex128)
        output.append(
            RankOwnedMPSState(wires, batch, rank, world, MPSConfig(), local, ownership)
        )
    return tuple(output)


def _terms(state: RankOwnedMPSState, slot: int):
    target = torch.linspace(
        -0.3, 0.2, state.bsz, device=next(iter(state.local_tensors.values())).device
    )
    return (({1: "z"}, target + slot * 0.01), ({4: "z", 5: "z"}, target - 0.1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cpu")
    dist.init_process_group("gloo")
    states = _states(5, device=device)
    terms = tuple(_terms(state, slot) for slot, state in enumerate(states))
    sequential = tuple(
        _fused_z_zz_mse_and_adjoints(state, _parse_z_zz_terms(state, item))
        for state, item in zip(states, terms)
    )
    drained = []
    pipelined = site_sharded_z_zz_objective_pipeline(
        states, terms, max_pipeline_slots=3, on_window_drained=drained.append
    )
    value_error = max(
        float(torch.abs(a[0] - b[0])) for a, b in zip(sequential, pipelined)
    )
    grad_error = max(
        float(torch.max(torch.abs(a[1][wire] - b[1][wire])))
        for a, b in zip(sequential, pipelined)
        for wire in a[1]
    )
    cancelled = site_sharded_z_zz_objective_pipeline(
        states, terms, max_pipeline_slots=2, cancelled=lambda: True
    )
    one = site_sharded_z_zz_objective_pipeline(
        states[:1], terms[:1], max_pipeline_slots=4
    )
    fault_drained = False
    try:
        site_sharded_z_zz_objective_pipeline(states[:1], ((({0: "z", 2: "z"}, 0.0),),))
    except ValueError:
        fault_drained = True
    payload = {
        "value_error": value_error,
        "gradient_error": grad_error,
        "partial_final_count": len(pipelined),
        "cancelled_drain_count": len(cancelled),
        "batch_one_count": len(one),
        "window_drain_points": drained,
        "preflight_fault_drained": fault_drained,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(gathered, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
