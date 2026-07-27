from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)
from flagquantum.runtime.distributed.engine import (
    clear_mps_static_descriptor_cache,
    mps_p2p_stats,
    reset_mps_p2p_stats,
)


def run(device, max_bond):
    theta = torch.tensor(0.31, device=device, requires_grad=True)
    circuit = Circuit(4, device=device)
    circuit.ry(1, theta)
    circuit.rxx(1, 2, theta)
    circuit.ry(2, theta)
    result = execute_torch_distributed_mps_reverse(
        circuit,
        observable={1: "z"},
        device=device,
        max_bond=max_bond,
        gradient_policy="approximate" if max_bond == 1 else "exact",
        gradient_tolerance=10.0 if max_bond == 1 else 0.0,
    )
    result.backward()
    return float(result.value), float(theta.grad)


def main():
    device = torch.device("cpu")
    dist.init_process_group("gloo")
    # Any warm-path regression to Python object collectives fails immediately.
    gather_objects = dist.all_gather_object
    dist.broadcast_object_list = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("broadcast_object_list used")
    )
    dist.all_gather_object = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("all_gather_object used")
    )
    clear_mps_static_descriptor_cache()
    reset_mps_p2p_stats()
    first = run(device, 4)
    cold = mps_p2p_stats()
    reset_mps_p2p_stats()
    second = run(device, 4)
    warm = mps_p2p_stats()
    reset_mps_p2p_stats()
    changed = run(device, 1)
    changed_stats = mps_p2p_stats()
    payload = {
        "rank": dist.get_rank(),
        "cold": cold,
        "warm": warm,
        "changed": changed_stats,
        "value_error": abs(first[0] - second[0]),
        "gradient_error": abs(first[1] - second[1]),
        "bond_generation_changed_result_finite": all(
            torch.isfinite(torch.tensor(changed))
        ),
    }
    records = [None] * dist.get_world_size()
    gather_objects(records, payload)
    if dist.get_rank() == 0:
        Path(os.environ["FQ_ISSUE103_OUTPUT"]).write_text(json.dumps(records, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
