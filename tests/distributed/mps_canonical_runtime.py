from __future__ import annotations

import json

import torch
import torch.distributed as dist

from flagquantum.circuit import Circuit
from flagquantum.runtime.executors.mps.forward import (
    execute_torch_distributed_mps_forward,
    gather_mps_for_validation,
)
from flagquantum.simulation.mps.entrypoints import run_mps


def workload(world: int) -> Circuit:
    n_wires = max(6, 2 * world)
    circuit = Circuit(n_wires)
    for layer in range(3):
        for wire in range(n_wires):
            circuit.ry(wire, 0.13 * (wire + layer + 1))
        for wire in range(n_wires - 1):
            circuit.rxx(wire, wire + 1, 0.21 + 0.03 * (wire + layer))
        for wire in range(n_wires - 2, -1, -1):
            circuit.rzz(wire, wire + 1, -0.17 - 0.02 * layer)
    return circuit


def main() -> None:
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        circuit = workload(world)
        center = circuit.n_wires // 2
        result = execute_torch_distributed_mps_forward(
            circuit,
            max_bond=2,
            canonical_center=center,
            global_error_budget=1.0,
            truncation_gradient_policy="approximate",
            rebalance_threshold=1.01,
        )
        summary = result.summary()
        assert summary["orthogonality_center"] == center
        assert summary["mixed_canonical_residual"] < 2e-5
        assert summary["truncation_error"] > 0
        assert summary["truncation_semantics"] == "approximate"
        assert summary["error_budget_satisfied"]
        assert summary["canonicalization_bytes_by_rank"]
        minimum_collective_bytes = 32 + 8 * circuit.bsz
        assert (
            min(summary["canonicalization_bytes_by_rank"]) >= minimum_collective_bytes
        )
        assert min(summary["canonicalization_messages_by_rank"]) >= 3
        assert max(summary["canonicalization_temporary_bytes_by_rank"]) > 0
        assert summary["truncation_gradient_metadata"]["exact"] is False

        gathered = gather_mps_for_validation(result)
        if rank == 0:
            assert gathered is not None
            reference = run_mps(circuit, max_bond=2)
            torch.testing.assert_close(
                gathered.to_statevector(),
                reference.to_statevector(),
                atol=4e-5,
                rtol=4e-5,
            )
            expected_error = sum(reference.truncation_errors)
            assert abs(summary["truncation_error"] - expected_error) < 1e-6
            assert gathered.mixed_canonical_residual(center) < 2e-5
            torch.testing.assert_close(
                gathered.state_norm(),
                torch.tensor(summary["state_norms"]),
                atol=2e-5,
                rtol=2e-5,
            )
            print(
                json.dumps(
                    {"world_size": world, "status": "passed", **summary}, default=str
                )
            )
        try:
            execute_torch_distributed_mps_forward(
                circuit,
                max_bond=2,
                global_error_budget=0.0,
                error_budget_policy="enforce",
                truncation_gradient_policy="approximate",
            )
        except RuntimeError as error:
            assert "exceeded global_error_budget" in str(error)
        else:
            raise AssertionError("error budget violation did not fail closed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
