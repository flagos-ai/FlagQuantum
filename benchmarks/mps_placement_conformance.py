"""Compare a sharded MPS gradient with the unsharded one on the same program.

The representation-agreement grid of #709 and the gradient extension of #899 both
compare numerical paths on one device. Neither answers the question a user with a
sharded deployment has: does moving the same circuit to several ranks change the
objective or the derivative, beyond what the truncation policy already changes?

That is two questions and this producer keeps them apart. With a bond cap that
discards nothing, any disagreement is placement. With a cap that discards weight,
both placement and truncation contribute, and the cell records which regime it is
in rather than blending them.

This is a single-host CPU semantic check. It does not measure capacity, speed, or
accelerator behaviour, and a sharded result here is not scalability evidence.

Run under a launcher, which is what the pytest wrapper does::

    torchrun --standalone --nproc-per-node=2 \\
        benchmarks/mps_placement_conformance.py --output /tmp/placement.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flagquantum.circuit import Circuit  # noqa: E402
from flagquantum.runtime.executors.mps.reverse import (  # noqa: E402
    execute_torch_distributed_mps_reverse,
)
from flagquantum.simulation.mps.entrypoints import run_mps  # noqa: E402

#: The bond cap at which a chain of this width is held exactly. A cap at or above
#: it cannot discard weight, so that arm isolates placement.
EXACT_CAP = 64
#: A cap narrow enough that the two-site splits discard weight at this depth.
TRUNCATING_CAP = 2

#: Widths chosen so each world size is a different program rather than a
#: relabelling of the previous one.
WIDTH_AT_WORLD = {1: 6, 2: 6, 3: 8, 4: 10}


def build_circuit(n_qubits: int, angles: torch.Tensor) -> Circuit:
    """Return the shared half filling and entangling ladder."""

    circuit = Circuit(n_qubits, dtype=torch.complex128)
    for qubit in range(n_qubits):
        circuit.ry(qubit, angles[0])
    for qubit in range(n_qubits - 1):
        circuit.rxx(qubit, qubit + 1, angles[1])
    circuit.ry(n_qubits // 2, angles[2])
    for qubit in range(n_qubits - 2, -1, -1):
        circuit.rzz(qubit, qubit + 1, angles[1])
    return circuit


def shared_angles(seed: int) -> torch.Tensor:
    """Return the same three angles on every rank, drawn from one seed."""

    generator = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(3, generator=generator, dtype=torch.float64)


def observable_qubits(n_qubits: int) -> tuple[int, ...]:
    return (1, n_qubits - 2)


def unsharded_reference(
    n_qubits: int, angles: torch.Tensor, max_bond: int
) -> tuple[float, torch.Tensor]:
    """Return the single-device value and gradient for the same program.

    The truncating arm authorizes the retained-subspace rule explicitly: the
    default refuses it because that rule is not the derivative of the forward
    map, and this arm exists to compare that rule across placements rather than
    to certify it.
    """

    leaf = angles.detach().clone().requires_grad_(True)
    state = run_mps(
        build_circuit(n_qubits, leaf),
        max_bond=max_bond,
        truncated_gradient="projected_stop_subspace",
    )
    value = state.expectation_ps(z=observable_qubits(n_qubits)).mean()
    (gradient,) = torch.autograd.grad(value, leaf)
    return float(value.detach()), gradient.detach()


def sharded_arm(
    n_qubits: int,
    angles: torch.Tensor,
    max_bond: int,
    device: torch.device,
    gradient_policy: str,
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    """Return the sharded value, gradient and the executor's own summary.

    ``gradient_policy`` is explicit because the executor refuses an exact
    gradient through a split that discarded weight; the truncating arm has to
    authorize the approximation it is measuring rather than inherit it.
    """

    leaf = angles.detach().clone().requires_grad_(True)
    result = execute_torch_distributed_mps_reverse(
        build_circuit(n_qubits, leaf),
        observable=dict.fromkeys(observable_qubits(n_qubits), "z"),
        max_bond=max_bond,
        device=device,
        gradient_policy=gradient_policy,
        # The executor refuses an approximation larger than the tolerance the
        # caller accepts, so the truncating arm declares one and the arm records
        # the discarded weight it had to cover.
        gradient_tolerance=1.0,
    )
    result.backward()
    gradient = leaf.grad
    if gradient is None:
        raise RuntimeError("sharded MPS backward produced no parameter gradient")
    return float(result.value), gradient.detach(), result.summary()


def compare(measured: float, reference: float) -> float | None:
    """Return the relative difference, or None when the reference is zero."""

    if reference == 0.0:
        return None
    return abs(measured - reference) / abs(reference)


def gradient_comparison(
    measured: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | None]:
    """Return the same two numbers the unsharded sweep reports for a gradient."""

    scale = float(reference.norm())
    return {
        "relative_l2_error": (
            None if scale == 0.0 else float((measured - reference).norm()) / scale
        ),
        "cosine_similarity": (
            None
            if scale == 0.0 or float(measured.norm()) == 0.0
            else float(
                torch.dot(measured, reference) / (measured.norm() * reference.norm())
            )
        ),
    }


def measure_arm(
    n_qubits: int,
    angles: torch.Tensor,
    max_bond: int,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    """Measure one truncation regime on every rank, comparing on rank zero."""

    gradient_policy = "exact" if max_bond >= EXACT_CAP else "approximate"
    value, gradient, summary = sharded_arm(
        n_qubits, angles, max_bond, device, gradient_policy
    )
    arm: dict[str, Any] = {
        "max_bond": max_bond,
        "gradient_policy": gradient_policy,
        "sharded_value": value,
        "sharded_gradient": [float(item) for item in gradient],
        "rank": rank,
        "distribution_semantics": summary["distribution_semantics"],
        "world_size": summary["world_size"],
        "local_world_size": summary["local_world_size"],
        "node_count": summary["node_count"],
        "discarded_weight": summary["discarded_weight"],
        "svd_factorization_count": summary["svd_factorization_count"],
        "qr_factorization_count": summary["qr_factorization_count"],
        "gradient_accuracy": summary["gradient_accuracy"],
        "replicated_autograd": summary["replicated_autograd"],
        "full_mps_reconstruction": summary["full_mps_reconstruction"],
        "saved_tensor_bytes": summary["saved_tensor_bytes"],
        "reverse_tape_records": summary["reverse_tape_records"],
        "blockers": summary["blockers"],
    }
    if rank == 0:
        reference_value, reference_gradient = unsharded_reference(
            n_qubits, angles, max_bond
        )
        arm["unsharded_value"] = reference_value
        arm["unsharded_gradient"] = [float(item) for item in reference_gradient]
        arm["value_relative_error"] = compare(value, reference_value)
        arm["gradient"] = gradient_comparison(gradient, reference_gradient)
        arm["truncated"] = bool(summary["discarded_weight"] > 0)
    return arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(arguments.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        n_qubits = WIDTH_AT_WORLD.get(world, 2 * world)
        angles = shared_angles(arguments.seed).to(device)
        arms = [
            measure_arm(n_qubits, angles, cap, device, rank)
            for cap in (EXACT_CAP, TRUNCATING_CAP)
        ]
        payload: dict[str, Any] = {
            "schema": "flagquantum.mps_placement_conformance.v1",
            "n_qubits": n_qubits,
            "seed": arguments.seed,
            "world_size": world,
            "rank": rank,
            "backend": arguments.backend,
            "device": str(device),
            "arms": arms,
            "claim_boundary": (
                "single-host CPU placement semantics; not capacity, speed, "
                "accelerator or scalability evidence"
            ),
        }
        if arguments.output is not None:
            target = arguments.output.with_suffix(f".rank{rank}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2) + "\n")
        if rank == 0:
            print(json.dumps(payload))
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
