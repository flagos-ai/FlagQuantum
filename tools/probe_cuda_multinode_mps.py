#!/usr/bin/env python
"""Verify CUDA MPS site shards across two hosts, forward, backward and resume.

The pair of hosts this runs on is the whole point and the whole limit. Each host
runs as many ranks as the caller asked for, one device each, and the artifact
records the shape it got. What it establishes is that one logical MPS workload
is partitioned across that shape by site ownership, that a gate spanning the
owned site boundary is exchanged over the inter-node transport rather than
reconstructed locally, that the partitioned gradient agrees with an exact
statevector reference, and that an owner-sharded optimizer step survives a
checkpoint and a restart. A forward-only probe cannot support a training claim,
and a training claim that stops at the first optimizer step is not a training
path either.

The reference is the complex128 statevector path rather than a single-device
MPS, because agreeing with another MPS run would not show that site sharding
preserved the state.

It makes no scale claim: the scope, the claim flags and the blockers in the
artifact say exactly how narrow the shape is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.experimental.distributed import train_distributed_mps
from flagquantum.runtime.executors.mps.forward import (
    execute_torch_distributed_mps_forward,
    gather_mps_for_validation,
)
from flagquantum.runtime.executors.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)

ROOT = Path(__file__).resolve().parents[1]

#: The forward scope. Six wires at two ranks gives three owned sites per rank,
#: so the pair of wires in the middle straddles the ownership boundary and the
#: exchange is exercised by an ordinary adjacent gate.
N_WIRES = 6
COMPLEX_DTYPE = torch.complex128
REAL_DTYPE = torch.float64

#: The bond limit for every leg. Six wires cannot carry a bond dimension above
#: thirty-two, so this is headroom rather than truncation: an exact MPS here,
#: which is what makes a round-off tolerance meaningful.
MAX_BOND = 64

#: Numerical tolerance for every comparison against the exact statevector
#: reference. Complex128 with an exact reverse should agree to round-off, so
#: this is round-off rather than a modelling allowance.
NUMERICAL_TOLERANCE = 1e-10
#: The same, for the multi-step training trajectory: two runs of four steps
#: accumulate more round-off than a single expectation does.
TRAINING_TOLERANCE = 1e-9

TRAINING_OPTIMIZER = "sgd"
TRAINING_LR = 0.05
#: The first leg stops here and the second leg resumes to `RESUMED_STEPS`.
CHECKPOINT_STEPS = 2
RESUMED_STEPS = 4

#: The parameters the trainable circuit binds. Fixed so that the reference and
#: the distributed run are the same function of the same numbers.
TRAINABLE_VALUES = (0.23, -0.37)

#: The owned-site boundary at two ranks: rank 0 owns wires 0..2 and rank 1 owns
#: wires 3..5, so this pair is the one adjacent gate that has to be exchanged.
BOUNDARY_PAIR = (N_WIRES // 2 - 1, N_WIRES // 2)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _synchronize(device: torch.device) -> None:
    """Wait for the device's work, so a reported failure is the one that happened.

    Every real run is on an accelerator -- the probe refuses to start without
    CUDA -- and this stays device-guarded so that the numerical helpers can be
    exercised on a CPU-only checkout.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _forward_circuit() -> fq.Circuit:
    """An adjacent-gate circuit whose middle gate crosses the site boundary."""

    return (
        fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
        .h(0)
        .ry(1, 0.31)
        .cx(0, 1)
        .rx(2, -0.27)
        .cx(1, 2)
        .cx(*BOUNDARY_PAIR)
        .ry(3, 0.41)
        .rz(4, 0.19)
        .cx(3, 4)
        .cx(4, 5)
    )


def _bind_trainable(
    circuit: fq.Circuit, theta: torch.Tensor, phi: torch.Tensor
) -> None:
    """Bind both trainable parameters, one of them twice.

    One parameter is bound twice and the entangling gate spans the shard
    boundary, so a gradient that is wrong in its reduction or in its shard
    ownership cannot agree with the reference by accident.
    """

    (circuit.h(0).ry(N_WIRES - 1, theta).rxx(*BOUNDARY_PAIR, phi).rz(1, theta))


def _trainable_circuit(
    *,
    device: torch.device,
    dtype: torch.dtype = COMPLEX_DTYPE,
    real_dtype: torch.dtype = REAL_DTYPE,
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...]]:
    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=real_dtype, device=device)
        # Allocation followed by an explicit copy, so an accidental narrowing to
        # float32 is an observable mismatch rather than a silent test of FP32.
        tensor.copy_(torch.tensor(value, dtype=real_dtype))
        parameters.append(tensor.requires_grad_())
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    _bind_trainable(circuit, *parameters)
    return circuit, tuple(parameters)


#: The two measured wires. They sit in different ranks -- wire 2 is the last site
#: rank 0 owns and wire 5 the last site rank 1 owns -- so contracting the
#: observable needs both, and the pair is not degenerate on this circuit: the
#: expectation is 0.9078 rather than 0 or 1, and it has a non-zero gradient with
#: respect to both trainable parameters. A degenerate pair would let a broken
#: reduction agree with the reference, because a constant is its own gradient.
OBSERVABLE_WIRES = (N_WIRES // 2 - 1, N_WIRES - 1)


def _z_parity_expectation(
    probabilities: torch.Tensor, wires: tuple[int, ...]
) -> torch.Tensor:
    """Sum |amplitude|^2 with a sign per measured wire, in the basis bit order."""

    indices = torch.arange(probabilities.numel(), device=probabilities.device)
    sign = torch.ones_like(probabilities)
    for wire in wires:
        bit = (indices >> (N_WIRES - wire - 1)) & 1
        sign = sign * (1 - 2 * bit)
    return torch.sum(probabilities * sign)


def _reference_expectation() -> tuple[float, tuple[float, ...]]:
    """The same expectation and gradient on one CPU device, for the differential.

    This is the exact statevector path, deliberately not a single-device MPS:
    the claim under test is that site sharding preserved the state, and two MPS
    runs agreeing would not show that.
    """

    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=REAL_DTYPE)
        tensor.copy_(torch.tensor(value, dtype=REAL_DTYPE))
        parameters.append(tensor.requires_grad_())
    circuit = fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
    _bind_trainable(circuit, *parameters)
    state = circuit.state(refresh=True)
    expectation = _z_parity_expectation(
        state.abs().square().reshape(-1), OBSERVABLE_WIRES
    )
    expectation.backward()
    return (
        float(expectation.detach()),
        tuple(float(parameter.grad.detach()) for parameter in parameters),
    )


def _reference_statevector() -> torch.Tensor:
    """The exact single-device statevector the distributed forward must match."""

    return _forward_circuit().state(refresh=True)[0].detach().cpu()


def _network_observation(
    path: Path | None, *, local_world_size: int = 1
) -> dict[str, Any]:
    configured_interface = os.environ.get("NCCL_SOCKET_IFNAME")
    ib_disabled = os.environ.get("NCCL_IB_DISABLE") == "1"
    observation: dict[str, Any] = {
        "backend": "nccl",
        "configured_interface": configured_interface,
        "ib_disabled": ib_disabled,
        "route": "socket" if ib_disabled else "nccl_auto",
        "evidence_level": "configured_only",
        # The launcher gives one `torchrun` per node one debug log, so above one
        # rank per node the file holds every local rank's view rather than one
        # rank's. Recorded rather than assumed: the route check reads the same
        # either way, but how much of the traffic it covers does not.
        "debug_log_scope": "node" if local_world_size > 1 else "rank",
    }
    if path is None:
        return observation
    content = path.read_text(encoding="utf-8", errors="replace")
    socket_observed = "NET/Socket" in content
    interface_observed = bool(configured_interface and configured_interface in content)
    observation.update(
        {
            "debug_log_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "socket_transport_observed": socket_observed,
            "configured_interface_observed": interface_observed,
            "evidence_level": "observed_debug_log",
        }
    )
    if ib_disabled and not (socket_observed and interface_observed):
        raise RuntimeError(
            "NCCL debug log does not confirm the configured socket route"
        )
    return observation


def _collective_verdict(device: torch.device, *, accepted: bool) -> bool:
    """Let rank 0 decide and both ranks act on the same answer.

    A rank that raised on its own would leave its peer waiting in the next
    collective, which reports as a hang rather than as the reason the run was
    refused.
    """

    verdict = torch.ones((), dtype=torch.int32, device=device)
    if dist.get_rank() == 0 and not accepted:
        verdict.zero_()
    dist.broadcast(verdict, src=0)
    return bool(int(verdict.item()))


class _WorkloadGates(NamedTuple):
    """What the forward leg's gates did, summed over the world.

    The per-rank counts are not the workload's. Ownership is rebalanced as the
    circuit runs, so a rank can observe a gate that spans two other ranks' sites
    and take no part in it; and a rank can own a site that no one-site gate acts
    on. What the probe has to refuse is a workload that never crossed an
    owned-site boundary, and one that left a rank with nothing to do. Both are
    properties of the sum, and neither is a property of one rank's count, so a
    per-rank count is asserted against nothing here.
    """

    local_gates: int
    boundary_gates: int
    boundary_messages: int
    boundary_bytes: int
    least_busy_rank_gates: int


def _workload_gates(
    device: torch.device,
    *,
    local_gates: int,
    boundary_gates: int,
    boundary_messages: int,
    boundary_bytes: int,
) -> _WorkloadGates:
    """Reduce this rank's gate accounting to the workload's.

    Every rank reaches the same totals, so every rank reaches the same verdict
    and a refusal is raised by all of them rather than leaving the others
    waiting in the next collective.
    """

    totals = torch.tensor(
        [local_gates, boundary_gates, boundary_messages, boundary_bytes],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(totals)
    # The sum cannot show an idle rank: one rank's four gates and another's none
    # add up to the same total as two each, so the least busy rank is asked
    # separately.
    least_busy = torch.tensor(
        [local_gates + boundary_gates], dtype=torch.int64, device=device
    )
    dist.all_reduce(least_busy, op=dist.ReduceOp.MIN)
    values = [int(value) for value in totals.tolist()]
    return _WorkloadGates(
        local_gates=values[0],
        boundary_gates=values[1],
        boundary_messages=values[2],
        boundary_bytes=values[3],
        least_busy_rank_gates=int(least_busy.item()),
    )


def _workload_total(device: torch.device, value: int) -> int:
    """Sum one rank's byte counter into the workload's.

    The reverse attributes a layer-boundary halo to the rank that received it,
    so at a shape where the host boundary cuts the wire the halo crosses, the
    ranks inside a node report nothing for it. A workload total is what says
    the exchange happened at all; every rank reaches the same number, so every
    rank reaches the same verdict.
    """

    total = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total.item())


def _split_parameter_ownership(
    ownership: Sequence[Mapping[str, Any]], *, local_world_size: int
) -> Mapping[str, Any]:
    """The one parameter the reverse attributed to two ranks, across two hosts.

    The parameter bound on both rank-owned halves of the site partition is why
    the reverse reduces parameter gradients at all: a reverse that attributed
    every parameter to a single rank would be shedding a partial gradient rather
    than an owner-sharded one. Which ranks share it follows from where the
    sharding put the wires the parameter is bound on, and that moves with the
    shape, so the pair is derived from the placement instead of being written
    down as one shape's answer. Two owners inside one host are refused: the
    reduction would then never cross the node boundary this lane exists to
    exercise.
    """

    spanning = [item for item in ownership if len(item["owner_ranks"]) == 2]
    if len(spanning) != 1:
        raise RuntimeError(
            "the reverse did not attribute exactly one parameter to two ranks: "
            f"{list(ownership)}"
        )
    owners = tuple(int(owner) for owner in spanning[0]["owner_ranks"])
    if len({owner // local_world_size for owner in owners}) != 2:
        raise RuntimeError(
            f"the parameter the reverse attributed to ranks {owners} is owned "
            "inside one host, so the reduction it needs never crosses the node "
            "boundary"
        )
    return spanning[0]


def _prepare_checkpoint_directory(
    directory: Path, *, rank: int, device: torch.device
) -> None:
    """Start from a directory holding no checkpoints, or not at all.

    The resumed leg reads what the first leg wrote, so a checkpoint left by an
    earlier run would be restored instead and the artifact would describe a
    trajectory this run never computed. The caller empties the directory; this
    is the second opinion, because the two must not disagree.
    """

    verdict = torch.ones((), dtype=torch.int32, device=device)
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            existing = sorted(directory.glob("rank-*.pt"))
            if existing:
                verdict.zero_()
        except OSError:
            verdict.zero_()
    dist.broadcast(verdict, src=0)
    if int(verdict.item()) == 0:
        raise RuntimeError(
            f"{directory} could not be used as an empty checkpoint directory; "
            "this probe resumes from what its own first leg wrote"
        )
    dist.barrier()


def _training_observations(
    *,
    device: torch.device,
    checkpoint_directory: Path,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Run the legs that make the training claim, and describe them.

    The resumed run and the uninterrupted run are the same computation over the
    same steps. Comparing their losses is what shows the checkpoint restored the
    parameters, the optimizer state and the step counter rather than merely
    reloading a file.

    Each leg gets its own circuit, because a training step writes through to the
    tensors the circuit binds: three legs sharing one circuit would run the
    second and third legs from wherever the first left the parameters, and the
    comparison would then be between two different computations.

    Each leg also gets its own checkpoint directory, because the engine refuses
    to start on a directory that already holds a committed generation.
    """

    interrupted, _ = _trainable_circuit(device=device)
    first = train_distributed_mps(
        interrupted,
        steps=CHECKPOINT_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        device=device,
        max_bond=MAX_BOND,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
    )
    _synchronize(device)

    uninterrupted, _ = _trainable_circuit(device=device)
    fresh = train_distributed_mps(
        uninterrupted,
        steps=RESUMED_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        device=device,
        max_bond=MAX_BOND,
        checkpoint_dir=checkpoint_directory / "uninterrupted",
        checkpoint_interval=1,
    )
    _synchronize(device)

    resuming, _ = _trainable_circuit(device=device)
    resumed = train_distributed_mps(
        resuming,
        steps=RESUMED_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        device=device,
        max_bond=MAX_BOND,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
        resume=True,
    )
    _synchronize(device)

    first_summary = first.summary()
    resumed_summary = resumed.summary()
    fresh_summary = fresh.summary()
    # Every leg is a separate process-group execution, and the resume claim is
    # only meaningful if the two legs it compares ran under the same topology.
    for leg, summary in (
        ("interrupted", first_summary),
        ("uninterrupted", fresh_summary),
        ("resumed", resumed_summary),
    ):
        _require_two_node_placement(
            summary, leg=f"training/{leg}", expected_world_size=world_size
        )
    metrics = {
        "resume_prefix_max_abs_error": max(
            (
                abs(actual - expected)
                for actual, expected in zip(
                    first.losses, fresh.losses[:CHECKPOINT_STEPS], strict=True
                )
            ),
            default=0.0,
        ),
        "resume_trajectory_max_abs_error": max(
            (
                abs(actual - expected)
                for actual, expected in zip(
                    resumed.losses, fresh.losses[CHECKPOINT_STEPS:], strict=True
                )
            ),
            default=0.0,
        ),
    }
    # The artifact carries the comparison; the per-rank records carry the
    # runtime summary. Keeping them separate means neither the trajectories nor
    # the placement evidence is recorded twice.
    observations = {
        "optimizer": TRAINING_OPTIMIZER,
        "learning_rate": TRAINING_LR,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "resumed_steps": RESUMED_STEPS,
        "first_leg_losses": list(first.losses),
        "resumed_leg_losses": list(resumed.losses),
        "uninterrupted_leg_losses": list(fresh.losses),
        "resumed_start_step": resumed.start_step,
        "uninterrupted_start_step": fresh.start_step,
        "checkpoint_files": list(resumed.checkpoint_files),
        "checkpoint_storage_semantics": resumed_summary["checkpoint_storage_semantics"],
        "checkpoint_integrity": resumed_summary["checkpoint_integrity"],
        "optimizer_state_ownership_semantics": resumed_summary[
            "optimizer_state_ownership_semantics"
        ],
        "interrupted_leg_checkpoint_files": list(first.checkpoint_files),
        "uninterrupted_leg_checkpoint_files": list(fresh.checkpoint_files),
    }
    return observations, metrics, resumed_summary


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    training: dict[str, Any],
    world_size: int,
    local_world_size: int,
) -> dict[str, Any]:
    evidence = {
        "schema": "flagquantum.cuda_multinode_mps_probe.v1",
        "status": "passed",
        "environment": {
            "source_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
        },
        "scope": {
            "dtype": "complex128",
            "n_wires": N_WIRES,
            "max_bond": MAX_BOND,
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": world_size // local_world_size,
            # Which sites each rank owned, read off the records rather than
            # written down here: the partitions change with the world size, and
            # a fixed pair of lists would have been a fact about one shape.
            "site_ownership": [record["owned_sites"] for record in rank_records],
            "observable_wires": list(OBSERVABLE_WIRES),
            "distribution_semantics": "sharded_across_ranks",
            "execution": "forward_backward_optimizer_and_checkpoint_resume",
        },
        "observations": {
            "numerical_metrics": metrics,
            "training": training,
            "rank_records": rank_records,
            "validation_full_mps_materialization": True,
            "production_full_mps_materialization": False,
        },
        "claim_blockers": [
            "validation_only_tiny_full_mps_gather",
            "inter_node_cut_width_not_swept",
            "rdma_not_tested",
            "production_performance_not_measured",
            "two_node_pair_only_no_wider_topology",
            "toy_circuit_parameters_only",
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = _canonical_sha256(evidence)
    return {
        "schema": "flagquantum.cuda_multinode_mps_artifact.v1",
        "evidence_sha256": digest,
        "evidence": evidence,
    }


def _require_two_node_placement(
    summary: dict[str, Any], *, leg: str, expected_world_size: int
) -> None:
    """Refuse a leg the planner placed differently from the launched shape.

    `expected_world_size` is what torchrun was told; what the runtime resolved
    is compared against it, so a leg that quietly ran somewhere else -- or on
    one node while the artifact says two -- is refused rather than recorded.
    """

    if summary["node_count"] != 2:
        raise RuntimeError(
            f"the {leg} leg did not resolve the two-node placement: "
            f"local_world_size={summary['local_world_size']} "
            f"node_count={summary['node_count']}"
        )
    if summary["world_size"] != expected_world_size:
        raise RuntimeError(
            f"the {leg} leg ran at {summary['world_size']} ranks where "
            f"{expected_world_size} were launched"
        )
    if summary["distribution_semantics"] != "sharded_across_ranks":
        raise RuntimeError(
            f"the {leg} leg did not report sharded semantics: "
            f"{summary['distribution_semantics']}"
        )


def _declared_shape() -> tuple[int, int]:
    """The 2xN shape this probe was launched in, refused unless it is one.

    The lane is a pair of hosts; how many ranks each host runs is the caller's
    choice, and the artifact records whichever answer it got. Anything else --
    one host standing in for a pair, or ranks placed unevenly across the two --
    is refused here, before the process group is built, because a shape mismatch
    found after it reports as a collective that never completes rather than as
    the shape that was refused.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world_size < 1 or local_world_size < 1:
        raise RuntimeError(
            "probe needs a positive WORLD_SIZE and LOCAL_WORLD_SIZE, received "
            f"world_size={world_size} local_world_size={local_world_size}"
        )
    if world_size % local_world_size:
        raise RuntimeError(
            "probe needs ranks placed evenly across the nodes, received "
            f"world_size={world_size} local_world_size={local_world_size}"
        )
    node_count = world_size // local_world_size
    if node_count != 2:
        raise RuntimeError(
            f"probe requires exactly two nodes, received {node_count} "
            f"(world_size={world_size} local_world_size={local_world_size})"
        )
    return world_size, local_world_size


def probe(
    *, network_log: Path | None, checkpoint_directory: Path
) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size, local_world_size = _declared_shape()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    payload: dict[str, Any] | None = None
    try:
        _prepare_checkpoint_directory(checkpoint_directory, rank=rank, device=device)
        circuit = _forward_circuit()
        forward = execute_torch_distributed_mps_forward(
            circuit,
            device=device,
            max_bond=MAX_BOND,
        )
        _synchronize(device)
        forward_summary = forward.summary()
        _require_two_node_placement(
            forward_summary, leg="forward", expected_world_size=world_size
        )
        if forward_summary["claim_evidence_type"] != "accelerator_semantics":
            raise RuntimeError(
                "the forward leg did not report accelerator semantics: "
                f"{forward_summary['claim_evidence_type']}"
            )
        if forward.local_gate_count < 0 or forward.boundary_gate_count < 0:
            raise RuntimeError(
                "the forward leg reported a negative gate count: "
                f"local={forward.local_gate_count} "
                f"boundary={forward.boundary_gate_count}"
            )

        # The forward result counts boundary messages and bytes but does not
        # split them by node tier, so the split is not claimed here. At one rank
        # per node the placement is the split: each rank is alone on its host,
        # so every counted exchange left it. Above that the sum mixes the two
        # tiers -- ranks on the same host exchange across their own boundaries
        # too -- and the reverse leg's halo bytes, which are attributed by tier,
        # are what keeps the inter-node transport evidenced at those shapes.
        gates = _workload_gates(
            device,
            local_gates=forward.local_gate_count,
            boundary_gates=forward.boundary_gate_count,
            boundary_messages=forward.boundary_messages,
            boundary_bytes=forward.boundary_bytes,
        )
        metrics: dict[str, float] = {
            "forward_boundary_messages": float(gates.boundary_messages),
            "forward_boundary_bytes": float(gates.boundary_bytes),
        }
        if not _collective_verdict(
            device,
            accepted=(
                gates.boundary_messages >= 1
                and gates.boundary_bytes > 0
                and gates.least_busy_rank_gates >= 1
            ),
        ):
            raise RuntimeError(
                "the workload did not exercise an owned-site boundary on every "
                f"rank and exchange across it: {gates}"
            )
        if forward_summary["full_state_materialization"]:
            raise RuntimeError("the forward leg materialized the full state")
        if forward_summary["full_mps_reconstruction_count"] != 0:
            raise RuntimeError(
                "the forward leg reconstructed an MPS: "
                f"{forward_summary['full_mps_reconstruction_count']}"
            )

        gathered = gather_mps_for_validation(forward)
        accepted = True
        statevector_metrics: dict[str, float] = {}
        if rank == 0:
            materialized = gathered.to_statevector()[0].detach().cpu()
            reference = _reference_statevector()
            statevector_metrics = {
                "statevector_max_abs_error": float(
                    torch.max(torch.abs(materialized - reference)).item()
                ),
                "statevector_norm_error": float(
                    torch.abs(torch.linalg.vector_norm(materialized) - 1).item()
                ),
                "statevector_infidelity": max(
                    0.0,
                    float(
                        1
                        - torch.abs(torch.vdot(reference, materialized)).square().item()
                    ),
                ),
            }
            accepted = all(
                value <= NUMERICAL_TOLERANCE for value in statevector_metrics.values()
            )
        if not _collective_verdict(device, accepted=accepted):
            raise RuntimeError(
                f"distributed forward validation failed: {statevector_metrics}"
            )
        metrics.update(statevector_metrics)

        trainable, parameters = _trainable_circuit(device=device)
        gradient = execute_torch_distributed_mps_reverse(
            trainable,
            observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
            device=device,
            max_bond=MAX_BOND,
            # Explicit rather than left to `_resolve_compile_site_kernels`: the
            # auto policy would decline for a checkpointing run and this leg
            # checkpoints nothing, and the compiled bucket path is what the
            # accelerator configuration uses for a repeated training workload.
            # It is also the only path on which the reverse prefetches a
            # layer-boundary halo, so without it the reverse would cross the
            # host boundary only through the parameter-gradient all-reduce and
            # the artifact would carry no measured inter-node reverse bytes.
            compile_site_kernels=True,
        )
        gradient.backward()
        _synchronize(device)
        gradient_summary = gradient.summary()
        _require_two_node_placement(
            gradient_summary, leg="backward", expected_world_size=world_size
        )
        if gradient_summary["mps_backward_execution"] != "completed":
            raise RuntimeError(
                "the backward leg did not complete: "
                f"{gradient_summary['mps_backward_execution']}"
            )
        if gradient_summary["gradient_accuracy"] != "exact":
            raise RuntimeError(
                "the backward leg did not report an exact gradient: "
                f"{gradient_summary['gradient_accuracy']}"
            )
        if gradient_summary["replicated_autograd"]:
            raise RuntimeError("the backward leg ran a replicated autograd")
        if gradient_summary["full_mps_reconstruction"]:
            raise RuntimeError("the backward leg reconstructed the full MPS")
        # The tier attribution is what keeps the inter-node transport evidenced
        # once a node holds more than one rank, where a rank's own forward
        # boundary count no longer implies that the exchange left its host. The
        # halo belongs to the rank that received it, so the figure the workload
        # is judged on is the total over the ranks, not any one rank's.
        halo_inter_node_bytes = _workload_total(
            device, gradient_summary["layer_halo_inter_node_bytes"]
        )
        halo_intra_node_bytes = _workload_total(
            device, gradient_summary["layer_halo_intra_node_bytes"]
        )
        if not _collective_verdict(device, accepted=halo_inter_node_bytes > 0):
            raise RuntimeError(
                "the reverse layer halo was not attributed to the inter-node "
                f"tier anywhere in the workload: {halo_inter_node_bytes} bytes"
            )
        if local_world_size == 1 and halo_intra_node_bytes != 0:
            # Above one rank per node an intra-node halo is expected -- the
            # ranks sharing a host do exchange across their own site boundaries.
            # At exactly one rank per node every rank is alone on its host, so a
            # byte attributed to the intra-node tier means the tier split is
            # wrong rather than that the workload grew.
            raise RuntimeError(
                "one rank per node must attribute no reverse halo bytes to the "
                f"intra-node tier: {halo_intra_node_bytes}"
            )
        if (
            gradient_summary["gradient_collective_count"] < 1
            or gradient_summary["gradient_collective_bytes"] <= 0
        ):
            raise RuntimeError(
                "the parameter gradients were never reduced across the ranks: "
                f"{gradient_summary['gradient_collective_count']} collectives, "
                f"{gradient_summary['gradient_collective_bytes']} bytes"
            )
        split = _split_parameter_ownership(
            gradient_summary["parameter_ownership"],
            local_world_size=local_world_size,
        )
        if int(split["occurrence_count"]) != 2:
            raise RuntimeError(
                "the parameter attributed to both ranks is not the one bound "
                f"twice: {gradient_summary['parameter_ownership']}"
            )

        reference_expectation, reference_gradients = _reference_expectation()
        actual_expectation = float(gradient.value.detach().cpu())
        actual_gradients = tuple(
            float(parameter.grad.detach().cpu()) for parameter in parameters
        )
        gradient_metrics = {
            "expectation_value_error": abs(actual_expectation - reference_expectation),
            "gradient_max_abs_error": max(
                abs(actual - expected)
                for actual, expected in zip(
                    actual_gradients, reference_gradients, strict=True
                )
            ),
        }
        # Counts and byte totals, kept out of the error metrics so that nothing
        # downstream can read a byte count as a tolerance. Every figure here is
        # this rank's own; the workload's totals are the ones in the numerical
        # metrics, and the two only agree at a shape where no rank can observe a
        # boundary gate without taking part in it.
        communication = {
            "forward_boundary_messages": forward.boundary_messages,
            "forward_boundary_bytes": forward.boundary_bytes,
            "forward_communication_messages": forward_summary["communication_messages"],
            "forward_communication_bytes": forward_summary["communication_bytes"],
            "forward_local_tensor_bytes_by_rank": list(
                forward_summary["local_tensor_bytes_by_rank"]
            ),
            "backward_layer_halo_inter_node_bytes": gradient_summary[
                "layer_halo_inter_node_bytes"
            ],
            "backward_layer_halo_intra_node_bytes": gradient_summary[
                "layer_halo_intra_node_bytes"
            ],
            "backward_gradient_collective_count": gradient_summary[
                "gradient_collective_count"
            ],
            "backward_gradient_collective_bytes": gradient_summary[
                "gradient_collective_bytes"
            ],
            "backward_shared_parameter_reduction": gradient_summary[
                "shared_parameter_reduction"
            ],
        }
        gradients_ok = _collective_verdict(
            device,
            accepted=all(
                value <= NUMERICAL_TOLERANCE for value in gradient_metrics.values()
            ),
        )
        if not gradients_ok:
            raise RuntimeError(
                f"distributed gradient validation failed: {gradient_metrics}"
            )
        metrics.update(gradient_metrics)

        training, training_metrics, training_summary = _training_observations(
            device=device,
            checkpoint_directory=checkpoint_directory,
            world_size=world_size,
        )
        training_ok = all(
            value <= TRAINING_TOLERANCE for value in training_metrics.values()
        )
        if not _collective_verdict(device, accepted=training_ok):
            raise RuntimeError(
                f"distributed training validation failed: {training_metrics}"
            )
        metrics.update(training_metrics)
        # A frozen expectation would make every resume comparison agree for the
        # wrong reason, so the uninterrupted trajectory is required to move. In
        # which direction it moves is recorded rather than asserted: SGD on this
        # objective takes a legitimate step either way at this learning rate, and
        # the claim under test is that the checkpoint reproduces the trajectory,
        # not that the learning rate was well chosen.
        uninterrupted_losses = training["uninterrupted_leg_losses"]
        if len(set(uninterrupted_losses)) != len(uninterrupted_losses):
            raise RuntimeError(
                "the optimizer step did not change the expectation, so the resume "
                f"comparison would agree for the wrong reason: {uninterrupted_losses}"
            )
        metrics["training_loss_decrease"] = float(
            uninterrupted_losses[0] - uninterrupted_losses[-1]
        )
        metrics["training_final_loss"] = float(uninterrupted_losses[-1])
        # The optimizer minimizes the expectation the reverse leg computed, so
        # the first training loss is that same number reached from the same
        # parameters. Comparing them ties the training objective to the exact
        # statevector reference instead of leaving it self-referential.
        metrics["first_leg_initial_expectation_error"] = abs(
            float(training["first_leg_losses"][0]) - reference_expectation
        )
        if int(training["resumed_start_step"]) != CHECKPOINT_STEPS:
            raise RuntimeError(
                "the resumed leg did not continue from the checkpoint: "
                f"start_step={training['resumed_start_step']}"
            )
        if len(training["checkpoint_files"]) < 2:
            raise RuntimeError(
                "the checkpoint leg did not write one shard per rank: "
                f"{training['checkpoint_files']}"
            )
        # The training legs checkpoint, so the engine's auto site-kernel policy
        # keeps them eager and their step metrics report no layer halo. Recorded
        # rather than left implicit: a reader comparing the legs' exchange
        # totals should be able to see why they differ.
        last_step = training_summary["step_metrics"][-1]
        communication.update(
            {
                "training_site_kernel_execution": training_summary[
                    "site_kernel_execution"
                ],
                "training_site_kernel_selection_reason": training_summary[
                    "site_kernel_selection_reason"
                ],
                "training_communication_protocol": training_summary[
                    "communication_protocol"
                ],
                "training_optimizer_ownership_semantics": training_summary[
                    "optimizer_state_ownership_semantics"
                ],
                "training_last_step_boundary_exchanges": last_step[
                    "boundary_exchanges"
                ],
                "training_last_step_boundary_bytes": last_step["boundary_bytes"],
                "training_last_step_gradient_collective_count": last_step[
                    "gradient_collective_count"
                ],
                "training_last_step_layer_halo_inter_node_bytes": last_step[
                    "layer_halo_inter_node_bytes"
                ],
            }
        )

        properties = torch.cuda.get_device_properties(device)
        record = forward_summary
        record.update(
            {
                "hostname_sha256": hashlib.sha256(
                    platform.node().encode("utf-8")
                ).hexdigest(),
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "")),
                "owned_sites": list(forward.shard_state.ownership[rank]),
                "boundary_gate_count": forward.boundary_gate_count,
                "communication": communication,
                "backward": gradient_summary,
                "training_step": training_summary["step_metrics"][-1],
            }
        )
        # One slot per rank: `all_gather_object` fills exactly the list it is
        # given, so a fixed-length list would truncate a wider shape.
        rank_records: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_records, record)
        if rank == 0:
            payload = _artifact(
                rank_records=[item for item in rank_records if item is not None],
                metrics=metrics,
                training=training,
                world_size=world_size,
                local_world_size=local_world_size,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0 and payload is not None:
        network = _network_observation(network_log, local_world_size=local_world_size)
        evidence = payload["evidence"]
        evidence["observations"]["network"] = network
        payload["evidence_sha256"] = _canonical_sha256(evidence)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
        help=(
            "shared, empty directory the training legs write rank-local "
            "checkpoints into and resume from"
        ),
    )
    parser.add_argument(
        "--network-log",
        type=Path,
        help="rank-zero NCCL debug log used to verify the selected network route",
    )
    args = parser.parse_args()
    if not shutil.which("git") and not os.environ.get("FLAGQUANTUM_SOURCE_REVISION"):
        # The artifact has to name a revision; a checkout without git can still
        # be pinned by the caller, and otherwise it cannot say what it measured.
        raise SystemExit(
            "FLAGQUANTUM_SOURCE_REVISION is required when the staged tree "
            "carries no git checkout"
        )
    payload = probe(
        network_log=args.network_log,
        checkpoint_directory=args.checkpoint_directory,
    )
    if payload is None:
        return 0
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
