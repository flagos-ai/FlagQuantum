#!/usr/bin/env python
"""Verify one CUDA tensor-network slice per rank across forward, gradient, training and resume.

The pair this runs on is the whole point and the whole limit: two ranks, one
device each, on two hosts. What it establishes is that one logical
tensor-network workload is partitioned across the two by internal-edge slices,
that each rank contracts only the slices it owns, that the sliced amplitudes and
the sliced gradient agree with an exact statevector reference, and that an
owner-sharded optimizer step survives a checkpoint and a restart. A forward-only
probe cannot support a training claim, and a training claim that stops at the
first optimizer step is not a training path either.

The reference is the complex128 statevector path rather than a single-device
tensor-network run, because agreeing with another contraction of the same graph
would not show that slicing preserved the state.

The sliced labels are a scope constant rather than a runtime choice: the
automatic slicer picks the budget-cheapest cut, and an arbitrary cut can leave
every slice but one of a key exactly zero, which would let a reduction that
dropped a rank still reproduce the reference value. The pair below gives four
non-zero slices, two per rank, and each rank a non-zero share of the gradient.

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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.execution import (
    distributed_tensor_network_amplitude,
    distributed_tensor_network_amplitudes,
    distributed_tensor_network_expectation,
)
from flagquantum.runtime.executors.tensor_network.training import (
    train_distributed_tensor_network,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    build_tensor_network_expectation,
)

ROOT = Path(__file__).resolve().parents[1]

#: Five wires keeps the exact reference cheap while still giving the contraction
#: a deep enough graph that slicing splits it into independent work.
N_WIRES = 5
COMPLEX_DTYPE = torch.complex128
REAL_DTYPE = torch.float64

#: Numerical tolerance for every comparison against the exact statevector
#: reference. Complex128 with exact slicing should agree to round-off, so this
#: is round-off rather than a modelling allowance.
NUMERICAL_TOLERANCE = 1e-10
#: The same, for the multi-step training trajectory: two runs of four steps
#: accumulate more round-off than a single expectation does.
TRAINING_TOLERANCE = 1e-9

#: The only optimizer the TN training entry implements.
TRAINING_OPTIMIZER = "sgd"
TRAINING_LR = 0.05
#: The first leg stops here and the second leg resumes to `RESUMED_STEPS`.
CHECKPOINT_STEPS = 2
RESUMED_STEPS = 4

#: The parameters the trainable circuit binds. Fixed so that the reference and
#: the distributed run are the same function of the same numbers.
TRAINABLE_VALUES = (0.23, -0.37)
#: A second, distinct parameter point, so the forward legs are not measured at
#: the point the gradient legs and the training trajectory start from.
FORWARD_VALUES = (0.31, -0.23)

#: A fixed rotation on wire 0, so the two trainable parameters act on a state
#: that is not an eigenstate of the measured observable.
PHASE_ROTATION = 0.6

#: The two sliced internal labels, two slices each across two ranks. They are
#: the contracted edges of the two multi-index nodes that carry the entangling
#: `rxx`, which is what makes the pair well conditioned: the four slice values
#: are 0.8177, -0.0109, -0.1431 and +0.0019, and the two rank sums are 0.6746 and
#: -0.0090 against a total of 0.6656. A reduction that silently dropped either
#: rank would be wrong by 0.0090, which is twelve orders of magnitude above the
#: tolerance, rather than agreeing by accident.
SLICED_LABELS = (6, 7)

#: The slice count the task plan must resolve to: two sliced labels of dimension
#: two each, so four combinations and two per rank.
EXPECTED_SLICE_COUNT = 4
#: How many of those slices each rank owns.
EXPECTED_SLICES_PER_RANK = 2

#: The measured wires. They are the two ends of the entangling gate, so the
#: observable spans the whole graph and the pair is not degenerate on this
#: circuit: the expectation is 0.6656 rather than 0 or 1, and both trainable
#: parameters have non-zero gradient. A degenerate pair would let a broken
#: reduction agree with the reference, because a constant is its own gradient.
OBSERVABLE_WIRES = (0, N_WIRES - 1)

#: The amplitudes the forward leg checks. They are pairwise distinct and all
#: non-zero, and they differ in the low and the high wire, so a bit-order error
#: cannot agree by accident.
CHECKED_BITSTRINGS = ("00000", "10000", "00001", "10001")


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


def _bind_trainable(
    circuit: fq.Circuit, theta: torch.Tensor, phi: torch.Tensor
) -> None:
    """Bind both trainable parameters, one of them twice.

    The first rotation is a fixed angle, the second and the fourth carry the
    trainable angle on either side of the entangling gate. The gate therefore
    sits between two layers that do not commute with it, which is what gives the
    entangling parameter a non-zero derivative; a circuit whose only non-commuting
    layer sat before the gate would make that derivative exactly zero and the
    gradient comparison would pass for a contraction that ignored the parameter.
    """

    circuit.ry(0, PHASE_ROTATION)
    circuit.ry(N_WIRES - 1, theta)
    circuit.rxx(*OBSERVABLE_WIRES, phi)
    circuit.ry(0, theta)


def _forward_circuit(
    *, device: torch.device | None = None, dtype: torch.dtype = COMPLEX_DTYPE
) -> fq.Circuit:
    """The same circuit shape at a fixed parameter point."""

    parameters = [
        torch.tensor(value, dtype=REAL_DTYPE, device=device) for value in FORWARD_VALUES
    ]
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    _bind_trainable(circuit, *parameters)
    return circuit


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

    This is the exact statevector path, deliberately not a single-device
    tensor-network contraction: the claim under test is that slicing preserved
    the state, and two contractions of the same graph agreeing would not show it.
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
    """The exact single-device statevector the distributed amplitudes must match."""

    return _forward_circuit().state(refresh=True)[0].detach().cpu()


def _sliced_label_multiplicities() -> dict[int, int]:
    """How many nodes carry each sliced label, for the scope record.

    A label carried by a single node cannot be sliced, so recording the
    multiplicities is what shows the chosen pair is a real cut of the graph.
    """

    plan = build_tensor_network_expectation(
        build_tensor_network(_forward_circuit(), device="cpu", dtype=COMPLEX_DTYPE),
        z=list(OBSERVABLE_WIRES),
    )
    counts = Counter(label for node in plan.nodes for label in node.labels)
    return {label: counts[label] for label in SLICED_LABELS}


def _network_observation(path: Path | None) -> dict[str, Any]:
    configured_interface = os.environ.get("NCCL_SOCKET_IFNAME")
    ib_disabled = os.environ.get("NCCL_IB_DISABLE") == "1"
    observation: dict[str, Any] = {
        "backend": "nccl",
        "configured_interface": configured_interface,
        "ib_disabled": ib_disabled,
        "route": "socket" if ib_disabled else "nccl_auto",
        "evidence_level": "configured_only",
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


def _gather_floats(
    values: tuple[float, ...], *, device: torch.device
) -> list[list[float]]:
    """Gather one float row per rank, over a tensor collective.

    The launched lanes install no NumPy, so the object collectives are not
    available and this evidence has to travel as a tensor.
    """

    local = torch.tensor([list(values)], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [row.tolist() for tensor in gathered for row in tensor]


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


def _verify_checkpoint_generation(
    directory: Path,
    *,
    resumed_step: int,
    expected_task_plan_identity: str,
    device: torch.device,
) -> dict[str, Any]:
    """Check the committed generation by reading it, not by trusting the writer.

    Both ranks checkpoint into the same shared directory, so the directory is
    the one place that shows the run produced exactly one generation per rank
    and that each generation is intact and stamped with the final step. A
    leftover generation would mean the resume restored something an earlier run
    wrote, which is the failure this whole leg exists to rule out.
    """

    expected_ranks = list(range(2))
    observation: dict[str, Any] = {}
    accepted = True
    if dist.get_rank() == 0:
        try:
            shards: dict[int, dict[str, Any]] = {}
            for path in sorted(directory.glob("rank-*-tn-training.pt")):
                # The rank is in the file name, and the sidecar carries only the
                # digests, so the payload is read to check the rank and step it
                # was written with rather than taking the sidecar's word for them.
                checksum_path = path.with_name(path.name + ".sha256")
                metadata = json.loads(checksum_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("file_sha256")
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise RuntimeError(f"checkpoint failed its integrity check: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                shard_rank = int(payload["rank"])
                if path.name != f"rank-{shard_rank}-tn-training.pt":
                    raise RuntimeError(
                        f"checkpoint file name disagrees with its payload: {path.name}"
                    )
                if payload["step"] != metadata["step"]:
                    raise RuntimeError(
                        f"checkpoint sidecar disagrees with its payload: {path.name}"
                    )
                if payload["schema_version"] != metadata["schema_version"]:
                    raise RuntimeError(
                        f"checkpoint schema disagrees with its sidecar: {path.name}"
                    )
                shards[shard_rank] = {
                    "file": path.name,
                    "step": int(payload["step"]),
                    "schema_version": payload["schema_version"],
                    "parameter_sha256": metadata["parameter_sha256"],
                    "world_size": int(payload["world_size"]),
                    "local_world_size": int(payload["local_world_size"]),
                    "node_count": int(payload["node_count"]),
                    "task_plan_identity": payload["task_plan_identity"],
                }
            if sorted(shards) != expected_ranks:
                raise RuntimeError(
                    "the training legs did not leave one checkpoint per rank: "
                    f"{sorted(shards)}"
                )
            if any(item["step"] != resumed_step for item in shards.values()):
                raise RuntimeError(
                    "the committed generation is not the last step of the run: "
                    f"{ {rank: item['step'] for rank, item in shards.items()} }"
                )
            for rank, item in shards.items():
                if (
                    item["world_size"] != 2
                    or item["local_world_size"] != 1
                    or item["node_count"] != 2
                ):
                    raise RuntimeError(
                        f"rank {rank} committed a checkpoint for another placement: "
                        f"world_size={item['world_size']} "
                        f"local_world_size={item['local_world_size']} "
                        f"node_count={item['node_count']}"
                    )
            if len({item["parameter_sha256"] for item in shards.values()}) != 1:
                # The owner all-gather replicates the updated parameters onto
                # every rank before the write, so a rank-local file is the whole
                # resumed state and the two files must hold the same numbers. A
                # difference here would mean a rank resumes from a parameter
                # vector the optimizer never produced.
                raise RuntimeError(
                    "the ranks committed checkpoints holding different parameter "
                    "values, so a rank-local file is not the whole resumed state"
                )
            if len({item["task_plan_identity"] for item in shards.values()}) != 1:
                raise RuntimeError(
                    "the ranks committed checkpoints for different slice task plans"
                )
            committed_plan = next(iter(shards.values()))["task_plan_identity"]
            if committed_plan != expected_task_plan_identity:
                raise RuntimeError(
                    "the committed generation belongs to a different slice task "
                    f"plan than the run reported: {committed_plan} != "
                    f"{expected_task_plan_identity}"
                )
            observation = {
                "committed_step": resumed_step,
                "task_plan_identity": committed_plan,
                "rank_local_files_hold_replicated_parameters": True,
                "shards": {
                    str(rank): {
                        **item,
                        # The digests themselves are long and add nothing a reader
                        # can act on; the equality checks above are what matter.
                        "parameter_sha256_prefix": item["parameter_sha256"][:16],
                    }
                    for rank, item in sorted(shards.items())
                },
            }
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            observation = {"error": str(error)}
            accepted = False
    if not _collective_verdict(device, accepted=accepted):
        raise RuntimeError(f"checkpoint generation validation failed: {observation}")
    return observation


def _require_two_node_placement(summary: dict[str, Any], *, leg: str) -> None:
    if summary["local_world_size"] != 1 or summary["node_count"] != 2:
        raise RuntimeError(
            f"the {leg} leg did not resolve the two-node placement: "
            f"local_world_size={summary['local_world_size']} "
            f"node_count={summary['node_count']}"
        )
    if summary["world_size"] != 2:
        raise RuntimeError(
            f"the {leg} leg did not run at two ranks: world_size={summary['world_size']}"
        )
    if summary["distribution_semantics"] != "sharded_across_ranks":
        raise RuntimeError(
            f"the {leg} leg did not report sharded semantics: "
            f"{summary['distribution_semantics']}"
        )
    if summary["claim_evidence_type"] != "production_runtime":
        raise RuntimeError(
            f"the {leg} leg did not report accelerator runtime semantics: "
            f"{summary['claim_evidence_type']}"
        )


def _require_slice_partition(summary: dict[str, Any], *, leg: str) -> None:
    """Every rank owns a slice, and the slices are the scope's.

    A leg whose plan gave one rank nothing, or resolved to a single slice, would
    be reporting the scope but running a different shape.
    """

    if summary["slice_tasks"] != EXPECTED_SLICE_COUNT:
        raise RuntimeError(
            f"the {leg} leg resolved to {summary['slice_tasks']} slices rather "
            f"than {EXPECTED_SLICE_COUNT}"
        )
    if tuple(summary["slice_labels"]) != SLICED_LABELS:
        raise RuntimeError(
            f"the {leg} leg sliced {summary['slice_labels']} rather than "
            f"{list(SLICED_LABELS)}"
        )
    tasks_by_rank = {
        int(rank): int(count) for rank, count in summary["tasks_by_rank"].items()
    }
    if set(tasks_by_rank) != {0, 1} or any(
        count != EXPECTED_SLICES_PER_RANK for count in tasks_by_rank.values()
    ):
        raise RuntimeError(
            f"the {leg} leg did not give each rank one slice: {tasks_by_rank}"
        )
    if summary["scalability_claim_allowed"] is not False:
        raise RuntimeError(f"the {leg} leg allowed a scalability claim")


def _require_sliced_labels_are_a_cut() -> dict[int, int]:
    """The scope's labels are contracted edges, not single-node indices."""

    multiplicities = _sliced_label_multiplicities()
    if any(count < 2 for count in multiplicities.values()):
        raise RuntimeError(
            f"the scope's sliced labels are not carried by two nodes: {multiplicities}"
        )
    return multiplicities


def _contraction_arguments(
    *, device: torch.device, rank: int, local_rank: int
) -> dict[str, Any]:
    """The placement, slice scope, and dtype every distributed leg shares.

    The two ranks are one process each on two hosts. Declaring the placement
    rather than reading it back from the environment is what makes the
    node_count each leg reports a claim the probe can be held to.
    """

    return {
        "world_size": 2,
        "local_world_size": 1,
        "distributed_executor": "torch",
        "rank": rank,
        "local_rank": local_rank,
        "sliced_labels": SLICED_LABELS,
        "device": device,
        "dtype": COMPLEX_DTYPE,
    }


def _training_observations(
    *,
    device: torch.device,
    checkpoint_directory: Path,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Run the legs that make the training claim, and describe them.

    The resumed run and the uninterrupted run are the same computation over the
    same steps. Comparing their losses is what shows the checkpoint restored the
    parameters and the step counter rather than merely reloading a file.

    Each leg gets its own circuit, because a training step writes through to the
    tensors the circuit binds: three legs sharing one circuit would run the
    second and third legs from wherever the first left the parameters, and the
    comparison would then be between two different computations.

    The uninterrupted leg gets its own checkpoint directory as well. Its job is
    to describe the trajectory a restart has to reproduce, and writing its own
    checkpoints over the interrupted leg's would leave the resumed leg with
    nothing to resume from.
    """

    interrupted, interrupted_parameters = _trainable_circuit(device=device)
    first = train_distributed_tensor_network(
        interrupted,
        interrupted_parameters,
        steps=CHECKPOINT_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        learning_rate=TRAINING_LR,
        sliced_labels=SLICED_LABELS,
        checkpoint_directory=checkpoint_directory,
        checkpoint_interval=1,
    )
    _synchronize(device)

    uninterrupted, uninterrupted_parameters = _trainable_circuit(device=device)
    fresh = train_distributed_tensor_network(
        uninterrupted,
        uninterrupted_parameters,
        steps=RESUMED_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        learning_rate=TRAINING_LR,
        sliced_labels=SLICED_LABELS,
        checkpoint_directory=checkpoint_directory / "uninterrupted",
        checkpoint_interval=1,
    )
    _synchronize(device)

    resuming, resuming_parameters = _trainable_circuit(device=device)
    resumed = train_distributed_tensor_network(
        resuming,
        resuming_parameters,
        steps=RESUMED_STEPS,
        observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
        learning_rate=TRAINING_LR,
        sliced_labels=SLICED_LABELS,
        checkpoint_directory=checkpoint_directory,
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
        _require_two_node_placement(summary, leg=f"training/{leg}")
        _require_slice_partition(summary, leg=f"training/{leg}")
    if first.task_plan_identity != resumed.task_plan_identity:
        raise RuntimeError(
            "the resumed leg restored parameters under a different slice task "
            "plan, so its losses are not the same trajectory"
        )
    if first.task_plan_identity != fresh.task_plan_identity:
        raise RuntimeError(
            "the uninterrupted leg ran a different slice task plan, so its "
            "losses are not comparable"
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
    # The artifact carries the comparison; the per-rank records carry the runtime
    # summary. Keeping them separate means neither the trajectories nor the
    # placement evidence is recorded twice.
    #
    # `checkpoint_files` is the write history of the resumed leg -- it rewrites
    # one rank-local generation per step -- so it is recorded as a count and as
    # the set of names rather than as a list of distinct files.
    observations = {
        "optimizer": TRAINING_OPTIMIZER,
        "learning_rate": TRAINING_LR,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "resumed_steps": RESUMED_STEPS,
        "objective": resumed_summary["objective"],
        "first_leg_losses": list(first.losses),
        "resumed_leg_losses": list(resumed.losses),
        "uninterrupted_leg_losses": list(fresh.losses),
        "resumed_start_step": resumed.start_step,
        "uninterrupted_start_step": fresh.start_step,
        "resumed_completed_steps": resumed.completed_steps,
        "slice_labels": list(resumed_summary["slice_labels"]),
        "tasks_by_rank": {
            str(rank): count for rank, count in resumed_summary["tasks_by_rank"].items()
        },
        "parameter_owner_ranks": list(resumed_summary["parameter_owner_ranks"]),
        "checkpoint_storage_semantics": resumed_summary["checkpoint_storage_semantics"],
        "resumed_leg_checkpoint_writes": len(resumed_summary["checkpoint_files"]),
        # These are the files the artifact's own rank wrote; the cross-rank
        # generation check is `checkpoint_generation`, which reads the shared
        # directory rather than either rank's view of it.
        "artifact_rank_checkpoint_names": sorted(
            {Path(path).name for path in resumed_summary["checkpoint_files"]}
        ),
        "interrupted_leg_checkpoint_writes": len(first_summary["checkpoint_files"]),
        "uninterrupted_leg_checkpoint_writes": len(fresh_summary["checkpoint_files"]),
    }
    return observations, metrics, resumed_summary


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    training: dict[str, Any],
    sliced_label_multiplicities: dict[int, int],
) -> dict[str, Any]:
    evidence = {
        "schema": "flagquantum.cuda_multinode_tn_probe.v1",
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
            "world_size": 2,
            "local_world_size": 1,
            "node_count": 2,
            "sliced_labels": list(SLICED_LABELS),
            "sliced_label_multiplicities": {
                str(label): count
                for label, count in sorted(sliced_label_multiplicities.items())
            },
            "slice_count": EXPECTED_SLICE_COUNT,
            "slices_per_rank": EXPECTED_SLICES_PER_RANK,
            "observable_wires": list(OBSERVABLE_WIRES),
            "checked_bitstrings": list(CHECKED_BITSTRINGS),
            "distribution_semantics": "sharded_across_ranks",
            "execution": "amplitudes_gradient_optimizer_and_checkpoint_resume",
        },
        "observations": {
            "numerical_metrics": metrics,
            "training": training,
            "rank_records": rank_records,
            "production_full_state_materialization": False,
            "reference": "exact_complex128_statevector_single_device",
            # The executor all-reduces the slice partials of the expectation and
            # returns the reduced value, so `distributed_expectation` is a
            # distributed result. It does not reduce parameter gradients on this
            # path, so the global gradient the probe compares is the sum of the
            # per-rank contributions it gathered itself.
            "gradient_reduction": "probe_summed_rank_partials",
        },
        "claim_blockers": [
            "two_node_pair_only_no_wider_topology",
            "inter_node_cut_width_not_swept",
            "rdma_not_tested",
            "production_performance_not_measured",
            "toy_circuit_parameters_only",
            "slice_count_fixed_at_world_size",
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = _canonical_sha256(evidence)
    return {
        "schema": "flagquantum.cuda_multinode_tn_artifact.v1",
        "evidence_sha256": digest,
        "evidence": evidence,
    }


def probe(
    *, network_log: Path | None, checkpoint_directory: Path
) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("probe requires exactly two torchrun ranks")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise RuntimeError("probe requires exactly one rank per node")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    payload: dict[str, Any] | None = None
    try:
        _prepare_checkpoint_directory(checkpoint_directory, rank=rank, device=device)
        sliced_label_multiplicities = _require_sliced_labels_are_a_cut()

        # The amplitudes leg. Each rank contracts the slices it owns and the
        # partial amplitudes are reduced; the reference is the exact statevector.
        forward_circuit = _forward_circuit(device=device)
        amplitudes = distributed_tensor_network_amplitudes(
            forward_circuit,
            list(CHECKED_BITSTRINGS),
            **_contraction_arguments(device=device, rank=rank, local_rank=local_rank),
        )
        _synchronize(device)
        amplitudes_summary = amplitudes.summary()
        _require_two_node_placement(amplitudes_summary, leg="amplitudes")
        _require_slice_partition(amplitudes_summary, leg="amplitudes")
        if amplitudes_summary["full_state_materialized"]:
            raise RuntimeError("the amplitudes leg materialized the full state")

        # The single-amplitude leg takes a different code path -- one projected
        # output rather than a shared batch -- so it is checked separately.
        single = distributed_tensor_network_amplitude(
            forward_circuit,
            CHECKED_BITSTRINGS[0],
            **_contraction_arguments(device=device, rank=rank, local_rank=local_rank),
        )
        _synchronize(device)
        single_summary = single.summary()
        _require_two_node_placement(single_summary, leg="amplitude")
        _require_slice_partition(single_summary, leg="amplitude")

        accepted = True
        amplitude_metrics: dict[str, float] = {}
        if rank == 0:
            reference = _reference_statevector()
            expected = torch.tensor(
                [reference[int(bitstring, 2)] for bitstring in CHECKED_BITSTRINGS],
                dtype=COMPLEX_DTYPE,
            )
            actual = amplitudes.values.detach().cpu().reshape(-1)
            amplitude_metrics = {
                "amplitude_batch_max_abs_error": float(
                    torch.max(torch.abs(actual - expected)).item()
                ),
                "single_amplitude_max_abs_error": float(
                    torch.abs(
                        single.value.detach().cpu().reshape(-1)[0] - expected[0]
                    ).item()
                ),
            }
            accepted = all(
                value <= NUMERICAL_TOLERANCE for value in amplitude_metrics.values()
            )
        if not _collective_verdict(device, accepted=accepted):
            raise RuntimeError(
                f"distributed amplitude validation failed: {amplitude_metrics}"
            )

        # The gradient leg. Slicing makes the autograd graph rank-local: each
        # rank's `parameter.grad` is its own slice's contribution to the
        # gradient, and the reduced gradient is their sum. That is what makes the
        # gradient comparison able to catch a dropped rank, so the per-rank
        # contributions are gathered and required to be non-zero rather than
        # merely summed.
        trainable, parameters = _trainable_circuit(device=device)
        expectation = distributed_tensor_network_expectation(
            trainable,
            z=list(OBSERVABLE_WIRES),
            **_contraction_arguments(device=device, rank=rank, local_rank=local_rank),
        )
        _synchronize(device)
        expectation_summary = expectation.summary()
        _require_two_node_placement(expectation_summary, leg="expectation")
        _require_slice_partition(expectation_summary, leg="expectation")
        if expectation_summary["full_state_materialized"]:
            raise RuntimeError("the expectation leg materialized the full state")
        expectation.value.reshape(-1)[0].backward()
        _synchronize(device)
        # Every rank sees the same expectation because the executor all-reduces
        # the slice partials before it returns, so this value is a distributed
        # result rather than a rank-local one. The parameter gradients are the
        # opposite: autograd leaves this rank's own partial on `parameter.grad`,
        # and the reduction below is the probe summing them, which is the only
        # reason the number it checks is a global gradient.
        rank_gradients = tuple(
            float(parameter.grad.detach().cpu()) for parameter in parameters
        )
        distributed_expectation = float(expectation.value.detach().cpu().reshape(-1)[0])
        gathered_gradients = _gather_floats(rank_gradients, device=device)
        reduced_gradients = tuple(
            sum(row[index] for row in gathered_gradients)
            for index in range(len(rank_gradients))
        )
        rank_contribution_magnitudes = [
            max(abs(value) for value in row) for row in gathered_gradients
        ]

        reference_expectation, reference_gradients = _reference_expectation()
        gradient_metrics = {
            "expectation_value_error": abs(
                distributed_expectation - reference_expectation
            ),
            "gradient_max_abs_error": max(
                abs(actual - expected)
                for actual, expected in zip(
                    reduced_gradients, reference_gradients, strict=True
                )
            ),
        }
        # Each rank has to move the gradient on its own. If one rank's slices
        # were all exactly zero the reduced gradient would still be right, and
        # the leg would be claiming a partition it did not exercise.
        if min(rank_contribution_magnitudes) <= NUMERICAL_TOLERANCE:
            raise RuntimeError(
                "a rank's slices contributed nothing to the gradient, so the "
                f"reduction was not exercised: {rank_contribution_magnitudes}"
            )
        if not _collective_verdict(
            device,
            accepted=all(
                value <= NUMERICAL_TOLERANCE for value in gradient_metrics.values()
            ),
        ):
            raise RuntimeError(
                f"distributed gradient validation failed: {gradient_metrics}"
            )
        metrics: dict[str, float] = {}
        metrics.update(amplitude_metrics)
        metrics.update(gradient_metrics)

        training, training_metrics, training_summary = _training_observations(
            device=device, checkpoint_directory=checkpoint_directory
        )
        if not _collective_verdict(
            device,
            accepted=all(
                value <= TRAINING_TOLERANCE for value in training_metrics.values()
            ),
        ):
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
        # The optimizer minimizes the expectation the gradient leg computed, so
        # the training trajectory starts from the same number. Comparing them
        # ties the training objective to the exact statevector reference instead
        # of leaving it self-referential.
        metrics["first_leg_initial_expectation_error"] = abs(
            float(training["first_leg_losses"][0]) - reference_expectation
        )
        if int(training["resumed_start_step"]) != CHECKPOINT_STEPS:
            raise RuntimeError(
                "the resumed leg did not continue from the checkpoint: "
                f"start_step={training['resumed_start_step']}"
            )
        if int(training["resumed_completed_steps"]) != RESUMED_STEPS:
            raise RuntimeError(
                "the resumed leg did not reach the requested total step count: "
                f"{training['resumed_completed_steps']}"
            )
        # The resumed leg rewrites one rank-local generation per step, so its
        # write history has to have one entry for each step it ran.
        resumed_writes = int(training["resumed_leg_checkpoint_writes"])
        if resumed_writes != RESUMED_STEPS - CHECKPOINT_STEPS:
            raise RuntimeError(
                "the resumed leg did not write one checkpoint per step it ran: "
                f"writes={resumed_writes} steps={RESUMED_STEPS - CHECKPOINT_STEPS}"
            )
        # Both ranks checkpoint into the same shared directory, so one file per
        # rank has to be there and the generation it holds has to be the last
        # step of the run: a leftover generation would mean the resume read
        # something other than what its own leg wrote.
        final_generation = _verify_checkpoint_generation(
            checkpoint_directory,
            resumed_step=RESUMED_STEPS,
            expected_task_plan_identity=training_summary["task_plan_identity"],
            device=device,
        )
        training["checkpoint_generation"] = final_generation

        communication = {
            "amplitudes_collective_bytes": amplitudes_summary["communication_tiers"][
                "reduction_tensor_bytes"
            ],
            "amplitudes_inter_node_collective_bytes": amplitudes_summary[
                "communication_tiers"
            ]["inter_node_collective_bytes"],
            "amplitudes_intra_node_collective_bytes": amplitudes_summary[
                "communication_tiers"
            ]["intra_node_collective_bytes"],
            "expectation_inter_node_collective_bytes": expectation_summary[
                "communication_tiers"
            ]["inter_node_collective_bytes"],
            "expectation_rank_partial_bytes_by_rank": {
                str(rank_index): count
                for rank_index, count in expectation_summary[
                    "rank_partial_bytes_by_rank"
                ].items()
            },
            "training_communication_bytes": training_summary["communication_bytes"],
            "training_communication_tiers": training_summary["communication_tiers"],
            "training_owned_parameter_bytes_by_rank": list(
                training_summary["owned_parameter_bytes_by_rank"]
            ),
            "training_local_memory_bytes_by_rank": list(
                training_summary["local_memory_bytes_by_rank"]
            ),
        }

        properties = torch.cuda.get_device_properties(device)
        record = expectation_summary
        record.update(
            {
                # The runtime resolved the placement, so its host digest is the
                # authority; the record repeats it at the top level only to keep
                # the per-rank shape the other two probes' artifacts already have.
                "hostname_sha256": expectation_summary["rank_placement"][
                    "hostname_sha256"
                ],
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "")),
                "distributed_expectation": distributed_expectation,
                "rank_gradient_contribution": list(rank_gradients),
                "reduced_gradient": list(reduced_gradients),
                "reference_gradient": list(reference_gradients),
                "communication": communication,
                "amplitudes": amplitudes_summary,
                "single_amplitude": single_summary,
                "training_step": (
                    training_summary["step_records"][-1]
                    if training_summary["step_records"]
                    else None
                ),
            }
        )
        rank_records: list[dict[str, Any] | None] = [None, None]
        dist.all_gather_object(rank_records, record)
        if rank == 0:
            payload = _artifact(
                rank_records=[item for item in rank_records if item is not None],
                metrics=metrics,
                training=training,
                sliced_label_multiplicities=sliced_label_multiplicities,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0 and payload is not None:
        network = _network_observation(network_log)
        evidence = payload["evidence"]
        evidence["observations"]["network"] = network
        # One digest per artifact, over the evidence and nothing else. Recording
        # it inside `evidence` too would put two disagreeing digests in one file
        # and leave a reader no way to tell which one was authoritative.
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
