#!/usr/bin/env python
"""Verify CUDA statevector shards across two hosts, forward, backward and resume.

The pair of hosts this runs on is the whole point and the whole limit. Each host
runs as many ranks as the caller asked for, one device each, and the artifact
records the shape it got. What it establishes is that a single logical
statevector workload is partitioned across that shape, that the partitioned
gradient agrees with a single-device reference, and that an owner-sharded
optimizer step survives a checkpoint and a restart -- because a forward-only
probe cannot support a training claim and a training claim that stops at the
first optimizer step is not a training path either.

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.audit.claim_boundary import (
    BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,
    BLOCKER_TWO_NODE_PAIR_ONLY,
    BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
    ClaimBoundaryError,
    claim_blockers,
    configured_infiniband_state,
    gather_claim_blockers,
    measured_performance,
    measurement_claim_blockers,
    observed_network_route,
    staging_claim_blockers,
    transport_claim_blockers,
)
from flagquantum.runtime.distributed.transport_observability import (
    classify_explicit_host_transfer,
)
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.reverse import (
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)
from flagquantum.runtime.executors.statevector.training import (
    train_distributed_statevector,
)

ROOT = Path(__file__).resolve().parents[1]

#: The forward scope. Five wires leaves `5 - log2(world_size)` wires per shard,
#: so the circuit crosses the shard boundary at every shape this probe accepts
#: and the exchange is exercised.
N_WIRES = 5
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

#: Numerical tolerance for every comparison against a single-device reference.
#: Complex128 with an exact adjoint should agree to round-off, so this is
#: round-off rather than a modelling allowance.
NUMERICAL_TOLERANCE = 1e-10
#: The same, for the multi-step training trajectory: two runs of four steps
#: accumulate more round-off than a single expectation does.
TRAINING_TOLERANCE = 1e-9

TRAINING_OPTIMIZER = "sgd"
TRAINING_LR = 0.05
#: The first leg stops here and the second leg resumes to `RESUMED_STEPS`.
#: `train_distributed_statevector` refuses a run shorter than two steps, because
#: a single step cannot show that the second one continued from the first.
CHECKPOINT_STEPS = 2
RESUMED_STEPS = 4

#: The parameters the trainable circuit binds. Fixed so that the reference and
#: the distributed run are the same function of the same numbers.
TRAINABLE_VALUES = (0.23, -0.37)

#: The measurement leg `--measure` asks for. The training legs already report a
#: duration per phase, but each is one unsynchronized reading of work thousands
#: of times shorter than the launch that precedes it. These samples are taken
#: between device synchronizations after a warmup every rank runs, and all of
#: them are kept.
MEASUREMENT = "sharded_statevector_forward"
MEASUREMENT_WARMUP_ITERATIONS = 2
MEASUREMENT_ITERATIONS = 5


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _synchronize(device: torch.device) -> None:
    """Wait for the device's work, so a reported failure is the one that happened.

    Every real run is on an accelerator -- the probe refuses to start without
    CUDA -- and this stays device-guarded so that the training legs can be
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
    return (
        fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
        .h(0)
        .ry(4, 0.31)
        .cx(4, 1)
        .rx(3, -0.27)
        .cx(0, 4)
        .rz(4, 0.19)
        .cx(3, 2)
    )


def _trainable_circuit(
    *,
    device: torch.device,
    dtype: torch.dtype = COMPLEX_DTYPE,
    real_dtype: torch.dtype = REAL_DTYPE,
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...]]:
    """The circuit the training legs differentiate.

    One parameter is bound twice and one gate spans the shard boundary, so a
    gradient that is wrong in its reduction or in its shard ownership cannot
    agree with the reference by accident.
    """

    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=real_dtype, device=device)
        # Allocation followed by an explicit copy, so an accidental narrowing to
        # float32 is an observable mismatch rather than a silent test of FP32.
        tensor.copy_(torch.tensor(value, dtype=real_dtype))
        parameters.append(tensor.requires_grad_())
    theta, phi = parameters
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    circuit.h(0).ry(N_WIRES - 1, theta).rxx(0, N_WIRES - 1, phi).rz(1, theta)
    return circuit, tuple(parameters)


def _observable_wire() -> int:
    # The last logical wire is the least-significant basis bit, which is what
    # `_reference_expectation` below assumes.
    return N_WIRES - 1


def _reference_expectation() -> tuple[float, tuple[float, ...]]:
    """The same expectation and gradient on one CPU device, for the differential."""

    parameters: list[torch.Tensor] = []
    for value in TRAINABLE_VALUES:
        tensor = torch.empty((), dtype=REAL_DTYPE)
        tensor.copy_(torch.tensor(value, dtype=REAL_DTYPE))
        parameters.append(tensor.requires_grad_())
    theta, phi = parameters
    circuit = (
        fq.Circuit(N_WIRES, dtype=COMPLEX_DTYPE)
        .h(0)
        .ry(N_WIRES - 1, theta)
        .rxx(0, N_WIRES - 1, phi)
        .rz(1, theta)
    )
    state = circuit.state(refresh=True)
    probabilities = state.abs().square().reshape(-1)
    expectation = probabilities[::2].sum() - probabilities[1::2].sum()
    expectation.backward()
    return (
        float(expectation.detach()),
        tuple(float(parameter.grad.detach()) for parameter in parameters),
    )


def _validation_state(result: Any) -> torch.Tensor:
    """Materialize a tiny state solely to compare with the CPU reference."""

    local = result.shard_state.amplitudes[0]
    gathered = [torch.empty_like(local) for _ in range(result.plan.world_size)]
    dist.all_gather(gathered, local)
    internal = torch.empty(
        result.plan.total_amplitudes,
        dtype=local.dtype,
        device=local.device,
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


def _network_observation(
    path: Path | None, *, local_world_size: int = 1
) -> dict[str, Any]:
    """The route this run took, read from the debug log rather than the plan.

    The log is read by the artifact's rank only, after the process group is
    gone, so this is a report about the run rather than part of it.
    """

    content = (
        None if path is None else path.read_text(encoding="utf-8", errors="replace")
    )
    try:
        return observed_network_route(
            content,
            configured_interface=os.environ.get("NCCL_SOCKET_IFNAME"),
            infiniband_disabled=configured_infiniband_state(os.environ),
            debug_log_scope="node" if local_world_size > 1 else "rank",
        )
    except ClaimBoundaryError as error:
        # The probe's failures are `RuntimeError`s, and this is one: the log the
        # lane was told to expect is not the log the run produced.
        raise RuntimeError(
            f"NCCL debug log does not confirm the configured route: {error}"
        ) from error


def _performance_observations(
    *, device: torch.device, local_world_size: int
) -> dict[str, Any]:
    """Time the sharded forward between synchronizations, with warmup and repeats.

    Every rank runs the same loop, because the loop contains collectives. The
    warmup iterations let the costs that belong to the first call -- a kernel's
    first launch, the allocator's first growth, the transport's first handshake
    -- land outside the samples, and all the samples are kept so that the spread
    is visible rather than summarized away.
    """

    circuit = _forward_circuit()

    def once() -> float:
        _synchronize(device)
        started = time.perf_counter()
        result = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=COMPLEX_DTYPE,
            local_world_size=local_world_size,
        )
        _synchronize(device)
        if result.inter_node_communication_count < 1:
            raise RuntimeError(
                "the measured workload did not exercise inter-node communication"
            )
        return time.perf_counter() - started

    for _ in range(MEASUREMENT_WARMUP_ITERATIONS):
        once()
    observation = measured_performance(
        [once() for _ in range(MEASUREMENT_ITERATIONS)],
        warmup_iterations=MEASUREMENT_WARMUP_ITERATIONS,
        measurement=MEASUREMENT,
    )
    observation["n_wires"] = N_WIRES
    return observation


def _profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    """The two scopes the staging question needs, CPU calls and device copies.

    A CPU-only profile would show the launch of a transfer without showing
    whether the device performed one, and a device-only profile would omit the
    host side of a staging copy. Both are recorded, so a reader can see the
    audit was scoped to the interchange rather than to one of its halves.
    """

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _profiler_unavailable_reason(device: torch.device) -> str | None:
    """Start and stop an empty profile, to find out before the workload runs.

    The answer has to be known before the profiled run rather than discovered
    from its failure: a rank that skipped the run while its peer entered it
    would be waiting in a different collective, which reports as a hang rather
    than as the missing profiler that caused it.
    """

    try:
        with torch.profiler.profile(activities=_profiler_activities(device)):
            pass
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _host_staging_observation(
    *, device: torch.device, local_world_size: int
) -> dict[str, Any]:
    """Profile the sharded forward for explicit host transfers.

    What a profiler can show here is bounded, and the record keeps the bound. An
    explicit host-to-device or device-to-host copy inside the profiled region is
    a fact about this interchange; the absence of one is not proof that the
    fabric carried it device-direct, and a profiler that could not be started
    has shown nothing at all. Both leave the blocker in place, so a missing
    profiler is recorded rather than raised -- fail-closed for the claim, and a
    run that cannot audit its staging still produces an artifact that says so.

    Availability is agreed on collectively. A rank that recorded "not audited"
    while its peer entered the profiled run would be in a different collective
    from it, and the lane would report a hang instead of the reason.
    """

    reason = _profiler_unavailable_reason(device)
    agreed = torch.tensor([0 if reason else 1], device=device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    if int(agreed.item()) != 1:
        return {
            "profiled": False,
            "profiler_error": reason or "the profiler is unavailable on another rank",
        }

    circuit = _forward_circuit()
    with torch.profiler.profile(activities=_profiler_activities(device)) as profile:
        execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=COMPLEX_DTYPE,
            local_world_size=local_world_size,
        )
        _synchronize(device)
    events = list(profile.key_averages())
    transfers = sorted(
        (
            {
                "name": event.key,
                "count": int(event.count),
                "direction": direction,
            }
            for event in events
            if (direction := classify_explicit_host_transfer(event.key)) is not None
        ),
        key=lambda transfer: (transfer["direction"], transfer["name"]),
    )
    return {
        "profiled": True,
        "profiled_activities": [
            str(activity) for activity in _profiler_activities(device)
        ],
        "profiled_workload": MEASUREMENT,
        "profiler_event_count": len(events),
        "host_transfer_observed": bool(transfers),
        "host_transfer_events": transfers,
    }


def _prepare_checkpoint_directory(
    directory: Path, *, rank: int, device: torch.device
) -> None:
    """Start from a directory holding no checkpoints, or not at all.

    The resumed leg reads what the first leg wrote, so a checkpoint left by an
    earlier run would be restored instead and the artifact would describe a
    trajectory this run never computed. The caller empties the directory; this
    is the second opinion, because the two must not disagree.

    Rank 0 decides and both ranks act on the same answer. A rank that raised on
    its own would leave its peer waiting in the next collective, which reports
    as a hang rather than as the reason the run was refused.
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
    """

    interrupted, _ = _trainable_circuit(device=device)
    first = train_distributed_statevector(
        interrupted,
        steps=CHECKPOINT_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
    )
    _synchronize(device)

    uninterrupted, _ = _trainable_circuit(device=device)
    fresh = train_distributed_statevector(
        uninterrupted,
        steps=RESUMED_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory / "uninterrupted",
        checkpoint_interval=1,
    )
    _synchronize(device)

    resuming, _ = _trainable_circuit(device=device)
    resumed = train_distributed_statevector(
        resuming,
        steps=RESUMED_STEPS,
        observable_wire=_observable_wire(),
        optimizer=TRAINING_OPTIMIZER,
        lr=TRAINING_LR,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=1,
        resume=True,
    )
    _synchronize(device)

    first_summary = first.summary()
    resumed_summary = resumed.summary()
    fresh_summary = fresh.summary()
    expectation = _reference_expectation()[0]
    metrics = {
        "first_leg_initial_expectation_error": abs(first.losses[0] - expectation),
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
        "first_leg_summary": first_summary,
        "resumed_leg_summary": resumed_summary,
        "uninterrupted_leg_summary": fresh_summary,
    }
    return observations, metrics, first_summary


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    network: dict[str, Any],
    training: dict[str, Any],
    performance: dict[str, Any] | None,
    host_staging: dict[str, Any] | None,
    world_size: int,
    local_world_size: int,
) -> dict[str, Any]:
    # One dict, put into the evidence and then read back by the derivation, so
    # the blockers cannot be decided from a different set of observations than
    # the artifact publishes.
    observations: dict[str, Any] = {
        "numerical_metrics": metrics,
        "training": training,
        "rank_records": rank_records,
        "network": network,
        "performance": performance,
        "host_staging": host_staging,
        "validation_full_state_materialization": True,
        "production_full_state_materialization": False,
    }
    evidence = {
        "schema": "flagquantum.cuda_multinode_statevector_probe.v2",
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
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": world_size // local_world_size,
            "distribution_semantics": "sharded_across_ranks",
            "execution": "forward_backward_optimizer_and_checkpoint_resume",
        },
        "observations": observations,
        # Two kinds of boundary, kept apart because only one of them can move.
        # What this run declares -- a pair of hosts, one fixed circuit -- cannot
        # be retracted by anything inside it, so it is stated. What the run
        # observed is derived from the observations above, so a later run that
        # records the missing observation drops its blocker without anyone
        # editing a literal here.
        "claim_blockers": claim_blockers(
            [
                BLOCKER_TWO_NODE_PAIR_ONLY,
                BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,
            ],
            gather_claim_blockers(
                production_materialization=observations[
                    "production_full_state_materialization"
                ],
                blocker=BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
            ),
            transport_claim_blockers(observations),
            measurement_claim_blockers(observations),
            staging_claim_blockers(observations),
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = _canonical_sha256(evidence)
    return {
        "schema": "flagquantum.cuda_multinode_statevector_artifact.v1",
        "evidence_sha256": digest,
        "evidence": evidence,
    }


def _declared_shape() -> tuple[int, int]:
    """The 2xN shape this probe was launched in, refused unless it is one.

    The lane is a pair of hosts; how many ranks each host runs is the caller's
    choice, and the artifact records whichever answer it got. Anything else --
    one host standing in for a pair, ranks placed unevenly across the two, or a
    world size the planner cannot shard -- is refused here, before the process
    group is built, because a shape mismatch found after it reports as a
    collective that never completes rather than as the shape that was refused.
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
            f"probe needs ranks placed evenly across the nodes, received "
            f"world_size={world_size} local_world_size={local_world_size}"
        )
    node_count = world_size // local_world_size
    if node_count != 2:
        raise RuntimeError(
            f"probe requires exactly two nodes, received {node_count} "
            f"(world_size={world_size} local_world_size={local_world_size})"
        )
    # The statevector planner shards amplitudes by rank address bits, so ranks
    # own unequal amplitude counts at any other world size and the executor
    # refuses that distribution rather than running it.
    if world_size & (world_size - 1):
        raise RuntimeError(
            f"probe requires a power-of-two world size, received {world_size}"
        )
    return world_size, local_world_size


def _require_declared_placement(
    summary: dict[str, Any],
    *,
    leg: str,
    world_size: int,
    local_world_size: int,
) -> None:
    """One leg has to have resolved the shape the launch declared.

    A leg that ran at a different width, or with its ranks split differently
    across the two hosts, is not the workload this probe set out to measure,
    whatever it computed.
    """

    if summary["world_size"] != world_size:
        raise RuntimeError(
            f"the {leg} leg ran at {summary['world_size']} ranks where "
            f"{world_size} were launched"
        )
    if summary["local_world_size"] != local_world_size:
        raise RuntimeError(
            f"the {leg} leg resolved {summary['local_world_size']} ranks per "
            f"node where {local_world_size} were launched"
        )
    if summary["node_count"] != 2:
        raise RuntimeError(
            f"the {leg} leg resolved {summary['node_count']} nodes rather than "
            f"the two hosts this probe requires "
            f"(world_size={summary['world_size']} "
            f"local_world_size={summary['local_world_size']})"
        )


def probe(
    *,
    network_log: Path | None,
    checkpoint_directory: Path,
    measure: bool = False,
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
    rank_records: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    training: dict[str, Any] = {}
    performance: dict[str, Any] | None = None
    host_staging: dict[str, Any] | None = None
    try:
        _prepare_checkpoint_directory(checkpoint_directory, rank=rank, device=device)
        circuit = _forward_circuit()
        result = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=COMPLEX_DTYPE,
            # Pinned to the launched shape rather than left to the environment:
            # the summary then reports `local_world_size_source == "caller"`, so
            # the recorded placement is the one this probe asked for rather than
            # whatever a stray variable happened to say.
            local_world_size=local_world_size,
        )
        _synchronize(device)
        if result.shard_state.amplitudes.dtype != COMPLEX_DTYPE:
            raise RuntimeError("distributed shard lost complex128 precision")
        if result.shard_state.amplitudes.device.type != "cuda":
            raise RuntimeError("distributed shard left CUDA")
        if result.inter_node_communication_count < 1:
            raise RuntimeError("workload did not exercise inter-node communication")

        candidate = _validation_state(result)
        reference = _forward_circuit().state()[0].to(device=device)
        metrics = {
            "statevector_max_abs_error": float(
                torch.max(torch.abs(candidate - reference)).item()
            ),
            "statevector_norm_error": float(
                torch.abs(torch.linalg.vector_norm(candidate) - 1).item()
            ),
            "statevector_infidelity": max(
                0.0,
                float(1 - torch.abs(torch.vdot(reference, candidate)).square().item()),
            ),
        }
        if any(value > NUMERICAL_TOLERANCE for value in metrics.values()):
            raise RuntimeError(f"distributed numerical validation failed: {metrics}")

        trainable, parameters = _trainable_circuit(device=device)
        gradient_probe = execute_torch_distributed_statevector_reverse(
            trainable,
            observable_wire=_observable_wire(),
            checkpoint_policy=StatevectorCheckpointPolicy(
                strategy="interval", interval=2
            ),
            device=device,
        )
        gradient_probe.backward()
        _synchronize(device)
        reference_expectation, reference_gradients = _reference_expectation()
        actual_gradients = tuple(
            float(parameter.grad.detach()) for parameter in parameters
        )
        _require_declared_placement(
            gradient_probe.summary(),
            leg="backward",
            world_size=world_size,
            local_world_size=local_world_size,
        )
        metrics.update(
            {
                "expectation_value_error": abs(
                    float(gradient_probe.value.detach()) - reference_expectation
                ),
                "gradient_max_abs_error": max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        actual_gradients, reference_gradients, strict=True
                    )
                ),
            }
        )
        if any(value > NUMERICAL_TOLERANCE for value in metrics.values()):
            raise RuntimeError(f"distributed gradient validation failed: {metrics}")

        training, training_metrics, training_summary = _training_observations(
            device=device, checkpoint_directory=checkpoint_directory
        )
        metrics.update(training_metrics)
        if any(value > TRAINING_TOLERANCE for value in training_metrics.values()):
            raise RuntimeError(
                f"distributed training validation failed: {training_metrics}"
            )
        _require_declared_placement(
            training_summary,
            leg="training",
            world_size=world_size,
            local_world_size=local_world_size,
        )
        if float(training_summary["losses"][-1]) == float(
            training_summary["losses"][0]
        ):
            # A frozen expectation would make every comparison below agree for
            # the wrong reason. Whether it moved towards or away from the
            # minimum is recorded rather than asserted: SGD on this observable
            # takes a legitimate step in either direction at this learning rate,
            # and the claim under test is that the checkpoint reproduces the
            # trajectory, not that the learning rate was well chosen.
            raise RuntimeError(
                "the optimizer step did not change the expectation: "
                f"{training_summary['losses']}"
            )
        metrics["training_loss_decrease"] = float(
            training_summary["losses"][0] - training_summary["losses"][-1]
        )
        if int(training["resumed_start_step"]) != CHECKPOINT_STEPS:
            raise RuntimeError(
                "the resumed leg did not continue from the checkpoint: "
                f"start_step={training['resumed_start_step']}"
            )

        if measure:
            # Both legs contain collectives, so every rank runs both of them and
            # in this order. A rank that skipped one would reach the next
            # collective alone and the lane would report a hang.
            performance = _performance_observations(
                device=device, local_world_size=local_world_size
            )
        host_staging = _host_staging_observation(
            device=device, local_world_size=local_world_size
        )

        properties = torch.cuda.get_device_properties(device)
        record = result.summary()
        record.update(
            {
                "hostname_sha256": hashlib.sha256(
                    platform.node().encode("utf-8")
                ).hexdigest(),
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "")),
                "owned_global_basis_indices": (
                    f"{rank} + k*{result.plan.world_size}, "
                    f"0 <= k < {result.shard_state.shard.local_amplitudes}"
                ),
                "backward": gradient_probe.summary(),
                "training": training_summary,
            }
        )
        # One slot per rank: `all_gather_object` fills exactly the list it is
        # given, so a fixed-length list would truncate a wider shape.
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, record)
        rank_records = [item for item in gathered if item is not None]
        if rank == 0:
            payload = _artifact(
                rank_records=rank_records,
                metrics=metrics,
                network={},
                training=training,
                performance=performance,
                host_staging=host_staging,
                world_size=world_size,
                local_world_size=local_world_size,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0 and payload is not None:
        # The route is read after the group is gone, because the debug log is
        # still being written while it exists.
        evidence = payload["evidence"]
        evidence["observations"]["network"] = _network_observation(
            network_log, local_world_size=local_world_size
        )
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
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "time the sharded workload with warmup and repeats and record every "
            "sample. Without it the artifact reports no measurement at all "
            "rather than the single unsynchronized reading the training legs "
            "already carry. Every rank must be given the same flag"
        ),
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
        measure=args.measure,
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
