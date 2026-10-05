#!/usr/bin/env python
"""Verify CUDA tensor-network slices across two hosts, forward, gradient, training and resume.

The pair of hosts this runs on is the whole point and the whole limit. Each host
runs as many ranks as the caller asked for, one device each, and the artifact
records the shape it got. What it establishes is that one logical tensor-network
workload is partitioned across that shape by internal-edge slices, that each
rank contracts only the slices it owns, that the sliced amplitudes and the
sliced gradient agree with an exact statevector reference, and that an
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
import time
from collections import Counter
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
    cut_width_claim_blockers,
    gather_claim_blockers,
    measured_performance,
    measurement_claim_blockers,
    observed_network_route,
    slice_count_claim_blockers,
    staging_claim_blockers,
    transport_claim_blockers,
)
from flagquantum.runtime.distributed.transport_observability import (
    classify_explicit_host_transfer,
)
from flagquantum.runtime.executors.tensor_network.execution import (
    distributed_tensor_network_amplitude,
    distributed_tensor_network_amplitudes,
    distributed_tensor_network_expectation,
    run_distributed_tensor_network,
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

#: The measurement leg `--measure` asks for. The training legs already report a
#: duration per phase, but each is one unsynchronized reading of work thousands
#: of times shorter than the launch that precedes it. These samples are taken
#: between device synchronizations after a warmup every rank runs, and all of
#: them are kept.
MEASUREMENT = "sharded_tensor_network_amplitudes"
MEASUREMENT_WARMUP_ITERATIONS = 2
MEASUREMENT_ITERATIONS = 5

#: The staging audit profiles the steady state for the same reason the samples
#: warm up: a circuit's first execution also does the planning it will not
#: repeat, and that work is not what the interchange under audit consists of.
#: The count is the sample's own, so the audited region and the measured region
#: are the same region.
HOST_STAGING_WARMUP_EXECUTIONS = MEASUREMENT_WARMUP_ITERATIONS

#: The export leg. The amplitude legs reduce a handful of projected outputs; this
#: leg asks the runtime for the state itself, which is what a caller of
#: `run_distributed_tensor_network` receives and nothing else in the probe
#: produces. The runtime's own all-reduce of the rank partials is what builds it,
#: so the state is the leg's product rather than a gather taken to check an answer
#: computed somewhere else. The timings below are the export's own and the
#: artifact never presents them as a speedup -- an all-reduce has no speedup to
#: report, and the state it returns is replicated rather than sharded.
EXPORT_MEASUREMENT = "distributed_tensor_network_full_state_export"
EXPORT_WARMUP_ITERATIONS = 2
EXPORT_ITERATIONS = 5
EXPORTED_PRODUCT = "full_tensor_network_state"

#: The one state-distribution the export leg may report. The runtime contracts
#: the slices by owner and then all-reduces the partials, so the object it
#: returns holds the whole state on every rank; a leg reporting anything else
#: would mean the state it recorded came from a different act.
EXPORTED_STATE_DISTRIBUTION = "replicated_full_state_after_all_reduce"

#: A fixed rotation on wire 0, so the two trainable parameters act on a state
#: that is not an eigenstate of the measured observable.
PHASE_ROTATION = 0.6

#: The two sliced internal labels, two slices each. They are the contracted
#: edges of the two multi-index nodes that carry the entangling `rxx`, which is
#: what makes the pair well conditioned: the four slice values are 0.8177,
#: -0.0109, -0.1431 and +0.0019, and the rank sums against a total of 0.6656 are
#: 0.6746 and -0.0090 at two ranks, or the four slice values themselves at four.
#: A reduction that silently dropped any rank would be wrong by at least 0.0019,
#: which is seven orders of magnitude above the tolerance, rather than agreeing
#: by accident.
SLICED_LABELS = (6, 7)

#: The slice count the task plan must resolve to: two sliced labels of dimension
#: two each, so four combinations. How many of those each rank owns is the world
#: size divided into this, and the probe refuses a world size this does not
#: divide rather than run a shape whose plan it cannot predict.
EXPECTED_SLICE_COUNT = 4

#: The cut widths the sweep contracts, counted in sliced labels. The scope's cut
#: is the widest; every shorter prefix of the same labels is a genuinely
#: different partition of the same graph, with half as many slices. A width is
#: only wide enough to count once it has crossed the hosts, which is why the
#: observation carries the inter-node bytes alongside the width instead of the
#: width alone.
CUT_WIDTHS = tuple(range(1, len(SLICED_LABELS) + 1))

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
    circuit: fq.Circuit,
    theta: torch.Tensor,
    phi: torch.Tensor,
    phase: torch.Tensor | float,
) -> None:
    """Bind both trainable parameters, one of them twice.

    The first rotation is a fixed angle, the second and the fourth carry the
    trainable angle on either side of the entangling gate. The gate therefore
    sits between two layers that do not commute with it, which is what gives the
    entangling parameter a non-zero derivative; a circuit whose only non-commuting
    layer sat before the gate would make that derivative exactly zero and the
    gradient comparison would pass for a contraction that ignored the parameter.

    The fixed angle is a device tensor wherever a device is in play, for the
    same reason the trainable ones are: a rotation handed a host scalar has to
    lift that scalar onto the device, and that lift would land inside whatever
    region is being profiled rather than in the plan the circuit already is.
    """

    circuit.ry(0, phase)
    circuit.ry(N_WIRES - 1, theta)
    circuit.rxx(*OBSERVABLE_WIRES, phi)
    circuit.ry(0, theta)


def _phase(device: torch.device | None) -> torch.Tensor | float:
    """The fixed rotation as a host float or a device scalar."""

    if device is None:
        return PHASE_ROTATION
    return torch.tensor(PHASE_ROTATION, dtype=REAL_DTYPE, device=device)


def _forward_circuit(
    *, device: torch.device | None = None, dtype: torch.dtype = COMPLEX_DTYPE
) -> fq.Circuit:
    """The same circuit shape at a fixed parameter point."""

    parameters = [
        torch.tensor(value, dtype=REAL_DTYPE, device=device) for value in FORWARD_VALUES
    ]
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    _bind_trainable(circuit, *parameters, _phase(device))
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
    _bind_trainable(circuit, *parameters, _phase(device))
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
    _bind_trainable(circuit, *parameters, _phase(None))
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
    *, device: torch.device, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Time the sliced amplitude reduction between synchronizations, repeatedly.

    Every rank runs the same loop, because the loop contains collectives. The
    warmup iterations let the costs that belong to the first call -- a kernel's
    first launch, the allocator's first growth, the transport's first handshake
    -- land outside the samples, and all the samples are kept so that the spread
    is visible rather than summarized away.

    The placement comes from the caller rather than from the environment, for
    the same reason the other legs take it that way: a measured leg that
    resolved its own placement could measure a shape the artifact does not
    claim.
    """

    circuit = _forward_circuit(device=device)

    def once() -> float:
        _synchronize(device)
        started = time.perf_counter()
        amplitudes = distributed_tensor_network_amplitudes(
            circuit, list(CHECKED_BITSTRINGS), **arguments
        )
        _synchronize(device)
        if int(amplitudes.values.numel()) != len(CHECKED_BITSTRINGS):
            raise RuntimeError("the measured workload returned no amplitudes")
        return time.perf_counter() - started

    for _ in range(MEASUREMENT_WARMUP_ITERATIONS):
        once()
    observation = measured_performance(
        [once() for _ in range(MEASUREMENT_ITERATIONS)],
        warmup_iterations=MEASUREMENT_WARMUP_ITERATIONS,
        measurement=MEASUREMENT,
    )
    observation["n_wires"] = N_WIRES
    observation["sliced_labels"] = list(SLICED_LABELS)
    observation["checked_bitstrings"] = list(CHECKED_BITSTRINGS)
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
    *, device: torch.device, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Profile the sliced amplitude reduction for explicit host transfers.

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

    circuit = _forward_circuit(device=device)
    for _ in range(HOST_STAGING_WARMUP_EXECUTIONS):
        distributed_tensor_network_amplitudes(
            circuit, list(CHECKED_BITSTRINGS), **arguments
        )
    _synchronize(device)
    with torch.profiler.profile(activities=_profiler_activities(device)) as profile:
        distributed_tensor_network_amplitudes(
            circuit, list(CHECKED_BITSTRINGS), **arguments
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
        "profiled_warmup_executions": HOST_STAGING_WARMUP_EXECUTIONS,
        "profiler_event_count": len(events),
        "host_transfer_observed": bool(transfers),
        "host_transfer_events": transfers,
    }


def _require_export_placement(
    summary: dict[str, Any], *, expected_world_size: int
) -> None:
    """Refuse an export leg the planner placed differently from the launched shape.

    This leg is the one whose result is the state, so it reports the full-state
    facade rather than `sharded_across_ranks`: the slices are contracted by their
    owners and the partials are then all-reduced, which leaves every rank holding
    the whole vector. The guard is therefore written against that distribution
    instead, and it asks the object itself whether the state it carries is the
    whole state -- a leg that reported the facade while holding one rank's
    partial would otherwise be recorded as a materialization.
    """

    if summary["node_count"] != 2:
        raise RuntimeError(
            "the export leg did not resolve the two-node placement: "
            f"local_world_size={summary['local_world_size']} "
            f"node_count={summary['node_count']}"
        )
    if summary["world_size"] != expected_world_size:
        raise RuntimeError(
            f"the export leg ran at {summary['world_size']} ranks where "
            f"{expected_world_size} were launched"
        )
    if summary["claim_evidence_type"] != "production_runtime":
        raise RuntimeError(
            "the export leg did not report accelerator runtime semantics: "
            f"{summary['claim_evidence_type']}"
        )
    if summary["state_distribution_semantics"] != EXPORTED_STATE_DISTRIBUTION:
        raise RuntimeError(
            "the export leg did not distribute its state the way the export "
            f"contract says: {summary['state_distribution_semantics']}"
        )
    if summary["full_state_materialized"] is not True:
        raise RuntimeError("the export leg did not materialize the full state")
    if summary["scalability_claim_allowed"] is not False:
        raise RuntimeError("the export leg allowed a scalability claim")


def _materializes_the_full_state(export: dict[str, Any] | None) -> bool:
    """Whether the export leg produced the whole state as its product.

    Read from the recorded result rather than from a bare flag beside it, so the
    fact the blocker turns on and the fact the artifact publishes cannot drift
    apart. Three things have to hold: the state was materialized in full, it was
    the tensor-network state rather than another result shape, and it was the
    product rather than a shard moved aside to check an answer.
    """

    if export is None:
        return False
    result = export.get("result")
    if not isinstance(result, dict):
        return False
    return (
        result.get("full_state_materialized") is True
        and result.get("state_mode") == "distributed_tensor_network"
        and export.get("product") == EXPORTED_PRODUCT
    )


def _export_observation(
    *,
    device: torch.device,
    arguments: dict[str, Any],
    world_size: int,
    local_world_size: int,
) -> dict[str, Any]:
    """Materialize the whole tensor-network state, and time the act that builds it.

    This is a separate act from the amplitude legs. Those reduce a handful of
    projected outputs and never materialize the state -- the probe raises if they
    do -- so the runtime's own all-reduce is not exercised by them as a product.
    This leg calls the entry point whose contract is to return the state, so the
    collective that assembles it is what the leg is for rather than a check on an
    answer computed somewhere else.

    Each sample is compared with the exact single-device statevector as it is
    taken, so the duration and the vector it belongs to cannot come from
    different calls, and a permuted or truncated product fails the run instead of
    being published. Every rank runs this, because the reduction is a collective.
    """

    reference = _reference_statevector().to(device=device)

    durations: list[float] = []
    errors: list[float] = []
    latest: Any = None
    for index in range(EXPORT_WARMUP_ITERATIONS + EXPORT_ITERATIONS):
        _synchronize(device)
        started = time.perf_counter()
        exported = run_distributed_tensor_network(
            _forward_circuit(device=device), **arguments
        )
        state = exported.state()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        errors.append(float(torch.max(torch.abs(state[0] - reference)).item()))
        if index >= EXPORT_WARMUP_ITERATIONS:
            durations.append(elapsed)
        latest = exported
    if latest is None:  # pragma: no cover - both counts are positive constants
        raise RuntimeError("the export leg has no iterations to run")

    summary = latest.summary()
    _require_export_placement(summary, expected_world_size=world_size)
    _require_slice_partition(summary, leg="export", expected_world_size=world_size)

    observed_error = max(errors)
    if observed_error > NUMERICAL_TOLERANCE:
        raise RuntimeError(
            "the exported tensor-network state disagrees with the single-device "
            f"reference: max_abs_error={observed_error}"
        )

    product = latest.state()[0].detach().to(device="cpu", dtype=COMPLEX_DTYPE)
    return {
        "measurement": EXPORT_MEASUREMENT,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "product": EXPORTED_PRODUCT,
        "product_is_the_full_state": True,
        "state_sha256": hashlib.sha256(product.numpy().tobytes()).hexdigest(),
        "max_abs_error": observed_error,
        "export": measured_performance(
            durations,
            warmup_iterations=EXPORT_WARMUP_ITERATIONS,
            measurement=EXPORT_MEASUREMENT,
        ),
        # The result's own record, published whole rather than paraphrased: it
        # carries the distribution the state ended up with, the rank ownership,
        # the byte counts and the blockers that keep these durations from being
        # read as a speedup.
        "result": summary,
    }


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
    world_size: int,
    local_world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Check the committed generation by reading it, not by trusting the writer.

    Every rank checkpoints into the same shared directory, so the directory is
    the one place that shows the run produced exactly one generation per rank
    and that each generation is intact and stamped with the final step. A
    leftover generation would mean the resume restored something an earlier run
    wrote, which is the failure this whole leg exists to rule out.
    """

    expected_ranks = list(range(world_size))
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
                    item["world_size"] != world_size
                    or item["local_world_size"] != local_world_size
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


def _require_two_node_placement(
    summary: dict[str, Any], *, leg: str, expected_world_size: int
) -> None:
    """Refuse a leg the planner placed differently from the launched shape.

    The probe declares the placement to every leg rather than reading it back,
    so this compares what each leg reported against what the probe declared: a
    leg that resolved somewhere else means the declared placement never reached
    it, and the artifact would otherwise describe a shape nothing ran in.
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
    if summary["claim_evidence_type"] != "production_runtime":
        raise RuntimeError(
            f"the {leg} leg did not report accelerator runtime semantics: "
            f"{summary['claim_evidence_type']}"
        )


def _require_slice_partition(
    summary: dict[str, Any], *, leg: str, expected_world_size: int
) -> None:
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
    every_rank_owns = EXPECTED_SLICE_COUNT // expected_world_size
    if set(tasks_by_rank) != set(range(expected_world_size)) or any(
        count != every_rank_owns for count in tasks_by_rank.values()
    ):
        raise RuntimeError(
            f"the {leg} leg did not give each of {expected_world_size} ranks "
            f"{every_rank_owns} slice(s): {tasks_by_rank}"
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
    *,
    device: torch.device,
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
) -> dict[str, Any]:
    """The placement, slice scope, and dtype every distributed leg shares.

    The shape is declared to the runtime rather than read back from the
    environment, which is what makes the node_count each leg reports a claim the
    probe can be held to: the legs cannot resolve a placement the probe did not
    hand them.
    """

    return {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "distributed_executor": "torch",
        "rank": rank,
        "local_rank": local_rank,
        "sliced_labels": SLICED_LABELS,
        "device": device,
        "dtype": COMPLEX_DTYPE,
    }


def _slices_for_width(width: int) -> int:
    """The slice count a cut of `width` labels resolves to."""

    return EXPECTED_SLICE_COUNT >> (len(SLICED_LABELS) - width)


def _cut_width_observations(
    *,
    device: torch.device,
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
) -> dict[str, Any]:
    """Contract the same workload at every cut width this shape can partition.

    The blockers this feeds are about a cut that never moved and a slice count
    that never varied, so clearing them needs the widths the cut did move to and
    the exchange each one actually performed. Every width is contracted from the
    arguments the declared scope uses, with the sliced labels shortened to a
    prefix of it, and a width only counts once the run crossed the hosts and
    reproduced the exact reference. A width whose slice count does not divide the
    world size is recorded as unexercised with the reason rather than dropped, so
    a narrow shape reports the sweep it could not run instead of an empty one.

    A width that runs and disagrees with the reference is a defect rather than a
    narrow sweep, and it raises: recording it as a width that did not count would
    quietly turn a wrong answer into a shorter list.
    """

    reference = _reference_statevector()
    expected = torch.tensor(
        [reference[int(bitstring, 2)] for bitstring in CHECKED_BITSTRINGS],
        dtype=COMPLEX_DTYPE,
    )
    inter_node_bytes_by_width: dict[str, int] = {}
    slices_by_count: dict[str, int] = {}
    ranks_by_width: dict[str, list[int]] = {}
    unexercised: list[dict[str, Any]] = []
    for width in CUT_WIDTHS:
        slices = _slices_for_width(width)
        if slices % world_size:
            unexercised.append(
                {
                    "width": width,
                    "slices": slices,
                    "reason": f"{slices} slices do not divide {world_size} ranks",
                }
            )
            continue
        arguments = _contraction_arguments(
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        arguments["sliced_labels"] = SLICED_LABELS[:width]
        amplitudes = distributed_tensor_network_amplitudes(
            _forward_circuit(device=device), list(CHECKED_BITSTRINGS), **arguments
        )
        _synchronize(device)
        summary = amplitudes.summary()
        _require_two_node_placement(
            summary, leg=f"cut_width_{width}", expected_world_size=world_size
        )
        if int(summary["slice_tasks"]) != slices:
            raise RuntimeError(
                f"the width-{width} leg resolved to {summary['slice_tasks']} "
                f"slices rather than {slices}"
            )
        if summary["full_state_materialized"]:
            raise RuntimeError(f"the width-{width} leg materialized the full state")
        if summary["scalability_claim_allowed"] is not False:
            raise RuntimeError(f"the width-{width} leg allowed a scalability claim")
        error = None
        if rank == 0:
            actual = amplitudes.values.detach().cpu().reshape(-1)
            error = float(torch.max(torch.abs(actual - expected)).item())
        if not _collective_verdict(
            device, accepted=error is None or error <= NUMERICAL_TOLERANCE
        ):
            raise RuntimeError(
                f"the width-{width} leg disagreed with the exact reference: {error}"
            )
        bytes_crossed = int(
            summary["communication_tiers"]["inter_node_collective_bytes"]
        )
        partial_bytes = {
            int(rank_index): int(count)
            for rank_index, count in summary["rank_partial_bytes_by_rank"].items()
        }
        # Every rank has to have contracted something at this width. A rank that
        # owned no slice would leave the reduction correct while taking no part
        # in it, which is the replicated shape the narrower width is meant to
        # rule out rather than reproduce at a smaller size.
        if set(partial_bytes) != set(range(world_size)) or any(
            count <= 0 for count in partial_bytes.values()
        ):
            raise RuntimeError(
                f"the width-{width} leg did not give each of {world_size} ranks a "
                f"slice: {partial_bytes}"
            )
        inter_node_bytes_by_width[str(width)] = bytes_crossed
        slices_by_count[str(slices)] = bytes_crossed
        ranks_by_width[str(width)] = sorted(partial_bytes)
    return {
        "widths": list(CUT_WIDTHS),
        "inter_node_bytes_by_width": inter_node_bytes_by_width,
        "inter_node_bytes_by_slice_count": slices_by_count,
        "ranks_by_width": ranks_by_width,
        "unexercised_widths": unexercised,
    }


def _training_observations(
    *,
    device: torch.device,
    checkpoint_directory: Path,
    world_size: int,
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
        _require_two_node_placement(
            summary, leg=f"training/{leg}", expected_world_size=world_size
        )
        _require_slice_partition(
            summary, leg=f"training/{leg}", expected_world_size=world_size
        )
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


def _derived_claim_blockers(observations: dict[str, Any]) -> list[str]:
    """Every blocker this run's own observations can retract.

    The declared part of the boundary -- the pair of hosts and the toy circuit
    -- is not decided here, because nothing inside the run can retract it. What
    is decided here is only what the run observed.

    Called once while the artifact is assembled and again after the route has
    been read, because that observation arrives after the process group is gone
    and the group is what writes the log it is read from. Deriving the list in
    one place and applying it twice keeps the two passes from disagreeing.
    """

    cut_widths = observations["cut_widths"]
    return claim_blockers(
        [BLOCKER_TWO_NODE_PAIR_ONLY, BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY],
        gather_claim_blockers(
            production_materialization=bool(
                observations["production_full_state_materialization"]
            ),
            blocker=BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
        ),
        transport_claim_blockers(observations),
        measurement_claim_blockers(observations),
        staging_claim_blockers(observations),
        cut_width_claim_blockers(cut_widths["inter_node_bytes_by_width"]),
        slice_count_claim_blockers(cut_widths["inter_node_bytes_by_slice_count"]),
    )


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    training: dict[str, Any],
    sliced_label_multiplicities: dict[int, int],
    world_size: int,
    local_world_size: int,
    network: dict[str, Any],
    performance: dict[str, Any] | None,
    host_staging: dict[str, Any],
    cut_widths: dict[str, Any],
    export: dict[str, Any] | None,
) -> dict[str, Any]:
    # One dict, put into the evidence and then read back by the derivation, so
    # the blockers cannot be decided from a different set of observations than
    # the artifact publishes.
    #
    # The full-state question has one answer here and it is kept apart from the
    # amplitude legs: those reduce a handful of projected outputs and never
    # materialize the state. The export calls the entry point whose contract is
    # to return the state, so its materialization is one the workload needs, and
    # only that one retracts the blocker.
    observations: dict[str, Any] = {
        "numerical_metrics": metrics,
        "training": training,
        "rank_records": rank_records,
        "export": export,
        "production_full_state_materialization": _materializes_the_full_state(export),
        "reference": "exact_complex128_statevector_single_device",
        # The executor all-reduces the slice partials of the expectation and
        # returns the reduced value, so `distributed_expectation` is a
        # distributed result. It does not reduce parameter gradients on this
        # path, so the global gradient the probe compares is the sum of the
        # per-rank contributions it gathered itself.
        "gradient_reduction": "probe_summed_rank_partials",
        "network": network,
        "performance": performance,
        "host_staging": host_staging,
        "cut_widths": cut_widths,
    }
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
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": world_size // local_world_size,
            "sliced_labels": list(SLICED_LABELS),
            "sliced_label_multiplicities": {
                str(label): count
                for label, count in sorted(sliced_label_multiplicities.items())
            },
            "slice_count": EXPECTED_SLICE_COUNT,
            "slices_per_rank": EXPECTED_SLICE_COUNT // world_size,
            "observable_wires": list(OBSERVABLE_WIRES),
            "checked_bitstrings": list(CHECKED_BITSTRINGS),
            "distribution_semantics": "sharded_across_ranks",
            "execution": "amplitudes_gradient_optimizer_and_checkpoint_resume",
            "exported_product": EXPORTED_PRODUCT,
        },
        "observations": observations,
        # Both topology blockers are decided by what this run observed rather
        # than by what the probe intended: the cut width and the slice count are
        # whatever the sweep above managed to move, and the route, the
        # measurement and the staging audit say for themselves whether they
        # happened. The route arrives after this call and is applied by `probe`.
        "claim_blockers": _derived_claim_blockers(observations),
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


def _declared_shape() -> tuple[int, int]:
    """The 2xN shape this probe was launched in, refused unless it is one.

    The lane is a pair of hosts; how many ranks each host runs is the caller's
    choice, and the artifact records whichever answer it got. The slice count is
    fixed by `SLICED_LABELS`, so a world size that does not divide it would give
    the ranks an uneven share of the slices; that is refused here, before the
    process group is built, because a shape mismatch found after it reports as a
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
            "probe needs ranks placed evenly across the nodes, received "
            f"world_size={world_size} local_world_size={local_world_size}"
        )
    node_count = world_size // local_world_size
    if node_count != 2:
        raise RuntimeError(
            f"probe requires exactly two nodes, received {node_count} "
            f"(world_size={world_size} local_world_size={local_world_size})"
        )
    if EXPECTED_SLICE_COUNT % world_size:
        raise RuntimeError(
            f"probe requires a world size dividing the {EXPECTED_SLICE_COUNT} "
            f"slice scope, received {world_size}"
        )
    return world_size, local_world_size


def _apply_network_observation(
    payload: dict[str, Any] | None,
    *,
    network_log: Path | None,
    local_world_size: int,
) -> dict[str, Any] | None:
    """Read the route out of the debug log and settle the blockers again.

    The route is the one observation that cannot be taken while the process
    group exists, because the group is what writes the log it is read from. It
    still decides `rdma_not_tested`, so the blockers are derived a second time
    here rather than left as they were assembled without it: a probe that
    derived them once, before the log was read, would report an untested fabric
    on a run that used one.

    The digest is recomputed because it covers the evidence, and the evidence
    just changed.
    """

    if payload is None:
        return None
    evidence = payload["evidence"]
    evidence["observations"]["network"] = _network_observation(
        network_log, local_world_size=local_world_size
    )
    evidence["claim_blockers"] = _derived_claim_blockers(evidence["observations"])
    payload["evidence_sha256"] = _canonical_sha256(evidence)
    return payload


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
    try:
        _prepare_checkpoint_directory(checkpoint_directory, rank=rank, device=device)
        sliced_label_multiplicities = _require_sliced_labels_are_a_cut()
        placement = _contraction_arguments(
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )

        # The amplitudes leg. Each rank contracts the slices it owns and the
        # partial amplitudes are reduced; the reference is the exact statevector.
        forward_circuit = _forward_circuit(device=device)
        amplitudes = distributed_tensor_network_amplitudes(
            forward_circuit,
            list(CHECKED_BITSTRINGS),
            **_contraction_arguments(
                device=device,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
            ),
        )
        _synchronize(device)
        amplitudes_summary = amplitudes.summary()
        _require_two_node_placement(
            amplitudes_summary, leg="amplitudes", expected_world_size=world_size
        )
        _require_slice_partition(
            amplitudes_summary, leg="amplitudes", expected_world_size=world_size
        )
        if amplitudes_summary["full_state_materialized"]:
            raise RuntimeError("the amplitudes leg materialized the full state")

        # The single-amplitude leg takes a different code path -- one projected
        # output rather than a shared batch -- so it is checked separately.
        single = distributed_tensor_network_amplitude(
            forward_circuit,
            CHECKED_BITSTRINGS[0],
            **_contraction_arguments(
                device=device,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
            ),
        )
        _synchronize(device)
        single_summary = single.summary()
        _require_two_node_placement(
            single_summary, leg="amplitude", expected_world_size=world_size
        )
        _require_slice_partition(
            single_summary, leg="amplitude", expected_world_size=world_size
        )

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
            **_contraction_arguments(
                device=device,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
            ),
        )
        _synchronize(device)
        expectation_summary = expectation.summary()
        _require_two_node_placement(
            expectation_summary, leg="expectation", expected_world_size=world_size
        )
        _require_slice_partition(
            expectation_summary, leg="expectation", expected_world_size=world_size
        )
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
            device=device,
            checkpoint_directory=checkpoint_directory,
            world_size=world_size,
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
            world_size=world_size,
            local_world_size=local_world_size,
            device=device,
        )
        training["checkpoint_generation"] = final_generation

        # The cut-width sweep, the measured leg and the staging audit all
        # contain collectives, so they run while the process group is still up.
        # The sweep goes first because it is the leg that re-partitions the
        # workload; the measurement follows it so the first widths' plans are out
        # of the allocator's way, and the profiled staging run goes last so its
        # profiler cannot observe either of the others.
        cut_widths = _cut_width_observations(
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        performance = (
            _performance_observations(device=device, arguments=placement)
            if measure
            else None
        )
        host_staging = _host_staging_observation(device=device, arguments=placement)
        # The export runs after the staging audit so the profiled region stays
        # the region the previous revision profiled. It runs on every rank and
        # unconditionally: it is a workload whose product is the state, not a
        # timed leg the caller opts into, and the blocker it retracts is about
        # the workload rather than about having taken a measurement.
        export = _export_observation(
            device=device,
            arguments=placement,
            world_size=world_size,
            local_world_size=local_world_size,
        )

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
            "export_reduction_bytes_by_rank": {
                str(rank_index): count
                for rank_index, count in export["result"][
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
        # One slot per rank: `all_gather_object` fills exactly the list it is
        # given, so a fixed-length list would truncate a wider shape.
        rank_records: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_records, record)
        if rank == 0:
            payload = _artifact(
                rank_records=[item for item in rank_records if item is not None],
                metrics=metrics,
                training=training,
                sliced_label_multiplicities=sliced_label_multiplicities,
                world_size=world_size,
                local_world_size=local_world_size,
                # The route is read from the debug log after the group is gone,
                # so the artifact is assembled with an empty report and the
                # observation is written into it, together with the blockers it
                # decides, by the rank that reads the log.
                network={},
                performance=performance,
                host_staging=host_staging,
                cut_widths=cut_widths,
                export=export,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        # The route is read after the group is gone, because the debug log is
        # still being written while it exists.
        payload = _apply_network_observation(
            payload, network_log=network_log, local_world_size=local_world_size
        )
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
            "time the sliced amplitude reduction between synchronizations after "
            "a warmup, so the artifact carries a measured duration rather than "
            "a single unsynchronized reading of the launch"
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
