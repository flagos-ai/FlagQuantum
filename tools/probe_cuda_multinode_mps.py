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

The owned-site boundary is then swept: the same reverse is re-run under every
site plan this shape can hold, and each leg's inter-node layer halo is what says
the cut it placed was carried across the hosts. A cut that never moved is not
evidence about a cut, so the probe measures the widths instead of declaring one.

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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.experimental.distributed import train_distributed_mps
from flagquantum.runtime.audit.claim_boundary import (
    BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,
    BLOCKER_TWO_NODE_PAIR_ONLY,
    BLOCKER_VALIDATION_ONLY_TINY_FULL_MPS_GATHER,
    ClaimBoundaryError,
    claim_blockers,
    configured_infiniband_state,
    cut_width_claim_blockers,
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
from flagquantum.runtime.executors.mps.forward import (
    execute_torch_distributed_mps_forward,
    gather_mps_for_validation,
)
from flagquantum.runtime.executors.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)
from flagquantum.runtime.executors.mps.state import (
    initial_mps_ownership,
    validate_mps_ownership,
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

#: The measurement leg `--measure` asks for. The training legs already report a
#: duration per phase, but each is one unsynchronized reading of work thousands
#: of times shorter than the launch that precedes it. These samples are taken
#: between device synchronizations after a warmup every rank runs, and all of
#: them are kept.
MEASUREMENT = "sharded_mps_forward"
MEASUREMENT_WARMUP_ITERATIONS = 2
MEASUREMENT_ITERATIONS = 5

#: The owned-site boundary at two ranks: rank 0 owns wires 0..2 and rank 1 owns
#: wires 3..5, so this pair is the one adjacent gate that has to be exchanged.
BOUNDARY_PAIR = (N_WIRES // 2 - 1, N_WIRES // 2)

#: The parameters the cut sweep binds. It takes no gradient of its own; it is
#: bound because a circuit with no trainable parameter cannot be reversed, and
#: the leg that measures a cut exchange is a reverse.
SWEEP_VALUES = (0.37, 0.29, 0.31)


def _bind_sweep(circuit: fq.Circuit, parameters: Sequence[torch.Tensor]) -> None:
    """Bind the sweep's three parameters on every adjacent pair in turn.

    Each pair in the wire is rotated, so the circuit's Schmidt profile is not
    flat and the cuts can tell each other apart. Every gate is a two-site
    rotation on adjacent wires, which is all the executor's compiled site
    kernels accept: the sweep then measures the partitioning rather than a gate
    set the declared circuit does not use.
    """

    circuit.h(0).ry(0, parameters[0])
    for left in range(N_WIRES - 1):
        circuit.rxx(left, left + 1, parameters[0])
    for left in range(1, N_WIRES - 1):
        circuit.ryy(left, left + 1, parameters[1])
    for left in range(N_WIRES - 1):
        circuit.ryy(left, left + 1, parameters[2])


def _cut_sweep_circuit(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = COMPLEX_DTYPE,
    real_dtype: torch.dtype = REAL_DTYPE,
) -> tuple[fq.Circuit, tuple[torch.Tensor, ...]]:
    """The circuit the cut sweep partitions, and the parameters it binds.

    The declared circuit cannot answer the question the sweep asks. Its Schmidt
    profile is flat at two across all five cuts -- a rank-two state at every
    position -- so moving the ownership boundary along it exchanges the same
    bond wherever it lands and any pair of numbers would be the same number.
    This circuit is not flat, so a boundary that moved has something to show.
    """

    parameters: list[torch.Tensor] = []
    for value in SWEEP_VALUES:
        tensor = torch.empty((), dtype=real_dtype, device=device)
        # Allocation followed by an explicit copy, so an accidental narrowing to
        # float32 is an observable mismatch rather than a silent test of FP32.
        tensor.copy_(torch.tensor(value, dtype=real_dtype))
        parameters.append(tensor.requires_grad_())
    circuit = fq.Circuit(N_WIRES, dtype=dtype, device=device)
    _bind_sweep(circuit, parameters)
    return circuit, tuple(parameters)


def _exact_schmidt_ranks() -> tuple[int, ...]:
    """The exact Schmidt rank of the sweep circuit at each of the five cuts.

    The bond dimension a site boundary carries is a property of the state, so
    this is what predicts it rather than the run being trusted for it. Rank is
    read from the singular values of the statevector split at each cut, which
    also fixes the wire order the sweep's widths are numbered in: cut `k` is the
    bond between wire `k` and wire `k + 1`.
    """

    circuit, _ = _cut_sweep_circuit(device=torch.device("cpu"))
    flat = circuit.state(refresh=True)[0].detach().cpu().reshape(-1)
    ranks: list[int] = []
    for cut in range(N_WIRES - 1):
        matrix = flat.reshape(2 ** (cut + 1), 2 ** (N_WIRES - cut - 1))
        singular_values = torch.linalg.svdvals(matrix)
        ranks.append(int(torch.count_nonzero(singular_values > NUMERICAL_TOLERANCE)))
    return tuple(ranks)


def _ownership_sweep_plans(
    world_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Every site plan this shape can hold with one owned boundary moved.

    A plan is an ordered, contiguous, complete partition of the wires, so moving
    the boundary between two neighbouring ranks is the whole of what a
    re-partition can do. The declared plan puts the boundaries at the balanced
    positions; this walks each of them across the placements that leave both of
    its neighbours non-empty and holds the rest still. At two ranks that is one
    boundary over five positions, and a wider shape has more boundaries, each
    walked in turn -- so the same sweep answers for a shape wider than a pair
    without pretending the two are the same experiment.
    """

    base = initial_mps_ownership(N_WIRES, world_size)
    plans: list[tuple[tuple[int, ...], ...]] = []
    for index in range(world_size - 1):
        start = base[index][0]
        stop = base[index + 1][-1]
        for position in range(start, stop):
            blocks: list[tuple[int, ...]] = []
            for rank, block in enumerate(base):
                if rank == index:
                    blocks.append(tuple(range(start, position + 1)))
                elif rank == index + 1:
                    blocks.append(tuple(range(position + 1, stop + 1)))
                else:
                    blocks.append(tuple(block))
            plan = tuple(blocks)
            validate_mps_ownership(plan, N_WIRES, world_size)
            if plan not in plans:
                plans.append(plan)
    return tuple(plans)


def _plan_width(plan: Sequence[Sequence[int]], ranks: Sequence[int]) -> int:
    """The widest bond a plan cuts, named by the exact rank of the state there.

    A plan wider than a pair holds several boundaries at once, so the number
    that describes it is the widest one it has to carry: a plan whose heaviest
    bond is rank four has exercised a rank-four cut however light its other
    boundaries are. Naming it by a lighter one would claim a rank-four bond was
    carried by a leg that crossed nothing of the sort.
    """

    cuts = [int(block[-1]) for block in plan[:-1]]
    return max(ranks[cut] for cut in cuts)


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


def _sweep_reference_expectation() -> tuple[float, tuple[float, ...]]:
    """The sweep circuit's expectation and gradient on one CPU device.

    Every sweep leg is a reverse, so the differential that says a leg cut the
    circuit it claims to have cut is the same one the declared leg uses: the
    exact complex128 statevector rather than a second MPS, because two MPS runs
    agreeing would say nothing about whether the sharding preserved the state.
    """

    circuit, parameters = _cut_sweep_circuit(device=torch.device("cpu"))
    state = circuit.state(refresh=True)
    expectation = _z_parity_expectation(
        state.abs().square().reshape(-1), OBSERVABLE_WIRES
    )
    expectation.backward()
    return (
        float(expectation.detach()),
        tuple(float(parameter.grad.detach()) for parameter in parameters),
    )


def _performance_observations(*, device: torch.device) -> dict[str, Any]:
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
        forward = execute_torch_distributed_mps_forward(
            circuit, device=device, max_bond=MAX_BOND
        )
        _synchronize(device)
        if forward.boundary_messages < 1 or forward.boundary_bytes <= 0:
            raise RuntimeError(
                "the measured workload did not exchange across the site boundary"
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
    observation["max_bond"] = MAX_BOND
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


def _host_staging_observation(*, device: torch.device) -> dict[str, Any]:
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
        execute_torch_distributed_mps_forward(circuit, device=device, max_bond=MAX_BOND)
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


def _cut_width_observations(
    *,
    device: torch.device,
    world_size: int,
) -> dict[str, Any]:
    """Partition one circuit at every site boundary the shape can hold.

    The blocker this feeds stands while a cut has never moved, so retracting it
    needs the cut to move and to carry an exchange wherever it landed. Only the
    reverse takes a plan, so each leg here is the reverse the training engine
    runs -- same call, same site kernels, no optimizer -- handed one plan from
    the family above; what it reports at that plan's boundary is the layer halo
    the inter-node transport actually carried.

    A leg that disagrees with the exact reference is a defect rather than a
    narrow sweep, and it raises: recording it as a width that did not count
    would quietly turn a wrong answer into a shorter list. A leg that crossed
    nothing is recorded with the nothing it crossed, and the blocker's own rule
    is what stops that from counting as a swept width.
    """

    ranks = _exact_schmidt_ranks()
    reference_value, reference_gradients = _sweep_reference_expectation()
    inter_node_bytes_by_width: dict[str, int] = {}
    plans: list[dict[str, Any]] = []
    for plan in _ownership_sweep_plans(world_size):
        width = _plan_width(plan, ranks)
        circuit, parameters = _cut_sweep_circuit(device=device)
        reverse = execute_torch_distributed_mps_reverse(
            circuit,
            observable=dict.fromkeys(OBSERVABLE_WIRES, "z"),
            device=device,
            max_bond=MAX_BOND,
            site_ownership=plan,
            compile_site_kernels=True,
        )
        reverse.backward()
        _synchronize(device)
        summary = reverse.summary()
        _require_two_node_placement(
            summary, leg=f"cut_width_{width}", expected_world_size=world_size
        )
        if summary["gradient_accuracy"] != "exact" or summary["discarded_weight"] != 0:
            raise RuntimeError(
                f"the width-{width} leg truncated the sweep circuit: "
                f"accuracy={summary['gradient_accuracy']} "
                f"discarded_weight={summary['discarded_weight']}"
            )
        if summary["full_mps_reconstruction"] or summary["replicated_autograd"]:
            raise RuntimeError(
                f"the width-{width} leg did not run as a sharded reverse"
            )
        if summary["scalability_claim_allowed"] is not False:
            raise RuntimeError(f"the width-{width} leg allowed a scalability claim")
        errors = [abs(float(reverse.value.detach()) - reference_value)]
        errors.extend(
            abs(float(parameter.grad.detach()) - expected)
            for parameter, expected in zip(parameters, reference_gradients, strict=True)
        )
        worst = max(errors)
        if not _collective_verdict(device, accepted=worst <= NUMERICAL_TOLERANCE):
            raise RuntimeError(
                f"the width-{width} leg disagreed with the exact reference: "
                f"{worst:.3e}"
            )
        crossed = _workload_total(device, summary["layer_halo_inter_node_bytes"])
        same_node = _workload_total(device, summary["layer_halo_intra_node_bytes"])
        payload = _workload_total(device, summary["layer_halo_payload_bytes"])
        messages = _workload_total(device, summary["layer_halo_message_count"])
        collectives = _workload_total(device, summary["gradient_collective_count"])
        # The two tiers are a partition of the halo rather than two independent
        # counters, so a leg that reported a payload it did not attribute to
        # either tier is refused instead of being recorded as an exchange.
        if crossed + same_node != payload:
            raise RuntimeError(
                f"the width-{width} leg attributed {crossed + same_node} of its "
                f"{payload} halo bytes to a tier"
            )
        # The halo is attributed to the rank that received it, so a rank inside
        # one host can honestly report none of it. What has to have happened is
        # the owner-sharded parameter reduction, because that is the gradient
        # the comparison above held to the reference.
        if collectives < 1:
            raise RuntimeError(
                f"the width-{width} leg reduced no gradients across {world_size} ranks"
            )
        inter_node_bytes_by_width[str(width)] = (
            inter_node_bytes_by_width.get(str(width), 0) + crossed
        )
        plans.append(
            {
                "plan": [list(block) for block in plan],
                "cut_positions": [int(block[-1]) for block in plan[:-1]],
                "width": width,
                "inter_node_bytes": crossed,
                "inter_node_messages": messages,
                "worst_error": worst,
            }
        )
    # A plan at `world_size` ranks carries one boundary per adjacent rank pair,
    # so it holds several cuts at once and the heaviest of them is what names the
    # leg. A height that only ever appears at a lighter boundary is therefore not
    # recorded as swept, because no leg carried it; the profile the reference
    # holds is recorded whole in `exact_schmidt_ranks`, so what the sweep placed
    # and what it did not are both readable from this record.
    return {
        "widths": sorted({int(plan["width"]) for plan in plans}),
        "exact_schmidt_ranks": list(ranks),
        "inter_node_bytes_by_width": inter_node_bytes_by_width,
        "plans": plans,
    }


def _derived_claim_blockers(observations: dict[str, Any]) -> list[str]:
    """Every blocker this run's own observations can retract.

    The declared part of the boundary -- the pair of hosts and the fixed
    circuit -- is not decided here, because nothing inside the run can retract
    it. What is decided here is only what the run observed, and the cut width is
    one of those: only the reverse accepts a plan, so the sweep above moves the
    site boundary and the widths the run crossed are the ones this reads. A
    sweep that never moved the cut leaves the blocker standing, which is what
    makes it a retraction rather than a relabelling.

    Called once while the artifact is assembled and again after the route has
    been read, because that observation arrives after the process group is gone
    and the group is what writes the log it is read from. Deriving the list in
    one place and applying it twice keeps the two passes from disagreeing.
    """

    return claim_blockers(
        [
            BLOCKER_TWO_NODE_PAIR_ONLY,
            BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,
        ],
        gather_claim_blockers(
            production_materialization=bool(
                observations["production_full_mps_materialization"]
            ),
            blocker=BLOCKER_VALIDATION_ONLY_TINY_FULL_MPS_GATHER,
        ),
        transport_claim_blockers(observations),
        measurement_claim_blockers(observations),
        staging_claim_blockers(observations),
        cut_width_claim_blockers(
            observations["cut_widths"]["inter_node_bytes_by_width"]
        ),
    )


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    training: dict[str, Any],
    cut_widths: dict[str, Any],
    network: dict[str, Any],
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
        "cut_widths": cut_widths,
        "network": network,
        "performance": performance,
        "host_staging": host_staging,
        "validation_full_mps_materialization": True,
        "production_full_mps_materialization": False,
    }
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
        "observations": observations,
        # The declared blockers, and why the pair and the circuit are among them,
        # are explained where the list is derived, as is the cut width, which is
        # retracted by the sweep rather than declared. The route arrives after
        # this call and is applied by `probe`.
        "claim_blockers": _derived_claim_blockers(observations),
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
    performance: dict[str, Any] | None = None
    host_staging: dict[str, Any] | None = None
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

        gathered_state = gather_mps_for_validation(forward)
        accepted = True
        statevector_metrics: dict[str, float] = {}
        if rank == 0:
            materialized = gathered_state.to_statevector()[0].detach().cpu()
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

        # The cut sweep is last of the execution legs, so nothing it leaves on
        # the device can reach the measured region or the staging audit.
        cut_widths = _cut_width_observations(device=device, world_size=world_size)

        if measure:
            # Both legs contain collectives, so every rank runs both of them and
            # in this order. A rank that skipped one would reach the next
            # collective alone and the lane would report a hang.
            performance = _performance_observations(device=device)
        host_staging = _host_staging_observation(device=device)

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
                cut_widths=cut_widths,
                network={},
                performance=performance,
                host_staging=host_staging,
                world_size=world_size,
                local_world_size=local_world_size,
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
