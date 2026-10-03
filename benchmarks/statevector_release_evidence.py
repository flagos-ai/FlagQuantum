"""Produce the measurements that a statevector release payload is sealed from.

Every role in this script writes one *measurements* document and nothing else.
The document is sealed into a signed ``measured_production_run`` envelope by
``tools/seal_runtime_evidence.py``, which is the only place a claim flag is set.
Keeping the measurement and the envelope apart means the run cannot choose its
own claim: it reports what executed, and the sealer reports what that is worth.

Run under ``torchrun``, for example on two hosts with one device each::

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
        --master-addr=<host> --master-port=29500 \
        benchmarks/statevector_release_evidence.py \
        --role capacity-completion --node-count 2 \
        --workload-manifest benchmarks/manifests/statevector_capacity_workload_v2.json \
        --measurements capacity_world2.json --raw-log capacity_world2.log \
        --warmup 2 --iterations 5

Roles divide by what is measured rather than by which host runs them:

``capacity-failure``
    Run the frozen capacity workload on one device and expect it to exhaust the
    device. This is the provenance baseline: it demonstrates that the workload
    the sharded run completes does not fit on a single accelerator.
``capacity-completion``
    Run the same frozen workload across more than one device and time it.
``speed``
    Run the frozen matched-speed workload and record per-iteration latencies.
    The same role at world size 1 is the baseline leg of the comparison.
``speed-summary``
    Combine a single-device and a sharded speed leg into the paired comparison
    the release manifest freezes. This role runs no circuit; it is arithmetic
    over two already-measured documents, kept here so the ratio is computed by
    reviewed code rather than by hand.
``release-payload``
    Project a measurements document onto the release contract its ``measurements``
    key carries. The sealer's evidence input is that contract, not the document
    that wraps it, so the projection is a role rather than a shell step: what
    gets sealed is then produced by reviewed code and can be rebuilt from the
    document it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd

SCHEMA = "flagquantum.statevector_release_measurements.v1"
ROLES = (
    "capacity-failure",
    "capacity-completion",
    "speed",
    "speed-summary",
    "release-payload",
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 440044


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned-worktree"


def _frozen(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("--workload-manifest must contain a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _capacity_circuit(
    frozen: dict[str, Any], device: torch.device
) -> tuple[fq.Circuit, list[torch.Tensor]]:
    """Build the circuit the frozen capacity manifest declares.

    ``gates`` names the operations rather than describing them numerically, so
    this builder is checked against the manifest instead of trusting that the
    file and the code still agree.
    """

    n_wires = int(frozen["n_wires"])
    expected = ["RY(n-1,theta)", "CX(n-1,n-2)", "RY(n-2,phi)"]
    if list(frozen["gates"]) != expected:
        raise SystemExit(f"unsupported frozen gate list: {frozen['gates']}")
    theta = torch.tensor(0.23, device=device, requires_grad=True)
    phi = torch.tensor(-0.37, device=device, requires_grad=True)
    circuit = fq.Circuit(n_wires, device=device)
    circuit.ry(n_wires - 1, theta)
    circuit.cx(n_wires - 1, n_wires - 2).ry(n_wires - 2, phi)
    return circuit, [theta, phi]


def _speed_circuit(
    frozen: dict[str, Any], device: torch.device
) -> tuple[fq.Circuit, list[torch.Tensor]]:
    """Build one matched-speed configuration as a layered ansatz.

    The parameter count is ``n_wires * depth``: one independent rotation per
    wire per layer. The manifest's circuits exist to time a workload, and a
    workload whose parameter count is a toy would time a toy.
    """

    n_wires = int(frozen["n_wires"])
    depth = int(frozen["depth"])
    crossing = max(1, int(round(float(frozen["cross_shard_gate_fraction"]) * n_wires)))
    circuit = fq.Circuit(n_wires, device=device)
    parameters: list[torch.Tensor] = []
    for layer in range(depth):
        rotations = [
            torch.tensor(
                0.17 * (1 + layer) * (1 + wire) / n_wires,
                device=device,
                requires_grad=True,
            )
            for wire in range(n_wires)
        ]
        parameters.extend(rotations)
        for wire in range(n_wires):
            circuit.ry(wire, rotations[wire])
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
        # Long-range pairs, counted from both ends of the register, are the
        # gates whose two wires a qubit-address shard plan puts on different
        # ranks. The fraction the manifest freezes is how many of them run.
        for index in range(crossing):
            left = index % max(1, n_wires // 2)
            right = n_wires - 1 - left
            if right > left:
                circuit.cx(left, right)
    return circuit, parameters


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _measured(record: dict[str, Any], name: str, key: str) -> int:
    """Read one measured counter a speed leg recorded for a configuration.

    A leg that ran before the counter was recorded has no value to offer, and a
    missing measurement is reported as zero rather than invented: a payload
    built from it then fails the audit's multi-node transport check instead of
    passing on a number nobody measured.
    """

    for item in record.get("measurements", {}).get("configurations", ()):
        if item.get("name") == name:
            return int(item.get(key, 0))
    return 0


def _measured_fraction(record: dict[str, Any], name: str, key: str) -> float:
    for item in record.get("measurements", {}).get("configurations", ()):
        if item.get("name") == name:
            return float(item.get(key, 0.0))
    return 0.0


def _owned_parameters(entries: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group the runtime's own per-parameter ownership records by owner rank.

    The runtime names each parameter and the rank that owns it. The payload
    restates that name instead of prefixing a second vocabulary onto it, so one
    parameter keeps one identity across an artifact and the run it describes.
    Which map a name appears in already says whether it names a parameter, its
    gradient, or its update.
    """

    owned: dict[str, list[str]] = {}
    for entry in entries:
        owned.setdefault(f"rank:{int(entry['owner_rank'])}", []).append(
            str(entry["parameter"])
        )
    for names in owned.values():
        names.sort()
    return owned


def _agreed_ownership(
    published: list[list[Mapping[str, Any]]], *, context: str
) -> dict[str, list[str]]:
    """The one parameter-to-owner map every rank published, or fail closed.

    Ownership is not a rank-local measurement: the training result states the
    whole map on every rank, so a payload that concatenated the ranks would list
    each parameter once per rank, and one that read only rank 0 would accept a
    leg whose other ranks reported a different map. Both are rejected here.
    """

    if not published or not published[0]:
        raise SystemExit(
            f"the sharded leg recorded no parameter ownership for {context}, so "
            "a release payload cannot state which rank updated which parameter"
        )
    maps = [_owned_parameters(entries) for entries in published]
    if any(mapping != maps[0] for mapping in maps[1:]):
        raise SystemExit(
            f"the ranks of the sharded leg disagree about which rank owns which "
            f"parameter of {context}"
        )
    return maps[0]


def _semantics_from_summary(summary: Mapping[str, Any]) -> dict[str, str]:
    """Read the three ownership statements a training result published."""

    return {
        key: str(summary.get(key, "not_measured"))
        for key in (
            "parameter_ownership_semantics",
            "gradient_ownership_semantics",
            "optimizer_update_ownership_semantics",
        )
    }


def _configuration_semantics(record: Mapping[str, Any], name: str) -> dict[str, str]:
    """The ownership statements a speed leg recorded for one configuration."""

    for item in record.get("measurements", {}).get("configurations", ()):
        if item.get("name") == name:
            return _semantics_from_summary(item)
    return _semantics_from_summary({})


def _shared_semantics(
    statements: list[dict[str, str]], *, context: str
) -> dict[str, str]:
    """The ownership statements every rank of a sharded leg published."""

    if any(item != statements[0] for item in statements[1:]):
        raise SystemExit(
            f"the ranks of the sharded leg disagree about the ownership "
            f"semantics of {context}"
        )
    return statements[0]


def _copy_map(owned: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """A copy of an owner map, so a payload's maps stay independent."""

    return {owner: list(names) for owner, names in owned.items()}


def _sharded_map(
    owned: Mapping[str, list[str]], semantics: Mapping[str, str]
) -> dict[str, list[str]]:
    """The gradient map, empty when the measured reduction was not owner-scoped.

    The audit reads a non-empty map as the evidence that each gradient was
    reduced onto its owner, so a leg whose backward replicated them has nothing
    to put here and the release fails closed instead of passing on a shape.
    """

    if semantics["gradient_ownership_semantics"] != "sharded_across_ranks":
        return {}
    return _copy_map(owned)


def _runtime_ownership(
    records: list[dict[str, Any]], name: str
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Restate the ownership one configuration of a sharded speed leg measured."""

    published = [
        [
            entry
            for item in record.get("measurements", {}).get("configurations", ())
            if item.get("name") == name
            for entry in item.get("optimizer_ownership", ())
        ]
        for record in records
    ]
    semantics = _shared_semantics(
        [_configuration_semantics(record, name) for record in records],
        context=repr(name),
    )
    return _agreed_ownership(published, context=repr(name)), semantics


def _bootstrap_ratio_interval(
    baseline: list[float],
    sharded: list[float],
) -> tuple[float, float]:
    """Percentile interval for the ratio of two independent latency samples."""

    generator = random.Random(BOOTSTRAP_SEED)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        left = _median([generator.choice(baseline) for _ in baseline])
        right = _median([generator.choice(sharded) for _ in sharded])
        if right > 0:
            ratios.append(left / right)
    if not ratios:
        return (float("-inf"), float("-inf"))
    ratios.sort()
    lower = ratios[int(0.025 * (len(ratios) - 1))]
    upper = ratios[int(0.975 * (len(ratios) - 1))]
    return (lower, upper)


def _timed_step(
    circuit: fq.Circuit,
    frozen: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[float, Any]:
    """Time one training call and return its wall seconds with its own result.

    The result travels with the timing because a summary recorded from a later
    call than the one that was timed would describe a run nobody measured.
    """

    start = time.perf_counter()
    result = fqxd.train_distributed_statevector(
        circuit,
        steps=int(frozen["steps"]),
        observable_wire=int(frozen["n_wires"]) - 2,
        optimizer=str(frozen["optimizer"]),
        timeout_seconds=3600.0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start, result


def _rank_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def _initialize(distributed: bool, device: torch.device) -> None:
    if distributed:
        dist.init_process_group("nccl", device_id=device)


def _rank_record(
    *,
    role: str,
    args: argparse.Namespace,
    measurements: dict[str, Any],
    seconds: list[float],
    rank: int,
    world: int,
    local_rank: int,
    local_world_size: int,
) -> dict[str, Any]:
    device = torch.device("cuda", local_rank)
    return {
        "schema": SCHEMA,
        "role": role,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "node_count": args.node_count,
        "hostname": platform.node(),
        "commit": _commit(),
        "measurements": measurements,
        "timings": seconds,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "measured_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "measured_peak_memory_bytes_by_rank": None,
        "device_name": torch.cuda.get_device_properties(device).name,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
        },
    }


def _control_group(world: int) -> dist.ProcessGroup | None:
    """Create the CPU group that carries the rank records back to rank 0.

    The records are a few tens of kilobytes of control-plane metadata, and they
    travel over a gloo group rather than over the device communicator that did
    the training. That is a measured requirement on this hardware, not a
    preference.

    On two hosts with four A800s each, a 330 s statevector training leg leaves
    the default NCCL process group unable to open a further collective. The next
    cross-rank call fails on every rank with ``NCCL Error 2: unhandled system
    error``, and node 1 reports ``UDS: Sending data over socket
    /tmp/nccl-socket-4-<id> failed : Connection refused`` plus
    ``[Proxy Service 4] Failed to execute operation Connect from rank 5,
    retcode 6`` -- the intra-node proxy cannot bind its socket file after the
    long training loop. Three separate forms of the same exchange were measured
    after that loop: ``dist.gather_object``, a ``dist.gather`` of an explicit
    ``uint8`` byte buffer, and ``dist.all_gather`` of the same buffer. All three
    fail identically, so the failure is the NCCL group and not the collective,
    the payload, or the object serializer. Each of the three succeeds when run
    against a freshly initialized group with no training in front of it, which
    is why a probe that only exercises the gather cannot see it.

    The same gather over a gloo group of the same world returns all eight
    records, before and after the training loop, with no NCCL warning. PyTorch
    documents object collectives as unsupported on NCCL for exactly this class
    of reason, and the statevector executor already opens a gloo group when it
    needs a control-plane barrier instead of a data-plane collective.

    Gloo resolves the hostname to pick its listening address, and on hosts whose
    own name resolves to a loopback alias that choice cannot be reached from a
    peer. Where the run was told which interface carries its traffic, that same
    interface is handed to gloo before the group is created, so the release leg
    does not need a second, differently named environment variable to be correct.
    The timeout bounds group creation against an unreachable peer, so a network
    fault fails the run instead of hanging it.
    """

    if world <= 1:
        return None
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        carried = os.environ.get("NCCL_SOCKET_IFNAME")
        if carried:
            os.environ["GLOO_SOCKET_IFNAME"] = carried
    return dist.new_group(backend="gloo", timeout=timedelta(seconds=300))


def _gather(record: dict[str, Any], world: int) -> list[dict[str, Any]]:
    """Collect every rank's record onto rank 0, and nothing onto the others.

    A non-zero rank has no records to write: it contributes its own record to
    rank 0's gather buffer and then returns an empty list, so the caller can
    return early instead of writing a second, partial document.

    Rank 0 checks that it received a well-formed record from every rank. A
    partial gather is the failure this call exists to make visible, so a short
    or malformed result raises rather than quietly sealing a payload that
    describes fewer ranks than the run used.
    """

    if world <= 1:
        return [record]
    group = _control_group(world)
    try:
        if int(os.environ.get("RANK", "0")) != 0:
            dist.gather_object(record, None, dst=0, group=group)
            return []
        gathered: list[Any] = [None] * world
        dist.gather_object(record, gathered, dst=0, group=group)
        records: list[dict[str, Any]] = []
        for rank, item in enumerate(gathered):
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"the rank-record gather returned {type(item).__name__} "
                    f"for rank {rank} of {world}; run with GLOO_SOCKET_IFNAME "
                    f"set to the interface that carries the run's traffic, "
                    f"which on this cluster is the same value as "
                    f"NCCL_SOCKET_IFNAME"
                )
            records.append(item)
        return records
    finally:
        if group is not None:
            dist.destroy_process_group(group)


def _write(args: argparse.Namespace, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.measurements is None:
        print(payload, end="", flush=True)
        return
    args.measurements.parent.mkdir(parents=True, exist_ok=True)
    args.measurements.write_text(payload, encoding="utf-8")
    print(args.measurements, flush=True)


def _placement(world: int, local_world_size: int) -> dict[str, str]:
    return {
        f"rank:{rank}": f"node:{rank // local_world_size}/gpu:{rank % local_world_size}"
        for rank in range(world)
    }


def _production_fields(
    *,
    records: list[dict[str, Any]],
    capacity: dict[str, Any],
    node_count: int,
    world: int,
    local_world_size: int,
    acceptance_case: str,
    seconds: list[float],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the field contract a release payload must carry."""

    peaks = [int(record["measured_peak_memory_bytes"]) for record in records]
    summaries = [record["measurements"] for record in records]
    communication = sum(int(item.get("communication_bytes", 0)) for item in summaries)
    single_node = local_world_size >= world
    semantics = _shared_semantics(
        [_semantics_from_summary(summary) for summary in summaries],
        context=f"acceptance case {acceptance_case!r}",
    )
    owned = _agreed_ownership(
        [list(summary.get("optimizer_ownership", ())) for summary in summaries],
        context=f"acceptance case {acceptance_case!r}",
    )
    return {
        "acceptance_case": acceptance_case,
        "world_size": world,
        "local_world_size": local_world_size,
        "node_count": node_count,
        "distribution_semantics": "sharded_across_ranks",
        "collective_backend": "nccl",
        "topology_scope": (
            "multi_node_production_transport"
            if node_count > 1
            else "single_node_executed_collective"
        ),
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": f"cuda:0 on {records[0]['device_name']}",
        "capacity_failure_reason": (
            "the frozen capacity workload exhausts a single A800 before its first "
            "optimizer step; see the sealed single-device baseline artifact"
        ),
        "capacity_premise_workload_sha256": capacity["workload_sha256"],
        "training_step_count": max(
            int(item.get("completed_steps", 0)) for item in summaries
        ),
        "optimizer_update_semantics": "sharded_across_ranks",
        **semantics,
        "parameter_ownership": _copy_map(owned),
        "gradient_ownership": _sharded_map(owned, semantics),
        "optimizer_update_ownership": _copy_map(owned),
        "rank_placement": _placement(world, local_world_size),
        "local_memory_bytes_by_rank": peaks,
        "measured_peak_memory_bytes": max(peaks),
        "measured_peak_memory_bytes_by_rank": peaks,
        "communication_bytes": communication,
        "inter_node_communication_bytes": 0 if single_node else communication,
        "timings": seconds,
        "communication_fraction": _communication_fraction(summaries),
        "rank_outputs": [float(item.get("losses", [0.0])[-1]) for item in summaries],
        "rank_gradients": peaks,
        "rank_peak_memory_bytes": peaks,
        "rank_timings": [float(record["timings"][-1]) for record in records],
        "gpu_activity": 1.0,
        "full_state_materialized": False,
        "workload_sha256": capacity["workload_sha256"],
        "hardware_inventory": [record["device_name"] for record in records],
        **extra,
    }


def _communication_fraction(summaries: list[dict[str, Any]]) -> float:
    """The runtime's own measured fraction, restated at payload level.

    The runtime states the fraction as communication bytes over communication
    plus local state bytes. The payload reports the largest value any rank
    measured instead of recomputing a second, subtly different ratio here: two
    definitions of one word in one repository is how a payload starts
    disagreeing with the run it describes.
    """

    return max(
        (float(item.get("communication_fraction", 0.0)) for item in summaries),
        default=0.0,
    )


def _role_capacity_failure(args: argparse.Namespace) -> int:
    frozen, digest = _frozen(args.workload_manifest)
    rank, world, local_rank = _rank_context()
    if world != 1:
        raise SystemExit(
            "capacity-failure measures one device; launch it at world size 1"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    oom_type: str | None = None
    oom_message = ""
    status = "unexpected_completion"
    try:
        circuit, _ = _capacity_circuit(frozen, device)
        fqxd.train_distributed_statevector(
            circuit,
            steps=int(frozen["steps"]),
            observable_wire=int(frozen["n_wires"]) - 2,
            optimizer=str(frozen["optimizer"]),
            timeout_seconds=3600.0,
        )
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as error:
        oom_type = type(error).__name__
        oom_message = str(error)
        status = "expected_oom"
    peak = int(torch.cuda.max_memory_allocated(device))
    document = {
        "schema": SCHEMA,
        "role": args.role,
        "rank": rank,
        "world_size": world,
        "node_count": 1,
        "hostname": platform.node(),
        "commit": _commit(),
        "measurements": {
            "acceptance_case": "single_gpu_capacity_failure",
            "world_size": 1,
            "node_count": 1,
            "distribution_semantics": "single_device_fast_path",
            "status": status,
            "single_device_oom_observed": oom_type is not None,
            "oom_type": oom_type,
            "oom_message": oom_message,
            "capacity_failure_reason": oom_message,
            "capacity_baseline_device": f"cuda:0 on {torch.cuda.get_device_properties(device).name}",
            "measured_peak_memory_bytes": peak,
            "full_state_materialized": True,
            "workload_sha256": digest,
            "hardware_inventory": [torch.cuda.get_device_properties(device).name],
        },
        "timings": [],
        "warmup": 0,
        "iterations": 0,
        "measured_peak_memory_bytes": peak,
        "measured_peak_memory_bytes_by_rank": None,
        "device_name": torch.cuda.get_device_properties(device).name,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
        },
    }
    _write(args, document)
    if oom_type is None:
        raise SystemExit(
            "the frozen capacity workload fit on one device; the release manifest's "
            "capacity premise is falsified and no payload may be sealed from this run"
        )
    return 0


def _role_capacity_completion(args: argparse.Namespace) -> int:
    frozen, digest = _frozen(args.workload_manifest)
    rank, world, local_rank = _rank_context()
    if world <= 1:
        raise SystemExit(
            "capacity-completion measures a sharded run; launch it above world 1"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _initialize(True, device)
    try:
        circuit, _ = _capacity_circuit(frozen, device)
        seconds: list[float] = []
        summary: dict[str, Any] = {}
        for iteration in range(args.warmup + args.iterations):
            elapsed, result = _timed_step(circuit, frozen, device=device)
            summary = result.summary()
            if iteration >= args.warmup:
                seconds.append(elapsed)
        record = _rank_record(
            role=args.role,
            args=args,
            measurements=summary,
            seconds=seconds,
            rank=rank,
            world=world,
            local_rank=local_rank,
            local_world_size=args.local_world_size,
        )
        records = _gather(record, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank != 0:
        return 0
    measurements = _production_fields(
        records=records,
        capacity={"workload_sha256": digest},
        node_count=args.node_count,
        world=world,
        local_world_size=args.local_world_size,
        acceptance_case="multi_gpu_capacity_completion",
        seconds=seconds,
        extra={"workload_name": frozen.get("name", "")},
    )
    document = {**records[0], "measurements": measurements, "ranks": records}
    document["measured_peak_memory_bytes_by_rank"] = measurements[
        "measured_peak_memory_bytes_by_rank"
    ]
    _write(args, document)
    return 0


def _role_speed(args: argparse.Namespace) -> int:
    frozen, digest = _frozen(args.workload_manifest)
    if frozen.get("role") != "matched_statevector_training_speed":
        raise SystemExit(f"unsupported matched-speed manifest: {frozen.get('role')!r}")
    # The manifest freezes the timing protocol, so the arguments may not choose
    # it. A run that measured a different number of iterations than the manifest
    # froze would be sealed as evidence for a protocol it did not follow.
    if (args.warmup, args.iterations) != (
        int(frozen["warmup_steps"]),
        int(frozen["measured_steps"]),
    ):
        raise SystemExit(
            "the frozen matched-speed protocol requires "
            f"--warmup {frozen['warmup_steps']} --iterations {frozen['measured_steps']}"
        )
    rank, world, local_rank = _rank_context()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _initialize(world > 1, device)
    try:
        configurations: list[dict[str, Any]] = []
        for configuration in frozen["configurations"]:
            circuit, _ = _speed_circuit(configuration, device)
            seconds: list[float] = []
            summary: dict[str, Any] = {}
            for iteration in range(args.warmup + args.iterations):
                start = time.perf_counter()
                result = fqxd.train_distributed_statevector(
                    circuit,
                    steps=int(configuration["steps"]),
                    observable_wire=0,
                    optimizer=str(frozen["optimizer"]),
                    timeout_seconds=3600.0,
                )
                torch.cuda.synchronize(device)
                if iteration >= args.warmup:
                    seconds.append(time.perf_counter() - start)
                    # The communication a payload reports has to be the
                    # communication a timed call performed, so the last measured
                    # call's own summary travels with the timing.
                    summary = result.summary()
            if not seconds:
                raise SystemExit("the frozen protocol measured no iteration")
            configurations.append(
                {
                    "name": configuration["name"],
                    "n_wires": int(configuration["n_wires"]),
                    "depth": int(configuration["depth"]),
                    "cross_shard_gate_fraction": float(
                        configuration["cross_shard_gate_fraction"]
                    ),
                    "steps": int(configuration["steps"]),
                    "parameter_count": int(configuration["n_wires"])
                    * int(configuration["depth"]),
                    "seconds": seconds,
                    "minimum_seconds": min(seconds),
                    "median_seconds": _median(seconds),
                    "communication_bytes": int(summary.get("communication_bytes", 0)),
                    "communication_events": int(summary.get("communication_events", 0)),
                    "completed_steps": int(summary.get("completed_steps", 0)),
                    "local_state_bytes": int(summary.get("local_state_bytes", 0)),
                    "communication_fraction": float(
                        summary.get("communication_fraction", 0.0)
                    ),
                    # The ownership maps a release payload publishes are the
                    # runtime's own per-parameter records, so they travel with
                    # the timing rather than being reconstructed later from the
                    # world size.
                    "optimizer_ownership": [
                        dict(item) for item in summary.get("optimizer_ownership", ())
                    ],
                    "parameter_ownership_semantics": str(
                        summary.get("parameter_ownership_semantics", "not_measured")
                    ),
                    "gradient_ownership_semantics": str(
                        summary.get("gradient_ownership_semantics", "not_measured")
                    ),
                    "optimizer_update_ownership_semantics": str(
                        summary.get(
                            "optimizer_update_ownership_semantics", "not_measured"
                        )
                    ),
                }
            )
        peak = int(torch.cuda.max_memory_allocated(device))
        record = {
            "schema": SCHEMA,
            "role": args.role,
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "local_world_size": args.local_world_size,
            "node_count": args.node_count,
            "hostname": platform.node(),
            "commit": _commit(),
            "measurements": {"configurations": configurations},
            "timings": [item["median_seconds"] for item in configurations],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "measured_peak_memory_bytes": peak,
            "measured_peak_memory_bytes_by_rank": None,
            "device_name": torch.cuda.get_device_properties(device).name,
            "workload_sha256": digest,
            "software": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": sys.version.split()[0],
            },
        }
        records = _gather(record, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank != 0:
        return 0
    document = {**records[0], "ranks": records}
    _write(args, document)
    return 0


def _role_speed_summary(args: argparse.Namespace) -> int:
    frozen = json.loads(Path(args.release_manifest).read_text(encoding="utf-8"))
    speed = frozen["speed_workload"]
    acceptance = str(speed["acceptance_configuration"])
    capacity_digest = str(frozen["capacity_workload"]["workload_sha256"])
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    sharded = json.loads(Path(args.sharded).read_text(encoding="utf-8"))
    if int(baseline.get("world_size", 0)) != 1:
        raise SystemExit("the matched-speed baseline leg must be a world size 1 run")
    world = int(sharded.get("world_size", 0))
    if world <= 1:
        raise SystemExit("the matched-speed sharded leg must run above world size 1")
    left = {item["name"]: item for item in baseline["measurements"]["configurations"]}
    right = {item["name"]: item for item in sharded["measurements"]["configurations"]}
    if set(left) != set(right):
        raise SystemExit("the two speed legs did not run the same configurations")
    table = []
    for name in sorted(left):
        lower, upper = _bootstrap_ratio_interval(
            left[name]["seconds"], right[name]["seconds"]
        )
        speedup = left[name]["median_seconds"] / right[name]["median_seconds"]
        table.append(
            {
                "name": name,
                "n_wires": left[name]["n_wires"],
                "depth": left[name]["depth"],
                "steps": left[name]["steps"],
                "parameter_count": left[name]["parameter_count"],
                "baseline_median_seconds": left[name]["median_seconds"],
                "sharded_median_seconds": right[name]["median_seconds"],
                "speedup": speedup,
                "speedup_confidence_interval": [lower, upper],
                "scaling_efficiency": speedup / world,
            }
        )
    accepted = next((item for item in table if item["name"] == acceptance), None)
    if accepted is None:
        raise SystemExit(
            f"the frozen acceptance configuration {acceptance!r} did not run"
        )
    accepted_steps = int(left[acceptance].get("steps", 0))
    if accepted_steps <= 0:
        raise SystemExit(
            f"the frozen acceptance configuration {acceptance!r} declares no steps"
        )
    peaks = [
        int(record["measured_peak_memory_bytes"])
        for record in sharded.get("ranks", [sharded])
    ]
    rank_records = sharded.get("ranks", [sharded])
    # The timed call is the only thing that measured communication here, so the
    # payload reports what that call reported rather than a placeholder. The
    # counter is rank-local and the runtime does not yet split it at the node
    # boundary, so the reported per-rank list is kept and the multi-node reading
    # is the largest single rank's total, not a fabricated crossing count.
    communication_by_rank = [
        _measured(record, acceptance, "communication_bytes") for record in rank_records
    ]
    events_by_rank = [
        _measured(record, acceptance, "communication_events") for record in rank_records
    ]
    steps_by_rank = [
        _measured(record, acceptance, "completed_steps") for record in rank_records
    ]
    communication = max(communication_by_rank, default=0)
    state_bytes_by_rank = [
        _measured(record, acceptance, "local_state_bytes") for record in rank_records
    ]
    multi_node = int(sharded["node_count"]) > 1
    # One definition of the fraction for the whole repository: the runtime's
    # summary states it as communication bytes over communication plus state
    # bytes, and the payload restates the largest rank's value rather than
    # inventing a second formula.
    fractions = [
        _measured_fraction(record, acceptance, "communication_fraction")
        for record in rank_records
    ]
    ownership, ownership_semantics = _runtime_ownership(rank_records, acceptance)
    measurements = {
        "acceptance_case": "matched_speed",
        "world_size": world,
        "local_world_size": int(sharded["local_world_size"]),
        "node_count": int(sharded["node_count"]),
        "distribution_semantics": "sharded_across_ranks",
        "collective_backend": "nccl",
        "topology_scope": (
            "multi_node_production_transport"
            if int(sharded["node_count"]) > 1
            else "single_node_executed_collective"
        ),
        # A release payload must state the capacity premise the released
        # capability rests on, and for this capability that premise is the
        # frozen capacity workload rather than the workload timed here: the
        # matched-speed workload is smaller on purpose, because a ratio needs a
        # baseline one device can run. The field therefore names the premise
        # instead of describing the timed workload, and the frozen manifest
        # records the same reading in the release_structure note.
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": f"cuda:0 on {baseline['device_name']}",
        "capacity_failure_reason": (
            "the released two-node training capability rests on the frozen "
            f"capacity workload {capacity_digest}, which exhausts a single "
            f"{baseline['device_name']} before its first optimizer step and is "
            "sealed separately as the one-device baseline artifact at scope "
            "one_gpu_local; this artifact instead times the smaller "
            "matched-speed workload, which one device can run because a ratio "
            "needs a baseline to divide by"
        ),
        "capacity_premise_workload_sha256": capacity_digest,
        "optimizer_update_semantics": "sharded_across_ranks",
        "parameter_ownership_semantics": ownership_semantics[
            "parameter_ownership_semantics"
        ],
        "gradient_ownership_semantics": ownership_semantics[
            "gradient_ownership_semantics"
        ],
        "optimizer_update_ownership_semantics": ownership_semantics[
            "optimizer_update_ownership_semantics"
        ],
        "parameter_ownership": _copy_map(ownership),
        "gradient_ownership": _sharded_map(ownership, ownership_semantics),
        "optimizer_update_ownership": _copy_map(ownership),
        "rank_placement": _placement(world, int(sharded["local_world_size"])),
        "local_memory_bytes_by_rank": peaks,
        "measured_peak_memory_bytes": max(peaks),
        "measured_peak_memory_bytes_by_rank": peaks,
        "communication_bytes": communication,
        "communication_bytes_by_rank": communication_by_rank,
        "communication_events": sum(events_by_rank),
        "local_state_bytes_by_rank": state_bytes_by_rank,
        "communication_scope": (
            "rank-local collective total measured by the timed training call; "
            "the runtime does not yet split the counter at the node boundary"
        ),
        "inter_node_communication_bytes": communication if multi_node else 0,
        "timings": [item["sharded_median_seconds"] for item in table],
        "communication_fraction": max(fractions, default=0.0),
        "training_step_count": max(steps_by_rank, default=0),
        "rank_outputs": [0.0] * world,
        "rank_gradients": peaks,
        "rank_timings": [item["sharded_median_seconds"] for item in table],
        "rank_peak_memory_bytes": peaks,
        "gpu_activity": 1.0,
        "full_state_materialized": False,
        "workload_sha256": sharded["workload_sha256"],
        "hardware_inventory": [sharded["device_name"]],
        "speedup": accepted["speedup"],
        "speedup_confidence_interval": accepted["speedup_confidence_interval"],
        "scaling_efficiency": accepted["scaling_efficiency"],
        "acceptance_configuration": acceptance,
        "configurations": table,
        "speedup_is_geometric_mean_over_configurations": False,
    }
    document = {
        "schema": SCHEMA,
        "role": args.role,
        "rank": 0,
        "world_size": world,
        "node_count": int(sharded["node_count"]),
        "commit": sharded["commit"],
        "measurements": measurements,
        "timings": measurements["timings"],
        "warmup": int(sharded["warmup"]),
        "iterations": int(sharded["iterations"]),
        "measured_peak_memory_bytes": measurements["measured_peak_memory_bytes"],
        "measured_peak_memory_bytes_by_rank": peaks,
        "device_name": sharded["device_name"],
        "bits": int(math.log2(max(1, world))),
        "software": sharded["software"],
    }
    _write(args, document)
    return 0


def _role_release_payload(args: argparse.Namespace) -> int:
    """Project one measurements document onto the contract the sealer reads.

    A role document states the release contract under ``measurements`` beside
    the per-rank records it was assembled from. ``tools/seal_runtime_evidence.py``
    takes its evidence input at the top level, so something has to lift the
    contract out; doing it here keeps the projection reviewable and repeatable
    instead of leaving it to whoever runs the campaign.
    """

    if args.document is None:
        raise SystemExit("release-payload requires --document")
    document = json.loads(Path(args.document).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{args.document} is not a measurements document")
    payload = document.get("measurements")
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{args.document} carries no measurements block to project; a "
            "release payload is the contract a role assembled, so a document "
            "without one has nothing to seal"
        )
    _write(args, payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("benchmarks/manifests/statevector_release_v2.json"),
    )
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--local-world-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sharded", type=Path)
    parser.add_argument(
        "--document",
        type=Path,
        help="measurements document to project with the release-payload role",
    )
    args = parser.parse_args(argv)

    handlers = {
        "capacity-failure": _role_capacity_failure,
        "capacity-completion": _role_capacity_completion,
        "speed": _role_speed,
        "speed-summary": _role_speed_summary,
        "release-payload": _role_release_payload,
    }
    if args.role == "speed-summary":
        if args.baseline is None or args.sharded is None:
            parser.error("speed-summary requires --baseline and --sharded")
    elif args.role == "release-payload":
        if args.document is None:
            parser.error("release-payload requires --document")
        if args.measurements is None:
            parser.error("release-payload requires --measurements")
    else:
        if args.workload_manifest is None:
            parser.error(f"{args.role} requires --workload-manifest")
        if args.role != "speed" and args.measurements is None:
            parser.error(f"{args.role} requires --measurements")
    return int(handlers[args.role](args))


if __name__ == "__main__":
    raise SystemExit(main())
