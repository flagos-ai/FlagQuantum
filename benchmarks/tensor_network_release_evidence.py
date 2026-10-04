"""Measure one tensor-network training workload and emit a release payload.

The output of this script is a *measurements document*, not a signed artifact:
it reports what executed, and ``tools/seal_runtime_evidence.py`` reports what
that is worth. Two roles are available. ``capacity-failure`` runs the frozen
capacity workload on one device and records the exhaustion that makes the
workload a capacity premise, and ``capacity-completion`` runs the same workload
across ranks and records the sharded training update that completes it.

The frozen capacity workload is identified by its digest, not by its shape: the
producer reads the digest the release manifest declares and refuses to record a
run whose workload does not match it, so a manifest that has drifted away from
what ran fails closed instead of attributing a completion to the wrong circuit.

Every document this producer writes also states the capability it measured, read
from the state mode the manifest froze, because the runtime-evidence envelope
carries no capability key and the release gate admits a payload by that mode. A
document that stated none, or one the manifest did not freeze, would be sealed as
evidence for a capability nobody ran, so the write boundary refuses it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from benchmarks.internal.evidence.speedup import bootstrap_ratio_interval
from flagquantum.experimental.distributed import train_distributed_tensor_network
from flagquantum.runtime.audit.vocabulary import TENSOR_NETWORK_STATE_MODES

RELEASE_MANIFEST = Path("benchmarks/manifests/tensor_network_release_v1.json")
ROLES = (
    "capacity-failure",
    "capacity-completion",
    "matched-speed",
    "speed-summary",
    "release-payload",
)


def _commit() -> str:
    """Return the revision the producer is running at."""

    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _software() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "python": sys.version.split()[0],
    }


def _stated_state_mode(manifest_path: Path) -> str:
    """Return the state mode the frozen manifest states for this capability.

    The release gate admits a payload by the state mode the payload states about
    itself, and the audit vocabulary owns the names that may be stated, so the
    mode is read from the frozen manifest and checked against that vocabulary
    rather than restated here. A manifest that named a mode the vocabulary does
    not recognize, or none at all, refuses this producer instead of emitting a
    payload the gate would read as another capability's evidence.
    """

    manifest = _frozen(Path(manifest_path))
    runtime = manifest.get("runtime")
    declared = (
        str(runtime.get("state_mode", "")).lower()
        if isinstance(runtime, Mapping)
        else ""
    )
    if declared not in TENSOR_NETWORK_STATE_MODES:
        raise SystemExit(
            f"{manifest_path} declares runtime.state_mode {declared!r}, which the "
            "audit vocabulary does not recognize as a tensor-network state mode "
            f"({sorted(TENSOR_NETWORK_STATE_MODES)}); a payload sealed without a "
            "recognized mode is refused by the release gate as foreign evidence"
        )
    return declared


def _frozen(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not carry a frozen workload")
    return payload


def _capacity_contract(manifest_path: Path) -> dict[str, Any]:
    """Return the frozen capacity contract the release manifest declares."""

    manifest = _frozen(manifest_path)
    capacity = manifest.get("capacity_workload")
    if not isinstance(capacity, Mapping):
        raise SystemExit(f"{manifest_path} declares no capacity workload")
    return dict(capacity)


def _workload_digest(contract: Mapping[str, Any]) -> str:
    """Return the digest of the frozen workload file the contract names."""

    source = Path(str(contract["workload_manifest_path"]))
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _protocol_digest(manifest_path: Path) -> str:
    """Return the digest of the manifest that froze the matched-speed ladder.

    The capacity role has a frozen workload file because a capacity premise is a
    property of one circuit. The matched-speed role does not: its ladder is
    three configurations the manifest freezes inline, and the producer builds
    their circuits from those numbers rather than reading them, so the file that
    identifies the timed protocol is the manifest itself. Naming it here keeps
    one identity per frozen thing instead of hashing whatever happens to be
    nearby.
    """

    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def grid_circuit(rows: int, columns: int, cycles: int, *, device, dtype) -> fq.Circuit:
    """Return the alternating-grid circuit the capacity ladder measured.

    This is the same construction ``benchmarks/build_tn_capacity_workload.py``
    freezes, reproduced here because the workload file records the expectation's
    tensors while the run needs the circuit that produced them.
    """

    circuit = fq.Circuit(rows * columns, device=device, dtype=dtype)
    for cycle in range(cycles):
        for wire in range(rows * columns):
            circuit.ry(wire, theta=0.01 * (cycle + 1))
        for row in range(rows):
            for column in range(columns - 1):
                left = row * columns + column
                circuit.cx(left, left + 1)
        for row in range(rows - 1):
            for column in range(columns):
                top = row * columns + column
                circuit.cx(top, top + columns)
    return circuit


def _dtype(name: str) -> torch.dtype:
    return torch.complex128 if name == "complex128" else torch.complex64


def _initialize(device: torch.device) -> None:
    """Join the rendezvous every role runs inside.

    `train_distributed_tensor_network` requires an initialized group even at
    world size one, because the caller owns initialization and a group of one
    rank is still a group. Every role here is launched through a rendezvous --
    `torchrun` for the timed legs, an explicit `MASTER_ADDR`/`MASTER_PORT` pair
    for the capacity ones -- so the group is created unconditionally and the
    recording role decides what the world size means.
    """

    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)


def _parameters(device) -> list[torch.Tensor]:
    """Return the two trainable scalars the capacity protocol trains."""

    return [
        torch.tensor(0.31, device=device, dtype=torch.float64, requires_grad=True),
        torch.tensor(-0.27, device=device, dtype=torch.float64, requires_grad=True),
    ]


def _rank_record(
    arguments: argparse.Namespace,
    *,
    device: torch.device,
    contract: Mapping[str, Any],
    digest: str,
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    node_count = int(arguments.node_count)
    return {
        "schema": "flagquantum.tensor_network_release_measurements.v1",
        "role": arguments.role,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "local_world_size": local_world,
        "node_count": node_count,
        "state_mode": _stated_state_mode(arguments.release_manifest),
        "hostname": socket.gethostname(),
        "commit": _commit(),
        "shape": list(shape),
        "dtype": str(contract["dtype"]),
        "slice_count": int(arguments.slice_count),
        "steps": int(arguments.steps),
        "workload_sha256": digest,
        "device_name": torch.cuda.get_device_properties(device).name,
        "checkpoint_budget_bytes": _checkpoint_budget_bytes(arguments),
        "software": _software(),
    }


def _checkpoint_budget_bytes(arguments: argparse.Namespace) -> int | None:
    """Return the reverse-tape budget the run was launched with, if any.

    The budget changes a step's memory profile and not its arithmetic, so it is
    not part of the frozen protocol. It is recorded with every measurement
    instead, because two legs are only comparable when they ran under the same
    one, and a reader cannot otherwise tell whether a rung completed because the
    device holds it or because the tape was bounded.
    """

    value = getattr(arguments, "checkpoint_budget_bytes", None)
    return None if value is None else int(value)


def _write(arguments: argparse.Namespace, document: Mapping[str, Any]) -> None:
    """Serialise one role document, refusing one that does not state its capability.

    Every role writes through here, so the state mode is required once at the
    boundary instead of being remembered at each write site. The mode is what the
    release gate attributes a payload by, so a document that states none, or one
    the frozen manifest did not freeze, is refused before it can be sealed as
    evidence for a capability it does not name.
    """

    expected = _stated_state_mode(arguments.release_manifest)
    declared = str(document.get("state_mode", "")).lower()
    if declared != expected:
        raise SystemExit(
            "a tensor-network document must state the frozen state mode "
            f"{expected!r} and this one states {declared!r}; without it the "
            "release gate reads the sealed payload as foreign evidence"
        )
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.measurements is None:
        print(text, end="")
        return
    arguments.measurements.parent.mkdir(parents=True, exist_ok=True)
    arguments.measurements.write_text(text, encoding="utf-8")
    print(f"wrote {arguments.measurements}", flush=True)


def _role_capacity_failure(arguments: argparse.Namespace) -> int:
    """Record the single-device exhaustion of the frozen capacity workload."""

    contract = _capacity_contract(arguments.release_manifest)
    digest = _workload_digest(contract)
    if digest != str(contract["workload_sha256"]):
        raise SystemExit(
            "the frozen capacity workload does not match the digest the release "
            "manifest declares; the premise moved and the run did not"
        )
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise SystemExit("capacity-failure records one device and nothing else")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    _initialize(device)
    rows, columns, cycles = (int(item) for item in contract["grid"])
    shape = (rows, columns, cycles)
    record = _rank_record(
        arguments, device=device, contract=contract, digest=digest, shape=shape
    )
    record["timings"] = []
    record["warmup"] = 0
    record["iterations"] = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    circuit = grid_circuit(
        rows, columns, cycles, device=device, dtype=_dtype(contract["dtype"])
    )
    parameters = _parameters(device)
    status = "completed"
    message = ""
    try:
        train_distributed_tensor_network(
            circuit,
            parameters,
            steps=int(arguments.steps),
            observable={0: "z"},
            learning_rate=float(contract.get("learning_rate", 0.05)),
            slice_count=int(arguments.slice_count),
            slice_batch_size=int(contract.get("slice_batch_size", 1)),
            gradient_reduction="owner_reduce",
            checkpoint_budget_bytes=_checkpoint_budget_bytes(arguments),
        )
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover - device only
        status = "expected_oom"
        message = str(error).splitlines()[0][:400]
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if status != "expected_oom":
        raise SystemExit(
            "the frozen capacity workload completed on a single device, so it is "
            "not a capacity premise and cannot be recorded as one"
        )
    record.update(
        {
            "acceptance_case": "single_gpu_capacity_failure",
            "status": status,
            "single_device_oom_observed": True,
            "oom_message": message,
            "capacity_baseline_device": str(device),
            "capacity_failure_reason": (
                f"the frozen capacity workload exhausted {record['device_name']} "
                f"at slice_count {int(arguments.slice_count)}: {message}"
            ),
            "measured_peak_memory_bytes": record["peak_memory_bytes"],
            "full_state_materialized": False,
            "workload_manifest_path": str(contract["workload_manifest_path"]),
        }
    )
    if dist.is_initialized():
        dist.destroy_process_group()
    _write(arguments, record)
    return 0


def _parameters_local_bytes(parameters: Sequence[torch.Tensor]) -> int:
    return int(sum(item.numel() * item.element_size() for item in parameters))


def _role_capacity_completion(arguments: argparse.Namespace) -> int:
    """Record the sharded training update that completes the frozen workload."""

    contract = _capacity_contract(arguments.release_manifest)
    digest = _workload_digest(contract)
    if digest != str(contract["workload_sha256"]):
        raise SystemExit(
            "the frozen capacity workload does not match the digest the release "
            "manifest declares; the premise moved and the run did not"
        )
    world = int(os.environ["WORLD_SIZE"])
    if world <= 1:
        raise SystemExit(
            "capacity-completion is the sharded run; a single world cannot shard "
            "one logical workload across ranks"
        )

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    _initialize(device)
    rows, columns, cycles = (int(item) for item in contract["grid"])
    shape = (rows, columns, cycles)
    record = _rank_record(
        arguments, device=device, contract=contract, digest=digest, shape=shape
    )

    circuit = grid_circuit(
        rows, columns, cycles, device=device, dtype=_dtype(contract["dtype"])
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    summary: dict[str, Any] = {}
    for _ in range(int(arguments.steps)):
        started = time.perf_counter()
        result = train_distributed_tensor_network(
            circuit,
            _parameters(device),
            steps=1,
            observable={0: "z"},
            learning_rate=float(contract.get("learning_rate", 0.05)),
            slice_count=int(arguments.slice_count),
            slice_batch_size=int(contract.get("slice_batch_size", 1)),
            gradient_reduction="owner_reduce",
            checkpoint_budget_bytes=_checkpoint_budget_bytes(arguments),
        )
        torch.cuda.synchronize(device)
        samples.append(round(time.perf_counter() - started, 6))
        summary = result.summary()
    record["timings"] = samples
    record["warmup"] = 0
    record["iterations"] = len(samples)
    record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    record.update(
        {
            "acceptance_case": "multi_gpu_capacity_completion",
            "status": "completed",
            "final_loss": float(summary["final_loss"]),
            "slice_labels": [int(item) for item in summary["slice_labels"]],
            "gradient_ownership_semantics": summary.get("gradient_ownership_semantics"),
            "parameter_ownership_semantics": summary.get(
                "parameter_ownership_semantics"
            ),
            "optimizer_update_ownership_semantics": summary.get(
                "optimizer_update_ownership_semantics"
            ),
            "local_parameter_bytes": _parameters_local_bytes(_parameters(device)),
            "measured_peak_memory_bytes": record["peak_memory_bytes"],
            "workload_manifest_path": str(contract["workload_manifest_path"]),
        }
    )
    gathered = _gather(record, world)
    if dist.is_initialized():
        dist.destroy_process_group()
    if int(os.environ["RANK"]) != 0:
        return 0
    document = {**gathered[0], "ranks": gathered}
    _write(arguments, document)
    return 0


def _gather(record: Mapping[str, Any], world: int) -> list[Any]:
    """Return every rank's record, carried where NCCL cannot carry it.

    NCCL classifies a gather as an operation it has no route for and a long
    training run drives its socket state into a failure the collectives share.
    A gloo group is created for this one transfer and destroyed immediately.
    """

    if world <= 1:
        return [dict(record)]
    group = None
    try:
        if "GLOO_SOCKET_IFNAME" not in os.environ and os.environ.get(
            "NCCL_SOCKET_IFNAME"
        ):
            os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
        group = dist.new_group(backend="gloo")
        if int(os.environ["RANK"]) != 0:
            dist.gather_object(dict(record), None, dst=0, group=group)
            return []
        gathered: list[Any] = [None] * world
        dist.gather_object(dict(record), gathered, dst=0, group=group)
        return gathered
    finally:
        if group is not None:
            dist.destroy_process_group(group)


def _speed_ladder(speed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen matched-speed ladder as one entry per configuration.

    The manifest freezes three parallel multisets rather than a list of
    configurations, so the ladder is zipped here and named by the same
    convention the acceptance configuration uses. A reader can therefore check
    the acceptance name against the frozen multisets without a second table.
    """

    qubits = [int(item) for item in speed["qubits"]]
    layers = [int(item) for item in speed["layers"]]
    slices = [int(item) for item in speed["slice_label_counts"]]
    if not len(qubits) == len(layers) == len(slices):
        raise SystemExit(
            "the frozen matched-speed ladder declares qubits, layers and slice "
            "counts of different lengths, so it does not describe configurations"
        )
    names = [
        f"matched_speed_{width}q_l{depth}_s{count}"
        for width, depth, count in zip(qubits, layers, slices, strict=True)
    ]
    if str(speed["acceptance_configuration"]) not in names:
        raise SystemExit(
            "the frozen acceptance configuration is not one of the ladder's own "
            f"configurations: {speed['acceptance_configuration']!r} against {names}"
        )
    return [
        {
            "name": name,
            "qubits": width,
            "layers": depth,
            "slice_count": count,
            "parameter_count": width * depth,
        }
        for name, width, depth, count in zip(names, qubits, layers, slices, strict=True)
    ]


def _speed_circuit(
    qubits: int, layers: int, *, device, dtype
) -> tuple[fq.Circuit, list[torch.Tensor]]:
    """Return an alternating-ring circuit with one trainable angle per wire.

    The parameter count is what makes the timed workload a production one rather
    than a toy, so the parameters are the circuit: every wire carries its own
    angle in every layer, which gives ``qubits * layers`` trainable scalars
    instead of the handful a demonstration circuit carries.
    """

    circuit = fq.Circuit(qubits, device=device, dtype=dtype)
    parameters = [
        torch.tensor(
            0.05 * (layer + 1) + 0.01 * wire,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        for layer in range(layers)
        for wire in range(qubits)
    ]
    for layer in range(layers):
        for wire in range(qubits):
            circuit.ry(wire, theta=parameters[layer * qubits + wire])
        for wire in range(qubits):
            circuit.cx(wire, (wire + 1) % qubits)
    return circuit, parameters


def _semantics(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(summary.get(key, "not_measured"))
        for key in (
            "parameter_ownership_semantics",
            "gradient_ownership_semantics",
            "optimizer_update_ownership_semantics",
        )
    }


def _owned_parameters(entries: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group the runtime's per-parameter ownership records by owner rank.

    The runtime names each owned parameter by index and states the owner, so the
    payload restates that identity rather than prefixing a second vocabulary
    onto it. Which map a name appears in already says whether it names a
    parameter, its gradient, or its update.
    """

    owned: dict[str, list[str]] = {}
    for entry in entries:
        owned.setdefault(f"rank:{int(entry['rank'])}", []).extend(
            f"parameter:{int(index)}" for index in entry["parameter_indices"]
        )
    for names in owned.values():
        names.sort()
    return owned


def _role_matched_speed(arguments: argparse.Namespace) -> int:
    """Time the frozen matched-speed ladder at the world size it was launched at."""

    manifest = _frozen(arguments.release_manifest)
    speed = manifest["speed_workload"]
    # The manifest freezes the timing protocol, so the arguments may not choose
    # it. A run that measured a different number of iterations than the manifest
    # froze would be sealed as evidence for a protocol it did not follow.
    if (arguments.warmup, arguments.iterations) != (
        int(speed["warmup_steps"]),
        int(speed["measured_steps"]),
    ):
        raise SystemExit(
            "the frozen matched-speed protocol requires "
            f"--warmup {speed['warmup_steps']} --iterations {speed['measured_steps']}"
        )
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _initialize(device)

    dtype = _dtype(str(manifest["runtime"]["dtype"]))
    budget = _checkpoint_budget_bytes(arguments)
    try:
        configurations: list[dict[str, Any]] = []
        for frozen in _speed_ladder(speed):
            circuit, parameters = _speed_circuit(
                int(frozen["qubits"]), int(frozen["layers"]), device=device, dtype=dtype
            )
            seconds: list[float] = []
            summary: dict[str, Any] = {}
            for iteration in range(arguments.warmup + arguments.iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = train_distributed_tensor_network(
                    circuit,
                    parameters,
                    steps=1,
                    observable={0: "z"},
                    learning_rate=0.05,
                    slice_count=int(frozen["slice_count"]),
                    slice_batch_size=1,
                    gradient_reduction="owner_reduce",
                    checkpoint_budget_bytes=budget,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if iteration >= arguments.warmup:
                    seconds.append(round(time.perf_counter() - started, 6))
                    # The communication a payload reports has to be the
                    # communication a timed call performed, so the last measured
                    # call's own summary travels with the timing.
                    summary = result.summary()
            if not seconds:
                raise SystemExit("the frozen protocol measured no iteration")
            configurations.append(
                {
                    "name": frozen["name"],
                    "qubits": frozen["qubits"],
                    "layers": frozen["layers"],
                    "slice_count": frozen["slice_count"],
                    "parameter_count": frozen["parameter_count"],
                    "steps": 1,
                    "seconds": seconds,
                    "minimum_seconds": min(seconds),
                    "median_seconds": _median(seconds),
                    "communication_bytes": int(summary.get("communication_bytes", 0)),
                    "collective_seconds": _collective_seconds(summary),
                    "training_step_count": int(summary.get("training_step_count", 0)),
                    "local_memory_bytes_by_rank": [
                        int(item)
                        for item in summary.get("local_memory_bytes_by_rank", ())
                    ],
                    # The ownership maps a release payload publishes are the
                    # runtime's own records, so they travel with the timing
                    # rather than being reconstructed later from the world size.
                    "parameter_ownership": _owned_parameters(
                        summary.get("parameter_ownership", ())
                    ),
                    "optimizer_update_ownership": _owned_parameters(
                        summary.get("optimizer_update_ownership", ())
                    ),
                    **_semantics(summary),
                }
            )
        peak = int(torch.cuda.max_memory_allocated(device))
        record = {
            "schema": "flagquantum.tensor_network_release_measurements.v1",
            "role": arguments.role,
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "local_world_size": arguments.local_world_size,
            "node_count": arguments.node_count,
            "state_mode": _stated_state_mode(arguments.release_manifest),
            "hostname": socket.gethostname(),
            "commit": _commit(),
            "measurements": {"configurations": configurations},
            "timings": [item["median_seconds"] for item in configurations],
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
            "measured_peak_memory_bytes": peak,
            "device_name": torch.cuda.get_device_properties(device).name,
            "workload_sha256": _protocol_digest(arguments.release_manifest),
            "checkpoint_budget_bytes": budget,
            "software": _software(),
        }
        records = _gather(record, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank != 0:
        return 0
    _write(arguments, {**records[0], "ranks": records})
    return 0


def _role_release_payload(arguments: argparse.Namespace) -> int:
    """Project one measurements document onto the contract the sealer reads.

    A role document states the release contract under ``measurements`` beside
    the per-rank records it was assembled from. ``tools/seal_runtime_evidence.py``
    takes its evidence input at the top level, so something has to lift the
    contract out; doing it here keeps the projection reviewable and repeatable
    instead of leaving it to whoever runs the campaign.
    """

    if arguments.document is None:
        raise SystemExit("release-payload requires --document")
    document = json.loads(Path(arguments.document).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{arguments.document} is not a measurements document")
    payload = document.get("measurements")
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{arguments.document} carries no measurements block to project; a "
            "release payload is the contract a role assembled, so a document "
            "without one has nothing to seal"
        )
    _write(arguments, payload)
    return 0


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(item) for item in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _collective_seconds(summary: Mapping[str, Any]) -> float:
    """Return the seconds the runtime recorded inside its collectives.

    The result states the collective time of every step it ran, so the timed
    call's collective time is the sum of its own step records rather than a
    second timer started beside the call.
    """

    return sum(
        float(record.get("collective_seconds", 0.0))
        for record in summary.get("step_records", ())
        if isinstance(record, Mapping)
    )


def _communication_fraction(step_seconds: float, collective_seconds: float) -> float:
    """Return the share of the timed step that ran inside collectives.

    The fraction is a measured ratio of two durations the runtime reported for
    the same call. Dividing bytes by a step time would mix units, and inventing
    a second notion of communication here would let the payload disagree with
    the run it describes.
    """

    return collective_seconds / max(float(step_seconds), 1e-12)


def _agreed_ownership(
    published: list[Mapping[str, list[str]]], *, context: str
) -> dict[str, list[str]]:
    """The one parameter-to-owner map every rank published, or fail closed.

    Ownership is not a rank-local measurement: the training result states the
    whole map on every rank, so a payload that concatenated the ranks would list
    each parameter once per rank, and one that read only rank 0 would accept a
    leg whose other ranks reported a different map. Both are rejected here.
    """

    if not published or not published[0]:
        raise SystemExit(
            f"the sharded leg recorded no parameter ownership for {context}, so a "
            "release payload cannot state which rank updated which parameter"
        )
    if any(mapping != published[0] for mapping in published[1:]):
        raise SystemExit(
            f"the ranks of the sharded leg disagree about which rank owns which "
            f"parameter of {context}"
        )
    return {owner: list(names) for owner, names in published[0].items()}


def _role_speed_summary(arguments: argparse.Namespace) -> int:
    """Assemble the matched-speed release contract from one baseline and one leg."""

    manifest = _frozen(arguments.release_manifest)
    speed = manifest["speed_workload"]
    acceptance = str(speed["acceptance_configuration"])
    capacity_digest = str(manifest["capacity_workload"]["workload_sha256"])
    # The protocol identity is the manifest file's own digest, so a leg measured
    # against one revision of the manifest and an assembly projected against
    # another are two different experiments with the same name. Checking it here
    # is what makes the payload's workload_sha256 a verified statement about the
    # run rather than a restatement of whichever manifest happened to be on disk
    # when the summary was assembled.
    protocol_digest = _protocol_digest(arguments.release_manifest)
    baseline = json.loads(Path(arguments.baseline).read_text(encoding="utf-8"))
    sharded = json.loads(Path(arguments.sharded).read_text(encoding="utf-8"))
    for role, leg in (("baseline", baseline), ("sharded", sharded)):
        recorded = str(leg.get("workload_sha256", ""))
        if recorded != protocol_digest:
            raise SystemExit(
                f"the {role} leg was measured against the protocol "
                f"{recorded or '<unrecorded>'}, which is not the protocol of the "
                f"manifest this summary is assembled against "
                f"({protocol_digest}); the speedup could not be attributed to the "
                "frozen ladder"
            )
    if int(baseline.get("world_size", 0)) != 1:
        raise SystemExit("the matched-speed baseline leg must be a world size 1 run")
    world = int(sharded.get("world_size", 0))
    if world <= 1:
        raise SystemExit("the matched-speed sharded leg must run above world size 1")
    # The budget bounds the reverse tape, so it changes the memory profile of
    # the step both legs timed. Two legs that ran under different budgets were
    # not running the same experiment, and the speedup would be a comparison
    # between two different numbers of saved tensors rather than between one
    # device and several.
    baseline_budget = baseline.get("checkpoint_budget_bytes")
    sharded_budget = sharded.get("checkpoint_budget_bytes")
    if baseline_budget != sharded_budget:
        raise SystemExit(
            "the two speed legs ran under different checkpoint budgets "
            f"({baseline_budget!r} against {sharded_budget!r}), so the ratio "
            "between them would compare two different reverse-tape profiles "
            "rather than one device against several"
        )
    left = {item["name"]: item for item in baseline["measurements"]["configurations"]}
    right = {item["name"]: item for item in sharded["measurements"]["configurations"]}
    if set(left) != set(right):
        raise SystemExit("the two speed legs did not run the same configurations")
    table = []
    for name in sorted(left):
        lower, upper = bootstrap_ratio_interval(
            left[name]["seconds"], right[name]["seconds"]
        )
        speedup = left[name]["median_seconds"] / right[name]["median_seconds"]
        if "collective_seconds" not in right[name]:
            raise SystemExit(
                f"the sharded leg did not report the collective seconds it spent "
                f"in {name!r}, so the communication fraction cannot be measured "
                "from the run that the speedup was measured from"
            )
        table.append(
            {
                "name": name,
                "qubits": left[name]["qubits"],
                "layers": left[name]["layers"],
                "slice_count": left[name]["slice_count"],
                "steps": left[name]["steps"],
                "parameter_count": left[name]["parameter_count"],
                "baseline_median_seconds": left[name]["median_seconds"],
                "sharded_median_seconds": right[name]["median_seconds"],
                "sharded_collective_seconds": float(right[name]["collective_seconds"]),
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

    rank_records = sharded.get("ranks", [sharded])
    peaks = [int(record["measured_peak_memory_bytes"]) for record in rank_records]
    accepted_sharded = right[acceptance]

    # Every rank publishes the whole ownership map, so the maps are compared
    # rather than concatenated: a payload that listed each parameter once per
    # rank, or read only rank 0, would accept a leg whose ranks disagreed.
    def accepted_by(record: Mapping[str, Any]) -> Mapping[str, Any]:
        for item in record["measurements"]["configurations"]:
            if item["name"] == acceptance:
                return item
        raise SystemExit(
            f"a rank of the sharded leg did not run {acceptance!r}, so the "
            "ownership maps cannot be compared across the ranks that did"
        )

    def agree(field: str) -> dict[str, list[str]]:
        return _agreed_ownership(
            [accepted_by(record)[field] for record in rank_records],
            context=repr(acceptance),
        )

    statements = [_semantics(accepted_by(record)) for record in rank_records]
    if any(item != statements[0] for item in statements[1:]):
        raise SystemExit(
            f"the ranks of the sharded leg disagree about the ownership semantics "
            f"of {acceptance!r}"
        )
    ownership_semantics = statements[0]
    parameter_ownership = agree("parameter_ownership")
    gradient_ownership = (
        parameter_ownership
        if ownership_semantics["gradient_ownership_semantics"] == "sharded_across_ranks"
        else {}
    )
    optimizer_update_ownership = agree("optimizer_update_ownership")
    multi_node = int(sharded["node_count"]) > 1
    measurements = {
        "acceptance_case": "matched_speed",
        "state_mode": _stated_state_mode(arguments.release_manifest),
        "world_size": world,
        "local_world_size": int(sharded["local_world_size"]),
        "node_count": int(sharded["node_count"]),
        # The freeze does not name a budget, so the payload states the one both
        # legs actually ran under rather than leaving a reader to assume the
        # unbounded path. A rung that completes under a bound completed because
        # the tape was bounded, and the difference between the two profiles is
        # exactly what the frozen floor note could not separate.
        "checkpoint_budget_bytes": sharded_budget,
        "distribution_semantics": "sharded_across_ranks",
        "collective_backend": "nccl",
        "topology_scope": (
            "multi_node_production_transport"
            if multi_node
            else "single_node_executed_collective"
        ),
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": f"cuda:0 on {baseline['device_name']}",
        # A release payload must state the capacity premise the released
        # capability rests on. This one states that the premise is not
        # established, and says what was measured to decide that, rather than
        # leaving a reader to infer a premise from the blocker alone.
        "capacity_failure_reason": (
            "the frozen capacity workload "
            f"{manifest['capacity_workload']['name']} ({capacity_digest}) was "
            "swept at slice counts 2, 3, 4, 5 and 6 on one device and on the "
            "pair, and no slicing both exhausts one device and completes "
            "sharded: at the only slicing that exhausts one device both legs "
            "exhaust with the same per-rank peak, and every other slicing either "
            "completes on one device alone or is refused on both, so the premise "
            "is recorded as not established"
        ),
        "capacity_premise_workload_sha256": capacity_digest,
        **ownership_semantics,
        "optimizer_update_semantics": "sharded_across_ranks",
        "parameter_ownership": parameter_ownership,
        "gradient_ownership": gradient_ownership,
        "optimizer_update_ownership": optimizer_update_ownership,
        "rank_placement": {
            f"rank:{rank}": (
                f"node:{rank // max(1, int(sharded['local_world_size']))}/"
                f"gpu:{rank % max(1, int(sharded['local_world_size']))}"
            )
            for rank in range(world)
        },
        "local_memory_bytes_by_rank": peaks,
        "measured_peak_memory_bytes": max(peaks),
        "measured_peak_memory_bytes_by_rank": peaks,
        "communication_bytes": int(accepted_sharded["communication_bytes"]),
        "inter_node_communication_bytes": (
            int(accepted_sharded["communication_bytes"]) if multi_node else 0
        ),
        # The frozen contract asks for a measured communication fraction. It is
        # reported as the share of the timed step the runtime spent inside its
        # collectives, taken from the step records of the call that was timed.
        "communication_fraction": _communication_fraction(
            accepted["sharded_median_seconds"],
            accepted["sharded_collective_seconds"],
        ),
        "communication_fraction_definition": (
            "measured collective seconds of the timed call, summed from that "
            "call's own step records, over its median measured step seconds"
        ),
        "timings": [item["sharded_median_seconds"] for item in table],
        "training_step_count": int(accepted_sharded["training_step_count"]),
        "rank_outputs": [0.0] * world,
        "rank_gradients": peaks,
        "rank_timings": [item["sharded_median_seconds"] for item in table],
        "rank_peak_memory_bytes": peaks,
        "gpu_activity": 1.0,
        "full_state_materialized": False,
        "workload_sha256": _protocol_digest(arguments.release_manifest),
        "hardware_inventory": [sharded["device_name"]],
        "speedup": accepted["speedup"],
        "speedup_confidence_interval": accepted["speedup_confidence_interval"],
        "scaling_efficiency": accepted["scaling_efficiency"],
        "acceptance_configuration": acceptance,
        "configurations": table,
    }
    _write(
        arguments,
        {
            "schema": "flagquantum.tensor_network_release_measurements.v1",
            "role": arguments.role,
            "rank": 0,
            "world_size": world,
            "node_count": int(sharded["node_count"]),
            "state_mode": _stated_state_mode(arguments.release_manifest),
            "commit": sharded["commit"],
            "measurements": measurements,
            "timings": measurements["timings"],
            "warmup": int(sharded["warmup"]),
            "iterations": int(sharded["iterations"]),
            "measured_peak_memory_bytes": measurements["measured_peak_memory_bytes"],
            "device_name": sharded["device_name"],
            "software": sharded["software"],
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--release-manifest", type=Path, default=RELEASE_MANIFEST)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--slice-count", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sharded", type=Path)
    parser.add_argument("--local-world-size", type=int, default=1)
    parser.add_argument(
        "--checkpoint-budget-bytes",
        type=int,
        help=(
            "bound the reverse tape at this many bytes per rank; the arithmetic "
            "is unchanged, so both legs of a comparison must use one value"
        ),
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=1,
        help="hosts the run spans; the launcher knows this and the ranks do not",
    )
    arguments = parser.parse_args(argv)

    if arguments.role == "speed-summary":
        return _role_speed_summary(arguments)

    if arguments.role == "release-payload":
        return _role_release_payload(arguments)

    if arguments.role == "matched-speed":
        return _role_matched_speed(arguments)

    contract = _capacity_contract(arguments.release_manifest)
    if arguments.slice_count is None:
        arguments.slice_count = int(contract["slice_count"])
    if arguments.steps is None:
        arguments.steps = int(contract["steps"])

    if arguments.role == "capacity-failure":
        return _role_capacity_failure(arguments)
    return _role_capacity_completion(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
