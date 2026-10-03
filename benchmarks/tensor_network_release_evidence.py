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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.experimental.distributed import train_distributed_tensor_network

RELEASE_MANIFEST = Path("benchmarks/manifests/tensor_network_release_v1.json")
ROLES = ("capacity-failure", "capacity-completion")


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
        "hostname": socket.gethostname(),
        "commit": _commit(),
        "shape": list(shape),
        "dtype": str(contract["dtype"]),
        "slice_count": int(arguments.slice_count),
        "steps": int(arguments.steps),
        "workload_sha256": digest,
        "device_name": torch.cuda.get_device_properties(device).name,
        "software": _software(),
    }


def _write(arguments: argparse.Namespace, document: Mapping[str, Any]) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--release-manifest", type=Path, default=RELEASE_MANIFEST)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--slice-count", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--node-count",
        type=int,
        default=1,
        help="hosts the run spans; the launcher knows this and the ranks do not",
    )
    arguments = parser.parse_args(argv)

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
