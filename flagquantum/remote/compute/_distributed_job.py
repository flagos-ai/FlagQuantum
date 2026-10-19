"""Single-node torchrun entrypoint for native Jiuding multi-device jobs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from ...core.ir import CircuitIR
from ...runtime.distributed import destroy_torch_distributed, init_torch_distributed
from ...runtime.executors.statevector import (
    execute_torch_distributed_statevector,
    gather_distributed_statevector,
)
from ...runtime.measurements import execute_measurements
from ._workspace_executor import (
    SCHEMA,
    VERSION,
    _encode_measurement_value,
    _encode_tensor,
)

_MAX_RESULT = 8 * 1024 * 1024


def _emit_result(value: dict[str, Any], run_id: str) -> None:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > _MAX_RESULT:
        raise RuntimeError("Result exceeds the 8 MiB job log transport limit")
    encoded = base64.b64encode(payload).decode()
    chunks = [encoded[index : index + 3072] for index in range(0, len(encoded), 3072)]
    digest = hashlib.sha256(payload).hexdigest()
    for index, chunk in enumerate(chunks):
        print(
            f"FQ_RESULT_{run_id} {index}/{len(chunks)} {digest} {chunk}",
            flush=True,
        )


def _response(request: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    program = CircuitIR.from_dict(request["program"])
    measurements = tuple(program.measurements)
    execution_program = replace(program, measurements=())
    precision = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }.get(program.dtype)
    if precision is None:
        raise ValueError(f"unsupported distributed precision {program.dtype!r}")

    started = time.perf_counter()
    sharded = execute_torch_distributed_statevector(
        execution_program,
        device=device,
        dtype=precision,
        local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    gathered = gather_distributed_statevector(sharded)
    elapsed_seconds = time.perf_counter() - started
    response: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "ok": True,
        "request_id": request.get("request_id"),
        "evidence": {
            "target": request["target"],
            "device": str(device),
            "dtype": str(precision).removeprefix("torch."),
            "cpu_fallback_used": False,
            "elapsed_seconds": elapsed_seconds,
            "devices": dist.get_world_size(),
            "distributed_execution": sharded.summary(),
            "result_materialization": gathered.summary(),
        },
    }
    if request["operation"] == "statevector":
        response["state"] = _encode_tensor(gathered.state)
        return response
    if request["operation"] != "measurements" or not measurements:
        raise ValueError("distributed job requires statevector or measurements")
    results = execute_measurements(
        gathered.state,
        measurements,
        n_qubits=program.n_wires,
    )
    response["measurements"] = [
        {
            "kind": item.kind,
            "wires": list(item.wires),
            "value": _encode_measurement_value(item.value),
            "shots": item.shots,
            "metadata": dict(item.metadata),
            "statistics": dict(item.statistics),
        }
        for item in results
    ]
    return response


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("expected an encoded request and run id")
    request = json.loads(base64.b64decode(sys.argv[1], validate=True))
    if not isinstance(request, dict):
        raise TypeError("distributed job request must be a JSON object")
    if request.get("schema") != SCHEMA or request.get("version") != VERSION:
        raise ValueError("unsupported distributed job protocol")
    if request.get("operation") not in {"statevector", "measurements"}:
        raise ValueError("unsupported distributed job operation")
    target = request.get("target")
    if not isinstance(target, str) or not target.startswith("jiuding:gpu/"):
        raise ValueError("distributed Jiuding jobs require a concrete GPU target")
    expected = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if (
        expected < 2
        or int(os.environ.get("WORLD_SIZE", "0")) != expected
        or request.get("devices") != expected
    ):
        raise RuntimeError("distributed Jiuding jobs require one multi-device node")
    context = init_torch_distributed(device="cuda")
    try:
        if context.world_size != expected or context.local_world_size != expected:
            raise RuntimeError("torchrun world size does not match requested devices")
        value = _response(request, device=context.device)
        if context.rank == 0:
            _emit_result(value, sys.argv[2])
    finally:
        destroy_torch_distributed()


if __name__ == "__main__":
    main()
