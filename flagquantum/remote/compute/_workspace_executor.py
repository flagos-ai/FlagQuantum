"""Small resident executor for a trusted Jiuding development workspace."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import time
from dataclasses import replace
from typing import Any

SCHEMA = "flagquantum.jiuding.workspace_executor"
VERSION = "1.1"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _read_exact(stream: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("workspace executor connection closed early")
        chunks.extend(chunk)
    return bytes(chunks)


def read_message(stream: Any) -> dict[str, Any]:
    """Read one bounded, length-prefixed JSON object."""

    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError("workspace executor message size is invalid")
    value = json.loads(_read_exact(stream, size))
    if not isinstance(value, dict):
        raise TypeError("workspace executor message must be a JSON object")
    return value


def write_message(stream: Any, value: dict[str, Any]) -> None:
    """Write one bounded, length-prefixed JSON object."""

    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("workspace executor response exceeds the size limit")
    stream.write(struct.pack("!I", len(payload)) + payload)
    stream.flush()


def _encode_tensor(value: Any) -> dict[str, Any]:
    import torch

    flat = value.detach().contiguous().reshape(-1).cpu()
    if flat.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(f"unsupported result dtype {flat.dtype}")
    storage = (
        flat.view(torch.float32)
        if flat.dtype == torch.complex64
        else flat.view(torch.float64) if flat.dtype == torch.complex128 else flat
    )
    return {
        "dtype": str(flat.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "data": base64.b64encode(storage.numpy().tobytes()).decode("ascii"),
    }


def execute(request: dict[str, Any], *, target: str, device: str) -> dict[str, Any]:
    """Execute one request while keeping framework and device context warm."""

    if request.get("schema") != SCHEMA or request.get("version") != VERSION:
        raise ValueError("unsupported workspace executor protocol")
    operation = request.get("operation")
    if operation == "health":
        from flagquantum.compute import get_platform_runtime

        platform = get_platform_runtime(device)
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "ok": True,
            "target": target,
            "device": device,
            "cuda_available": (
                platform.is_available() if device.startswith("cuda") else False
            ),
        }
    if operation not in {"statevector", "measurements"}:
        raise ValueError(f"unsupported workspace executor operation {operation!r}")
    if request.get("target") != target:
        raise ValueError("request target does not match the resident executor target")

    from flagquantum.core.ir import CircuitIR
    from flagquantum.runtime.execution import run
    from flagquantum.runtime.options import ExecutionOptions

    program = CircuitIR.from_dict(request["program"])
    measurements = program.measurements
    if operation == "measurements":
        if not measurements:
            raise ValueError("measurement execution requires at least one request")
        supported = {"expectation_identity", "expectation_ps", "probabilities"}
        if any(item.kind not in supported for item in measurements):
            raise ValueError(
                "resident execution supports probabilities and expectations only"
            )
    program = replace(program, measurements=())
    started = time.perf_counter()
    result = run(
        program,
        options=ExecutionOptions(
            mode="statevector", device=device, precision=program.dtype
        ),
    )
    state = result.to_statevector()
    if state.device.type != device.split(":", 1)[0]:
        raise RuntimeError(
            f"execution returned {state.device} instead of requested {device}"
        )
    from flagquantum.compute import get_platform_runtime

    platform = get_platform_runtime(state.device.type)
    accelerator = next(
        (item.name for item in platform.discover() if item.device == state.device), ""
    )
    response: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "ok": True,
        "request_id": request.get("request_id"),
        "evidence": {
            "target": target,
            "device": str(state.device),
            "dtype": str(state.dtype).removeprefix("torch."),
            "cpu_fallback_used": False,
            "accelerator": accelerator,
        },
    }
    if operation == "statevector":
        platform.synchronize(state.device)
        response["evidence"]["elapsed_seconds"] = time.perf_counter() - started
        response["state"] = _encode_tensor(state)
        return response

    from flagquantum.runtime.measurements import execute_measurements

    results = execute_measurements(state, measurements, n_wires=program.n_wires)
    platform.synchronize(state.device)
    response["evidence"]["elapsed_seconds"] = time.perf_counter() - started
    response["measurements"] = [
        {
            "kind": item.kind,
            "wires": list(item.wires),
            "value": _encode_tensor(item.value),
            "shots": item.shots,
            "metadata": dict(item.metadata),
            "statistics": dict(item.statistics),
        }
        for item in results
    ]
    return response


def serve(*, host: str, port: int, target: str, device: str) -> None:
    """Serve sequential requests; one failure does not terminate the worker."""

    with socket.create_server((host, port), reuse_port=False) as listener:
        while True:
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb", buffering=0) as stream:
                while True:
                    try:
                        request = read_message(stream)
                    except EOFError:
                        break
                    try:
                        response = execute(request, target=target, device=device)
                    except Exception as exc:
                        response = {
                            "schema": SCHEMA,
                            "version": VERSION,
                            "ok": False,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    try:
                        write_message(stream, response)
                    except (BrokenPipeError, ConnectionError, ValueError):
                        break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=57621)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        raise ValueError("workspace executor must listen on loopback")
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    serve(host=args.host, port=args.port, target=args.target, device=args.device)


if __name__ == "__main__":
    main()
