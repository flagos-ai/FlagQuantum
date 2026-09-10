"""Shared program bundles for Jiuding batch execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROGRAM_JOB_SCHEMA = "flagquantum.jiuding.program_job"
PROGRAM_JOB_VERSION = "1.0"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def write_program_bundle(
    program: Any,
    *,
    target: str,
    operation: str,
    receipt: Path,
) -> tuple[Path, Path]:
    """Persist one bounded request and a minimal shared entrypoint."""

    if operation not in {"statevector", "measurements"}:
        raise ValueError("unsupported Jiuding program job operation")
    request_path = receipt.with_name(receipt.stem + ".program.json")
    runner_path = receipt.with_name(receipt.stem + ".runner.py")
    payload = {
        "schema": PROGRAM_JOB_SCHEMA,
        "version": PROGRAM_JOB_VERSION,
        "target": target,
        "operation": operation,
        "program": program.to_dict(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise ValueError("Jiuding program request exceeds the 8 MiB limit")
    with request_path.open("xb") as file:
        file.write(encoded + b"\n")
    runner = (
        "from flagquantum.remote.compute._program_job import execute_request\n\n"
        "def main():\n"
        f"    return execute_request({str(request_path)!r})\n"
    )
    try:
        with runner_path.open("x", encoding="utf-8") as file:
            file.write(runner)
    except BaseException:
        request_path.unlink(missing_ok=True)
        raise
    return request_path, runner_path


def execute_request(path: str | Path) -> dict[str, Any]:
    """Execute one persisted request inside its scheduled container."""

    request_path = Path(path)
    if request_path.stat().st_size > _MAX_REQUEST_BYTES:
        raise ValueError("Jiuding program request exceeds the 8 MiB limit")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or set(request) != {
        "schema",
        "version",
        "target",
        "operation",
        "program",
    }:
        raise ValueError("invalid Jiuding program request")
    if (
        request["schema"] != PROGRAM_JOB_SCHEMA
        or request["version"] != PROGRAM_JOB_VERSION
    ):
        raise ValueError("unsupported Jiuding program request version")
    target = request["target"]
    if not isinstance(target, str) or not target:
        raise ValueError("Jiuding program request target is invalid")
    provider, separator, resource = target.partition(":")
    chip_type, model_separator, model = resource.partition("/")
    if (
        separator != ":"
        or provider != "jiuding"
        or chip_type not in {"cpu", "gpu"}
        or (chip_type == "cpu" and model_separator)
        or (chip_type == "gpu" and (not model_separator or not model))
    ):
        raise ValueError("Jiuding program request target is invalid")
    device = "cuda:0" if chip_type == "gpu" else "cpu"
    from ._workspace_executor import SCHEMA, VERSION, execute

    return execute(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "operation": request["operation"],
            "request_id": "batch-program",
            "target": target,
            "program": request["program"],
        },
        target=target,
        device=device,
    )


def decode_program_result(
    value: Any,
    *,
    requested_target: str,
    selected_target: str,
) -> Any:
    """Restore a normal ExecutionResult from the batch worker response."""

    from ._workspace_executor import SCHEMA, VERSION

    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("version") != VERSION
        or value.get("request_id") != "batch-program"
        or value.get("ok") is not True
    ):
        raise RuntimeError("Jiuding program job returned an invalid execution result")
    if "measurements" in value:
        from ._workspace_results import measurement_result

        return measurement_result(
            value,
            requested_target=requested_target,
            effective_target=selected_target,
        )
    if "state" not in value:
        raise RuntimeError("Jiuding program job returned no state or measurements")
    from ...runtime.result import ExecutionResult
    from ._workspace_results import decode_tensor

    evidence = dict(value.get("evidence", {}))
    return ExecutionResult(
        state=decode_tensor(value["state"]),
        runtime={
            "execution_path": "jiuding_batch_program",
            "device": evidence.get("device"),
            "elapsed_seconds": evidence.get("elapsed_seconds"),
        },
        provenance={
            "requested_target": requested_target,
            "selected_target": selected_target,
            "accelerator": evidence.get("accelerator"),
            "cpu_fallback_used": evidence.get("cpu_fallback_used"),
        },
    )


__all__ = (
    "PROGRAM_JOB_SCHEMA",
    "PROGRAM_JOB_VERSION",
    "decode_program_result",
    "execute_request",
    "write_program_bundle",
)
