"""Translate resident Jiuding responses into FlagQuantum results."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch

    from ...runtime.result import ExecutionResult


def decode_tensor(payload: dict[str, Any]) -> torch.Tensor:
    import torch

    dtype_name = payload.get("dtype")
    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "int64": torch.int64,
    }
    if not isinstance(dtype_name, str) or dtype_name not in dtypes:
        raise RuntimeError("Jiuding workspace returned an unsupported result dtype")
    raw = base64.b64decode(payload["data"], validate=True)
    storage_dtype = {
        "complex64": torch.float32,
        "complex128": torch.float64,
    }.get(dtype_name, dtypes[dtype_name])
    value = torch.frombuffer(bytearray(raw), dtype=storage_dtype)
    if dtype_name.startswith("complex"):
        value = value.view(dtypes[dtype_name])
    return value.reshape(tuple(int(item) for item in payload["shape"])).clone()


def _decode_value(value: object) -> torch.Tensor | list[dict[str | int, int]]:
    if isinstance(value, dict) and {"dtype", "shape", "data"} <= value.keys():
        return decode_tensor(value)
    if not isinstance(value, dict) or set(value) != {"counts"}:
        raise RuntimeError("Jiuding workspace returned an unsupported result value")
    rows = value["counts"]
    if not isinstance(rows, list):
        raise RuntimeError("Jiuding workspace returned invalid counts")
    decoded: list[dict[str | int, int]] = []
    for row in rows:
        if not isinstance(row, list):
            raise RuntimeError("Jiuding workspace returned invalid counts")
        counts: dict[str | int, int] = {}
        for entry in row:
            if not isinstance(entry, dict) or set(entry) != {"outcome", "count"}:
                raise RuntimeError("Jiuding workspace returned invalid counts")
            outcome, count = entry["outcome"], entry["count"]
            if (
                not isinstance(outcome, (str, int))
                or isinstance(outcome, bool)
                or type(count) is not int
                or count < 0
            ):
                raise RuntimeError("Jiuding workspace returned invalid counts")
            if outcome in counts:
                raise RuntimeError(
                    "Jiuding workspace returned duplicate count outcomes"
                )
            counts[outcome] = count
        decoded.append(counts)
    return decoded


def measurement_result(
    response: Mapping[str, Any],
    *,
    requested_target: str,
    effective_target: str,
    batch_index: int | None = None,
    batch_size: int | None = None,
    batch_elapsed_seconds: float | None = None,
) -> ExecutionResult:
    """Build one standard result from a validated workspace response."""

    import torch

    from ...runtime.result import ExecutionResult, MeasurementResult

    measurements = tuple(
        MeasurementResult(
            kind=item["kind"],
            wires=tuple(int(wire) for wire in item["wires"]),
            value=_decode_value(item["value"]),
            shots=item.get("shots"),
            metadata=dict(item.get("metadata", {})),
            statistics=dict(item.get("statistics", {})),
        )
        for item in response["measurements"]
    )
    first_samples = next(
        (
            item.value
            for item in measurements
            if item.kind in {"sample", "sample_ps"}
            and isinstance(item.value, torch.Tensor)
        ),
        None,
    )
    evidence = dict(response.get("evidence", {}))
    runtime = {
        "execution_path": "jiuding_workspace_executor",
        "device": evidence.get("device"),
        "elapsed_seconds": evidence.get("elapsed_seconds"),
        "result_transfer": evidence.get("result_transfer"),
        "counts_aggregation": evidence.get("counts_aggregation"),
    }
    if batch_index is not None:
        runtime.update(
            {
                "batch_index": batch_index,
                "batch_size": batch_size,
                "batch_elapsed_seconds": batch_elapsed_seconds,
            }
        )
    return ExecutionResult(
        samples=first_samples,
        measurements=measurements,
        runtime=runtime,
        provenance={
            "requested_target": requested_target,
            "selected_target": effective_target,
            "accelerator": evidence.get("accelerator"),
            "cpu_fallback_used": evidence.get("cpu_fallback_used"),
            "statevector_transferred": evidence.get("statevector_transferred"),
        },
    )


__all__ = ("decode_tensor", "measurement_result")
