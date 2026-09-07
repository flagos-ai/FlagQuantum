"""Shared stateless parsing helpers for quantum-cloud providers."""

from __future__ import annotations

from typing import Any, Mapping, cast


def _normalize_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): int(round(float(value))) for key, value in counts.items()}


def _flip_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key)[::-1]: int(round(float(value))) for key, value in counts.items()}


def _unwrap_result_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result", payload)
    if (
        isinstance(result, Mapping)
        and "ok" in result
        and isinstance(result.get("result"), Mapping)
    ):
        return cast(Mapping[str, Any], result["result"])
    if isinstance(result, Mapping):
        return result
    return payload


def _extract_counts(
    payload: Mapping[str, Any], *, provider: str, flip: bool = False
) -> dict[str, int]:
    current: Any = _unwrap_result_envelope(payload)
    if isinstance(current, Mapping) and "counts" in current:
        current = current["counts"]
    elif isinstance(current, Mapping) and "count" in current:
        current = current["count"]
    elif isinstance(current, Mapping) and isinstance(current.get("data"), Mapping):
        current = current["data"]
        if "counts" in current:
            current = current["counts"]
        elif "count" in current:
            current = current["count"]
    if not isinstance(current, Mapping):
        raise RuntimeError(f"{provider} result response does not contain counts.")
    return _flip_counts(current) if flip else _normalize_counts(current)


def _strip_barrier(qasm: str) -> str:
    return "\n".join(
        line for line in qasm.splitlines() if not line.strip().startswith("barrier")
    )


def _format_circuit_source(source: str) -> str:
    return "\n".join(line.strip() for line in str(source).splitlines() if line.strip())
