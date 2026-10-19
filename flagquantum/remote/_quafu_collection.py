"""Ordered Quafu collections over the existing task API transport."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypedDict
from urllib.parse import quote, urlencode

from ..deployment.cloud import DeploymentPackage, validate_deployment_package
from ..errors import CapabilityError, ExecutionError, ValidationError
from ..runtime.result import ExecutionResult
from .qpu.contracts import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_result,
)
from .qpu.execution import counts_result
from .qpu.quafu import QuafuProvider
from .qpu.result_parsing import _extract_counts

_SCHEMA = "flagquantum.quafu-collection.v1"
_PAGE_SIZE = 200


class _CollectionReceipt(TypedDict):
    schema: str
    items: list[dict[str, str]]
    shots: int
    n_qubits: int
    sha256: str


def _digest(items: list[dict[str, str]], shots: int, width: int) -> str:
    payload = json.dumps([items, shots, width], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_context(
    raw: Any, *, width: int, identity: Mapping[str, str]
) -> tuple[list[dict[str, str]], int]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "items",
        "shots",
        "n_qubits",
        "sha256",
    }:
        raise ValueError("Invalid Quafu collection receipt context")
    items, shots = raw["items"], raw["shots"]
    if (
        raw["schema"] != _SCHEMA
        or type(raw["n_qubits"]) is not int
        or raw["n_qubits"] != width
        or type(shots) is not int
        or shots <= 0
        or not isinstance(items, list)
        or not items
    ):
        raise ValueError("Invalid Quafu collection receipt dimensions")
    for item in items:
        if not isinstance(item, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in item.items()
        ):
            raise ValueError("Invalid Quafu collection submission identity")
        build_result_metadata(ProviderTaskHandle("quafu", "validation", "", item))
        if item.get("quafu_protocol") != "task_api_v1" or any(
            re.fullmatch(r"[0-9a-f]{64}", item.get(key, "")) is None
            for key in ("routing_evidence_sha256", "deployment_artifact_sha256")
        ):
            raise ValueError("Invalid Quafu collection artifact identity")
    digest = _digest(items, shots, width)
    if raw["sha256"] != digest or identity != items[0] | {"collection_sha256": digest}:
        raise ValueError("Quafu collection receipt identity mismatch")
    return items, shots


def submit_collection(
    client: QuafuProvider, packages: Sequence[DeploymentPackage]
) -> tuple[ProviderTaskHandle, _CollectionReceipt]:
    for package in packages:
        validate_deployment_package(package)
    if not client.api_key:
        raise ValidationError(
            "Quafu collections require QUAFU_API_KEY; no legacy fallback"
        )
    first = packages[0]
    metadata = client.transport.get_json(
        client._task_api_url(
            "/backends/{backend}/metadata", backend=quote(first.backend.name, safe="")
        ),
        {},
        client.timeout,
    )
    limits = metadata.get("limits") if isinstance(metadata, Mapping) else None
    maximum = limits.get("max_tasks_per_job") if isinstance(limits, Mapping) else None
    if type(maximum) is not int or maximum < 1:
        raise CapabilityError("Quafu did not publish a valid max_tasks_per_job limit")
    if len(packages) > maximum:
        raise CapabilityError(f"Quafu collection exceeds max_tasks_per_job={maximum}")
    items = [
        build_submission_receipt(package, {"quafu_protocol": "task_api_v1"})
        for package in packages
    ]
    digest = _digest(items, first.shots, first.n_qubits)
    context: _CollectionReceipt = {
        "schema": _SCHEMA,
        "items": items,
        "shots": first.shots,
        "n_qubits": first.n_qubits,
        "sha256": digest,
    }
    response = client.transport.post_json(
        client._task_api_url("/jobs"),
        {
            "mode": "collection",
            "circuit_level": "logical",
            "target": first.backend.name,
            "shots": first.shots,
            "circuits": [{"qasm": package.qasm} for package in packages],
            "name": first.name,
            "client_ref": str(uuid.uuid4()),
        },
        client._task_api_headers(),
        client.timeout,
    )
    job_id = response.get("job_id") if isinstance(response, Mapping) else None
    if not isinstance(job_id, str) or not job_id.strip():
        raise ExecutionError(
            "Quafu collection response does not contain a job ID; do not resubmit"
        )
    if "n_tasks" in response and (
        type(response["n_tasks"]) is not int or response["n_tasks"] != len(packages)
    ):
        raise ExecutionError("Quafu accepted an unexpected task count; do not resubmit")
    if "task_ids" in response and (
        not isinstance(response["task_ids"], list)
        or len(response["task_ids"]) != len(packages)
    ):
        raise ExecutionError("Quafu accepted an unexpected task count; do not resubmit")
    return (
        ProviderTaskHandle(
            "quafu",
            job_id,
            first.backend.name,
            items[0] | {"collection_sha256": digest},
        ),
        context,
    )


def _fetch_results(
    client: QuafuProvider,
    handle: ProviderTaskHandle,
    items: list[dict[str, str]],
    shots: int,
    width: int,
) -> list[DeploymentResult]:
    results: dict[int, DeploymentResult] = {}
    offset = 0
    while len(results) < len(items):
        query = urlencode(
            {
                "offset": offset,
                "limit": min(_PAGE_SIZE, len(items) - offset),
                "fields": "counts",
            }
        )
        response = client.transport.get_json(
            client._task_api_url("/jobs/{task_id}/results", task_id=handle.task_id)
            + "?"
            + query,
            client._task_api_headers(),
            client.timeout,
        )
        if not isinstance(response, Mapping):
            raise ExecutionError("Invalid Quafu collection result page")
        if response.get("job_id", handle.task_id) != handle.task_id:
            raise ExecutionError("Quafu collection result job ID mismatch")
        indices = response.get("seq")
        if not isinstance(indices, list) or not indices:
            raise ExecutionError("Quafu collection results are incomplete")
        columns = ("counts", "status", "shots_returned", "n_clbits")
        if any(
            not isinstance(response.get(key), list)
            or len(response[key]) != len(indices)
            for key in columns
        ):
            raise ExecutionError("Quafu collection result columns are misaligned")
        order = response.get("bit_order")
        if not isinstance(order, str) or order not in {"c0_rightmost", "c0_leftmost"}:
            raise ExecutionError("Unknown Quafu collection result bit order")
        for row, index in enumerate(indices):
            if (
                type(index) is not int
                or not 0 <= index < len(items)
                or index in results
            ):
                raise ExecutionError("Duplicate or invalid Quafu collection task index")
            status = response["status"][row]
            if not isinstance(status, str) or status not in {"Finished", "Completed"}:
                raise ExecutionError(f"Quafu collection task #{index} is {status}")
            raw = response["counts"][row]
            if raw is None:
                raise RuntimeError(f"quafu task {handle.task_id} has no result yet")
            returned = response["shots_returned"][row]
            if (
                not isinstance(raw, Mapping)
                or not raw
                or type(returned) is not int
                or not 0 < returned <= shots
                or type(response["n_clbits"][row]) is not int
                or response["n_clbits"][row] != width
                or any(
                    not isinstance(key, str)
                    or len(key) != width
                    or set(key) - {"0", "1"}
                    or type(value) is not int
                    or value < 0
                    for key, value in raw.items()
                )
                or sum(raw.values()) != returned
            ):
                raise ExecutionError(
                    f"Invalid Quafu collection counts for task #{index}"
                )
            counts = _extract_counts(
                {"counts": raw},
                provider="quafu",
                flip=client.reverse_result_bits and order == "c0_rightmost",
            )
            task_handle = ProviderTaskHandle(
                "quafu", handle.task_id, handle.backend_name, items[index]
            )
            results[index] = validate_deployment_result(
                DeploymentResult(
                    task_handle,
                    counts,
                    returned,
                    build_result_metadata(
                        task_handle,
                        {
                            "quafu_protocol": "task_api_v1",
                            "task_index": index,
                            "bit_order": order,
                            "status": status,
                        },
                    ),
                )
            )
        offset += len(indices)
    return [results[index] for index in range(len(items))]


def collection_result(
    client: QuafuProvider,
    handle: ProviderTaskHandle,
    context: Any,
    *,
    width: int,
    output_name: str | None,
    target: str,
    name: str | None,
) -> ExecutionResult:
    items, shots = validate_context(context, width=width, identity=handle.payload)
    native = _fetch_results(client, handle, items, shots, width)
    first = counts_result(
        native[0],
        n_qubits=width,
        output_name=output_name,
        compiler=None,
        target=target,
        target_qubits=(),
        name=name,
    )
    returned = [item.shots for item in native]
    counts: list[dict[str | int, int]] = [
        {key: value for key, value in item.counts.items()} for item in native
    ]
    measurement = replace(
        first.measurements[0],
        value=counts,
        shots=returned[0] if len(set(returned)) == 1 else None,
    )
    return replace(
        first,
        measurements=(measurement,),
        provenance={
            key: value
            for key, value in first.provenance.items()
            if key != "service_compiler"
        }
        | {
            "collection_sha256": context["sha256"],
            "batch_size": len(items),
            "deployment": [dict(item.metadata) for item in native],
        },
        runtime={
            "mode": "remote_qpu",
            "shots": shots,
            "batch_size": len(items),
            "shots_returned": returned,
        },
    )
