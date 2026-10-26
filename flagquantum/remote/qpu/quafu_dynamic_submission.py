"""Private exactly-once Quafu dynamic task contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...deployment.cloud import DeploymentPackage
from ...deployment.dynamic_sealing import _validate_dynamic_remote_package_seal
from ._quafu_sdk import _QuafuSdkAdapter
from .contracts import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_result,
)
from .result_parsing import _extract_counts

_SDK_PROTOCOL = "quafu_sdk_v1"
_RESULT_BIT_ORDER = "msb_left"


def _dynamic_receipt_identity(package: DeploymentPackage) -> dict[str, Any]:
    seal = package.metadata["dynamic_remote_package_seal"]
    return {
        "dynamic_remote_package_seal_sha256": package.metadata[
            "dynamic_remote_package_seal_sha256"
        ],
        "source_requirement_set_id": seal["source_requirement_set_id"],
        "target_snapshot_id": seal["target_snapshot_id"],
        "calibration_id": seal["calibration_id"],
        "provider_legality_identity": seal["provider_legality_identity"],
    }


def _submit_quafu_dynamic_once(
    package: DeploymentPackage,
    *,
    adapter: _QuafuSdkAdapter,
) -> ProviderTaskHandle:
    """Create at most one mocked SDK submission for one sealed package."""

    _validate_dynamic_remote_package_seal(package)
    if package.backend.provider != "quafu":
        raise ValueError("dynamic Quafu submission requires a Quafu package")
    seal_id = package.metadata["dynamic_remote_package_seal_sha256"]
    client_ref = f"fq-dynamic-{seal_id}"
    try:
        response = adapter.submit_physical(
            package.qasm,
            target=package.backend.name,
            shots=package.shots,
            name=client_ref,
        )
    except Exception:
        raise RuntimeError("quafu_dynamic_submission_ambiguous") from None
    job_id = response.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RuntimeError("quafu_dynamic_submission_ambiguous:missing_job_id")

    receipt = build_submission_receipt(package, response)
    receipt.update(_dynamic_receipt_identity(package))
    receipt.update(
        {
            "quafu_protocol": _SDK_PROTOCOL,
            "client_ref": client_ref,
            "submitted_shots": package.shots,
            "submitted_qubits": package.n_qubits,
        }
    )
    return ProviderTaskHandle("quafu", job_id, package.backend.name, receipt)


def _normalize_quafu_dynamic_mock_result(
    handle: ProviderTaskHandle,
    response: Mapping[str, Any],
) -> DeploymentResult:
    """Normalize the closed mocked result shape without claiming live support."""

    if (
        handle.provider != "quafu"
        or handle.payload.get("quafu_protocol") != _SDK_PROTOCOL
    ):
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:task_identity")
    if response.get("bit_order") != _RESULT_BIT_ORDER:
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:bit_order")
    if (
        "mid_circuit_measurements" in response
        or response.get("measurements") is not None
    ):
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:mid_circuit_data")
    rows = response.get("counts")
    if not isinstance(rows, Mapping):
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:counts")
    counts = _extract_counts({"counts": rows}, provider="quafu")
    submitted_shots = handle.payload.get("submitted_shots")
    if type(submitted_shots) is not int or submitted_shots <= 0:
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:shots")
    if response.get("shots_returned") != submitted_shots:
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:shots")
    submitted_qubits = handle.payload.get("submitted_qubits")
    if type(submitted_qubits) is not int or submitted_qubits <= 0:
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:bit_width")
    if response.get("n_clbits") != submitted_qubits:
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:bit_width")
    if any(
        len(outcome) != submitted_qubits or set(outcome) - {"0", "1"}
        for outcome in counts
    ):
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:bit_width")
    identity_names = (
        "dynamic_remote_package_seal_sha256",
        "source_requirement_set_id",
        "target_snapshot_id",
        "calibration_id",
        "provider_legality_identity",
    )
    identity = {name: handle.payload.get(name) for name in identity_names}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise RuntimeError("quafu_dynamic_result_contract_unconfirmed:identity")
    result = DeploymentResult(
        handle=handle,
        counts=counts,
        shots=submitted_shots,
        metadata=build_result_metadata(
            handle,
            {
                **identity,
                "counts_bit_order": _RESULT_BIT_ORDER,
                "counts_width": submitted_qubits,
                "quafu_protocol": _SDK_PROTOCOL,
            },
        ),
    )
    return validate_deployment_result(result)


__all__: tuple[str, ...] = ()
