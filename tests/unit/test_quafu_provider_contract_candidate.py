from __future__ import annotations

import json
from pathlib import Path

import torch

import flagquantum as fq

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "quafu-provider-contract-v1-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_quafu_candidate_does_not_change_the_stable_root_or_protected_schemas() -> None:
    contract = _contract()

    assert contract["status"] == "draft"
    assert contract["root_manifest_change"] is False
    assert contract["protected_contract_change_authorized"] is False
    assert contract["protected_schema_changes"] == []


def test_quafu_candidate_records_the_asymmetric_bit_order_boundary() -> None:
    contract = _contract()
    bit_order = contract["bit_order"]
    assert isinstance(bit_order, dict)
    fixture = bit_order["asymmetric_fixture"]
    assert isinstance(fixture, dict)

    circuit = fq.Circuit(2).x(0)
    counts = circuit.counts(8, generator=torch.Generator().manual_seed(7))

    assert counts == [{fixture["local_wires_0_1"]: 8}]
    assert fixture["portal_counts"] == fixture["local_wires_0_1"][::-1]
    assert bit_order["provider_boundary_conversion_required"] is True


def test_quafu_adapter_candidate_does_not_reintroduce_flat_experimental_api() -> None:
    contract = _contract()
    adapter = contract["adapter"]
    assert isinstance(adapter, dict)

    assert adapter["forbidden_dependency"] == "fq.experimental.QPUTwin"
    assert adapter["contains_flagquantum_implementation_source"] is False
    assert adapter["allowed_contents"] == [
        "openapi",
        "schemas",
        "mock",
        "docs",
        "thin_binding",
    ]
    assert "QPUTwin" not in dir(fq.experimental)
    assert adapter["calibration_entry"] == (
        "flagquantum.deployment.quafu_noise_model_from_chip_info"
    )
    assert adapter["execution_entry"] == "fq.run"


def test_quafu_candidate_has_closed_status_and_metadata_vocabularies() -> None:
    contract = _contract()

    assert contract["task_states"] == [
        "Pending",
        "Running",
        "Finished",
        "Failed",
        "Cancelled",
    ]
    assert contract["result_metadata_minimum"] == [
        "backend",
        "duration_ms",
        "calibration_version",
        "noise_model_version",
        "counts_bit_order",
        "physical_qubits",
        "program_sha256",
    ]
