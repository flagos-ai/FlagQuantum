import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.core.contracts import (
    AccuracyContract,
    AuditRecordContract,
    CapabilityContract,
    ContractVersionError,
    EstimatedResources,
    ExecutionObservation,
    ExecutionRecordContract,
    MeasurementContract,
    OwnershipContract,
    ProvenanceContract,
    RequestedExecution,
    RuntimePlanContract,
    UnknownContractFieldError,
    migrate_contract,
)
from flagquantum.runtime.audit.contracts import audit_execution_record
from flagquantum.runtime.records import record_execution
from tools.runtime_contract_schema import render_schema

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _plan() -> RuntimePlanContract:
    return RuntimePlanContract(
        plan_id="plan-1",
        requested=RequestedExecution(world_size=2),
        estimated=EstimatedResources(memory_bytes=1024),
        capability=CapabilityContract(supports_distributed=True),
    )


def _record() -> ExecutionRecordContract:
    return record_execution(
        _plan(),
        observed=ExecutionObservation(
            executor="torchrun", elapsed_seconds=1.5, completed=True
        ),
        ownership=OwnershipContract(
            rank_owners=(("wire:0", 0), ("wire:1", 1)),
            distribution_semantics="sharded_across_ranks",
        ),
        measurements=(MeasurementContract(wires=(0,), values=(1.0,)),),
        accuracy=AccuracyContract(metric="fidelity", value=1.0, passed=True),
        provenance=ProvenanceContract(
            commit="a" * 40,
            workload_sha256="b" * 64,
            command=("python", "run.py"),
            devices=("cuda:0", "cuda:1"),
            rank_mapping=("0:cuda:0", "1:cuda:1"),
            raw_log_sha256="c" * 64,
        ),
    )


def test_plan_and_execution_json_are_deterministic_round_trips():
    plan = _plan()
    restored_plan = RuntimePlanContract.from_dict(json.loads(plan.to_json()))
    assert restored_plan == plan
    assert restored_plan.content_hash == plan.content_hash

    record = _record()
    restored_record = ExecutionRecordContract.from_dict(json.loads(record.to_json()))
    assert restored_record == record
    assert restored_record.content_hash == record.content_hash


def test_production_parser_rejects_unknown_outer_and_nested_fields():
    payload = _plan().to_dict()
    with pytest.raises(UnknownContractFieldError, match="mystery"):
        RuntimePlanContract.from_dict({**payload, "mystery": True})
    requested = {**payload["requested"], "undocumented": "value"}
    with pytest.raises(UnknownContractFieldError, match="undocumented"):
        RuntimePlanContract.from_dict({**payload, "requested": requested})


def test_legacy_contract_requires_explicit_supported_migration():
    legacy = {
        "kind": "runtime_plan",
        "version": "0.9",
        "plan_id": "old",
        "request": _plan().requested.to_dict(),
        "estimate": _plan().estimated.to_dict(),
        "capability": _plan().capability.to_dict(),
    }
    for item in (legacy["request"], legacy["estimate"], legacy["capability"]):
        item.pop("version")
    migrated = RuntimePlanContract.from_dict(migrate_contract(legacy))
    assert migrated.plan_id == "old"
    with pytest.raises(ContractVersionError, match="no migration"):
        migrate_contract({**legacy, "version": "0.8"})


def test_planner_contract_cannot_contain_measured_or_validated_fields():
    contract = fq.Circuit(2).h(0).cx(0, 1).plan().to_contract()
    assert set(contract.to_dict()) == {
        "kind",
        "version",
        "requested",
        "estimated",
        "capability",
        "plan_id",
    }
    assert "observed" not in contract.to_dict()
    assert "release_claim_approved" not in contract.to_json()


def test_executor_record_cannot_approve_and_only_auditor_validates():
    record = _record()
    assert "release_claim_approved" not in record.to_json()
    audit = audit_execution_record(record, validator="release-gate")
    assert isinstance(audit, AuditRecordContract)
    assert audit.validated.release_claim_approved
    assert AuditRecordContract.from_dict(json.loads(audit.to_json())) == audit


def test_generated_contract_schema_is_current():
    assert (
        ROOT / "docs" / "runtime_contracts.schema.json"
    ).read_text() == render_schema()
