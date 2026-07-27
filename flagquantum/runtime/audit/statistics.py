"""Canonical payload attachment helpers for distributed audit results."""

from typing import Any, Mapping


def attach_distributed_evidence_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    from . import evaluate_distributed_evidence_contract

    out = dict(payload)
    contract = evaluate_distributed_evidence_contract(out).summary()
    out["distributed_evidence_contract"] = contract
    out["claimability_status"] = contract["status"]
    out["claimable_production_training"] = contract["claimable_production_training"]
    out["distributed_evidence_contract_version"] = contract["contract_version"]
    out["transport_evidence"] = contract["transport_evidence"]
    out["transport_evidence_status"] = contract["transport_evidence"]["status"]
    return out


def attach_distributed_scalability_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    from . import audit_distributed_scalability

    out = dict(payload)
    out["scalability_audit"] = audit_distributed_scalability(payload).summary()
    return out


__all__ = [
    "attach_distributed_evidence_contract",
    "attach_distributed_scalability_audit",
]
