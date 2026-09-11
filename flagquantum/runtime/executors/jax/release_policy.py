"""Release classification kept separate from JAX execution code."""

from __future__ import annotations

from typing import Any, Mapping, cast


def attach_statevector_claimability(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...audit import (
        attach_distributed_evidence_contract,
        evaluate_statevector_training_claimability,
    )

    out = attach_distributed_evidence_contract(payload)
    gate = evaluate_statevector_training_claimability(out).summary()
    out["statevector_training_claimability_gate"] = gate
    out["statevector_training_claimability_status"] = gate["status"]
    out["claimable_production_training"] = gate["claimable_production_training"]
    return cast(dict[str, Any], out)


def attach_evidence_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...audit import attach_distributed_evidence_contract

    return cast(dict[str, Any], attach_distributed_evidence_contract(payload))


def attach_mps_backward_readiness(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...audit import attach_mps_runtime_summary

    return cast(dict[str, Any], attach_mps_runtime_summary(payload))
