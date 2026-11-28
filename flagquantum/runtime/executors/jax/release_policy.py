"""Release classification kept separate from JAX execution code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
    return out
