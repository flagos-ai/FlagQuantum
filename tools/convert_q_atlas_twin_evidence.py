#!/usr/bin/env python3
"""Convert a prospectively bound Q-ATLAS audit to canonical Twin evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flagquantum.twin import TwinEvidenceEnvelope

_AUDIT_SCHEMA = "flagquantum.q_atlas_connected_subgraph_future_audit.v1"
_BINDING_SCHEMA = "flagquantum.q_atlas_twin_evidence_binding.v1"
_BINDING_FIELDS = {
    "schema",
    "source_validation_identity",
    "snapshot_identity",
    "qpu_identity",
    "topology_identity",
    "ordered_physical_chain",
    "source_shadow_candidate_identity",
    "supported_operations",
    "maximum_instruction_count",
    "circuit_identities",
    "created_before_target_outcomes",
    "target_outcome_count_available_when_frozen",
    "binding_identity",
}


def _digest(payload: Mapping[str, Any], *, identity_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(identity_field, None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read evidence input {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence input must be a JSON object: {path}")
    return payload


def convert(
    audit: Mapping[str, Any], binding: Mapping[str, Any]
) -> TwinEvidenceEnvelope:
    """Validate a prospective identity bridge and build a canonical envelope."""

    if audit.get("schema") != _AUDIT_SCHEMA:
        raise ValueError("unsupported Q-ATLAS audit schema")
    if binding.get("schema") != _BINDING_SCHEMA:
        raise ValueError("unsupported Q-ATLAS evidence-binding schema")
    if set(binding) != _BINDING_FIELDS:
        raise ValueError("Q-ATLAS evidence binding fields do not match its schema")
    if audit.get("audit_identity") != _digest(audit, identity_field="audit_identity"):
        raise ValueError("Q-ATLAS audit identity changed")
    if binding.get("binding_identity") != _digest(
        binding, identity_field="binding_identity"
    ):
        raise ValueError("Q-ATLAS evidence binding identity changed")
    summary = audit.get("summary")
    provenance = audit.get("provenance")
    if not isinstance(summary, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("Q-ATLAS audit summary or provenance is invalid")
    if (
        audit.get("state") != "validated_fixed_program_envelope"
        or summary.get("all_predeclared_gates_passed") is not True
    ):
        raise ValueError("Q-ATLAS audit did not pass its predeclared gates")
    if (
        binding.get("created_before_target_outcomes") is not True
        or binding.get("target_outcome_count_available_when_frozen") != 0
    ):
        raise ValueError("Q-ATLAS identity binding was not frozen prospectively")
    if binding.get("source_validation_identity") != provenance.get(
        "source_draft_identity"
    ):
        raise ValueError("Q-ATLAS audit does not descend from the frozen validation")

    audit_binding = audit.get("binding", {})
    if not isinstance(audit_binding, Mapping):
        raise ValueError("Q-ATLAS audit binding is invalid")
    checks = {
        "qpu_identity": audit.get("qpu_identity"),
        "ordered_physical_chain": audit_binding.get("ordered_physical_chain"),
        "source_shadow_candidate_identity": audit_binding.get(
            "source_shadow_candidate_identity"
        ),
        "topology_identity": audit_binding.get("topology_identity"),
    }
    for name, expected in checks.items():
        if binding.get(name) != expected:
            raise ValueError(f"Q-ATLAS {name} changed")

    rows = audit.get("program_audits")
    circuit_bindings = binding.get("circuit_identities")
    if not isinstance(rows, list) or not rows or not isinstance(circuit_bindings, dict):
        raise ValueError("Q-ATLAS fixed-program evidence is incomplete")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Q-ATLAS program audit is invalid")
    program_ids = {str(row.get("physical_program_identity")) for row in rows}
    if program_ids != set(circuit_bindings):
        raise ValueError("Q-ATLAS physical programs lack exact IR identity bindings")

    bounds: list[float] = []
    for row in rows:
        if row.get("gate_passed") is not True:
            raise ValueError("Q-ATLAS program gate failed")
        try:
            error = float(row["prediction_error_tv"])
            tolerance = float(row["finite_shot_two_sample_tv_bound_95"])
        except (KeyError, TypeError, ValueError) as exception:
            raise ValueError("Q-ATLAS program TV evidence is invalid") from exception
        bound = error + tolerance
        if not 0.0 <= error <= 1.0 or not 0.0 <= tolerance <= 1.0:
            raise ValueError("Q-ATLAS TV evidence is outside [0, 1]")
        bounds.append(min(1.0, bound))

    try:
        return TwinEvidenceEnvelope(
            snapshot_identity=str(binding["snapshot_identity"]),
            physical_qubits=tuple(binding["ordered_physical_chain"]),
            supported_operations=tuple(binding["supported_operations"]),
            maximum_instruction_count=int(binding["maximum_instruction_count"]),
            verified_circuit_identities=tuple(
                str(circuit_bindings[program_id]) for program_id in sorted(program_ids)
            ),
            evidence_identity=str(audit["audit_identity"]),
            verified_tv_error_bound=max(bounds),
            estimated_tv_error_bound=None,
            confidence_level=0.95,
        )
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Q-ATLAS evidence binding is invalid") from exception


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    envelope = convert(_read(arguments.audit), _read(arguments.binding))
    _write_once(arguments.output, envelope.to_dict())
    print(json.dumps({"evidence_envelope_identity": envelope.identity}))


if __name__ == "__main__":
    main()
