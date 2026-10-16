"""Machine-readable JSON Schemas for the research capability service family."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _closed(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required or properties),
        "additionalProperties": False,
    }


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def _document(schema_id: str, title: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "$schema": _DIALECT,
        "$id": schema_id,
        "title": title,
        **dict(body),
    }


_ISSUE = _closed(
    {
        "code": {"type": "string", "minLength": 1},
        "message": {"type": "string", "minLength": 1},
        "context": {"type": "object"},
    }
)
_VALIDATION_REPORT = _closed(
    {
        "passed": {"type": "boolean"},
        "validator_id": {"type": "string", "minLength": 1},
        "blockers": {"type": "array", "items": _ISSUE},
        "warnings": {"type": "array", "items": _ISSUE},
        "metrics": {"type": "object"},
    }
)
_STRUCTURED_ERROR = _closed(
    {
        "code": {"type": "string", "minLength": 1},
        "stage": {
            "enum": [
                "catalog_match",
                "input_validation",
                "resource_estimation",
                "environment_preflight",
                "compilation",
                "execution",
                "result_parsing",
                "scientific_validation",
                "artifact_persistence",
            ]
        },
        "retryable": {"type": "boolean"},
        "message": {"type": "string", "minLength": 1},
        "context": {"type": "object"},
        "evidence_generated": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggested_actions": {
            "type": "array",
            "items": {"type": "object"},
        },
    }
)
_PAULI_TERM = _closed(
    {
        "coefficient": {"type": "number"},
        "pauli": {"type": "string", "pattern": "^[IXYZ]+$"},
    }
)
_PAULI_SUM = _closed(
    {
        "n_qubits": {"type": "integer", "minimum": 1},
        "terms": {"type": "array", "minItems": 1, "items": _PAULI_TERM},
        "coefficient_unit": {"const": "hartree"},
        "basis_convention": {"const": "wire_0_is_leftmost"},
        "schema": {"const": "flagquantum.pauli_sum.v1"},
    }
)
_PROVIDER_PROVENANCE = _closed(
    {
        "provider": {"type": "string", "minLength": 1},
        "package": {"type": "string", "minLength": 1},
        "package_version": {"type": "string", "minLength": 1},
        "tool": {"type": "string", "minLength": 1},
        "tool_version": {"type": "string", "minLength": 1},
        "execution_digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    }
)
_UNSUPPORTED = _closed(
    {
        "code": {"type": "string", "minLength": 1},
        "message": {"type": "string", "minLength": 1},
        "context": {"type": "object"},
    }
)
_COMPLEX_AMPLITUDE = _closed({"real": {"type": "number"}, "imag": {"type": "number"}})
_STATEVECTOR = _closed(
    {
        "basis_states": {"type": "array", "items": {"type": "string"}},
        "amplitudes": {"type": "array", "items": _COMPLEX_AMPLITUDE},
        "norm": {"type": "number"},
    }
)
_VQE_REQUEST = _closed(
    {
        "hamiltonian": _PAULI_SUM,
        "fixed_hamming_weight": _nullable({"type": "integer", "minimum": 0}),
        "ansatz": _closed(
            {
                "name": {"const": "two_qubit_single_excitation"},
                "initial_basis_state": {"type": "string", "pattern": "^[01]+$"},
                "initial_parameters": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            }
        ),
        "optimizer": _closed(
            {
                "name": {"const": "adam"},
                "max_evaluations": {"type": "integer", "minimum": 2},
                "learning_rate": {"type": "number", "exclusiveMinimum": 0},
            }
        ),
        "schema": {"const": "flagquantum.ground_state_vqe_request.v1"},
    }
)
_EXACT_REQUEST = _closed(
    {
        "hamiltonian": _PAULI_SUM,
        "fixed_hamming_weight": {"type": "integer", "minimum": 0},
    }
)
_VARIANCE_REQUEST = _closed(
    {
        "hamiltonian": _PAULI_SUM,
        "statevector": _STATEVECTOR,
        "tolerance": {"type": "number", "minimum": 0},
    }
)
_VQE_RESULT = _closed(
    {
        "status": {"enum": ["completed", "unsupported"]},
        "request": _VQE_REQUEST,
        "provenance": _PROVIDER_PROVENANCE,
        "energy": _nullable({"type": "number"}),
        "parameters": {"type": "array", "items": {"type": "number"}},
        "convergence_trace": {
            "type": "array",
            "items": _closed(
                {
                    "evaluation": {"type": "integer", "minimum": 1},
                    "energy": {"type": "number"},
                }
            ),
        },
        "term_evidence": {
            "type": "array",
            "items": _closed(
                {
                    "pauli": {"type": "string"},
                    "coefficient": {"type": "number"},
                    "expectation": {"type": "number"},
                    "contribution": {"type": "number"},
                }
            ),
        },
        "statevector": _nullable(_STATEVECTOR),
        "resources": _nullable(
            _closed(
                {
                    "n_qubits": {"type": "integer"},
                    "hamiltonian_terms": {"type": "integer"},
                    "ansatz_gates": {"type": "integer"},
                    "trainable_parameters": {"type": "integer"},
                    "optimizer_steps": {"type": "integer"},
                    "objective_evaluations": {"type": "integer"},
                    "backend_executions": {"type": "integer"},
                    "execution_mode": {"const": "local_statevector"},
                }
            )
        ),
        "unsupported": _nullable(_UNSUPPORTED),
        "schema": {"const": "flagquantum.ground_state_vqe_result.v1"},
    }
)
_EXACT_RESULT = _closed(
    {
        "status": {"enum": ["completed", "unsupported"]},
        "hamiltonian": _PAULI_SUM,
        "fixed_hamming_weight": {"type": "integer"},
        "provenance": _PROVIDER_PROVENANCE,
        "energy": _nullable({"type": "number"}),
        "sector_basis_states": {"type": "array", "items": {"type": "string"}},
        "unsupported": _nullable(_UNSUPPORTED),
        "schema": {"const": "flagquantum.exact_sector_reference.v1"},
    }
)
_VARIANCE_RESULT = _closed(
    {
        "status": {"enum": ["completed", "unsupported"]},
        "hamiltonian": _PAULI_SUM,
        "provenance": _PROVIDER_PROVENANCE,
        "tolerance": {"type": "number", "minimum": 0},
        "energy": _nullable({"type": "number"}),
        "variance": _nullable({"type": "number"}),
        "passed": {"type": "boolean"},
        "unsupported": _nullable(_UNSUPPORTED),
        "schema": {"const": "flagquantum.energy_variance_reference.v1"},
    }
)
_MANIFEST = _closed(
    {
        "capability_id": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "provider": {"type": "string", "minLength": 1},
        "adapter_version": {"type": "string", "minLength": 1},
        "domain": {"type": "string", "minLength": 1},
        "task": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "tags": {"type": "array", "items": {"type": "string"}},
        "input_schema": {"type": "string", "minLength": 1},
        "output_schema": {"type": "string", "minLength": 1},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "validators": {"type": "array", "items": {"type": "string"}},
        "fallback_capability_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "implementation_status": {"enum": ["available", "planned"]},
        "maturity_binding": _nullable({"type": "string"}),
        "schema": {"const": "flagquantum.research_capability_manifest.v1"},
    }
)
_EXECUTION = _closed(
    {
        "target": {"type": "string", "minLength": 1},
        "execution_mode": {"const": "local_statevector"},
        "started_at": {"type": "string", "minLength": 1},
        "completed_at": {"type": "string", "minLength": 1},
        "duration_seconds": {"type": "number", "minimum": 0},
        "provider_order": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "fallback_used": {"type": "boolean"},
        "stage_durations_seconds": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "call_counts": _closed(
            {
                "provider": {"type": "integer", "minimum": 0},
                "validators": {"type": "integer", "minimum": 0},
            }
        ),
    }
)
_EVIDENCE = _closed(
    {
        "schema": {"const": "flagquantum.research_capability_evidence_bundle.v1"},
        "run_id": {"type": "string", "minLength": 1},
        "capability": _closed(
            {
                "capability_id": {"type": "string", "minLength": 1},
                "version": {"type": "string", "minLength": 1},
                "provider": {"type": "string", "minLength": 1},
            }
        ),
        "status": {"enum": ["succeeded", "unsupported", "validation_failed", "failed"]},
        "result": {"type": "object"},
        "metrics": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "object"}},
        "execution": _EXECUTION,
        "validation": {"type": "array", "items": _VALIDATION_REPORT},
        "attempts": {
            "type": "array",
            "items": _closed(
                {
                    "capability_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "status": {"enum": ["succeeded", "unsupported", "failed"]},
                    "started_at": {"type": "string"},
                    "completed_at": {"type": "string"},
                    "error": _nullable(_STRUCTURED_ERROR),
                }
            ),
        },
        "warnings": {"type": "array", "items": _ISSUE},
        "provenance": _closed(
            {
                "package": {"const": "flagquantum"},
                "package_version": {"type": "string"},
                "python_version": {"type": "string"},
                "torch_version": _nullable({"type": "string"}),
                "fallback_used": {"type": "boolean"},
                "input_sha256": _nullable(
                    {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                ),
                "deterministic": {"type": "boolean"},
                "random_seed": _nullable({"type": "integer"}),
                "content_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            }
        ),
        "error": _nullable(_STRUCTURED_ERROR),
    }
)


@dataclass(frozen=True, slots=True)
class ResearchSchemaDocument:
    """One immutable-by-copy JSON Schema served by installed FlagQuantum."""

    schema_id: str
    kind: Literal["input", "output", "envelope", "value"]
    document: Mapping[str, Any]
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "kind": self.kind,
            "version": self.version,
            "document": deepcopy(dict(self.document)),
        }


def _entry(
    schema_id: str,
    kind: Literal["input", "output", "envelope", "value"],
    title: str,
    body: Mapping[str, Any],
) -> ResearchSchemaDocument:
    return ResearchSchemaDocument(schema_id, kind, _document(schema_id, title, body))


_SCHEMAS = (
    _entry("flagquantum.pauli_sum.v1", "value", "Pauli sum", _PAULI_SUM),
    _entry(
        "flagquantum.ground_state_vqe_request.v1", "input", "VQE request", _VQE_REQUEST
    ),
    _entry(
        "flagquantum.ground_state_vqe_result.v1", "output", "VQE result", _VQE_RESULT
    ),
    _entry(
        "flagquantum.exact_ground_state_request.v1",
        "input",
        "Exact request",
        _EXACT_REQUEST,
    ),
    _entry(
        "flagquantum.exact_sector_reference.v1", "output", "Exact result", _EXACT_RESULT
    ),
    _entry(
        "flagquantum.energy_variance_request.v1",
        "input",
        "Variance request",
        _VARIANCE_REQUEST,
    ),
    _entry(
        "flagquantum.energy_variance_reference.v1",
        "output",
        "Variance result",
        _VARIANCE_RESULT,
    ),
    _entry(
        "flagquantum.research_capability_manifest.v1",
        "value",
        "Capability manifest",
        _MANIFEST,
    ),
    _entry(
        "flagquantum.research_validation_report.v1",
        "value",
        "Validation report",
        _VALIDATION_REPORT,
    ),
    _entry(
        "flagquantum.research_capability_error.v1",
        "value",
        "Structured error",
        _STRUCTURED_ERROR,
    ),
    _entry(
        "flagquantum.research_capability_evidence_bundle.v1",
        "envelope",
        "Research capability evidence bundle",
        _EVIDENCE,
    ),
)
_SCHEMAS_BY_ID = MappingProxyType({item.schema_id: item for item in _SCHEMAS})


def list_research_schemas() -> tuple[ResearchSchemaDocument, ...]:
    """Return every installed research schema in stable identity order."""

    return tuple(describe_research_schema(item.schema_id) for item in _SCHEMAS)


def describe_research_schema(schema_id: str) -> ResearchSchemaDocument:
    """Return an isolated copy of one exact installed schema document."""

    if not isinstance(schema_id, str) or not schema_id:
        raise TypeError("schema_id must be a non-empty string")
    try:
        item = _SCHEMAS_BY_ID[schema_id]
    except KeyError as exc:
        raise KeyError(f"unknown research schema: {schema_id!r}") from exc
    return ResearchSchemaDocument(
        item.schema_id,
        item.kind,
        deepcopy(dict(item.document)),
        item.version,
    )


__all__ = (
    "ResearchSchemaDocument",
    "describe_research_schema",
    "list_research_schemas",
)
