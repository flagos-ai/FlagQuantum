"""Deterministic discovery and preflight for bounded research capabilities.

This module owns protocol-independent application-service composition. It does
not parse natural language, register protocol tools, persist runs, or execute
remote work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .ground_state import (
    ComplexAmplitude,
    GroundStateAnsatz,
    GroundStateOptimizer,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    StatevectorEvidence,
    validate_ground_state_vqe_request,
)

_VQE_ID = "ground_state.solve.vqe"
_EXACT_ID = "ground_state.solve.exact"
_VARIANCE_ID = "validate.energy_variance"
_MATURITY_BINDING = "research_capability_services"


def _closed_mapping(
    value: object,
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {unknown}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"missing {label} fields: {missing}")
    return value


def _pauli_sum_from_mapping(value: object) -> PauliSum:
    payload = _closed_mapping(
        value,
        label="Pauli sum",
        allowed=frozenset(
            {"n_qubits", "terms", "coefficient_unit", "basis_convention", "schema"}
        ),
        required=frozenset({"n_qubits", "terms"}),
    )
    raw_terms = payload["terms"]
    if not isinstance(raw_terms, (list, tuple)):
        raise TypeError("Pauli sum terms must be an array")
    terms = []
    for index, raw_term in enumerate(raw_terms):
        term = _closed_mapping(
            raw_term,
            label=f"Pauli sum term {index}",
            allowed=frozenset({"coefficient", "pauli"}),
            required=frozenset({"coefficient", "pauli"}),
        )
        terms.append(PauliSumTerm(term["coefficient"], term["pauli"]))
    return PauliSum(
        n_qubits=payload["n_qubits"],
        terms=tuple(terms),
        coefficient_unit=payload.get("coefficient_unit", "hartree"),
        basis_convention=payload.get("basis_convention", "wire_0_is_leftmost"),
        schema=payload.get("schema", "flagquantum.pauli_sum.v1"),
    )


def _statevector_from_mapping(value: object) -> StatevectorEvidence:
    payload = _closed_mapping(
        value,
        label="statevector",
        allowed=frozenset({"basis_states", "amplitudes", "norm"}),
        required=frozenset({"basis_states", "amplitudes", "norm"}),
    )
    raw_basis = payload["basis_states"]
    raw_amplitudes = payload["amplitudes"]
    if not isinstance(raw_basis, (list, tuple)):
        raise TypeError("statevector basis_states must be an array")
    if not isinstance(raw_amplitudes, (list, tuple)):
        raise TypeError("statevector amplitudes must be an array")
    amplitudes = []
    for index, raw_amplitude in enumerate(raw_amplitudes):
        amplitude = _closed_mapping(
            raw_amplitude,
            label=f"statevector amplitude {index}",
            allowed=frozenset({"real", "imag"}),
            required=frozenset({"real", "imag"}),
        )
        amplitudes.append(ComplexAmplitude(amplitude["real"], amplitude["imag"]))
    return StatevectorEvidence(tuple(raw_basis), tuple(amplitudes), payload["norm"])


def _normalize_research_request(capability_id: str, request: object) -> object:
    """Decode a JSON-shaped request at the FlagQuantum-owned domain boundary."""

    if not isinstance(request, Mapping):
        return request
    if capability_id == _VQE_ID:
        payload = _closed_mapping(
            request,
            label="VQE request",
            allowed=frozenset(
                {"hamiltonian", "fixed_hamming_weight", "ansatz", "optimizer", "schema"}
            ),
            required=frozenset({"hamiltonian", "fixed_hamming_weight"}),
        )
        raw_ansatz = payload.get("ansatz", {})
        ansatz = _closed_mapping(
            raw_ansatz,
            label="VQE ansatz",
            allowed=frozenset({"name", "initial_basis_state", "initial_parameters"}),
            required=frozenset(),
        )
        raw_initial_parameters = ansatz.get("initial_parameters", (0.0,))
        if not isinstance(raw_initial_parameters, (list, tuple)):
            raise TypeError("VQE ansatz initial_parameters must be an array")
        raw_optimizer = payload.get("optimizer", {})
        optimizer = _closed_mapping(
            raw_optimizer,
            label="VQE optimizer",
            allowed=frozenset({"name", "max_evaluations", "learning_rate"}),
            required=frozenset(),
        )
        return GroundStateVQERequest(
            hamiltonian=_pauli_sum_from_mapping(payload["hamiltonian"]),
            fixed_hamming_weight=payload["fixed_hamming_weight"],
            ansatz=GroundStateAnsatz(
                name=ansatz.get("name", "two_qubit_single_excitation"),
                initial_basis_state=ansatz.get("initial_basis_state", "01"),
                initial_parameters=tuple(raw_initial_parameters),
            ),
            optimizer=GroundStateOptimizer(
                name=optimizer.get("name", "adam"),
                max_evaluations=optimizer.get("max_evaluations", 121),
                learning_rate=optimizer.get("learning_rate", 0.1),
            ),
            schema=payload.get("schema", "flagquantum.ground_state_vqe_request.v1"),
        )
    if capability_id == _EXACT_ID:
        payload = _closed_mapping(
            request,
            label="exact ground-state request",
            allowed=frozenset({"hamiltonian", "fixed_hamming_weight"}),
            required=frozenset({"hamiltonian", "fixed_hamming_weight"}),
        )
        return ExactGroundStateRequest(
            _pauli_sum_from_mapping(payload["hamiltonian"]),
            payload["fixed_hamming_weight"],
        )
    if capability_id == _VARIANCE_ID:
        payload = _closed_mapping(
            request,
            label="energy-variance request",
            allowed=frozenset({"hamiltonian", "statevector", "tolerance"}),
            required=frozenset({"hamiltonian", "statevector"}),
        )
        return EnergyVarianceRequest(
            _pauli_sum_from_mapping(payload["hamiltonian"]),
            _statevector_from_mapping(payload["statevector"]),
            payload.get("tolerance", 1e-5),
        )
    return request


@dataclass(frozen=True, slots=True)
class ExactGroundStateRequest:
    """Input for exact diagonalization in the bounded ground-state sector."""

    hamiltonian: PauliSum
    fixed_hamming_weight: int


@dataclass(frozen=True, slots=True)
class EnergyVarianceRequest:
    """Input for independent energy-variance validation."""

    hamiltonian: PauliSum
    statevector: StatevectorEvidence
    tolerance: float = 1e-5


@dataclass(frozen=True, slots=True)
class ResearchCapabilityQuery:
    """Structured filters supplied by an application or protocol adapter."""

    capability_id: str | None = None
    domain: str | None = None
    task: str | None = None
    provider: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchCapabilityManifest:
    """One versioned scientific operation supplied by one provider."""

    capability_id: str
    version: str
    provider: str
    adapter_version: str
    domain: str
    task: str
    description: str
    tags: tuple[str, ...]
    input_schema: str
    output_schema: str
    limitations: tuple[str, ...]
    validators: tuple[str, ...]
    fallback_capability_ids: tuple[str, ...]
    implementation_status: Literal["available", "planned"]
    maturity_binding: str | None = None
    schema: str = "flagquantum.research_capability_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchCapabilityCandidate:
    """A deterministic catalog match with an inspectable reason."""

    capability_id: str
    version: str
    provider: str
    match_score: float
    applicability: Literal["supported", "planned"]
    validators: tuple[str, ...]
    fallback_capability_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchValidationIssue:
    """One machine-readable reason that validation did not pass."""

    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class ResearchValidationReport:
    """Input or scientific validation outcome for one capability."""

    passed: bool
    validator_id: str
    blockers: tuple[ResearchValidationIssue, ...] = ()
    warnings: tuple[ResearchValidationIssue, ...] = ()
    metrics: Mapping[str, float | int | str | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "validator_id": self.validator_id,
            "blockers": tuple(item.to_dict() for item in self.blockers),
            "warnings": tuple(item.to_dict() for item in self.warnings),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ResearchResourceEstimate:
    """Deterministic pre-execution resource estimate for the P0 scope."""

    capability_id: str
    supported: bool
    metrics: Mapping[str, float | int | str | bool | None] = field(default_factory=dict)
    blockers: tuple[ResearchValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "supported": self.supported,
            "metrics": dict(self.metrics),
            "blockers": tuple(item.to_dict() for item in self.blockers),
        }


_MANIFESTS = (
    ResearchCapabilityManifest(
        capability_id=_VQE_ID,
        version="1.0.0",
        provider="flagquantum-native",
        adapter_version="1.0.0",
        domain="ground_state",
        task="variational_energy_minimization",
        description="Solve the bounded two-qubit ground-state problem with VQE.",
        tags=("ground-state", "pauli-hamiltonian", "vqe", "variational"),
        input_schema="flagquantum.ground_state_vqe_request.v1",
        output_schema="flagquantum.ground_state_vqe_result.v1",
        limitations=(
            "exactly two qubits",
            "fixed Hamming weight one",
            "Hartree coefficients",
            "single-excitation ansatz and bounded Adam",
            "local statevector execution",
        ),
        validators=(_EXACT_ID, _VARIANCE_ID),
        fallback_capability_ids=(),
        implementation_status="available",
        maturity_binding=_MATURITY_BINDING,
    ),
    ResearchCapabilityManifest(
        capability_id=_EXACT_ID,
        version="1.0.0",
        provider="flagquantum-native",
        adapter_version="1.0.0",
        domain="ground_state",
        task="exact_sector_diagonalization",
        description="Compute an exact reference in the bounded Hamming sector.",
        tags=("exact", "ground-state", "pauli-hamiltonian", "validation"),
        input_schema="flagquantum.exact_ground_state_request.v1",
        output_schema="flagquantum.exact_sector_reference.v1",
        limitations=(
            "exactly two qubits",
            "fixed Hamming weight one",
            "local CPU dense diagonalization",
        ),
        validators=(),
        fallback_capability_ids=(),
        implementation_status="available",
        maturity_binding=_MATURITY_BINDING,
    ),
    ResearchCapabilityManifest(
        capability_id=_VARIANCE_ID,
        version="1.0.0",
        provider="flagquantum-native",
        adapter_version="1.0.0",
        domain="validation",
        task="energy_variance",
        description="Validate a final state with the Hamiltonian energy variance.",
        tags=("energy", "ground-state", "validation", "variance"),
        input_schema="flagquantum.energy_variance_request.v1",
        output_schema="flagquantum.energy_variance_reference.v1",
        limitations=(
            "exactly two qubits",
            "normalized full statevector evidence",
            "local CPU validation",
        ),
        validators=(),
        fallback_capability_ids=(),
        implementation_status="available",
        maturity_binding=_MATURITY_BINDING,
    ),
)
_MANIFESTS_BY_ID = MappingProxyType(
    {manifest.capability_id: manifest for manifest in _MANIFESTS}
)


def _normalized_query(
    query: ResearchCapabilityQuery | Mapping[str, Any] | None,
) -> ResearchCapabilityQuery:
    if query is None:
        return ResearchCapabilityQuery()
    if isinstance(query, ResearchCapabilityQuery):
        return query
    if not isinstance(query, Mapping):
        raise TypeError("query must be a ResearchCapabilityQuery or mapping")
    allowed = {"capability_id", "domain", "task", "provider", "tags"}
    unknown = sorted(set(query) - allowed)
    if unknown:
        raise ValueError(f"unknown research capability query fields: {unknown}")
    raw_tags = query.get("tags", ())
    if not isinstance(raw_tags, (list, tuple)) or any(
        not isinstance(tag, str) or not tag for tag in raw_tags
    ):
        raise TypeError("query tags must be a sequence of non-empty strings")
    scalar_values: dict[str, str | None] = {}
    for name in ("capability_id", "domain", "task", "provider"):
        value = query.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise TypeError(f"query {name} must be a non-empty string or None")
        scalar_values[name] = value
    return ResearchCapabilityQuery(
        **scalar_values,
        tags=tuple(raw_tags),
    )


def search_research_capabilities(
    query: ResearchCapabilityQuery | Mapping[str, Any] | None = None,
) -> tuple[ResearchCapabilityCandidate, ...]:
    """Return deterministic candidates satisfying every supplied filter."""

    normalized = _normalized_query(query)
    candidates: list[ResearchCapabilityCandidate] = []
    required_tags = frozenset(normalized.tags)
    for manifest in _MANIFESTS:
        if normalized.capability_id not in (None, manifest.capability_id):
            continue
        if normalized.domain not in (None, manifest.domain):
            continue
        if normalized.task not in (None, manifest.task):
            continue
        if normalized.provider not in (None, manifest.provider):
            continue
        if not required_tags.issubset(manifest.tags):
            continue
        supplied_filters = sum(
            value is not None
            for value in (
                normalized.capability_id,
                normalized.domain,
                normalized.task,
                normalized.provider,
            )
        ) + len(required_tags)
        score = min(1.0, 0.5 + 0.1 * supplied_filters)
        candidates.append(
            ResearchCapabilityCandidate(
                capability_id=manifest.capability_id,
                version=manifest.version,
                provider=manifest.provider,
                match_score=score,
                applicability=(
                    "supported"
                    if manifest.implementation_status == "available"
                    else "planned"
                ),
                validators=manifest.validators,
                fallback_capability_ids=manifest.fallback_capability_ids,
                reason="all supplied structured filters matched",
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (-item.match_score, item.capability_id))
    )


def describe_research_capability(
    capability_id: str, version: str | None = None
) -> ResearchCapabilityManifest:
    """Return the manifest for one exact research capability identity."""

    if not isinstance(capability_id, str) or not capability_id:
        raise TypeError("capability_id must be a non-empty string")
    try:
        manifest = _MANIFESTS_BY_ID[capability_id]
    except KeyError as exc:
        raise KeyError(f"unknown research capability: {capability_id!r}") from exc
    if version is not None and version != manifest.version:
        raise KeyError(
            f"research capability {capability_id!r} has no version {version!r}"
        )
    return manifest


def _unsupported_report(
    capability_id: str, code: str, message: str, context: Mapping[str, Any]
) -> ResearchValidationReport:
    return ResearchValidationReport(
        passed=False,
        validator_id=f"{capability_id}.input",
        blockers=(ResearchValidationIssue(code, message, dict(context)),),
    )


def validate_research_capability_input(
    capability_id: str, request: object
) -> ResearchValidationReport:
    """Validate one typed request without executing numerical work."""

    describe_research_capability(capability_id)
    request = _normalize_research_request(capability_id, request)
    if capability_id == _VQE_ID:
        if not isinstance(request, GroundStateVQERequest):
            return _unsupported_report(
                capability_id,
                "INVALID_REQUEST_TYPE",
                "ground_state.solve.vqe requires GroundStateVQERequest",
                {"received_type": type(request).__name__},
            )
        reason = validate_ground_state_vqe_request(request)
        if reason is not None:
            return _unsupported_report(
                capability_id, reason.code, reason.message, reason.context
            )
        return ResearchValidationReport(True, f"{capability_id}.input")

    if capability_id == _EXACT_ID:
        if not isinstance(request, ExactGroundStateRequest):
            return _unsupported_report(
                capability_id,
                "INVALID_REQUEST_TYPE",
                "ground_state.solve.exact requires ExactGroundStateRequest",
                {"received_type": type(request).__name__},
            )
        probe = GroundStateVQERequest(
            hamiltonian=request.hamiltonian,
            fixed_hamming_weight=request.fixed_hamming_weight,
        )
        reason = validate_ground_state_vqe_request(probe)
        if reason is not None:
            return _unsupported_report(
                capability_id, reason.code, reason.message, reason.context
            )
        return ResearchValidationReport(True, f"{capability_id}.input")

    if not isinstance(request, EnergyVarianceRequest):
        return _unsupported_report(
            capability_id,
            "INVALID_REQUEST_TYPE",
            "validate.energy_variance requires EnergyVarianceRequest",
            {"received_type": type(request).__name__},
        )
    probe = GroundStateVQERequest(
        hamiltonian=request.hamiltonian,
        fixed_hamming_weight=1,
    )
    reason = validate_ground_state_vqe_request(probe)
    if reason is not None:
        return _unsupported_report(
            capability_id, reason.code, reason.message, reason.context
        )
    if (
        not isinstance(request.tolerance, (int, float))
        or isinstance(request.tolerance, bool)
        or not math.isfinite(float(request.tolerance))
        or request.tolerance < 0
    ):
        return _unsupported_report(
            capability_id,
            "INVALID_VARIANCE_TOLERANCE",
            "variance tolerance must be finite and non-negative",
            {"received": request.tolerance},
        )
    expected_size = 1 << request.hamiltonian.n_qubits
    if (
        len(request.statevector.basis_states) != expected_size
        or len(request.statevector.amplitudes) != expected_size
    ):
        return _unsupported_report(
            capability_id,
            "UNSUPPORTED_STATEVECTOR_SHAPE",
            "statevector evidence must cover every computational-basis state",
            {
                "expected_amplitudes": expected_size,
                "received_amplitudes": len(request.statevector.amplitudes),
            },
        )
    amplitude_norm_squared = sum(
        item.real * item.real + item.imag * item.imag
        for item in request.statevector.amplitudes
    )
    if not math.isfinite(amplitude_norm_squared) or not math.isclose(
        amplitude_norm_squared, 1.0, rel_tol=0.0, abs_tol=2e-6
    ):
        return _unsupported_report(
            capability_id,
            "UNSUPPORTED_STATEVECTOR_NORM",
            "statevector evidence must be finite and normalized",
            {"received_norm_squared": amplitude_norm_squared},
        )
    return ResearchValidationReport(True, f"{capability_id}.input")


def estimate_research_capability(
    capability_id: str,
    request: object,
    target: str | None = None,
) -> ResearchResourceEstimate:
    """Estimate bounded local resources after input validation."""

    request = _normalize_research_request(capability_id, request)
    validation = validate_research_capability_input(capability_id, request)
    if not validation.passed:
        return ResearchResourceEstimate(
            capability_id=capability_id,
            supported=False,
            blockers=validation.blockers,
        )
    if target not in (None, "local-cpu"):
        blocker = ResearchValidationIssue(
            "UNSUPPORTED_TARGET",
            "the P0 research capabilities support only local-cpu",
            {"received": target},
        )
        return ResearchResourceEstimate(
            capability_id=capability_id,
            supported=False,
            blockers=(blocker,),
        )

    if capability_id == _VQE_ID:
        assert isinstance(request, GroundStateVQERequest)
        return ResearchResourceEstimate(
            capability_id=capability_id,
            supported=True,
            metrics={
                "execution_mode": "local_statevector",
                "statevector_amplitudes": 4,
                "statevector_bytes_complex64": 32,
                "hamiltonian_terms": len(request.hamiltonian.terms),
                "trainable_parameters": 1,
                "optimizer_steps": request.optimizer.max_evaluations - 1,
                "objective_evaluations": request.optimizer.max_evaluations,
            },
        )
    if capability_id == _EXACT_ID:
        assert isinstance(request, ExactGroundStateRequest)
        return ResearchResourceEstimate(
            capability_id=capability_id,
            supported=True,
            metrics={
                "execution_mode": "local_dense_eigendecomposition",
                "full_matrix_dimension": 4,
                "sector_matrix_dimension": 2,
                "matrix_bytes_complex128": 256,
                "hamiltonian_terms": len(request.hamiltonian.terms),
            },
        )
    assert isinstance(request, EnergyVarianceRequest)
    return ResearchResourceEstimate(
        capability_id=capability_id,
        supported=True,
        metrics={
            "execution_mode": "local_dense_validation",
            "statevector_amplitudes": 4,
            "matrix_bytes_complex128": 256,
            "hamiltonian_terms": len(request.hamiltonian.terms),
        },
    )


__all__ = (
    "EnergyVarianceRequest",
    "ExactGroundStateRequest",
    "ResearchCapabilityCandidate",
    "ResearchCapabilityManifest",
    "ResearchCapabilityQuery",
    "ResearchResourceEstimate",
    "ResearchValidationIssue",
    "ResearchValidationReport",
    "describe_research_capability",
    "estimate_research_capability",
    "search_research_capabilities",
    "validate_research_capability_input",
)
