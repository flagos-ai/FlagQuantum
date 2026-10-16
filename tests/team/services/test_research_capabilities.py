from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.services import (
    ComplexAmplitude,
    EnergyVarianceRequest,
    ExactGroundStateRequest,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    ResearchCapabilityQuery,
    StatevectorEvidence,
    describe_research_capability,
    energy_variance_reference,
    estimate_research_capability,
    run_ground_state_vqe,
    search_research_capabilities,
    validate_research_capability_input,
)

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]


def _h2_hamiltonian() -> PauliSum:
    return PauliSum(
        n_qubits=2,
        terms=(
            PauliSumTerm(-1.052373245772859, "II"),
            PauliSumTerm(0.39793742484318045, "ZI"),
            PauliSumTerm(-0.39793742484318045, "IZ"),
            PauliSumTerm(-0.01128010425623538, "ZZ"),
            PauliSumTerm(0.18093119978423156, "XX"),
        ),
    )


def _vqe_request() -> GroundStateVQERequest:
    return GroundStateVQERequest(
        hamiltonian=_h2_hamiltonian(),
        fixed_hamming_weight=1,
    )


def test_catalog_exposes_exactly_the_p0_research_capabilities() -> None:
    candidates = search_research_capabilities()

    assert tuple(item.capability_id for item in candidates) == (
        "ground_state.solve.exact",
        "ground_state.solve.vqe",
        "validate.energy_variance",
    )
    assert all(item.applicability == "supported" for item in candidates)
    assert all(item.fallback_capability_ids == () for item in candidates)


def test_catalog_uses_structured_filters_and_stable_tie_breaking() -> None:
    candidates = search_research_capabilities(
        ResearchCapabilityQuery(domain="ground_state", tags=("ground-state",))
    )

    assert tuple(item.capability_id for item in candidates) == (
        "ground_state.solve.exact",
        "ground_state.solve.vqe",
    )
    assert candidates[0].match_score == candidates[1].match_score
    assert search_research_capabilities({"task": "energy_variance"})[
        0
    ].capability_id == ("validate.energy_variance")


def test_catalog_rejects_unknown_query_fields() -> None:
    with pytest.raises(ValueError, match="unknown research capability query fields"):
        search_research_capabilities({"prompt": "find a VQE tool"})


def test_manifest_keeps_operation_and_provider_identity_separate() -> None:
    manifest = describe_research_capability("ground_state.solve.vqe")

    assert manifest.capability_id == "ground_state.solve.vqe"
    assert manifest.provider == "flagquantum-native"
    assert manifest.validators == (
        "ground_state.solve.exact",
        "validate.energy_variance",
    )
    assert manifest.fallback_capability_ids == ()
    assert manifest.maturity_binding == "research_capability_services"


def test_every_manifest_binding_resolves_to_production_maturity_authority() -> None:
    maturity = tomllib.loads(
        (_ROOT / "capability-maturity.toml").read_text(encoding="utf-8")
    )["capabilities"]

    for candidate in search_research_capabilities():
        manifest = describe_research_capability(candidate.capability_id)
        assert manifest.maturity_binding in maturity
        assert maturity[manifest.maturity_binding]["level"] == "production_supported"


def test_input_validation_fails_before_execution() -> None:
    request = GroundStateVQERequest(
        hamiltonian=_h2_hamiltonian(),
        fixed_hamming_weight=0,
    )

    report = validate_research_capability_input("ground_state.solve.vqe", request)

    assert report.passed is False
    assert report.blockers[0].code == "UNSUPPORTED_SECTOR"
    assert report.validator_id == "ground_state.solve.vqe.input"


def test_estimates_are_available_before_numerical_execution() -> None:
    request = _vqe_request()

    estimate = estimate_research_capability("ground_state.solve.vqe", request)

    assert estimate.supported is True
    assert estimate.metrics["objective_evaluations"] == 121
    assert estimate.metrics["statevector_bytes_complex64"] == 32
    assert estimate.metrics["hamiltonian_terms"] == 5


def test_json_shaped_requests_are_decoded_by_the_service_boundary() -> None:
    request = _vqe_request().to_dict()

    validation = validate_research_capability_input("ground_state.solve.vqe", request)
    estimate = estimate_research_capability("ground_state.solve.vqe", request)

    assert validation.passed is True
    assert estimate.supported is True
    assert estimate.metrics["objective_evaluations"] == 121


def test_json_request_decoder_rejects_unknown_nested_fields() -> None:
    request = _vqe_request().to_dict()
    request["hamiltonian"]["vendor_hint"] = "ignored"

    with pytest.raises(ValueError, match="unknown Pauli sum fields"):
        validate_research_capability_input("ground_state.solve.vqe", request)


def test_estimator_rejects_targets_outside_the_p0_scope() -> None:
    estimate = estimate_research_capability(
        "ground_state.solve.exact",
        ExactGroundStateRequest(_h2_hamiltonian(), fixed_hamming_weight=1),
        target="remote-qpu",
    )

    assert estimate.supported is False
    assert estimate.blockers[0].code == "UNSUPPORTED_TARGET"


def test_energy_variance_is_computed_from_serialized_final_state() -> None:
    request = _vqe_request()
    result = run_ground_state_vqe(request)
    assert result.statevector is not None

    validation_request = EnergyVarianceRequest(
        hamiltonian=request.hamiltonian,
        statevector=result.statevector,
        tolerance=1e-5,
    )
    preflight = validate_research_capability_input(
        "validate.energy_variance", validation_request
    )
    reference = energy_variance_reference(
        validation_request.hamiltonian,
        validation_request.statevector,
        tolerance=validation_request.tolerance,
    )

    assert preflight.passed is True
    assert reference.status == "completed"
    assert reference.passed is True
    assert reference.variance is not None
    assert reference.variance <= validation_request.tolerance
    assert reference.energy == pytest.approx(result.energy, abs=2e-6)


def test_variance_preflight_rejects_a_partial_statevector() -> None:
    partial = StatevectorEvidence(
        basis_states=("00",),
        amplitudes=(ComplexAmplitude(1.0, 0.0),),
        norm=1.0,
    )

    report = validate_research_capability_input(
        "validate.energy_variance",
        EnergyVarianceRequest(_h2_hamiltonian(), partial),
    )

    assert report.passed is False
    assert report.blockers[0].code == "UNSUPPORTED_STATEVECTOR_SHAPE"


def test_variance_preflight_rejects_an_unnormalized_statevector() -> None:
    invalid = StatevectorEvidence(
        basis_states=("00", "01", "10", "11"),
        amplitudes=(
            ComplexAmplitude(2.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
        ),
        norm=2.0,
    )

    report = validate_research_capability_input(
        "validate.energy_variance",
        EnergyVarianceRequest(_h2_hamiltonian(), invalid),
    )

    assert report.passed is False
    assert report.blockers[0].code == "UNSUPPORTED_STATEVECTOR_NORM"


def test_variance_tolerance_must_be_finite_and_non_negative() -> None:
    full = StatevectorEvidence(
        basis_states=("00", "01", "10", "11"),
        amplitudes=(
            ComplexAmplitude(1.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
        ),
        norm=1.0,
    )

    for tolerance in (-1.0, math.inf, math.nan):
        report = validate_research_capability_input(
            "validate.energy_variance",
            EnergyVarianceRequest(_h2_hamiltonian(), full, tolerance=tolerance),
        )
        assert report.passed is False
        assert report.blockers[0].code == "INVALID_VARIANCE_TOLERANCE"
