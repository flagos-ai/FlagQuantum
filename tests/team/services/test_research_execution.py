from __future__ import annotations

import json

import pytest

from flagquantum.services import (
    ComplexAmplitude,
    EnergyVarianceRequest,
    ExactGroundStateRequest,
    GroundStateOptimizer,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    ResearchExecutionOptions,
    StatevectorEvidence,
    execute_research_capability,
    validate_research_evidence,
)

pytestmark = pytest.mark.unit


def _hamiltonian() -> PauliSum:
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


def _request(*, fixed_hamming_weight: int = 1) -> GroundStateVQERequest:
    return GroundStateVQERequest(
        hamiltonian=_hamiltonian(),
        fixed_hamming_weight=fixed_hamming_weight,
    )


def test_vqe_execution_returns_validated_evidence_bundle() -> None:
    evidence = execute_research_capability("ground_state.solve.vqe", _request())

    assert evidence.status == "succeeded"
    assert evidence.capability.capability_id == "ground_state.solve.vqe"
    assert evidence.capability.provider == "flagquantum-native"
    assert tuple(item.validator_id for item in evidence.validation) == (
        "ground_state.solve.exact",
        "validate.energy_variance",
    )
    assert all(item.passed for item in evidence.validation)
    assert evidence.attempts[0].status == "succeeded"
    assert evidence.execution["fallback_used"] is False
    assert evidence.metrics["optimizer_converged"] is True
    assert evidence.metrics["optimizer_budget_exhausted"] is False
    assert evidence.warnings == ()
    assert evidence.execution["call_counts"] == {"provider": 1, "validators": 2}
    assert set(evidence.execution["stage_durations_seconds"]) == {
        "input_validation",
        "resource_estimation",
        "environment_preflight",
        "execution",
        "scientific_validation",
    }
    assert all(
        duration >= 0.0
        for duration in evidence.execution["stage_durations_seconds"].values()
    )
    assert evidence.provenance["fallback_used"] is False
    assert evidence.provenance["deterministic"] is True
    assert evidence.provenance["random_seed"] is None
    assert len(evidence.provenance["input_sha256"]) == 64
    assert len(evidence.provenance["content_sha256"]) == 64
    json.dumps(evidence.to_dict(), allow_nan=False)


def test_vqe_execution_accepts_the_public_json_request_shape() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        _request().to_dict(),
    )

    assert evidence.status == "succeeded"
    assert evidence.capability.provider == "flagquantum-native"
    assert all(report.passed for report in evidence.validation)


def test_mandatory_scientific_failure_is_not_reported_as_success() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        _request(),
        ResearchExecutionOptions(
            exact_energy_tolerance=0.0,
            energy_variance_tolerance=0.0,
        ),
    )

    assert evidence.status == "validation_failed"
    assert evidence.attempts[0].status == "succeeded"
    assert evidence.error is not None
    assert evidence.error.code == "SCIENTIFIC_VALIDATION_FAILED"
    assert evidence.error.stage == "scientific_validation"
    assert any(not report.passed for report in evidence.validation)


def test_invalid_input_is_refused_before_provider_attempt() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.vqe", _request(fixed_hamming_weight=0)
    )

    assert evidence.status == "unsupported"
    assert evidence.attempts == ()
    assert evidence.result == {}
    assert evidence.error is not None
    assert evidence.error.code == "UNSUPPORTED_SECTOR"
    assert evidence.error.stage == "input_validation"
    assert evidence.execution["call_counts"] == {"provider": 0, "validators": 0}
    assert set(evidence.execution["stage_durations_seconds"]) == {"input_validation"}


def test_unsupported_target_is_refused_during_resource_estimation() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.exact",
        ExactGroundStateRequest(_hamiltonian(), fixed_hamming_weight=1),
        {"target": "remote-qpu"},
    )

    assert evidence.status == "unsupported"
    assert evidence.attempts == ()
    assert evidence.error is not None
    assert evidence.error.code == "UNSUPPORTED_TARGET"
    assert evidence.error.stage == "resource_estimation"


def test_direct_exact_execution_uses_the_same_bundle_shape() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.exact",
        ExactGroundStateRequest(_hamiltonian(), fixed_hamming_weight=1),
    )

    assert evidence.status == "succeeded"
    assert evidence.result["status"] == "completed"
    assert evidence.validation == ()
    assert evidence.attempts[0].status == "succeeded"


def test_direct_variance_execution_distinguishes_scientific_failure() -> None:
    state = StatevectorEvidence(
        basis_states=("00", "01", "10", "11"),
        amplitudes=(
            ComplexAmplitude(1.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
        ),
        norm=1.0,
    )
    evidence = execute_research_capability(
        "validate.energy_variance",
        EnergyVarianceRequest(_hamiltonian(), state, tolerance=0.0),
    )

    assert evidence.status == "validation_failed"
    assert evidence.attempts[0].status == "succeeded"
    assert evidence.validation[0].passed is False
    assert evidence.error is not None
    assert evidence.error.stage == "scientific_validation"


def test_recorded_validator_evidence_is_integrity_checked() -> None:
    evidence = execute_research_capability("ground_state.solve.vqe", _request())

    report = validate_research_evidence(
        "validate.energy_variance",
        evidence,
    )
    tampered = evidence.to_dict()
    tampered["metrics"]["energy_variance"] = 123.0
    tampered_report = validate_research_evidence(
        "validate.energy_variance",
        tampered,
    )

    assert report.passed is True
    assert tampered_report.passed is False
    assert tampered_report.blockers[0].code == "EVIDENCE_INTEGRITY_MISMATCH"


def test_missing_validator_record_fails_closed() -> None:
    evidence = execute_research_capability(
        "ground_state.solve.exact",
        ExactGroundStateRequest(_hamiltonian(), fixed_hamming_weight=1),
    )

    report = validate_research_evidence("validate.energy_variance", evidence)

    assert report.passed is False
    assert report.blockers[0].code == "VALIDATOR_EVIDENCE_MISSING"


def test_json_round_trip_preserves_verifiable_evidence() -> None:
    evidence = execute_research_capability("ground_state.solve.vqe", _request())
    round_tripped = json.loads(json.dumps(evidence.to_dict(), allow_nan=False))

    report = validate_research_evidence("validate.energy_variance", round_tripped)

    assert report.passed is True


def test_json_client_float_reformatting_preserves_verifiable_evidence() -> None:
    evidence = execute_research_capability("ground_state.solve.vqe", _request())
    encoded = json.dumps(evidence.to_dict(), allow_nan=False)
    scientific = "1.5798166104197264e-06"
    decimal = "0.0000015798166104197264"
    assert scientific in encoded

    report = validate_research_evidence(
        "validate.energy_variance",
        json.loads(encoded.replace(scientific, decimal)),
    )

    assert report.passed is True


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (
            lambda payload: payload.__setitem__("unknown", True),
            "INVALID_EVIDENCE_FIELDS",
        ),
        (
            lambda payload: payload.__setitem__("schema", "flagquantum.unknown.v1"),
            "INVALID_EVIDENCE_SCHEMA",
        ),
        (
            lambda payload: payload["provenance"].__setitem__(
                "python_version", "tampered"
            ),
            "EVIDENCE_INTEGRITY_MISMATCH",
        ),
    ),
)
def test_evidence_validation_rejects_schema_and_provenance_mutation(
    mutation: object, expected_code: str
) -> None:
    evidence = execute_research_capability("ground_state.solve.vqe", _request())
    payload = evidence.to_dict()

    assert callable(mutation)
    mutation(payload)
    report = validate_research_evidence("validate.energy_variance", payload)

    assert report.passed is False
    assert report.blockers[0].code == expected_code


def test_vqe_result_is_reproducible_for_the_same_request() -> None:
    first = execute_research_capability("ground_state.solve.vqe", _request())
    second = execute_research_capability("ground_state.solve.vqe", _request())

    assert first.status == second.status == "succeeded"
    assert first.result["energy"] == second.result["energy"]
    assert first.result["parameters"] == second.result["parameters"]
    assert first.result["convergence_trace"] == second.result["convergence_trace"]
    assert first.validation == second.validation
    assert first.provenance["input_sha256"] == second.provenance["input_sha256"]


def test_vqe_evidence_distinguishes_convergence_from_budget_exhaustion() -> None:
    request = GroundStateVQERequest(
        hamiltonian=_hamiltonian(),
        fixed_hamming_weight=1,
        optimizer=GroundStateOptimizer(max_evaluations=2),
    )

    evidence = execute_research_capability("ground_state.solve.vqe", request)

    assert evidence.status == "validation_failed"
    assert evidence.metrics["optimizer_converged"] is False
    assert evidence.metrics["optimizer_budget_exhausted"] is True
    assert evidence.warnings[0].code == "OPTIMIZER_CONVERGENCE_NOT_ESTABLISHED"
    assert evidence.warnings[0].context["observed_points"] == 2
