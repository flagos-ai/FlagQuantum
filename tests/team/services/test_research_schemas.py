from __future__ import annotations

from dataclasses import asdict

import pytest

from flagquantum.services import (
    ComplexAmplitude,
    EnergyVarianceRequest,
    ExactGroundStateRequest,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    ResearchCapabilityError,
    StatevectorEvidence,
    describe_research_capability,
    describe_research_schema,
    execute_research_capability,
    list_research_schemas,
    search_research_capabilities,
)

pytestmark = pytest.mark.unit

_SCHEMA_IDS = (
    "flagquantum.pauli_sum.v1",
    "flagquantum.ground_state_vqe_request.v1",
    "flagquantum.ground_state_vqe_result.v1",
    "flagquantum.exact_ground_state_request.v1",
    "flagquantum.exact_sector_reference.v1",
    "flagquantum.energy_variance_request.v1",
    "flagquantum.energy_variance_reference.v1",
    "flagquantum.research_capability_manifest.v1",
    "flagquantum.research_validation_report.v1",
    "flagquantum.research_capability_error.v1",
    "flagquantum.research_capability_evidence_bundle.v1",
)


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


def _assert_closed_shape(value: object, schema: dict[str, object]) -> None:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        if value is None:
            assert any(item == {"type": "null"} for item in alternatives)
            return
        non_null = [item for item in alternatives if item != {"type": "null"}]
        assert len(non_null) == 1
        _assert_closed_shape(value, non_null[0])
        return

    if schema.get("type") == "object" and "properties" in schema:
        assert isinstance(value, dict)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert schema.get("additionalProperties") is False
        assert set(value) == set(schema.get("required", ()))
        assert set(value) == set(properties)
        for name, child_schema in properties.items():
            assert isinstance(child_schema, dict)
            _assert_closed_shape(value[name], child_schema)
        return

    if schema.get("type") == "array":
        assert isinstance(value, (list, tuple))
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item in value:
                _assert_closed_shape(item, item_schema)


def _document(schema_id: str) -> dict[str, object]:
    document = describe_research_schema(schema_id).to_dict()["document"]
    assert isinstance(document, dict)
    return document


def test_catalog_has_a_stable_closed_identity_set() -> None:
    schemas = list_research_schemas()

    assert tuple(item.schema_id for item in schemas) == _SCHEMA_IDS
    assert all(item.version == "1.0.0" for item in schemas)
    for item in schemas:
        document = item.to_dict()["document"]
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["$id"] == item.schema_id
        assert document["type"] == "object"
        assert document["additionalProperties"] is False


def test_every_manifest_input_and_output_schema_resolves() -> None:
    installed = {item.schema_id for item in list_research_schemas()}

    for candidate in search_research_capabilities():
        manifest = describe_research_capability(candidate.capability_id)
        assert manifest.input_schema in installed
        assert manifest.output_schema in installed


def test_returned_documents_are_isolated_from_catalog_state() -> None:
    first = describe_research_schema("flagquantum.pauli_sum.v1")
    first.document["title"] = "mutated by caller"

    second = describe_research_schema("flagquantum.pauli_sum.v1")

    assert second.document["title"] == "Pauli sum"


@pytest.mark.parametrize("schema_id", ("", "flagquantum.unknown.v1", None))
def test_schema_lookup_fails_closed(schema_id: object) -> None:
    expected = (
        TypeError if not isinstance(schema_id, str) or not schema_id else KeyError
    )
    with pytest.raises(expected):
        describe_research_schema(schema_id)  # type: ignore[arg-type]


def test_published_input_shapes_match_installed_request_values() -> None:
    hamiltonian = _hamiltonian()
    statevector = StatevectorEvidence(
        basis_states=("00", "01", "10", "11"),
        amplitudes=(
            ComplexAmplitude(1.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
            ComplexAmplitude(0.0, 0.0),
        ),
        norm=1.0,
    )
    examples = {
        "flagquantum.ground_state_vqe_request.v1": GroundStateVQERequest(
            hamiltonian, fixed_hamming_weight=1
        ).to_dict(),
        "flagquantum.exact_ground_state_request.v1": asdict(
            ExactGroundStateRequest(hamiltonian, fixed_hamming_weight=1)
        ),
        "flagquantum.energy_variance_request.v1": asdict(
            EnergyVarianceRequest(hamiltonian, statevector)
        ),
    }

    for schema_id, value in examples.items():
        _assert_closed_shape(value, _document(schema_id))


def test_published_output_and_envelope_shapes_match_runtime_values() -> None:
    hamiltonian = _hamiltonian()
    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        GroundStateVQERequest(hamiltonian, fixed_hamming_weight=1),
    )
    exact = execute_research_capability(
        "ground_state.solve.exact",
        ExactGroundStateRequest(hamiltonian, fixed_hamming_weight=1),
    )
    variance = execute_research_capability(
        "validate.energy_variance",
        EnergyVarianceRequest(
            hamiltonian,
            StatevectorEvidence(
                basis_states=("00", "01", "10", "11"),
                amplitudes=(
                    ComplexAmplitude(1.0, 0.0),
                    ComplexAmplitude(0.0, 0.0),
                    ComplexAmplitude(0.0, 0.0),
                    ComplexAmplitude(0.0, 0.0),
                ),
                norm=1.0,
            ),
        ),
    )
    manifest = describe_research_capability("ground_state.solve.vqe")
    error = ResearchCapabilityError(
        code="EXAMPLE",
        stage="execution",
        retryable=False,
        message="example",
    )

    _assert_closed_shape(
        manifest.to_dict(), _document("flagquantum.research_capability_manifest.v1")
    )
    _assert_closed_shape(
        error.to_dict(), _document("flagquantum.research_capability_error.v1")
    )
    _assert_closed_shape(
        evidence.to_dict(),
        _document("flagquantum.research_capability_evidence_bundle.v1"),
    )
    _assert_closed_shape(
        evidence.result, _document("flagquantum.ground_state_vqe_result.v1")
    )
    _assert_closed_shape(
        exact.result, _document("flagquantum.exact_sector_reference.v1")
    )
    _assert_closed_shape(
        variance.result, _document("flagquantum.energy_variance_reference.v1")
    )
    _assert_closed_shape(
        evidence.validation[0].to_dict(),
        _document("flagquantum.research_validation_report.v1"),
    )
