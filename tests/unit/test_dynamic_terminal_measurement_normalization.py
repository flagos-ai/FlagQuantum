"""Pure terminal-measurement normalization contracts."""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ExecutionError
from flagquantum.runtime.dynamic.terminal_measurement import (
    normalize_terminal_measurements,
)

pytestmark = pytest.mark.unit


def _terminal_program(*, permuted: bool = False) -> DynamicCircuit:
    circuit = DynamicCircuit(2).h(0).cx(0, 1)
    if permuted:
        return circuit.measure(1, classical_bit=0).measure(0, classical_bit=1)
    return circuit.measure(0, classical_bit=0).measure(1, classical_bit=1)


def test_normalization_preserves_source_and_builds_an_inspectable_prefix() -> None:
    circuit = _terminal_program(permuted=True)
    source = circuit.to_ir()

    normalized = normalize_terminal_measurements(circuit, outputs=fq.counts())

    assert normalized is not None
    assert normalized.source_ir is source
    assert [item.name for item in source.instructions] == [
        "h",
        "cx",
        "measure",
        "measure",
    ]
    assert [item.name for item in normalized.unitary_ir.instructions] == ["h", "cx"]
    assert normalized.measurement_mapping == ((0, 1), (1, 0))
    assert len(normalized.normalization_id) == 64
    assert normalized.summary()["measurement_mapping"] == [
        {"qubit": 0, "classical_bit": 1},
        {"qubit": 1, "classical_bit": 0},
    ]


def test_normalization_identity_is_deterministic_and_binds_output_name() -> None:
    circuit = _terminal_program()
    first = normalize_terminal_measurements(circuit, outputs=fq.counts(name="first"))
    repeated = normalize_terminal_measurements(circuit, outputs=fq.counts(name="first"))
    renamed = normalize_terminal_measurements(circuit, outputs=fq.counts(name="second"))

    assert first is not None and repeated is not None and renamed is not None
    assert first.normalization_id == repeated.normalization_id
    assert first.normalization_id != renamed.normalization_id


def test_permuted_mapping_projects_qubit_order_into_classical_order() -> None:
    normalized = normalize_terminal_measurements(
        _terminal_program(permuted=True),
        outputs=None,
    )

    assert normalized is not None
    assert normalized.project_counts({"10": 7, "01": 5}) == {"01": 7, "10": 5}


@pytest.mark.parametrize(
    "program",
    (
        DynamicCircuit(2).h(0).measure(0, classical_bit=0),
        DynamicCircuit(2).measure(0, classical_bit=0).measure(0, classical_bit=1),
        DynamicCircuit(2).measure(0, classical_bit=0).measure(1, classical_bit=2),
        DynamicCircuit(2).measure(0, classical_bit=0).x(1),
        DynamicCircuit(2).reset(0).measure(0).measure(1),
        DynamicCircuit(2)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .measure(1, classical_bit=1),
    ),
)
def test_non_v1_measurement_shapes_are_not_normalized(
    program: DynamicCircuit,
) -> None:
    assert normalize_terminal_measurements(program, outputs=None) is None


@pytest.mark.parametrize(
    "output",
    (
        fq.samples(),
        fq.counts(qubits=0),
        fq.expectation(fq.Z(0)),
        (fq.counts(), fq.counts()),
    ),
)
def test_v1_rejects_incompatible_outputs_before_execution(output: object) -> None:
    with pytest.raises(ValueError, match="one full-register fq.counts"):
        normalize_terminal_measurements(_terminal_program(), outputs=output)


@pytest.mark.parametrize("counts", ({"0": 1}, {"2x": 1}, {"00": -1}))
def test_result_projection_rejects_invalid_provider_counts(
    counts: dict[str, int],
) -> None:
    normalized = normalize_terminal_measurements(_terminal_program(), outputs=None)

    assert normalized is not None
    with pytest.raises(ExecutionError, match="invalid provider"):
        normalized.project_counts(counts)
