"""Unit coverage for detector and observable layouts and the circuit builder."""

from __future__ import annotations

import pytest

from flagquantum.qec.circuit import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


class _TwoQubitParityCode:
    """A second code, to prove the builder is not pinned to the repetition code."""

    distance = 2
    num_data_qubits = 2
    num_ancilla_qubits = 1
    data_wires = (0, 1)
    ancilla_wires = (2,)
    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        ),
    )
    stabilizers = (Pauli(z_wires=(0, 1)),)
    logical_observables = (Pauli(z_wires=(0, 1)),)


def test_detector_count_matches_the_stated_formula() -> None:
    for distance, rounds in ((3, 3), (2, 1), (5, 5), (7, 7)):
        built = build_memory_circuit(RepetitionCode(distance), rounds=rounds)
        assert len(built.detectors) == (distance - 1) * (rounds + 1)


def test_detector_indices_are_dense_and_ordered() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert tuple(item.index for item in built.detectors.detectors) == tuple(range(8))


def test_round_zero_detectors_reference_only_that_round() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    first, second = built.detectors.detectors[0], built.detectors.detectors[1]
    assert first.parity == (MeasurementRef(0, 3),)
    assert second.parity == (MeasurementRef(0, 4),)


def test_later_round_detectors_compare_with_the_previous_round() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    third = built.detectors.detectors[2]
    assert set(third.parity) == {MeasurementRef(1, 3), MeasurementRef(0, 3)}


def test_final_boundary_detectors_join_the_last_round_to_the_data_readout() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    final_left, final_right = built.detectors.detectors[6], built.detectors.detectors[7]
    assert set(final_left.parity) == {
        MeasurementRef(2, 3),
        MeasurementRef(None, 0),
        MeasurementRef(None, 1),
    }
    assert set(final_right.parity) == {
        MeasurementRef(2, 4),
        MeasurementRef(None, 1),
        MeasurementRef(None, 2),
    }


def test_observable_layout_declares_the_terminal_data_parity() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert len(built.observables) == 1
    (observable,) = built.observables.observables
    assert observable.index == 0
    assert observable.pauli == Pauli(z_wires=(0, 1, 2))
    assert observable.measurement_parity == (
        MeasurementRef(None, 0),
        MeasurementRef(None, 1),
        MeasurementRef(None, 2),
    )


def test_source_declares_every_check_and_returns_the_last_measurement() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert "qp.CNOT(wires=[0, 3])" in built.source
    assert "qp.CNOT(wires=[1, 4])" in built.source
    assert "qp.measure(wires=4)" in built.source
    assert "qp.reset(wires=4)" in built.source
    assert built.source.rstrip().endswith("return last")


def test_builder_accepts_a_second_code_without_repetition_assumptions() -> None:
    built = build_memory_circuit(_TwoQubitParityCode(), rounds=1)

    assert built.code.distance == 2
    assert len(built.detectors) == 2
    assert "qp.CNOT(wires=[0, 2])" in built.source
    assert "qp.CNOT(wires=[1, 2])" in built.source


def test_builder_rounds_the_round_count_onto_the_record() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=4)

    assert built.rounds == 4
    assert isinstance(built, MemoryCircuit)


def test_builder_rejects_a_non_code_object() -> None:
    with pytest.raises(TypeError, match="StabilizerCode"):
        build_memory_circuit(object(), rounds=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("rounds", (0, -1))
def test_builder_rejects_a_non_positive_round_count(rounds: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_memory_circuit(RepetitionCode(3), rounds=rounds)


def test_builder_rejects_a_non_integer_round_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        build_memory_circuit(RepetitionCode(3), rounds=2.5)  # type: ignore[arg-type]


def test_memory_circuit_rejects_a_mismatched_detector_count() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    with pytest.raises(ValueError, match="detector count"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=DetectorLayout(built.detectors.detectors[:-1]),
            observables=built.observables,
        )


def test_memory_circuit_rejects_empty_source() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    with pytest.raises(ValueError, match="source"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source="   ",
            detectors=built.detectors,
            observables=built.observables,
        )


def test_measurement_reference_accepts_a_terminal_readout() -> None:
    assert MeasurementRef(None, 0).round_index is None


@pytest.mark.parametrize("round_index", (-1,))
def test_measurement_reference_rejects_a_negative_round(round_index: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MeasurementRef(round_index, 0)


def test_measurement_reference_rejects_a_negative_wire() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MeasurementRef(0, -1)


def test_measurement_reference_rejects_a_non_integer_wire() -> None:
    with pytest.raises(TypeError, match="integer"):
        MeasurementRef(0, 1.5)  # type: ignore[arg-type]


def test_detector_rejects_an_empty_parity() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        Detector(index=0, parity=())


def test_detector_rejects_a_repeated_measurement() -> None:
    reference = MeasurementRef(0, 3)

    with pytest.raises(ValueError, match="repeat"):
        Detector(index=0, parity=(reference, reference))


def test_detector_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        Detector(index=-1, parity=(MeasurementRef(0, 3),))


def test_detector_layout_requires_a_dense_ordering() -> None:
    with pytest.raises(ValueError, match="dense"):
        DetectorLayout(
            (Detector(index=1, parity=(MeasurementRef(0, 3),)),),
        )


def test_detector_layout_requires_at_least_one_detector() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorLayout(())


def test_logical_observable_rejects_an_identity_operator() -> None:
    with pytest.raises(ValueError, match="identity"):
        LogicalObservable(index=0, pauli=Pauli(), measurement_parity=())


def test_logical_observable_rejects_an_x_type_operator() -> None:
    with pytest.raises(ValueError, match="Z-type"):
        LogicalObservable(
            index=0,
            pauli=Pauli(x_wires=(0,)),
            measurement_parity=(MeasurementRef(None, 0),),
        )


def test_observable_layout_requires_a_dense_ordering() -> None:
    with pytest.raises(ValueError, match="dense"):
        ObservableLayout(
            (
                LogicalObservable(
                    index=2,
                    pauli=Pauli(z_wires=(0,)),
                    measurement_parity=(MeasurementRef(None, 0),),
                ),
            )
        )


def test_observable_layout_requires_at_least_one_observable() -> None:
    with pytest.raises(ValueError, match="at least one observable"):
        ObservableLayout(())
