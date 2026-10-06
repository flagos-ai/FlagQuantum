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
from flagquantum.qec.codes import CodeCheck, RepetitionCode, SteaneCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


class _TwoQubitParityCode:
    """A second code, to prove the builder is not pinned to the repetition code.

    It declares a logical observable in each basis, as the Steane code does, so
    it is also the smallest record that can be read out either way.
    """

    distance = 2
    num_data_qubits = 2
    num_ancilla_qubits = 1
    num_ancilla_x_qubits = 0
    num_ancilla_z_qubits = 1
    num_x_stabilizers = 0
    num_z_stabilizers = 1
    data_qubits = (0, 1)
    ancilla_qubits = (2,)
    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_qubits=(0, 1)),
            ancilla_qubit=2,
            cnot_qubits=((0, 2), (1, 2)),
        ),
    )
    stabilizers = (Pauli(z_qubits=(0, 1)),)
    logical_observables = (Pauli(z_qubits=(0, 1)), Pauli(x_qubits=(0, 1)))


class _ThreeCheckParityCode:
    """A code with more checks than ``distance - 1``, to pin the count formula."""

    distance = 3
    num_data_qubits = 3
    num_ancilla_qubits = 3
    num_ancilla_x_qubits = 0
    num_ancilla_z_qubits = 3
    num_x_stabilizers = 0
    num_z_stabilizers = 3
    data_qubits = (0, 1, 2)
    ancilla_qubits = (3, 4, 5)
    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_qubits=(0, 1)),
            ancilla_qubit=3,
            cnot_qubits=((0, 3), (1, 3)),
        ),
        CodeCheck(
            index=1,
            stabilizer=Pauli(z_qubits=(1, 2)),
            ancilla_qubit=4,
            cnot_qubits=((1, 4), (2, 4)),
        ),
        CodeCheck(
            index=2,
            stabilizer=Pauli(z_qubits=(0, 2)),
            ancilla_qubit=5,
            cnot_qubits=((0, 5), (2, 5)),
        ),
    )
    stabilizers = (
        Pauli(z_qubits=(0, 1)),
        Pauli(z_qubits=(1, 2)),
        Pauli(z_qubits=(0, 2)),
    )
    logical_observables = (Pauli(z_qubits=(0, 1, 2)),)


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
    assert observable.pauli == Pauli(z_qubits=(0, 1, 2))
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


def test_the_z_basis_source_carries_no_rotation_at_all() -> None:
    """The Z experiment is the runtime's own preparation and readout.

    The runtime starts every qubit in ``|0>`` and returns the terminal sample of
    each data qubit, so a Z-basis experiment needs no gate of its own: the source
    is exactly the round loop, and its first statement is the loop itself.
    """

    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert built.readout_basis == "z"
    assert "qp.H(" not in built.source
    lines = built.source.splitlines()
    assert lines[1] == "    last = False"
    assert lines[2] == "    for round_index in range(rounds):"


def test_the_x_basis_source_rotates_the_data_qubits_on_both_ends() -> None:
    """The X experiment is the Z one conjugated by ``H`` on the data qubits.

    The loop is untouched -- an ancilla's Z-basis readout is its check's
    eigenvalue under either data basis -- so the rotation is one gate per data
    qubit before the loop and the same gates again after it, which is what makes
    the terminal sample a readout in the X basis.
    """

    code = _TwoQubitParityCode()
    built = build_memory_circuit(code, rounds=2, readout_basis="x")

    body = built.source.splitlines()
    prep = ["    qp.H(wires=0)", "    qp.H(wires=1)"]
    assert body[1:3] == prep
    assert body[-3:-1] == prep
    # One anchor either way, so a forced flip has a single round loop to inject
    # into and the loop is not split by the rotation.
    assert built.source.count("for round_index in range(rounds):") == 1
    assert "qp.H(wires=2)" not in built.source


def test_x_type_checks_are_the_ones_deterministic_under_an_x_readout() -> None:
    """The two bases are the same layout read over the two check classes.

    Under a ``|+>`` preparation the code's X-type checks are deterministic in
    round zero and again at the terminal readout, and its Z-type checks are
    deterministic only against the round before, which is the mirror of the
    Z-basis layout rather than a second rule.
    """

    code = SteaneCode()
    z_layout = build_memory_circuit(code, rounds=3, readout_basis="z")
    x_layout = build_memory_circuit(code, rounds=3, readout_basis="x")

    x_ancillas = tuple(
        check.ancilla_qubit for check in code.checks if check.stabilizer.x_qubits
    )
    z_ancillas = tuple(
        check.ancilla_qubit for check in code.checks if check.stabilizer.z_qubits
    )
    assert len(x_ancillas) == len(z_ancillas) == 3

    assert len(x_layout.detectors) == len(z_layout.detectors)
    # Round zero: the readout basis' checks, and only those.
    assert tuple(item.parity for item in x_layout.detectors.detectors[:3]) == tuple(
        (MeasurementRef(0, qubit),) for qubit in x_ancillas
    )
    assert tuple(item.parity for item in z_layout.detectors.detectors[:3]) == tuple(
        (MeasurementRef(0, qubit),) for qubit in z_ancillas
    )
    # Terminal: again the readout basis' checks, joined to their own support.
    assert tuple(item.parity for item in x_layout.detectors.detectors[-3:]) == tuple(
        (MeasurementRef(2, check.ancilla_qubit),)
        + tuple(MeasurementRef(None, qubit) for qubit in check.stabilizer.support)
        for check in code.checks
        if check.stabilizer.x_qubits
    )
    # Between rounds: every check, in both bases.
    assert tuple(item.parity for item in x_layout.detectors.detectors[6:9]) == tuple(
        (MeasurementRef(1, qubit), MeasurementRef(0, qubit)) for qubit in x_ancillas
    )


def test_the_x_basis_observable_is_the_codes_x_type_logical_operator() -> None:
    """The Steane code declares one observable in each basis, so the basis picks one.

    The two share a support here, so the Pauli type is the whole difference and
    the parity that realizes it reads the same data qubits in either case; what
    changes is which basis those terminal samples were taken in.
    """

    code = SteaneCode()
    z_observable = build_memory_circuit(code, rounds=2).observables.observables[0]
    x_observable = build_memory_circuit(
        code, rounds=2, readout_basis="x"
    ).observables.observables[0]

    assert z_observable.pauli == Pauli(z_qubits=(0, 1, 2))
    assert x_observable.pauli == Pauli(x_qubits=(0, 1, 2))
    assert z_observable.measurement_parity == x_observable.measurement_parity


def test_the_readout_basis_must_name_a_basis() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        build_memory_circuit(RepetitionCode(3), rounds=2, readout_basis="y")

    with pytest.raises(TypeError, match="must be a string"):
        build_memory_circuit(
            RepetitionCode(3), rounds=2, readout_basis=0  # type: ignore[arg-type]
        )


def test_a_hand_built_circuit_defaults_to_the_z_basis() -> None:
    """The field is trailing and optional, so an existing caller is unchanged.

    A caller that states the five original fields gets the experiment those
    fields have always described, which is the Z one, rather than a refusal to
    choose.
    """

    built = build_memory_circuit(RepetitionCode(3), rounds=2)
    rebuilt = MemoryCircuit(
        code=built.code,
        rounds=built.rounds,
        source=built.source,
        detectors=built.detectors,
        observables=built.observables,
    )

    assert rebuilt.readout_basis == "z"
    assert rebuilt == built


def test_builder_accepts_a_second_code_without_repetition_assumptions() -> None:
    built = build_memory_circuit(_TwoQubitParityCode(), rounds=1)

    assert built.code.distance == 2
    assert len(built.detectors) == 2
    assert "qp.CNOT(wires=[0, 2])" in built.source
    assert "qp.CNOT(wires=[1, 2])" in built.source


@pytest.mark.parametrize("rounds", (1, 2, 3))
def test_builder_counts_detectors_from_the_declared_checks(rounds: int) -> None:
    built = build_memory_circuit(_ThreeCheckParityCode(), rounds=rounds)

    assert len(built.detectors) == 3 * (rounds + 1)


def test_builder_accepts_a_code_with_more_checks_than_distance_minus_one() -> None:
    built = build_memory_circuit(_ThreeCheckParityCode(), rounds=2)

    assert built.code.distance == 3
    assert len(built.code.checks) == 3
    assert "qp.CNOT(wires=[0, 5])" in built.source
    assert "qp.measure(wires=5)" in built.source


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


def test_logical_observable_accepts_a_pure_x_type_operator() -> None:
    """Both bases are readable, so neither pure Pauli type is refused.

    Which of the two an experiment reads is the experiment's readout basis, which
    is stated on the circuit rather than on the observable record. The record's
    own job is narrower: it refuses an operator that no single readout basis can
    measure, which is the mixed case below.
    """

    observable = LogicalObservable(
        index=0,
        pauli=Pauli(x_qubits=(0,)),
        measurement_parity=(MeasurementRef(None, 0),),
    )

    assert observable.pauli == Pauli(x_qubits=(0,))


def test_logical_observable_rejects_a_mixed_operator() -> None:
    with pytest.raises(ValueError, match="pure X or pure Z"):
        LogicalObservable(
            index=0,
            pauli=Pauli(x_qubits=(0,), z_qubits=(1,)),
            measurement_parity=(MeasurementRef(None, 0), MeasurementRef(None, 1)),
        )


def test_observable_layout_requires_a_dense_ordering() -> None:
    with pytest.raises(ValueError, match="dense"):
        ObservableLayout(
            (
                LogicalObservable(
                    index=2,
                    pauli=Pauli(z_qubits=(0,)),
                    measurement_parity=(MeasurementRef(None, 0),),
                ),
            )
        )


def test_observable_layout_requires_at_least_one_observable() -> None:
    with pytest.raises(ValueError, match="at least one observable"):
        ObservableLayout(())


class _DistanceOnlyCode:
    """A stand-in that declares a distance but no code interface."""

    distance = 3


class _ChecklessCode:
    """A code that declares its wire layout but no checks."""

    distance = 3
    num_data_qubits = 3
    num_ancilla_qubits = 2
    num_ancilla_x_qubits = 0
    num_ancilla_z_qubits = 0
    num_x_stabilizers = 0
    num_z_stabilizers = 0
    data_qubits = (0, 1, 2)
    ancilla_qubits = (3, 4)
    checks: tuple[CodeCheck, ...] = ()
    stabilizers: tuple[Pauli, ...] = ()
    logical_observables = (Pauli(z_qubits=(0, 1, 2)),)


class _RogueLogicalCode:
    """A code whose declared logical observable leaves its declared data wires.

    ``LogicalObservable`` already requires its readout to measure exactly the
    operator's support, so for a coherent code the readout-wire guard in
    ``MemoryCircuit`` agrees with that check. This stand-in reaches the guard
    through a hand-built pair, where the layout is internally consistent and only
    the code pairing is incoherent.
    """

    distance = 3
    num_data_qubits = 3
    num_ancilla_qubits = 2
    num_ancilla_x_qubits = 0
    num_ancilla_z_qubits = 2
    num_x_stabilizers = 0
    num_z_stabilizers = 2
    data_qubits = (0, 1, 2)
    ancilla_qubits = (3, 4)
    checks = RepetitionCode(3).checks
    stabilizers = RepetitionCode(3).stabilizers
    logical_observables = (Pauli(z_qubits=(0, 1, 99)),)


def _built_layout() -> MemoryCircuit:
    """A valid three-round layout whose single detectors can be substituted."""

    return build_memory_circuit(RepetitionCode(3), rounds=3)


def _with_detector(
    built: MemoryCircuit, position: int, parity: tuple[MeasurementRef, ...]
) -> DetectorLayout:
    substituted = list(built.detectors.detectors)
    substituted[position] = Detector(index=position, parity=parity)
    return DetectorLayout(tuple(substituted))


def test_memory_circuit_rejects_a_non_code_object() -> None:
    built = _built_layout()

    with pytest.raises(TypeError, match="StabilizerCode"):
        MemoryCircuit(
            code=_DistanceOnlyCode(),  # type: ignore[arg-type]
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=built.observables,
        )


def test_memory_circuit_rejects_a_detector_on_an_unknown_ancilla_wire() -> None:
    built = _built_layout()

    with pytest.raises(ValueError, match="declared ancilla qubit"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=_with_detector(built, 0, (MeasurementRef(0, 99),)),
            observables=built.observables,
        )


def test_memory_circuit_rejects_a_terminal_readout_on_an_ancilla_wire() -> None:
    built = _built_layout()

    with pytest.raises(ValueError, match="declared data qubit"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=_with_detector(built, 7, (MeasurementRef(None, 3),)),
            observables=built.observables,
        )


def test_memory_circuit_rejects_a_syndrome_round_beyond_the_configuration() -> None:
    built = _built_layout()

    with pytest.raises(ValueError, match="inside the configured rounds"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=_with_detector(built, 0, (MeasurementRef(9, 3),)),
            observables=built.observables,
        )


def test_logical_observable_rejects_an_empty_readout() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        LogicalObservable(index=0, pauli=Pauli(z_qubits=(0,)), measurement_parity=())


def test_logical_observable_rejects_a_round_indexed_readout() -> None:
    with pytest.raises(ValueError, match="terminal"):
        LogicalObservable(
            index=0,
            pauli=Pauli(z_qubits=(0,)),
            measurement_parity=(MeasurementRef(0, 0),),
        )


def test_logical_observable_rejects_a_readout_outside_the_operator_support() -> None:
    with pytest.raises(ValueError, match="support"):
        LogicalObservable(
            index=0,
            pauli=Pauli(z_qubits=(0, 1)),
            measurement_parity=(MeasurementRef(None, 0),),
        )


def test_builder_rejects_a_code_without_checks() -> None:
    with pytest.raises(ValueError, match="at least one check"):
        build_memory_circuit(_ChecklessCode(), rounds=1)


def test_memory_circuit_rejects_an_observable_that_disagrees_with_the_code() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)
    (observable,) = built.observables.observables
    incoherent = LogicalObservable(
        index=0,
        pauli=Pauli(z_qubits=(99,)),
        measurement_parity=(MeasurementRef(None, 99),),
    )

    with pytest.raises(ValueError, match="declared logical observable"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=ObservableLayout((incoherent,)),
        )
    assert observable.pauli == Pauli(z_qubits=(0, 1, 2))


def test_memory_circuit_rejects_an_observable_readout_on_an_undeclared_wire() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)
    incoherent = LogicalObservable(
        index=0,
        pauli=Pauli(z_qubits=(0, 1, 99)),
        measurement_parity=(
            MeasurementRef(None, 0),
            MeasurementRef(None, 1),
            MeasurementRef(None, 99),
        ),
    )

    with pytest.raises(ValueError, match="declared data qubit"):
        MemoryCircuit(
            code=_RogueLogicalCode(),
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=ObservableLayout((incoherent,)),
        )


def test_public_namespace_publishes_the_code_independent_layer() -> None:
    import flagquantum.qec as qec

    expected = (
        "CodeCheck",
        "Detector",
        "DetectorLayout",
        "LogicalObservable",
        "MeasurementRef",
        "MemoryCircuit",
        "ObservableLayout",
        "Pauli",
        "RepetitionCode",
        "StabilizerCode",
        "build_memory_circuit",
    )
    missing = [name for name in expected if not hasattr(qec, name)]
    assert not missing, f"flagquantum.qec is missing {missing}"
    for name in expected:
        assert name in qec.__all__, f"{name} is not in flagquantum.qec.__all__"
