"""Coverage for building a detector error model from a memory circuit."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from flagquantum.qec import dem as dem_module
from flagquantum.qec.circuit import (
    Detector,
    DetectorLayout,
    MeasurementRef,
    build_memory_circuit,
)
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.dem import (
    DetectorErrorModel,
    _forced_signature,
    _inject_measurement_flip,
    _Mechanism,
    _mechanisms,
)
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.integration


def test_repetition_model_shape() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    assert model.num_detectors == 8
    assert model.num_observables == 1
    # 9 data mechanisms plus 6 measurement mechanisms, all with distinct signatures
    assert model.num_errors == 15


def test_every_mechanism_has_a_unique_signature_at_distance_three() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    signatures = [(error.detectors, error.observables) for error in model.errors]
    assert len(set(signatures)) == len(signatures)


def test_a_noiseless_model_has_no_mechanisms() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(built, noise=PhenomenologicalNoise())
    assert model.num_errors == 0
    assert model.num_detectors == 8


def test_data_flips_alone_always_flip_the_observable() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05)
    )
    assert model.num_errors == 9
    assert all(error.observables == (0,) for error in model.errors)


def test_measurement_flips_alone_never_flip_the_observable() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_errors == 6
    assert all(error.observables == () for error in model.errors)


def test_probabilities_come_from_the_noise_record() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.25)
    )
    assert {error.probability for error in model.errors} == {0.25}


def test_distance_five_mechanism_count() -> None:
    built = build_memory_circuit(RepetitionCode(distance=5), rounds=5)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    assert model.num_detectors == 24
    assert model.num_errors == 45


@dataclass(frozen=True)
class _IndexSwappedCode:
    """Two checks whose declared indices disagree with their tuple positions.

    Stage 1 recorded that ``CodeCheck.index`` is not consumed by the library and
    that a code may declare indices disagreeing with position with no error. The
    classical-bit stride is positional, so a model that keyed on ``index`` would
    scramble this code's signatures.
    """

    distance: int = 3

    @property
    def num_data_qubits(self) -> int:
        return 3

    @property
    def num_ancilla_qubits(self) -> int:
        return 2

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3, 4)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_wires=(0, 1)),
                ancilla_wire=3,
                cnot_wires=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_wires=(1, 2)),
                ancilla_wire=4,
                cnot_wires=((1, 4), (2, 4)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=(0, 1, 2)),)


def test_signatures_key_on_check_position_not_on_declared_index() -> None:
    """A measurement flip is read at the check's *positional* classical bit.

    The classical register strides by tuple position, so a model keyed on
    ``CodeCheck.index`` reads the wrong bit. This test asserts the *mapping*
    from mechanism to signature, not the set of signatures: under this code the
    two keys differ by a transposition, so a set comparison is preserved by the
    wrong implementation and would pass.
    """

    code = _IndexSwappedCode()
    built = build_memory_circuit(code, rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_detectors == 6
    assert model.num_errors == 4
    signatures = {error.detectors for error in model.errors}
    assert signatures == {(0, 2), (1, 3), (2, 4), (3, 5)}

    # The positional stride: a round-``r`` measurement flip on the check at
    # position ``p`` moves detectors ``r * 2 + p`` and ``(r + 1) * 2 + p``.
    # Check position 0 is ancilla 3 (whose declared ``index`` is 1) and position
    # 1 is ancilla 4 (declared ``index`` 0), so a model keyed on ``index``
    # transposes exactly these two expectations and survives the set assertion
    # above.
    noise = PhenomenologicalNoise(measurement_flip=0.05)
    observed = {
        (record.round_index, record.wire): _forced_signature(
            built,
            _inject_measurement_flip(
                built, round_index=record.round_index, ancilla_wire=record.wire
            ),
        )
        for record in _mechanisms(built, noise)
    }
    assert observed[(0, 3)] == ((0, 2), ())
    assert observed[(0, 4)] == ((1, 3), ())
    assert observed[(1, 3)] == ((2, 4), ())
    assert observed[(1, 4)] == ((3, 5), ())


def test_mechanisms_enumerate_in_round_then_tuple_order() -> None:
    """Data flips come first, then measurement flips, each in the code's own order.

    The order is part of what a caller that fires several mechanisms at once
    relies on. Under this code the declared ``index`` order is not the tuple
    order, so an enumeration that sorted by index differs here.
    """

    built = build_memory_circuit(_IndexSwappedCode(), rounds=2)
    noise = PhenomenologicalNoise(data_flip=0.1, measurement_flip=0.2)
    assert [
        (mechanism.kind, mechanism.round_index, mechanism.wire)
        for mechanism in _mechanisms(built, noise)
    ] == [
        ("data", 0, 0),
        ("data", 0, 1),
        ("data", 0, 2),
        ("data", 1, 0),
        ("data", 1, 1),
        ("data", 1, 2),
        ("measurement", 0, 3),
        ("measurement", 0, 4),
        ("measurement", 1, 3),
        ("measurement", 1, 4),
    ]


@dataclass(frozen=True)
class _ThreeCheckCode:
    """A code whose check count is not ``distance - 1``.

    Stage 1's whole-branch review found the detector count tied to ``distance``
    rather than to the declared checks, and found that every test code was shaped
    like ``RepetitionCode``, so six independent edits could leave 194 tests green.
    This code keeps that hole closed for the model: three checks against a
    declared distance of three, so anything keyed on ``distance - 1`` enumerates
    two checks where the program performs three and reads the wrong classical
    bits.

    Wire 0 sits in checks 0 and 2, wire 1 in checks 0 and 1, wire 2 in checks 1
    and 2 — a triangle, so a data flip never touches two adjacent checks the way
    it does in a repetition code.
    """

    distance: int = 3

    @property
    def num_data_qubits(self) -> int:
        return 3

    @property
    def num_ancilla_qubits(self) -> int:
        return 3

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3, 4, 5)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_wires=(0, 1)),
                ancilla_wire=3,
                cnot_wires=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_wires=(1, 2)),
                ancilla_wire=4,
                cnot_wires=((1, 4), (2, 4)),
            ),
            CodeCheck(
                index=2,
                stabilizer=Pauli(z_wires=(0, 2)),
                ancilla_wire=5,
                cnot_wires=((0, 5), (2, 5)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=(0, 1, 2)),)


@dataclass(frozen=True)
class _TwoObservableCode:
    """A code declaring two logical observables.

    Nothing in Stage 1 checks that a declared observable is a genuine logical
    operator, so a second Z-type operator on a data wire is representable. The
    point of the test is that no code path hardcodes a single observable.
    """

    distance: int = 3

    @property
    def num_data_qubits(self) -> int:
        return 3

    @property
    def num_ancilla_qubits(self) -> int:
        return 2

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3, 4)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_wires=(0, 1)),
                ancilla_wire=3,
                cnot_wires=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_wires=(1, 2)),
                ancilla_wire=4,
                cnot_wires=((1, 4), (2, 4)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=(0, 1, 2)), Pauli(z_wires=(0,)))


def test_check_count_drives_the_mechanism_count_not_distance() -> None:
    """Three checks and a declared distance of three must give 3 * (rounds + 1) detectors."""

    built = build_memory_circuit(_ThreeCheckCode(), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_detectors == 9
    assert model.num_errors == 6
    signatures = [error.detectors for error in model.errors]
    assert len(set(signatures)) == len(signatures)


def test_data_flips_alone_flip_the_declared_observable_for_any_code() -> None:
    built = build_memory_circuit(_ThreeCheckCode(), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05)
    )
    assert model.num_errors == 6
    assert all(error.observables == (0,) for error in model.errors)


def test_two_declared_observables_are_both_modelled() -> None:
    """One model per noise class, so no assertion depends on a merged probability."""

    built = build_memory_circuit(_TwoObservableCode(), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05)
    )
    assert model.num_observables == 2
    assert model.observable_rates().shape == (2,)
    assert model.num_errors == 6
    # observable 0 spans every data wire, so any data flip moves it
    assert all(0 in error.observables for error in model.errors)
    # observable 1 spans wire 0 alone, so only some data flips move it
    assert any(1 in error.observables for error in model.errors)
    assert any(1 not in error.observables for error in model.errors)


def test_measurement_flips_never_move_a_declared_observable() -> None:
    built = build_memory_circuit(_TwoObservableCode(), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_observables == 2
    assert all(error.observables == () for error in model.errors)


@dataclass(frozen=True)
class _SpareAncillaCode(_TwoObservableCode):
    """A code that declares an ancilla wire no check owns.

    ``MemoryCircuit`` checks that a detector's syndrome reference names a
    *declared* ancilla, never that a check owns it, so this layout is
    representable by hand. It is the input a hand-built circuit can reach that
    ``build_memory_circuit`` cannot produce.
    """

    @property
    def num_ancilla_qubits(self) -> int:
        return 3

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3, 4, 5)


def test_a_detector_naming_an_unowned_ancilla_fails_closed() -> None:
    """An ancilla no check measures has no classical bit, and that is stated.

    The signature engine reads one syndrome bit per check, so a detector naming
    an ancilla that no check owns names a bit that was never recorded. The
    failure must be a stated ``ValueError`` rather than the bare ``KeyError`` a
    lookup without a guard would raise.
    """

    base = build_memory_circuit(_SpareAncillaCode(), rounds=2)
    detectors = list(base.detectors.detectors)
    detectors[0] = Detector(
        index=0, parity=(MeasurementRef(0, 5), MeasurementRef(None, 0))
    )
    circuit = replace(base, detectors=DetectorLayout(tuple(detectors)))
    with pytest.raises(ValueError, match="no check owns"):
        DetectorErrorModel.from_memory_circuit(
            circuit, noise=PhenomenologicalNoise(measurement_flip=0.05)
        )


def test_a_source_that_does_not_match_its_layout_fails_closed() -> None:
    """An injector refusal reaches the caller unchanged, not swallowed."""

    base = build_memory_circuit(_TwoObservableCode(), rounds=2)
    circuit = replace(base, source="def memory_experiment(rounds):\n    return False\n")
    with pytest.raises(ValueError, match="check measurement anchor"):
        DetectorErrorModel.from_memory_circuit(
            circuit, noise=PhenomenologicalNoise(measurement_flip=0.05)
        )


def test_a_circuit_must_be_a_memory_circuit() -> None:
    with pytest.raises(TypeError, match="circuit must be a MemoryCircuit"):
        DetectorErrorModel.from_memory_circuit(
            "not a circuit", noise=PhenomenologicalNoise(data_flip=0.05)
        )


def test_an_unknown_mechanism_kind_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kind that is neither flip is refused, not read as a measurement flip.

    The record's wire is a declared ancilla, which is exactly what a measurement
    record carries, so a dispatch whose fall-through is "measurement" would
    accept this record and build a model from it. ``_mechanisms`` never sets
    such a kind; the guard is for a caller that builds records by hand.
    """

    built = build_memory_circuit(_TwoObservableCode(), rounds=2)
    record = _Mechanism(kind="Data", round_index=0, wire=3, probability=0.05)
    monkeypatch.setattr(dem_module, "_mechanisms", lambda circuit, noise: (record,))
    with pytest.raises(ValueError, match="unknown mechanism kind 'Data'"):
        DetectorErrorModel.from_memory_circuit(
            built, noise=PhenomenologicalNoise(measurement_flip=0.05)
        )


def test_noise_must_be_a_phenomenological_noise_record() -> None:
    built = build_memory_circuit(_TwoObservableCode(), rounds=2)
    with pytest.raises(TypeError, match="noise must be a PhenomenologicalNoise"):
        DetectorErrorModel.from_memory_circuit(built, noise=0.05)


@dataclass(frozen=True)
class _RepeatedDataWireCode(_TwoObservableCode):
    """A code that declares data wire 1 twice.

    Membership is all Stage 1 checks a data wire for, in both
    ``MemoryCircuit`` and the observable layout, so a repeated wire is
    representable and builds. It is the input that makes the enumeration visit
    one physical location twice per round.
    """

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2, 1)


def test_duplicate_data_wires_are_refused() -> None:
    """A repeated data wire would state one location's rate as two flips.

    Both copies of the location carry the same signature, so the merge would
    combine ``p`` with itself and the model would claim ``2p(1-p)`` for a
    location the circuit has once.
    """

    built = build_memory_circuit(_RepeatedDataWireCode(), rounds=2)
    with pytest.raises(ValueError, match="repeated data wire"):
        DetectorErrorModel.from_memory_circuit(
            built, noise=PhenomenologicalNoise(data_flip=0.05)
        )


def test_merging_combines_identical_signatures() -> None:
    """Two mechanisms with one signature merge by XOR probability."""

    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.2, (0,), ())], num_detectors=1, num_observables=0
    )
    assert merged.num_errors == 1
    assert merged.errors[0].probability == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)


def test_distinct_signatures_are_kept_apart() -> None:
    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.1, (1,), ())], num_detectors=2, num_observables=0
    )
    assert merged.num_errors == 2


def test_signatures_that_differ_only_in_observables_are_kept_apart() -> None:
    """The merge key is the whole signature, never its detector half alone.

    The case above differs in detectors, so a key of ``(detectors, ())`` passes
    it; two mechanisms that agree on their detectors and disagree on their
    observables are what pins the observable component of the key.
    """

    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.1, (0,), (0,))], num_detectors=1, num_observables=1
    )
    assert merged.num_errors == 2


def test_empty_signature_mechanisms_are_discarded() -> None:
    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.1, (), ())], num_detectors=1, num_observables=0
    )
    assert merged.num_errors == 1
