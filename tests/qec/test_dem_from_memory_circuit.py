"""Coverage for building a detector error model from a memory circuit."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import dem_construction as dem_module
from flagquantum.qec.circuit import (
    Detector,
    DetectorLayout,
    MeasurementRef,
    MemoryCircuit,
    build_memory_circuit,
)
from flagquantum.qec.codes import (
    CodeCheck,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    ancilla_bands,
)
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.dem_construction import (
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
    _Mechanism,
    _mechanisms,
)
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
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
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return (3, 4)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_qubits=(0, 1)),
                ancilla_qubit=3,
                cnot_qubits=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_qubits=(1, 2)),
                ancilla_qubit=4,
                cnot_qubits=((1, 4), (2, 4)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=(0, 1, 2)),)


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
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return (3, 4, 5)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
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

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=(0, 1, 2)),)


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
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return (3, 4)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
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
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=(0, 1, 2)), Pauli(z_qubits=(0,)))


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


@pytest.mark.parametrize(
    ("basis", "read_family", "invisible_family"),
    (("z", "data_flip", "phase_flip"), ("x", "phase_flip", "data_flip")),
)
def test_the_readout_basis_decides_which_data_fault_moves_the_observable(
    basis: str, read_family: str, invisible_family: str
) -> None:
    """A fault is logical when it anticommutes with the observable that is read.

    The Steane code declares a logical observable in each basis, so the same
    record describes two experiments and the readout basis is the whole
    difference between them. The Z-basis experiment reads the Z-type operator, so
    an X fault on its support moves the observable and a Z fault cannot; the
    X-basis experiment reads the X-type operator and the two families swap roles.
    Both halves are asserted, because either one alone is also what a model that
    ignored the basis would produce whenever it happened to read the right
    operator.
    """

    code = SteaneCode()
    read = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=2, readout_basis=basis),
        noise=PhenomenologicalNoise(**{read_family: 0.01}),
    )
    invisible = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=2, readout_basis=basis),
        noise=PhenomenologicalNoise(**{invisible_family: 0.01}),
    )

    assert read.num_errors > 0
    assert invisible.num_errors > 0
    assert any(error.observables == (0,) for error in read.errors)
    assert all(error.observables == () for error in invisible.errors)


def test_two_readout_bases_of_one_code_are_two_models() -> None:
    """The basis is not a relabelling: it changes which mechanisms the model has.

    The two experiments have the same shape, because the Steane code has the same
    number of checks of each type, so the counts cannot tell them apart. The
    observable rate can: each basis' terminal detectors read the checks in which
    that basis' invisible faults are the ones that moved the logical operator,
    and the two rates therefore differ.
    """

    code = SteaneCode()
    noise = PhenomenologicalNoise(
        data_flip=0.01, phase_flip=0.02, both_flip=0.005, measurement_flip=0.03
    )
    z_model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=3, readout_basis="z"), noise=noise
    )
    x_model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=3, readout_basis="x"), noise=noise
    )

    assert z_model.num_detectors == x_model.num_detectors
    assert z_model.num_observables == x_model.num_observables == 1
    assert z_model.num_errors == x_model.num_errors
    assert z_model.observable_rates()[0] != x_model.observable_rates()[0], (
        "the two bases produced the same observable rate, so the basis never "
        "reached the model"
    )


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
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
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


def test_a_terminal_detector_is_read_off_the_layout_not_the_check_support() -> None:
    """A terminal detector reads the data wires the layout names, nothing else.

    ``MemoryCircuit`` requires a terminal reference to name a declared data
    wire, never a wire the owning check touches, so a layout whose terminal
    detector reads a wire its check does not touch is representable by hand.
    The two readings agree on every circuit ``build_memory_circuit`` emits,
    because the terminal parity it generates *is* the check's support there, so
    only a hand-built layout can tell them apart — which is what this does.
    """

    base = build_memory_circuit(_TwoObservableCode(), rounds=2)
    detectors = list(base.detectors.detectors)
    terminal = len(detectors) - len(base.code.checks)
    # The generated first terminal detector reads its own check's support (0, 1).
    assert detectors[terminal].parity == (
        MeasurementRef(1, 3),
        MeasurementRef(None, 0),
        MeasurementRef(None, 1),
    )
    # Check 0 touches wires 0 and 1 and never wire 2, and this layout reads wire
    # 2 anyway. Wire 2 is a declared data wire, so the circuit is still legal.
    detectors[terminal] = Detector(
        index=terminal, parity=(MeasurementRef(1, 3), MeasurementRef(None, 2))
    )
    circuit = replace(base, detectors=DetectorLayout(tuple(detectors)))

    # A last-round flip on wire 2 moves check 1's round-1 syndrome, which is
    # detector 3, and — through this layout's terminal reference — detector 4.
    # Read off check 0's support instead, detector 4 stays put and the signature
    # is (3,), so the two readings are distinguishable here.
    signature = _forced_signature(
        circuit, _inject_data_flip(circuit, round_index=1, wire=2)
    )
    assert signature == ((3, 4), (0,))


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
    def data_qubits(self) -> tuple[int, ...]:
        return (0, 1, 2, 1)


def test_duplicate_data_wires_are_refused() -> None:
    """A repeated data wire would state one location's rate as two flips.

    Both copies of the location carry the same signature, so the merge would
    combine ``p`` with itself and the model would claim ``2p(1-p)`` for a
    location the circuit has once.
    """

    built = build_memory_circuit(_RepeatedDataWireCode(), rounds=2)
    with pytest.raises(ValueError, match="repeated data qubit"):
        DetectorErrorModel.from_memory_circuit(
            built, noise=PhenomenologicalNoise(data_flip=0.05)
        )


@dataclass(frozen=True)
class _RepeatedAncillaCode(_TwoObservableCode):
    """A code whose two checks measure one shared ancilla wire.

    Nothing in Stage 1 requires two checks to own distinct ancillas: a code
    declares the wires it uses, and ``CodeCheck`` ties each check's CNOTs to its
    own ancilla, which a second check may name as well. The emitted loop then
    measures wire 3 twice per round, so both checks' syndrome bits land at one
    position in the classical register.
    """

    @property
    def num_ancilla_qubits(self) -> int:
        return 1

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return (3,)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_qubits=(0, 1)),
                ancilla_qubit=3,
                cnot_qubits=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_qubits=(1, 2)),
                ancilla_qubit=3,
                cnot_qubits=((1, 3), (2, 3)),
            ),
        )


def test_repeated_check_ancilla_wires_are_refused() -> None:
    """Two checks on one ancilla fail closed before a mechanism is enumerated.

    The shape is representable and its program builds, and a data flip never
    reaches the injection engine's measurement anchor, so without this guard the
    model builds: both checks' detectors would read the one classical bit at the
    shared position, and a mechanism that flips nothing as misread would be
    dropped rather than refused.
    """

    built = build_memory_circuit(_RepeatedAncillaCode(), rounds=2)
    with pytest.raises(ValueError, match="repeated check ancilla qubit"):
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


# Composite faults: the Y family read as one fault or as the X and Z faults it is
# the XOR of. The decoder-facing question is which of the two readings states a
# model a minimum-weight matcher can weight, and the answer is measured on a
# rotated surface code rather than asserted, because the repetition code's
# composite faults are already graphlike under either reading and would pass
# either way.

_COMPOSITE_READINGS = "decompose_composite_faults"


def _signatures(
    model: DetectorErrorModel,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """Return each signature's probability, which a model states once."""

    return {
        (error.detectors, error.observables): error.probability
        for error in model.errors
    }


def _every_noise() -> PhenomenologicalNoise:
    """Return a record with all four families at one rate."""

    return PhenomenologicalNoise(
        data_flip=0.01, phase_flip=0.01, both_flip=0.01, measurement_flip=0.01
    )


def _surface(rounds: int) -> MemoryCircuit:
    """Return a distance-three rotated surface code's memory experiment."""

    return build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=rounds)


def test_the_option_defaults_to_the_combined_reading() -> None:
    """A caller that states nothing gets the model every earlier caller got.

    This is the whole compatibility claim of the option: the combined reading is
    what the route has always produced, so naming it explicitly and omitting it
    have to reach the same model rather than two that agree on the marginals.
    """

    built = _surface(2)
    omitted = DetectorErrorModel.from_memory_circuit(built, noise=_every_noise())
    stated = DetectorErrorModel.from_memory_circuit(
        built, noise=_every_noise(), decompose_composite_faults=False
    )
    assert _signatures(omitted) == _signatures(stated)


def test_the_option_refuses_a_flag_that_is_not_a_boolean() -> None:
    """``1`` is an integer and not a decision, so it fails closed by type."""

    built = _surface(2)
    with pytest.raises(TypeError, match="decompose_composite_faults must be a bool"):
        DetectorErrorModel.from_memory_circuit(
            built, noise=_every_noise(), decompose_composite_faults=1
        )


def test_a_composite_fault_is_the_x_and_z_faults_at_its_own_rate() -> None:
    """The decomposed model is the two single-Pauli families at the same rate.

    The reference is the route's own answer for an X fault and for a Z fault: a
    record whose composite family is the only noisy one, read apart, has to reach
    the model two records with the composite rate placed in the X field and in
    the Z field reach. Nothing here re-derives a signature; both sides are read
    off the program by the construction route's own injector.
    """

    built = _surface(2)
    composite = DetectorErrorModel.from_memory_circuit(
        built,
        noise=PhenomenologicalNoise(both_flip=0.01),
        decompose_composite_faults=True,
    )
    parts: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for field in ("data_flip", "phase_flip"):
        model = DetectorErrorModel.from_memory_circuit(
            built, noise=PhenomenologicalNoise(**{field: 0.01})
        )
        for signature, probability in _signatures(model).items():
            # The two halves touch disjoint detector and observable index ranges,
            # so a signature is stated by at most one of them and a sum cannot
            # hide an overlap that the decomposed model would have merged.
            assert signature not in parts
            parts[signature] = probability
    assert set(_signatures(composite)) == set(parts)
    for signature, probability in parts.items():
        assert _signatures(composite)[signature] == pytest.approx(probability)


def test_decomposing_widens_the_matcher_to_a_model_it_refused() -> None:
    """The authority matcher refuses the combined model and answers the split one.

    The refusal is the gap the option exists to close, so it is asserted by the
    name it is reported under: the combined model states five mechanisms the
    matcher calls hyperedges, four of three detectors and one of four, and the
    decomposed model states none above two and the same matcher builds on it.
    """

    built = _surface(2)
    combined = DetectorErrorModel.from_memory_circuit(built, noise=_every_noise())
    decomposed = DetectorErrorModel.from_memory_circuit(
        built, noise=_every_noise(), decompose_composite_faults=True
    )
    hyperedges = [error for error in combined.errors if len(error.detectors) > 2]
    assert sorted(len(error.detectors) for error in hyperedges) == [3, 3, 3, 3, 4]
    assert max(len(error.detectors) for error in decomposed.errors) == 2
    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(combined)
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(decomposed)
    assert matcher.decode(()).observables == ()
    assert matcher.decode((0,)).weight > 0.0
    assert (
        matcher.decode(tuple(error.detectors[0] for error in hyperedges)).weight > 0.0
    )


def test_decomposing_preserves_the_detector_and_observable_rates() -> None:
    """Each detector's and each observable's own flip rate survives the reading.

    A rate is the marginal of one target, and the parts of a composite fault
    touch disjoint targets -- the X part reaches Z-type detectors and the Z part
    X-type ones -- so replacing one fault by two changes the joint law and not a
    single marginal. The tolerance is floating-point rounding and nothing wider:
    the two models fold the same probabilities in a different order and no step
    of either arithmetic is approximate.
    """

    built = _surface(2)
    combined = DetectorErrorModel.from_memory_circuit(built, noise=_every_noise())
    decomposed = DetectorErrorModel.from_memory_circuit(
        built, noise=_every_noise(), decompose_composite_faults=True
    )
    assert combined.num_detectors == decomposed.num_detectors
    assert combined.num_observables == decomposed.num_observables
    for name in ("detector_rates", "observable_rates"):
        left = getattr(combined, name)()
        right = getattr(decomposed, name)()
        assert bool((left - right).abs().max().item() < 1e-15)


def test_a_noiseless_record_still_builds_an_empty_model() -> None:
    """The reading does not invent a part where the program states no fault.

    A location whose own probability is zero is not enumerated at all, and the
    enumeration is the one place that rule is stated, so it holds under either
    reading: a record that states no rate has no composite fault to read apart
    and the decomposed model is the empty one with the circuit's shape, rather
    than two parts of nothing at a rate nothing states.
    """

    built = _surface(2)
    decomposed = DetectorErrorModel.from_memory_circuit(
        built,
        noise=PhenomenologicalNoise(),
        decompose_composite_faults=True,
    )
    assert decomposed.num_errors == 0
    assert decomposed.num_detectors == 16


def test_a_code_whose_composite_faults_are_already_parts_changes_nothing() -> None:
    """Where the program already separates the parts the two readings agree.

    A repetition code's checks are Z-type alone, so a Z fault on a data qubit
    reaches no detector in any round and the composite fault's X part carries its
    whole signature; the two readings then state the same mechanisms at the same
    rates rather than two models that happen to share their marginals.
    """

    for distance in (3, 5):
        built = build_memory_circuit(RepetitionCode(distance=distance), rounds=3)
        combined = DetectorErrorModel.from_memory_circuit(built, noise=_every_noise())
        decomposed = DetectorErrorModel.from_memory_circuit(
            built, noise=_every_noise(), decompose_composite_faults=True
        )
        assert combined.num_errors > 0
        assert _signatures(combined) == _signatures(decomposed)


def test_a_one_round_experiment_changes_nothing() -> None:
    """One round places the composite fault where its parts are not separable yet.

    At one round the composite fault of a rotated surface code flips at most two
    detectors whole, so the combined model is already graphlike and the reading
    costs nothing; the option is not a rewrite that changes a model it has no
    room to change.
    """

    built = _surface(1)
    combined = DetectorErrorModel.from_memory_circuit(built, noise=_every_noise())
    decomposed = DetectorErrorModel.from_memory_circuit(
        built, noise=_every_noise(), decompose_composite_faults=True
    )
    assert max(len(error.detectors) for error in combined.errors) == 2
    assert _signatures(combined) == _signatures(decomposed)


def test_the_matrix_route_states_no_decomposition_option() -> None:
    """The code-capacity route makes no graphlike promise, so it has no keyword.

    The matrix route reads a fault's signature as the support of a column against
    two detector bands instead of off an executed program, so a part there reaches
    four detectors at two rounds and decomposing it would not hand the matcher the
    shape the option names. An option named for the reading on that route would
    promise a model it cannot deliver, so its absence is pinned and adding the
    keyword later is a decision rather than a slip.
    """

    matrix_route = inspect.signature(DetectorErrorModel.from_code_matrices)
    assert _COMPOSITE_READINGS not in matrix_route.parameters
    with pytest.raises(TypeError):
        DetectorErrorModel.from_code_matrices(  # type: ignore[call-arg]
            dem_module.css_code_matrices(RotatedSurfaceCode(distance=3)),
            noise=_every_noise(),
            decompose_composite_faults=True,
        )
