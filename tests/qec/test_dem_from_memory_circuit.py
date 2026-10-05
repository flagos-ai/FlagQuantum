"""Coverage for building a detector error model from a memory circuit."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from flagquantum.qec import dem_construction as dem_module
from flagquantum.qec.circuit import (
    Detector,
    DetectorLayout,
    MeasurementRef,
    build_memory_circuit,
)
from flagquantum.qec.codes import (
    CodeCheck,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
)
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.dem_construction import (
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
    _Mechanism,
    _mechanisms,
)
from flagquantum.qec.logical import derive_anticommuting_logical_product
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.integration


@pytest.fixture(
    params=[RepetitionCode(distance=3), SteaneCode(), RotatedSurfaceCode(distance=3)],
    ids=["repetition-3", "steane", "surface-3"],
)
def modelled_code(request: pytest.FixtureRequest) -> StabilizerCode:
    """One code per check family: Z-type only, both types, and a rotated patch."""

    return request.param


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
    """An empty mechanism set still reports the shape the circuit declares.

    The shape is read off the circuit's own layouts rather than counted from the
    mechanisms, so a record that contributes no mechanism contributes no errors
    while the detector and observable counts stay the circuit's. A shape derived
    from the mechanism set would report no observables at all here, because a
    mechanism is what carries an observable flip.
    """

    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(built, noise=PhenomenologicalNoise())
    assert model.num_errors == 0
    assert model.num_detectors == 8
    assert model.num_observables == 1
    assert model.errors == ()
    assert model.detector_error_matrix().shape == (8, 0)
    assert model.observables_flips_matrix().shape == (1, 0)


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


@pytest.mark.parametrize(
    ("fault", "gates"),
    [
        ("x", ("X",)),
        ("z", ("H", "X", "H")),
        ("y", ("X", "H", "X", "H")),
    ],
)
def test_data_faults_are_injected_as_themselves(
    fault: str, gates: tuple[str, ...]
) -> None:
    """``data_flip`` is an X fault, ``phase_flip`` a Z fault, ``both_flip`` a Y one.

    The injected sequence is compared against the whole source, because the
    language has no ``Z`` gate: a Z fault is written as the three gates whose
    product is ``Z``, and a Y fault as ``X`` followed by that same product. A
    count-based assertion would pass on a source injected at the wrong anchor.
    """

    built = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    anchor = "    for round_index in range(rounds):\n"
    head, _, tail = built.source.partition(anchor)
    assert head and tail
    block = "".join(
        [
            "        if round_index == 0:\n",
            *(f"            qp.{gate}(wires=1)\n" for gate in gates),
        ]
    )

    assert (
        _inject_data_flip(built, round_index=0, wire=1, fault=fault)  # type: ignore[arg-type]
        == f"{head}{anchor}{block}{tail}"
    )
    assert built.source.count(anchor) == 1
    assert "qp.Z(" not in built.source


def test_an_unknown_data_fault_is_refused() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    with pytest.raises(ValueError, match="unknown data fault 'w'"):
        _inject_data_flip(built, round_index=0, wire=1, fault="w")  # type: ignore[arg-type]


def test_the_three_fault_families_are_enumerated_separately() -> None:
    """Each family is a location of its own, in the code's own wire order."""

    built = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    noise = PhenomenologicalNoise(data_flip=0.01, phase_flip=0.02, both_flip=0.03)
    assert [
        (mechanism.fault, mechanism.wire) for mechanism in _mechanisms(built, noise)
    ] == [
        ("x", 0),
        ("z", 0),
        ("y", 0),
        ("x", 1),
        ("z", 1),
        ("y", 1),
        ("x", 2),
        ("z", 2),
        ("y", 2),
    ]


def test_a_family_whose_rate_is_zero_is_not_a_location() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    noise = PhenomenologicalNoise(phase_flip=0.02)
    mechanisms = _mechanisms(built, noise)
    assert {mechanism.fault for mechanism in mechanisms} == {"z"}
    assert len(mechanisms) == 3


def test_measurement_mechanisms_carry_the_x_fault_label() -> None:
    """A measurement flip is not a Pauli fault, and the field says so uniformly."""

    built = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    noise = PhenomenologicalNoise(measurement_flip=0.2)
    mechanisms = _mechanisms(built, noise)
    assert [(mechanism.kind, mechanism.fault) for mechanism in mechanisms] == [
        ("measurement", "x"),
        ("measurement", "x"),
    ]


def test_a_z_fault_in_the_z_basis_flips_only_the_x_type_checks() -> None:
    """The family a code-capacity model needs is the one its other checks see.

    The Steane code's Z-type checks read the data in the Z basis, so the ``|0>``
    preparation already fixes them and a Z fault in round zero is invisible to
    them: an X fault is what their round-zero detectors catch. A Z fault is caught
    by the X-type checks instead, and only by their round-to-round detectors,
    because an X-type check's round-zero outcome is random in this frame. Under
    this code the X-type checks hold detectors 6, 7 and 8.
    """

    built = build_memory_circuit(SteaneCode(), rounds=2)
    noise = PhenomenologicalNoise(phase_flip=0.05)
    model = DetectorErrorModel.from_memory_circuit(built, noise=noise)
    measured = {
        (record.round_index, record.wire): _forced_signature(
            built,
            _inject_data_flip(
                built,
                round_index=record.round_index,
                wire=record.wire,
                fault=record.fault,
            ),
        )
        for record in _mechanisms(built, noise)
    }

    assert measured[(0, 0)] == ((), ())
    assert measured[(1, 0)] == ((8,), ())
    assert measured[(1, 3)] == ((6,), ())
    assert measured[(1, 6)] == ((6, 7, 8), ())
    assert set(measured) == {
        (round_index, wire) for round_index in (0, 1) for wire in range(7)
    }
    flipped = {detector for detectors, _ in measured.values() for detector in detectors}
    assert flipped == {6, 7, 8}
    assert model.num_observables == 1
    assert all(not error.observables for error in model.errors)
    assert all(error.detectors or error.observables for error in model.errors)


def test_a_z_fault_in_the_x_basis_flips_the_rotated_observable() -> None:
    """Reading a code out in the X basis is what makes its Z faults matter.

    The Z-type checks are the ones whose round-zero outcome the ``|+>``
    preparation randomizes, so in this frame none of them keeps a boundary
    detector, and a Z fault in round zero is therefore caught by nothing at all.
    From round one on it is caught by the X-type checks' round-to-round detectors
    and it also flips the X-basis observable.
    """

    code = SteaneCode()
    product = derive_anticommuting_logical_product(code, 0)
    built = build_memory_circuit(code, rounds=2, product=product)
    noise = PhenomenologicalNoise(phase_flip=0.05)
    model = DetectorErrorModel.from_memory_circuit(built, noise=noise)
    measured = {
        (record.round_index, record.wire): _forced_signature(
            built,
            _inject_data_flip(
                built,
                round_index=record.round_index,
                wire=record.wire,
                fault=record.fault,
            ),
        )
        for record in _mechanisms(built, noise)
    }

    assert built.x_readout_wires == (2, 4, 5)
    assert product == Pauli(x_wires=(2, 4, 5))
    assert measured[(0, 0)] == ((), ())
    assert measured[(0, 2)] == ((), (0,))
    assert measured[(1, 2)] == ((4, 5), (0,))
    assert measured[(1, 6)] == ((3, 4, 5), ())
    live = [error for error in model.errors if error.observables]
    assert live
    assert all(tuple(error.observables) == (0,) for error in live)
    assert all(error.detectors or error.observables for error in model.errors)


def test_an_x_fault_in_the_x_basis_is_detected_but_never_flips_the_observable() -> None:
    """The rotated readout measures an X-type product, which an X fault commutes with."""

    code = SteaneCode()
    built = build_memory_circuit(
        code, rounds=2, product=derive_anticommuting_logical_product(code, 0)
    )
    noise = PhenomenologicalNoise(data_flip=0.05)
    model = DetectorErrorModel.from_memory_circuit(built, noise=noise)
    measured = {
        (record.round_index, record.wire): _forced_signature(
            built,
            _inject_data_flip(
                built,
                round_index=record.round_index,
                wire=record.wire,
                fault=record.fault,
            ),
        )
        for record in _mechanisms(built, noise)
    }

    assert model.num_errors == 7
    assert all(not error.observables for error in model.errors)
    assert measured[(1, 0)] == ((2,), ())
    assert measured[(1, 6)] == ((0, 1, 2), ())


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


@dataclass(frozen=True)
class _OrphanAncillaCode:
    """A code that declares one ancilla wire no check reads.

    The declared ancilla count and the declared wires are what the memory
    circuit's layout validation reads, so this is the smallest code that carries
    a detector reference nothing records a bit for.
    """

    inner: _TwoObservableCode
    orphan_wire: int = 5

    @property
    def distance(self) -> int:
        return self.inner.distance

    @property
    def num_data_qubits(self) -> int:
        return self.inner.num_data_qubits

    @property
    def num_ancilla_qubits(self) -> int:
        return self.inner.num_ancilla_qubits + 1

    @property
    def data_wires(self) -> tuple[int, ...]:
        return self.inner.data_wires

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return self.inner.ancilla_wires + (self.orphan_wire,)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return self.inner.checks

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return self.inner.stabilizers

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return self.inner.logical_observables


def test_a_syndrome_reference_no_check_owns_is_refused() -> None:
    """A reference nothing records a bit for is named, not read as even parity.

    ``MemoryCircuit`` ties a syndrome reference to the code's declared ancilla
    wires rather than to its checks, so a layout can name a wire no check
    measures. Every detector reading it would then compare against a bit that
    does not exist, and a lookup that finds nothing reports an even parity — a
    detector that silently never fires. The derivation states the failure
    instead. This is the derivation's own refusal and not the sampler's: the
    sampler never runs here.
    """

    code = _OrphanAncillaCode(_TwoObservableCode())
    base = build_memory_circuit(code, rounds=2)
    assert code.orphan_wire in code.ancilla_wires
    assert all(check.ancilla_wire != code.orphan_wire for check in code.checks)
    detectors = list(base.detectors.detectors)
    first = detectors[0]
    assert first.parity
    detectors[0] = Detector(
        index=first.index,
        parity=(MeasurementRef(first.parity[0].round_index, code.orphan_wire),)
        + first.parity[1:],
    )
    circuit = replace(base, detectors=DetectorLayout(tuple(detectors)))

    with pytest.raises(ValueError, match="which no check owns"):
        DetectorErrorModel.from_memory_circuit(
            circuit, noise=PhenomenologicalNoise(data_flip=0.05)
        )


def test_a_source_that_does_not_match_its_layout_fails_closed() -> None:
    """An injector refusal reaches the caller unchanged, not swallowed."""

    base = build_memory_circuit(_TwoObservableCode(), rounds=2)
    circuit = replace(base, source="def memory_experiment(rounds):\n    return False\n")
    with pytest.raises(ValueError, match="check measurement anchor"):
        DetectorErrorModel.from_memory_circuit(
            circuit, noise=PhenomenologicalNoise(measurement_flip=0.05)
        )


def test_the_round_loop_anchor_is_required_only_when_a_data_fault_is() -> None:
    """A source is held to the anchors the enumerated mechanisms actually read.

    Renaming the round loop variable leaves every measurement anchor in place, so
    a model built from data faults alone can no longer say which round a fault was
    injected at and is refused, while a model built from measurement faults alone
    reads no round loop at all and is accepted. The asymmetry is the point: a
    guard that required both anchors for every noise record would refuse a
    measurement-only model of a source that describes its measurements perfectly.
    """

    base = build_memory_circuit(_TwoObservableCode(), rounds=2)
    renamed = base.source.replace(
        "    for round_index in range(rounds):\n", "    for step in range(rounds):\n"
    )
    assert renamed != base.source
    circuit = replace(base, source=renamed)

    with pytest.raises(ValueError, match="the round loop anchor"):
        DetectorErrorModel.from_memory_circuit(
            circuit, noise=PhenomenologicalNoise(data_flip=0.05)
        )
    model = DetectorErrorModel.from_memory_circuit(
        circuit, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_errors == 4
    assert all(error.detectors for error in model.errors)


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
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3,)

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
                ancilla_wire=3,
                cnot_wires=((1, 3), (2, 3)),
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
    with pytest.raises(ValueError, match="repeated check ancilla wire"):
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


def test_the_derived_signatures_match_forcing_every_location(
    modelled_code: StabilizerCode,
) -> None:
    """Every derived signature is held against the execution that used to build it.

    ``from_memory_circuit`` derives a mechanism's flip set from the circuit's
    layouts; ``_forced_signature`` injects the mechanism into the stored program
    and executes it, and reads the same layouts off the result. Neither route is
    the authority for the other, and they read the layouts through different code,
    so agreeing on every location of every code here is what pins the derivation
    to the program it describes. The comparison is exhaustive over the
    mechanisms ``_mechanisms`` enumerates, so no statistic enters it.
    """

    built = build_memory_circuit(modelled_code, rounds=2)
    noise = PhenomenologicalNoise(
        data_flip=0.05, phase_flip=0.05, both_flip=0.05, measurement_flip=0.05
    )
    derived = {
        (error.detectors, error.observables)
        for error in DetectorErrorModel.from_memory_circuit(built, noise=noise).errors
    }
    forced = set()
    for record in _mechanisms(built, noise):
        if record.kind == "data":
            source = _inject_data_flip(
                built,
                round_index=record.round_index,
                wire=record.wire,
                fault=record.fault,
            )
        else:
            source = _inject_measurement_flip(
                built, round_index=record.round_index, ancilla_wire=record.wire
            )
        signature = _forced_signature(built, source)
        if signature != ((), ()):
            forced.add(signature)
    assert derived == forced


def test_the_derivation_reaches_a_width_no_execution_can() -> None:
    """A distance-seven surface patch builds, where forcing a location cannot.

    The derived route reads the layout, so the number of wires does not bound it.
    The executed route allocates the whole amplitude vector: its distance-three
    patch is 17 wires and completes, and its distance-four patch is 31 wires and
    fails on the allocator. The distances asserted here are the ones the executed
    route cannot reach, so this test is what fails if the derivation is replaced
    by an execution again.
    """

    code = RotatedSurfaceCode(distance=7)
    built = build_memory_circuit(code, rounds=7)
    assert len(code.data_wires) + len(code.ancilla_wires) == 97
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
    )
    assert model.num_detectors == 336
    assert model.num_errors == 637
    assert all(error.detectors or error.observables for error in model.errors)
