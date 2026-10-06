"""Coverage for certifying a logical product and reading it out in its own frame.

Every code here declares checks of a single type except the ZXXZ surface patch,
whose every check is a product of an ``X`` factor and a ``Z`` factor. That patch
is what makes the two questions this module asks -- is a candidate a product of
the checks, and which checks constrain a candidate -- questions about a check
rather than about a family of checks, and it is the reason both of them are
answered per check and over the whole operator rather than per type and over a
support.
"""

from __future__ import annotations

import pytest

from flagquantum.qec.circuit import (
    Detector,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    _check_source,
    _detector_layout,
    build_memory_circuit,
)
from flagquantum.qec.codes import RepetitionCode, SteaneCode
from flagquantum.qec.logical import (
    certify_logical_product,
    derive_anticommuting_logical_product,
)
from flagquantum.qec.pauli import Pauli
from flagquantum.qec.surface import RotatedSurfaceCode, ZxxzSurfaceCode

pytestmark = pytest.mark.unit


class _NotAPauli:
    """A stand-in for a caller that passes something other than an operator."""


def test_certification_rejects_a_non_pauli_product() -> None:
    with pytest.raises(TypeError, match="Pauli operator"):
        certify_logical_product(
            RepetitionCode(3), _NotAPauli()  # type: ignore[arg-type]
        )


def test_certification_rejects_the_identity() -> None:
    with pytest.raises(ValueError, match="must not be the identity"):
        certify_logical_product(RepetitionCode(3), Pauli())


def test_certification_rejects_a_y_type_product() -> None:
    with pytest.raises(ValueError, match="pure X-type or pure Z-type"):
        certify_logical_product(RepetitionCode(3), Pauli(x_wires=(0,), z_wires=(0,)))


def test_certification_rejects_an_undeclared_wire() -> None:
    with pytest.raises(ValueError, match=r"not declared data\s+wires"):
        certify_logical_product(RepetitionCode(3), Pauli(z_wires=(9,)))


def test_certification_rejects_a_product_that_anticommutes_with_a_check() -> None:
    """``X0`` does not preserve the repetition code's code space."""

    with pytest.raises(ValueError, match="anticommutes with check 0"):
        certify_logical_product(RepetitionCode(3), Pauli(x_wires=(0,)))


def test_certification_rejects_a_product_inside_the_stabilizer_span() -> None:
    """``Z0*Z1`` is the repetition code's first check, so its outcome is fixed."""

    with pytest.raises(ValueError, match="stabilizer span"):
        certify_logical_product(RepetitionCode(3), Pauli(z_wires=(0, 1)))


def test_certification_accepts_a_declared_observable() -> None:
    product = Pauli(z_wires=(0, 1, 2))
    assert certify_logical_product(RepetitionCode(3), product) is product


def test_certification_accepts_the_derived_x_type_conjugate() -> None:
    product = Pauli(x_wires=(0, 1, 2))
    assert certify_logical_product(RepetitionCode(3), product) is product


def test_certification_accepts_a_stabilizer_multiple_of_a_logical_class() -> None:
    """``Z0`` differs from the declared logical by the code's second check."""

    product = Pauli(z_wires=(0,))
    assert certify_logical_product(RepetitionCode(3), product) is product


def test_certification_accepts_another_representative_of_the_other_class() -> None:
    """Any representative of the class is measurable, not just the derived one."""

    product = Pauli(x_wires=(0, 3, 6))
    assert certify_logical_product(RotatedSurfaceCode(3), product) is product


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        (RepetitionCode(3), (0, 1, 2)),
        (RepetitionCode(5), (0, 1, 2, 3, 4)),
        (RotatedSurfaceCode(2), (1, 3)),
        (RotatedSurfaceCode(3), (2, 5, 8)),
        (SteaneCode(), (2, 4, 5)),
        (ZxxzSurfaceCode(3), (0, 4, 8)),
        (ZxxzSurfaceCode(5), (0, 6, 12, 18, 24)),
        (ZxxzSurfaceCode(7), (0, 8, 16, 24, 32, 40, 48)),
    ),
)
def test_derivation_returns_a_weight_d_partner(
    code: RepetitionCode | RotatedSurfaceCode | SteaneCode | ZxxzSurfaceCode,
    expected: tuple[int, ...],
) -> None:
    derived = derive_anticommuting_logical_product(code, 0)

    assert derived == Pauli(x_wires=expected)
    assert derived.weight == code.distance
    assert certify_logical_product(code, derived) == derived
    assert not derived.commutes_with(code.logical_observables[0])


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_derivation_on_a_mixed_check_code_reads_the_z_factors_alone(
    distance: int,
) -> None:
    """Only a check's ``Z`` factor constrains an X-type candidate, and only it.

    The derivation is a linear system over the checks, and the row a mixed check
    contributes is the one its ``Z`` factor gives. Reading the row off the check's
    *support* instead -- both factors at once -- over-constrains the system, and
    the solution is then the operator on every data wire: measured here as the
    weight-``d ** 2`` operator that is the wrong answer this test names rather
    than as a count. The correct system's solution is the patch's own declared
    X-type observable, of weight ``d``, which is where the declaration and the
    derivation have to meet: the two halves of this file are otherwise independent
    of each other.

    The all-mixed control is asserted first, because on a code whose checks are
    pure the two readings agree and the test would prove nothing.
    """

    patch = ZxxzSurfaceCode(distance)
    declared_x = patch.logical_observables[1]
    derived = derive_anticommuting_logical_product(patch, 0)

    assert all(
        check.stabilizer.x_wires and check.stabilizer.z_wires for check in patch.checks
    )
    assert derived == declared_x
    assert derived.weight == distance
    assert derived != Pauli(x_wires=patch.data_wires)
    assert set(derived.support) < set(patch.data_wires)
    assert not derived.commutes_with(patch.logical_observables[0])
    assert certify_logical_product(patch, derived) == derived


def test_derivation_refuses_a_code_with_no_x_type_partner() -> None:
    """An even repetition code's declared all-Z operator is a product of checks."""

    with pytest.raises(ValueError, match="cannot be read out in the X basis"):
        derive_anticommuting_logical_product(RepetitionCode(4), 0)


def test_derivation_refuses_an_index_with_no_declared_z_type_observable() -> None:
    with pytest.raises(ValueError, match="no Z-type observable at index 1"):
        derive_anticommuting_logical_product(RotatedSurfaceCode(2), 1)


def test_derivation_refuses_a_non_integer_index() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        derive_anticommuting_logical_product(RepetitionCode(3), True)


def test_derivation_recovers_the_class_the_code_already_declares() -> None:
    """Steane declares both classes, so the derived partner is the declared one up
    to an X-type check, which is what makes their product a stabilizer."""

    code = SteaneCode()
    declared_z, declared_x = code.logical_observables
    derived = derive_anticommuting_logical_product(code, 0)

    assert not declared_z.commutes_with(declared_x)
    assert not declared_z.commutes_with(derived)
    with pytest.raises(ValueError, match="stabilizer span"):
        certify_logical_product(code, derived * declared_x)


@pytest.mark.parametrize(
    "code",
    (RepetitionCode(3), RotatedSurfaceCode(2), SteaneCode()),
)
def test_declared_readout_reproduces_the_default_experiment(
    code: RepetitionCode | RotatedSurfaceCode | SteaneCode,
) -> None:
    """Asking for the declared logical explicitly is the default, not a variant."""

    default = build_memory_circuit(code, rounds=2)
    explicit = build_memory_circuit(code, rounds=2, product=code.logical_observables[0])

    assert explicit.source == default.source
    assert explicit.detectors == default.detectors
    assert explicit.observables == default.observables
    assert explicit.x_readout_wires == ()


def test_rotated_source_rotates_both_ends_and_nothing_else() -> None:
    rotated = build_memory_circuit(
        RepetitionCode(3), rounds=2, product=Pauli(x_wires=(0, 1, 2))
    )
    rotations = ["    qp.H(wires=0)\n", "    qp.H(wires=1)\n", "    qp.H(wires=2)\n"]

    head = "def memory_experiment(rounds):\n    last = False\n"
    body = (
        "    for round_index in range(rounds):\n"
        "        qp.CNOT(wires=[0, 3])\n"
        "        qp.CNOT(wires=[1, 3])\n"
        "        last = qp.measure(wires=3)\n"
        "        qp.reset(wires=3)\n"
        "        qp.CNOT(wires=[1, 4])\n"
        "        qp.CNOT(wires=[2, 4])\n"
        "        last = qp.measure(wires=4)\n"
        "        qp.reset(wires=4)\n"
    )
    expected = (
        head + "".join(rotations) + body + "".join(rotations) + "    return last\n"
    )
    assert rotated.source == expected


def test_rotating_a_subset_leaves_the_other_wires_in_the_z_basis() -> None:
    rotated = build_memory_circuit(
        RotatedSurfaceCode(3), rounds=2, product=Pauli(x_wires=(0, 3, 6))
    )
    counts: dict[int, int] = {}
    for line in rotated.source.splitlines():
        prefix = "    qp.H(wires="
        if line.startswith(prefix):
            wire = int(line[len(prefix) : -1])
            counts[wire] = counts.get(wire, 0) + 1

    for wire in (0, 3, 6):
        assert counts[wire] == 2
    for wire in (1, 2, 4, 5, 7, 8):
        assert counts.get(wire, 0) == 0
    assert rotated.x_readout_wires == (0, 3, 6)


def test_rotating_a_wire_already_rotated_by_its_check_adds_two_more_gates() -> None:
    """An X-type check rotates its own ancilla; that is not a readout rotation."""

    default = build_memory_circuit(RotatedSurfaceCode(3), rounds=2)
    rotated = build_memory_circuit(
        RotatedSurfaceCode(3), rounds=2, product=Pauli(x_wires=(0, 3, 6))
    )

    assert (
        rotated.source.count("qp.H(wires=") == default.source.count("qp.H(wires=") + 6
    )


def test_a_rotated_readout_declares_one_observable_on_exactly_its_x_wires() -> None:
    rotated = build_memory_circuit(
        RotatedSurfaceCode(3), rounds=2, product=Pauli(x_wires=(0, 3, 6))
    )
    (observable,) = rotated.observables.observables

    assert observable.index == 0
    assert observable.pauli == Pauli(x_wires=(0, 3, 6))
    assert observable.measurement_parity == (
        MeasurementRef(None, 0),
        MeasurementRef(None, 3),
        MeasurementRef(None, 6),
    )


def test_a_rotated_readout_keeps_only_the_checks_the_preparation_pins() -> None:
    """A Z-type check survives round zero only if no wire of it was rotated."""

    default = build_memory_circuit(RotatedSurfaceCode(3), rounds=2)
    rotated = build_memory_circuit(
        RotatedSurfaceCode(3), rounds=2, product=Pauli(x_wires=(0, 3, 6))
    )
    pinned_ancillas = (14, 16)

    def pinned(detector: Detector) -> bool:
        """Whether a default-frame detector needs a check the preparation pins.

        A round-to-round comparison is deterministic for every check, because it
        compares a check against its own previous value. Only the two boundary
        detectors -- the round-zero one and the terminal one -- need the
        preparation to pin the check, and those are the two whose parity names
        every wire of the check rather than just its syndrome history.
        """

        references = detector.parity
        if references[0].round_index == 0 and len(references) == 1:
            return references[0].wire in pinned_ancillas
        if any(reference.round_index is None for reference in references):
            return references[0].wire in pinned_ancillas
        return True

    assert len(default.detectors) == 16
    assert len(rotated.detectors) == 12
    assert tuple(detector.parity for detector in rotated.detectors.detectors) == tuple(
        detector.parity for detector in default.detectors.detectors if pinned(detector)
    )
    assert (
        tuple(
            detector.parity[0].wire
            for detector in rotated.detectors.detectors
            if detector.parity[0].round_index == 0
        )
        == pinned_ancillas
    )
    assert tuple(item.index for item in rotated.detectors.detectors) == tuple(range(12))


def test_a_fully_rotated_repetition_code_loses_every_pinned_check() -> None:
    """Every check becomes volatile, so only round-to-round detectors remain."""

    rotated = build_memory_circuit(
        RepetitionCode(3), rounds=3, product=Pauli(x_wires=(0, 1, 2))
    )

    assert len(rotated.detectors) == 4
    for detector in rotated.detectors.detectors:
        assert tuple(ref.round_index for ref in detector.parity) == (1, 0) or tuple(
            ref.round_index for ref in detector.parity
        ) == (2, 1)


def test_a_rotated_steane_patch_keeps_the_check_that_misses_the_rotated_wires() -> None:
    rotated = build_memory_circuit(
        SteaneCode(), rounds=2, product=Pauli(x_wires=(0, 1, 2))
    )

    assert len(rotated.detectors) == 8
    assert rotated.detectors.detectors[0].parity == (MeasurementRef(0, 7),)
    assert rotated.detectors.detectors[-1].parity == (
        MeasurementRef(1, 7),
        MeasurementRef(None, 3),
        MeasurementRef(None, 4),
        MeasurementRef(None, 5),
        MeasurementRef(None, 6),
    )


def test_a_single_round_rotated_experiment_with_no_pinned_check_is_refused() -> None:
    with pytest.raises(ValueError, match="increase the round count"):
        build_memory_circuit(
            RepetitionCode(3), rounds=1, product=Pauli(x_wires=(0, 1, 2))
        )


def test_builder_refuses_an_uncertified_product() -> None:
    with pytest.raises(ValueError, match="anticommutes with check 0"):
        build_memory_circuit(RepetitionCode(3), rounds=2, product=Pauli(x_wires=(0,)))


def test_memory_circuit_refuses_an_x_observable_whose_wires_were_not_rotated() -> None:
    code = SteaneCode()
    built = build_memory_circuit(
        code, rounds=2, product=derive_anticommuting_logical_product(code, 0)
    )
    elsewhere = LogicalObservable(
        index=0,
        pauli=Pauli(x_wires=tuple(code.data_wires)),
        measurement_parity=tuple(
            MeasurementRef(None, wire) for wire in code.data_wires
        ),
    )

    with pytest.raises(ValueError, match="exactly the wires the experiment"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=ObservableLayout((elsewhere,)),
            x_readout_wires=built.x_readout_wires,
        )


def test_memory_circuit_refuses_two_observables_in_a_rotated_readout() -> None:
    built = build_memory_circuit(
        RepetitionCode(3), rounds=2, product=Pauli(x_wires=(0, 1, 2))
    )
    doubled = ObservableLayout(
        (
            LogicalObservable(
                index=0,
                pauli=Pauli(x_wires=(0, 1, 2)),
                measurement_parity=(
                    MeasurementRef(None, 0),
                    MeasurementRef(None, 1),
                    MeasurementRef(None, 2),
                ),
            ),
            LogicalObservable(
                index=1,
                pauli=Pauli(z_wires=(0, 1, 2)),
                measurement_parity=(
                    MeasurementRef(None, 0),
                    MeasurementRef(None, 1),
                    MeasurementRef(None, 2),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="exactly one observable"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=doubled,
            x_readout_wires=built.x_readout_wires,
        )


def test_memory_circuit_refuses_an_undeclared_readout_wire() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=2)

    with pytest.raises(ValueError, match="declared data wires"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=built.observables,
            x_readout_wires=(0, 9),
        )


def test_memory_circuit_refuses_an_unordered_readout_frame() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=2)

    with pytest.raises(ValueError, match="unique and in ascending order"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=built.observables,
            x_readout_wires=(1, 0),
        )


def test_the_declared_frame_still_refuses_a_non_z_type_observable() -> None:
    """The Z-basis route keeps its original refusal; only the rotation widens it."""

    built = build_memory_circuit(RepetitionCode(3), rounds=2)
    x_type = LogicalObservable(
        index=0,
        pauli=Pauli(x_wires=(0, 1, 2)),
        measurement_parity=(
            MeasurementRef(None, 0),
            MeasurementRef(None, 1),
            MeasurementRef(None, 2),
        ),
    )

    with pytest.raises(ValueError, match="declared logical observable"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=ObservableLayout((x_type,)),
        )


def test_memory_circuit_refuses_an_uncertified_rotated_observable() -> None:
    """A rotated record is held to the certificate, not only the builder.

    No builder reaches this frame: the surface patch of distance two has no
    certified X-type product on a single wire, because such a product
    anticommutes with the patch's weight-four Z check. The frame is stated here
    directly so that the record's own certificate is what has to refuse it.
    """

    code = RotatedSurfaceCode(2)
    observable = LogicalObservable(
        index=0,
        pauli=Pauli(x_wires=(0,)),
        measurement_parity=(MeasurementRef(None, 0),),
    )

    with pytest.raises(ValueError, match="anticommutes with check 1"):
        MemoryCircuit(
            code=code,
            rounds=2,
            source=_check_source(code, x_readout=(0,)),
            detectors=_detector_layout(code, rounds=2, x_readout=(0,)),
            observables=ObservableLayout((observable,)),
            x_readout_wires=(0,),
        )


def test_memory_circuit_refuses_a_non_integer_readout_wire() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=2)

    with pytest.raises(TypeError, match="X readout wires must be integers"):
        type(built)(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=built.detectors,
            observables=built.observables,
            x_readout_wires=(1.5,),  # type: ignore[arg-type]
        )


def test_a_derived_conjugate_is_the_only_readout_a_repetition_patch_offers() -> None:
    """A repetition code has one logical class, so its X readout is the conjugate."""

    code = RepetitionCode(3)
    products = [check.stabilizer for check in code.checks]
    assert products == [Pauli(z_wires=(0, 1)), Pauli(z_wires=(1, 2))]

    conjugate = Pauli(x_wires=(0, 1, 2))
    assert certify_logical_product(code, conjugate) == conjugate
    assert code.logical_observables[0] == Pauli(z_wires=(0, 1, 2))


def test_a_surface_patch_reads_out_the_class_it_does_not_declare() -> None:
    """Rotating a patch reads the other logical class, not a relabelled copy."""

    code = RotatedSurfaceCode(2)
    (declared,) = code.logical_observables
    partner = derive_anticommuting_logical_product(code, 0)

    assert declared.z_wires and not declared.x_wires
    assert partner.x_wires and not partner.z_wires
    assert not declared.commutes_with(partner)
    assert code.logical_observables == (declared,)


def test_a_repetition_patch_declares_and_reads_out_the_two_classes_of_one_wire() -> (
    None
):
    """A repetition code has one logical qubit, and both classes act on all wires."""

    code = RepetitionCode(3)
    (declared,) = code.logical_observables
    partner = derive_anticommuting_logical_product(code, 0)

    assert declared == Pauli(z_wires=(0, 1, 2))
    assert partner == Pauli(x_wires=(0, 1, 2))
