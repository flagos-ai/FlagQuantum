"""Unit coverage for code descriptions and the repetition-code reference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum.qec.codes import CodeCheck, RepetitionCode, StabilizerCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


def test_default_distance_matches_the_reference_profile() -> None:
    assert RepetitionCode().distance == 3


def test_distance_three_wire_layout() -> None:
    code = RepetitionCode(3)

    assert code.num_data_qubits == 3
    assert code.num_ancilla_qubits == 2
    assert code.data_wires == (0, 1, 2)
    assert code.ancilla_wires == (3, 4)


def test_distance_three_checks_are_adjacent_z_pairs() -> None:
    code = RepetitionCode(3)

    assert len(code.checks) == 2
    for index, check in enumerate(code.checks):
        assert check.index == index
        assert check.stabilizer == Pauli(z_wires=(index, index + 1))
        assert check.ancilla_wire == 3 + index
        assert check.cnot_wires == ((index, 3 + index), (index + 1, 3 + index))


def test_stabilizers_are_the_declared_check_operators() -> None:
    code = RepetitionCode(5)

    assert code.stabilizers == tuple(check.stabilizer for check in code.checks)
    assert code.stabilizers == (
        Pauli(z_wires=(0, 1)),
        Pauli(z_wires=(1, 2)),
        Pauli(z_wires=(2, 3)),
        Pauli(z_wires=(3, 4)),
    )


def test_logical_observable_is_z_type_and_commutes_with_every_stabilizer() -> None:
    code = RepetitionCode(5)

    (observable,) = code.logical_observables
    assert observable == Pauli(z_wires=(0, 1, 2, 3, 4))
    assert not observable.x_wires
    assert all(observable.commutes_with(item) for item in code.stabilizers)


def test_scaled_layouts_keep_checks_and_ancillas_contiguous() -> None:
    code = RepetitionCode(7)

    assert code.num_ancilla_qubits == 6
    assert code.ancilla_wires == (7, 8, 9, 10, 11, 12)
    assert len(code.checks) == 6
    assert code.checks[-1].cnot_wires == ((5, 12), (6, 12))


def test_distance_two_is_the_smallest_supported_code() -> None:
    code = RepetitionCode(2)

    assert code.num_ancilla_qubits == 1
    assert code.ancilla_wires == (2,)
    assert len(code.checks) == 1
    assert code.checks[0].stabilizer == Pauli(z_wires=(0, 1))


@pytest.mark.parametrize("distance", (1, 0, -3))
def test_distances_below_two_are_rejected(distance: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        RepetitionCode(distance)


@pytest.mark.parametrize("distance", (2.5, "3", None, True))
def test_non_integer_distances_are_rejected(distance: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        RepetitionCode(distance)  # type: ignore[arg-type]


def test_repetition_code_satisfies_the_stabilizer_code_protocol() -> None:
    assert isinstance(RepetitionCode(3), StabilizerCode)


def test_repetition_codes_are_frozen_and_hashable() -> None:
    assert RepetitionCode(3) == RepetitionCode(3)
    assert len({RepetitionCode(3), RepetitionCode(3)}) == 1
    with pytest.raises(FrozenInstanceError):
        RepetitionCode(3).distance = 5  # type: ignore[misc]


def test_check_rejects_a_cnot_that_does_not_target_the_ancilla() -> None:
    with pytest.raises(ValueError, match="target"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 1), (1, 2)),
        )


def test_check_rejects_a_cnot_that_controls_the_ancilla() -> None:
    with pytest.raises(ValueError, match="control"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((2, 2),),
        )


def test_check_rejects_an_empty_cnot_schedule() -> None:
    with pytest.raises(ValueError, match="at least one CNOT"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=(),
        )


def test_check_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        CodeCheck(
            index=-1,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        )


@pytest.mark.parametrize(
    "stabilizer",
    (
        Pauli(z_wires=(5, 6)),
        Pauli(z_wires=(0,)),
        Pauli(z_wires=(0, 1, 2)),
        Pauli(),
    ),
)
def test_check_rejects_a_stabilizer_that_disagrees_with_the_cnot_controls(
    stabilizer: Pauli,
) -> None:
    with pytest.raises(ValueError, match="support"):
        CodeCheck(
            index=0,
            stabilizer=stabilizer,
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        )


def test_check_rejects_a_duplicated_cnot_schedule() -> None:
    with pytest.raises(ValueError, match="support"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0,)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (0, 2)),
        )


def test_check_rejects_an_identity_stabilizer_with_cnots() -> None:
    with pytest.raises(ValueError, match="support"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        )


def test_check_accepts_an_x_type_stabilizer_with_the_mirrored_direction() -> None:
    """An X-type check is expressed by reversing the CNOT direction.

    The gadget is the mirror of the Z-type one: the ancilla is prepared in
    ``|+>`` and controls into the stabilizer's support. Before this direction was
    expressible the record refused every X-type stabilizer, and the old refusal
    is kept below as a premise so the reason the guard existed stays visible.
    """

    check = CodeCheck(
        index=0,
        stabilizer=Pauli(x_wires=(0, 1)),
        ancilla_wire=2,
        cnot_wires=((2, 0), (2, 1)),
    )

    assert check.stabilizer == Pauli(x_wires=(0, 1))
    assert tuple(control for control, _ in check.cnot_wires) == (2, 2)
    assert tuple(target for _, target in check.cnot_wires) == (0, 1)


def test_check_still_rejects_an_x_type_stabilizer_measured_the_z_type_way() -> None:
    """The old refusal survives as a direction check rather than as a type check.

    An X-type stabilizer whose CNOTs point the Z-type way is still refused,
    because measuring it that way would collapse the ancilla instead of
    collecting the check's eigenvalue.
    """

    with pytest.raises(ValueError, match="target data wires"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(x_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        )


def test_check_accepts_a_mixed_x_and_z_stabilizer_on_one_ancilla() -> None:
    """One ancilla measures a mixed stabilizer, with the CNOTs in factor order.

    A mixed check used to be refused on the premise that it needs a second
    ancilla. It does not. The Z-factor pairs come first with the data wire
    controlling, the X-factor pairs follow with the ancilla controlling, and the
    single ``H`` pair the emission wraps around the second run is what makes the
    same ancilla's Z-basis readout equal to the product's eigenvalue. The two
    runs are what makes the order load-bearing: it is the boundary between them
    that says which pairs the ``H`` gates wrap, so a permutation of the pairs is
    a different operator rather than a different spelling of this one.
    """

    check = CodeCheck(
        index=0,
        stabilizer=Pauli(x_wires=(0,), z_wires=(1,)),
        ancilla_wire=2,
        cnot_wires=((1, 2), (2, 0)),
    )

    assert check.cnot_wires == ((1, 2), (2, 0))
    assert check.stabilizer == Pauli(x_wires=(0,), z_wires=(1,))


def test_check_rejects_a_mixed_stabilizer_whose_x_run_comes_first() -> None:
    """The run order is checked, and the mirrored order is not this check.

    ``(ancilla, data)`` then ``(data, ancilla)`` describes the same two CNOTs
    read in the opposite order, and the emission would wrap its ``H`` pair around
    the wrong half, so it is refused rather than silently reordered.
    """

    with pytest.raises(ValueError, match="after the Z-factor pairs"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(x_wires=(0,), z_wires=(1,)),
            ancilla_wire=2,
            cnot_wires=((2, 0), (1, 2)),
        )


def test_check_rejects_an_x_type_stabilizer_with_cnots_on_the_wrong_wires() -> None:
    with pytest.raises(ValueError, match="support"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(x_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((2, 0), (2, 3)),
        )


def test_check_rejects_an_x_type_stabilizer_whose_cnots_stay_on_the_data() -> None:
    """The X gadget's ancilla must be the control of every CNOT.

    Writing the X-type check as CNOTs among the data wires keeps the targets
    equal to the support, so only the direction guard distinguishes it from the
    gadget that actually measures the stabilizer.
    """

    with pytest.raises(ValueError, match="controlled by the declared ancilla"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(x_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 1), (1, 0)),
        )


def test_check_rejects_a_z_type_check_that_controls_from_its_ancilla() -> None:
    """A Z-type check must not treat its ancilla as a data wire.

    The support here happens to contain the ancilla wire, which is what leaves
    the direction guard as the only guard that rejects the record.
    """

    with pytest.raises(ValueError, match="data wires, not the ancilla"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1, 2)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2), (2, 2)),
        )


def test_public_namespace_publishes_the_code_records() -> None:
    import flagquantum.qec as qec

    expected = (
        "CodeCheck",
        "Pauli",
        "RepetitionCode",
        "RotatedSurfaceCode",
        "StabilizerCode",
    )
    missing = [name for name in expected if not hasattr(qec, name)]
    assert not missing, f"flagquantum.qec is missing {missing}"
    for name in expected:
        assert name in qec.__all__, f"{name} is not in flagquantum.qec.__all__"
